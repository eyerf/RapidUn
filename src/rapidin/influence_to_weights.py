"""
Map RapidIn influence scores (helpful and optionally harmful) to per-sample training weights.

Overview of the pipeline (for each group: forget / retain):

1) Load influence results from four views:
   - F2F: forget test → forget train
   - F2R: forget test → retain train
   - R2F: retain test → forget train
   - R2R: retain test → retain train
   Each test entry may contain:
     - 'helpful' (train indices), 'helpful_infl' (influence values)
     - optionally 'harmful', 'harmful_infl' (influence values)

2) For each view, aggregate per test:
   - optionally keep top-k entries per test
   - use either rank-based scores (k, k-1, ...) or raw values
   - optionally apply robust scaling (median / MAD) within each test
   - aggregate to a per-train vector via elementwise median across tests

3) Combine cross-view scores for each group:
   - helpful-only (default):
       Sf =  alpha * FF_help  -  beta  * FR_help
       Sr =  gamma * RR_help  -  delta * RF_help
   - if harmful components are enabled:
       Sf -= (alpha_harm * FF_harm + beta_harm  * FR_harm)
       Sr -= (gamma_harm * RR_harm + delta_harm * RF_harm)

4) Apply robust scaling (median / MAD) to Sf and Sr to obtain z-scores.

5) Map z-scores to weights using exponential mapping with per-group temperature
   and clipping:
   - wf = exp_clip_map(Sf_z, tau_f, wmin_f, wmax_f)
   - wr = exp_clip_map(Sr_z, tau_r, wmin_r, wmax_r)

6) Mean-normalize each group so that the average weight is 1.0.

7) Save weights:
   - forget_weights.jsonl  (one JSON object per line: {"index": i, "weight": w})
   - retain_weights.jsonl  (one JSON object per line: {"index": j, "weight": w})
"""

import json
import math
import argparse
from statistics import median
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable

import numpy as np

EPS = 1e-8


# ------------------------ Utilities ------------------------

def robust_scale(xs: List[float], mad_floor: float = 1e-3, use_sigma: bool = True) -> List[float]:
    """
    Compute a robust z-score: (x - median) / (1.4826 * MAD), with a lower bound on MAD.
    """
    if not xs:
        return []
    m = median(xs)
    abs_dev = [abs(x - m) for x in xs]
    mad = median(abs_dev)
    if use_sigma:
        mad *= 1.4826
    mad = max(mad, mad_floor)
    return [(x - m) / mad for x in xs]


def load_infl_file(path: str) -> List[Dict[str, Any]]:
    """
    Load influence results from a file.

    Supported formats:
      (A) Single JSON object:
          {"config": "...", "0": {...}, "1": {...}, ...}
      (B) JSONL file with one test entry per line.

    Returns a list of entries, where each entry may contain
    'helpful' / 'helpful_infl' and optionally 'harmful' / 'harmful_infl'.
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    if not txt:
        return []

    # Try single JSON object
    try:
        obj = json.loads(txt)
        entries = []
        for k, v in obj.items():
            if k == "config":
                continue
            if isinstance(v, dict) and any(key in v for key in ("helpful", "harmful")):
                entries.append(v)
        if entries:
            return entries
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL
    entries = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if any(key in o for key in ("helpful", "harmful")):
                entries.append(o)
    return entries


def _pairs_from_entry(entry: Dict[str, Any],
                      key_idx: str,
                      key_val: str) -> List[Tuple[int, float]]:
    """
    Extract (index, value) pairs from an entry using the given keys.

    If only indices or only values are present, the missing part
    is synthesized (indices as 0..N-1 or values as 1.0).
    """
    idxs = entry.get(key_idx, []) or []
    vals = entry.get(key_val, []) or []
    if len(idxs) != len(vals):
        if vals and not idxs:
            idxs = list(range(len(vals)))
        elif idxs and not vals:
            vals = [1.0] * len(idxs)
    try:
        pairs = [(int(i), float(v)) for i, v in zip(idxs, vals)]
    except Exception:
        pairs = []
    return pairs


def aggregate_one_kind(entries: List[Dict[str, Any]],
                       n_train: int,
                       mode: str = "helpful",
                       topk: int = None,
                       per_test_rank: bool = True,
                       per_test_robust: bool = False) -> List[float]:
    """
    Aggregate 'helpful' or 'harmful' influence into a per-train vector.

    Per test:
      - extract (idx, value) pairs using the given mode
      - optionally apply robust scaling within the test (value mode only)
      - sort by value in descending order
      - optionally keep top-k entries
      - accumulate into a vector of length n_train:
          * rank mode: top entry gets k, next gets k-1, etc.
          * value mode: sum raw or scaled values

    Across tests:
      - apply elementwise median to obtain a robust per-train vector.

    Returns a list of length n_train.
    """
    if n_train <= 0:
        return []

    key_idx = f"{mode}"
    key_val = f"{mode}_infl"

    per_test_scores = []
    for item in entries:
        pairs = _pairs_from_entry(item, key_idx, key_val)
        if not pairs:
            continue

        # Optional per-test robust scaling (value mode only)
        if not per_test_rank and per_test_robust:
            vals = [v for _, v in pairs]
            vals_rob = robust_scale(vals)
            pairs = [(idx, v_rob) for (idx, _), v_rob in zip(pairs, vals_rob)]

        pairs.sort(key=lambda x: float(x[1]), reverse=True)
        if topk is not None:
            pairs = pairs[:topk]

        vec = [0.0] * n_train
        if per_test_rank:
            k = len(pairs)
            for r, (idx, _) in enumerate(pairs):
                if 0 <= idx < n_train:
                    vec[idx] += float(k - r)
        else:
            for idx, val in pairs:
                if 0 <= idx < n_train:
                    vec[idx] += float(val)
        per_test_scores.append(vec)

    if not per_test_scores:
        return [0.0] * n_train

    mat = np.array(per_test_scores, dtype=float)  # shape [num_tests, n_train]
    med = np.median(mat, axis=0)
    return med.tolist()


def exp_clip_map(scores: Iterable[float],
                 tau: float = 1.0,
                 wmin: float = 0.2,
                 wmax: float = 5.0) -> List[float]:
    """
    Map scores to weights via w = exp(s / tau), with clipping in log-space
    to keep weights within [wmin, wmax].
    """
    tau = max(float(tau), EPS)
    log_wmin = -float("inf") if wmin is None else math.log(max(wmin, EPS))
    log_wmax = float("inf") if wmax is None else math.log(max(wmax, EPS))

    out = []
    for s in scores:
        z = s / tau
        # Clamp in log-space to keep exp(z) within [wmin, wmax].
        z = min(max(z, log_wmin), log_wmax)
        out.append(math.exp(z))
    return out


def mean_normalize(weights: List[float]) -> List[float]:
    """
    Normalize weights so that their mean is 1.0.
    """
    if not weights:
        return weights
    m = sum(weights) / max(len(weights), 1)
    if m <= 0:
        return weights
    return [w / m for w in weights]


def save_jsonl(path: str, weights: List[float]) -> None:
    """
    Save weights to a JSONL file, one entry per line with fields:
      - index: sample index
      - weight: weight value
    """
    with Path(path).open("w", encoding="utf-8") as f:
        for idx, w in enumerate(weights):
            f.write(json.dumps({"index": idx, "weight": float(w)}, ensure_ascii=False) + "\n")


# ------------------------ Main ------------------------

def main(args):
    # 1) Load four influence result files
    F2F_entries = load_infl_file(args.f2f)
    F2R_entries = load_infl_file(args.f2r)
    R2F_entries = load_infl_file(args.r2f)
    R2R_entries = load_infl_file(args.r2r)

    # 2) Aggregate helpful components
    FF_help = aggregate_one_kind(F2F_entries, args.n_forget,
                                 mode="helpful",
                                 topk=args.topk,
                                 per_test_rank=args.rank,
                                 per_test_robust=args.per_test_robust)
    FR_help = aggregate_one_kind(F2R_entries, args.n_forget,
                                 mode="helpful",
                                 topk=args.topk,
                                 per_test_rank=args.rank,
                                 per_test_robust=args.per_test_robust)
    RF_help = aggregate_one_kind(R2F_entries, args.n_retain,
                                 mode="helpful",
                                 topk=args.topk,
                                 per_test_rank=args.rank,
                                 per_test_robust=args.per_test_robust)
    RR_help = aggregate_one_kind(R2R_entries, args.n_retain,
                                 mode="helpful",
                                 topk=args.topk,
                                 per_test_rank=args.rank,
                                 per_test_robust=args.per_test_robust)

    # 3) Optionally aggregate harmful components
    if args.use_harmful:
        FF_harm = aggregate_one_kind(F2F_entries, args.n_forget,
                                     mode="harmful",
                                     topk=args.topk,
                                     per_test_rank=args.rank,
                                     per_test_robust=args.per_test_robust)
        FR_harm = aggregate_one_kind(F2R_entries, args.n_forget,
                                     mode="harmful",
                                     topk=args.topk,
                                     per_test_rank=args.rank,
                                     per_test_robust=args.per_test_robust)
        RF_harm = aggregate_one_kind(R2F_entries, args.n_retain,
                                     mode="harmful",
                                     topk=args.topk,
                                     per_test_rank=args.rank,
                                     per_test_robust=args.per_test_robust)
        RR_harm = aggregate_one_kind(R2R_entries, args.n_retain,
                                     mode="harmful",
                                     topk=args.topk,
                                     per_test_rank=args.rank,
                                     per_test_robust=args.per_test_robust)
    else:
        FF_harm = FR_harm = RF_harm = RR_harm = None

    # 4) Combine helpful scores
    Sf = [args.alpha * ff - args.beta * fr for ff, fr in zip(FF_help, FR_help)]
    Sr = [args.gamma * rr - args.delta * rf for rr, rf in zip(RR_help, RF_help)]

    # Subtract harmful contributions if enabled
    if args.use_harmful and FF_harm is not None:
        Sf = [s - (args.alpha_harm * f_h + args.beta_harm * r_h)
              for s, f_h, r_h in zip(Sf, FF_harm, FR_harm)]
        Sr = [s - (args.gamma_harm * r_h + args.delta_harm * f_h)
              for s, r_h, f_h in zip(Sr, RR_harm, RF_harm)]

    # 5) Robust scaling per group (z-scores)
    Sf_z = robust_scale(Sf) if len(set(Sf)) > 1 else [0.0] * len(Sf)
    Sr_z = robust_scale(Sr) if len(set(Sr)) > 1 else [0.0] * len(Sr)

    # 6) Map to weights with separate parameters for forget / retain
    wf = exp_clip_map(Sf_z, tau=args.tau_f, wmin=args.wmin_f, wmax=args.wmax_f)
    wr = exp_clip_map(Sr_z, tau=args.tau_r, wmin=args.wmin_r, wmax=args.wmax_r)

    # 7) Mean-normalize each group to have average weight 1.0
    wf = mean_normalize(wf)
    wr = mean_normalize(wr)

    # 8) Save weights
    save_jsonl(args.out_forget, wf)
    save_jsonl(args.out_retain, wr)

    print(f"[Done] Saved forget weights to {args.out_forget} (N={len(wf)})")
    print(f"[Done] Saved retain  weights to {args.out_retain} (N={len(wr)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map RapidIn influence scores to per-sample weights")

    # Paths
    parser.add_argument("--f2f", type=str, required=True, help="forget-test → forget-train influence file")
    parser.add_argument("--f2r", type=str, required=True, help="forget-test → retain-train influence file")
    parser.add_argument("--r2f", type=str, required=True, help="retain-test → forget-train influence file")
    parser.add_argument("--r2r", type=str, required=True, help="retain-test → retain-train influence file")

    # Train set sizes
    parser.add_argument("--n_forget", type=int, required=True, help="Number of forget train samples")
    parser.add_argument("--n_retain", type=int, required=True, help="Number of retain train samples")

    # Aggregation controls
    parser.add_argument("--topk", type=int, default=None, help="Top-k entries per test (None for all)")
    parser.add_argument("--rank", action="store_true", help="Use rank-based scores per test (default: value-based)")
    parser.add_argument("--no-rank", dest="rank", action="store_false")
    parser.set_defaults(rank=False)
    parser.add_argument(
        "--per_test_robust",
        action="store_true",
        help="Apply robust scaling within each test (value mode only) before top-k and summation"
    )

    # Score combination coefficients (helpful)
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for F→F helpful scores")
    parser.add_argument("--beta",  type=float, default=1.0, help="Coefficient for F→R helpful scores (penalty)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Coefficient for R→R helpful scores")
    parser.add_argument("--delta", type=float, default=1.0, help="Coefficient for R→F helpful scores (penalty)")

    # Optional harmful components
    parser.add_argument("--use_harmful", action="store_true", help="Include harmful components in the combined scores")
    parser.add_argument("--alpha_harm", type=float, default=1.0, help="Coefficient for F→F harmful scores (penalty)")
    parser.add_argument("--beta_harm",  type=float, default=1.0, help="Coefficient for F→R harmful scores (penalty)")
    parser.add_argument("--gamma_harm", type=float, default=1.0, help="Coefficient for R→R harmful scores (penalty)")
    parser.add_argument("--delta_harm", type=float, default=1.0, help="Coefficient for R→F harmful scores (penalty)")

    # Mapping parameters (separate for forget / retain)
    parser.add_argument("--tau_f",  type=float, default=1.0, help="Temperature for forget weights")
    parser.add_argument("--tau_r",  type=float, default=1.0, help="Temperature for retain weights")
    parser.add_argument("--wmin_f", type=float, default=1.0, help="Minimum forget weight after exponential mapping")
    parser.add_argument("--wmax_f", type=float, default=3.0, help="Maximum forget weight after exponential mapping")
    parser.add_argument("--wmin_r", type=float, default=0.5, help="Minimum retain weight after exponential mapping")
    parser.add_argument("--wmax_r", type=float, default=3.0, help="Maximum retain weight after exponential mapping")

    # Output paths
    parser.add_argument("--out_forget", type=str, default="forget_weights.jsonl")
    parser.add_argument("--out_retain", type=str, default="retain_weights.jsonl")

    args = parser.parse_args()
    main(args)
