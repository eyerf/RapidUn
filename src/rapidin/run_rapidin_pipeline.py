#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified RapidIn pipeline:
1) Cache gradients for retain / forget sets.
2) Run 4 influence estimations (F→F, F→R, R→F, R→R).
3) Rename influence result files to *_FTF/FTR/RTF/RTR.json.
4) Call the RapidIn→LoReUn weight mapping script to obtain forget / retain weights.
5) (Optional) Clean up intermediate gradient directories and raw influence_results_{N}.json files.

Usage (YAML-driven, recommended):

conda run -n RapidIn python run_rapidin_pipeline.py \
  --config configs/rapidin_llama3_uniform.yaml
"""

import argparse
import json
import subprocess
import shutil
from pathlib import Path

import yaml  # ensure PyYAML is installed: pip install pyyaml


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def count_jsonl(path_str: str) -> int:
    """Count non-empty lines in a JSONL file (used for n_forget / n_retain and filenames)."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def run_mp_main(mp_main: Path, config_path: Path):
    print(f"[RUN] MP_main.py with config = {config_path}")
    subprocess.run(
        ["python", str(mp_main), "--config", str(config_path)],
        check=True,
    )


def build_caching_config(train_path: str,
                         grads_path: str,
                         outdir: str,
                         model_path: str,
                         max_length: int,
                         seed: int,
                         load_in_4bit: bool,
                         rapidgrad_k: int,
                         shuffle_lambda: int):
    """
    Build config for the caching stage (only saves gradients to grads_path, no influence computation).
    """
    return {
        "data": {
            "train_data_path": train_path
        },
        "influence": {
            "outdir": outdir,
            "seed": seed,
            "cal_words_infl": False,
            "save_to_grads_path": True,
            "n_threads": 1,
            "RapidGrad": {
                "enable": True,
                "RapidGrad_K": rapidgrad_k,
                "shuffle_lambda": shuffle_lambda
            },
            "offload_train_grad": False,
            "skip_test": True,
            "skip_influence": True,
            "grads_path": grads_path
        },
        "model": {
            "model_path": model_path,
            "lora_path": None,
            "max_length": max_length,
            "load_in_4bit": bool(load_in_4bit)
        }
    }


def build_retrieval_config(train_path: str,
                           test_path: str,
                           grads_path: str,
                           outdir: str,
                           model_path: str,
                           max_length: int,
                           seed: int,
                           load_in_4bit: bool,
                           rapidgrad_k: int,
                           shuffle_lambda: int,
                           top_k: int):
    """
    Build config for the influence estimation stage
    (loads gradients from grads_path and computes influence).
    """
    return {
        "data": {
            "train_data_path": train_path,
            "test_data_path": test_path
        },
        "influence": {
            "outdir": outdir,
            "seed": seed,
            "cal_words_infl": False,
            "n_threads": 1,
            "RapidGrad": {
                "enable": True,
                "RapidGrad_K": rapidgrad_k,
                "shuffle_lambda": shuffle_lambda
            },
            "offload_test_grad": False,
            "offload_train_grad": False,
            "delete_model": True,
            "calculate_infl_in_gpu": True,
            "load_from_grads_path": True,
            "save_to_grads_path": False,
            "grads_path": grads_path,
            "top_k": top_k
        },
        "model": {
            "model_path": model_path,
            "lora_path": None,
            "max_length": max_length,
            "load_in_4bit": bool(load_in_4bit)
        }
    }


def rename_influence_file(outdir: Path,
                          n_train: int,
                          suffix: str) -> Path:
    """
    After one influence estimation run, RapidIn writes files like:
      influence_results_{n_train}_YYYY-MM-DD-HH-MM-SS.json

    This function:
      1) Finds all files matching influence_results_{n_train}_*.json.
      2) Picks the latest one by modification time.
      3) Renames it to influence_results_{n_train}_{suffix}.json
         (e.g. influence_results_40_FTF.json).
    """
    pattern = f"influence_results_{n_train}_*.json"
    candidates = list(outdir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No influence result files found in {outdir} "
            f"matching pattern {pattern}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime)
    src = candidates[-1]

    tgt = outdir / f"influence_results_{n_train}_{suffix}.json"

    print(f"[RENAME] {src.name} -> {tgt.name}")
    if tgt.exists():
        tgt.unlink()
    src.rename(tgt)
    return tgt


def main():
    parser = argparse.ArgumentParser(
        description="Unified RapidIn pipeline: caching + influence + mapping to weights"
    )

    # ===== 0) YAML config =====
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file for the RapidIn pipeline"
    )

    # ===== 1) Basic paths (can be overridden by YAML) =====
    parser.add_argument(
        "--retain_path",
        type=str,
        default="../../data/unlearn_packs/retain_set_rapidin.jsonl",
        help="Retain set JSONL for RapidIn"
    )
    parser.add_argument(
        "--forget_path",
        type=str,
        default="../../data/unlearn_packs/forget_set_rapidin.jsonl",
        help="Forget set JSONL for RapidIn"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../models/poisoned_baked_partial_unfreeze",
        help="Base model path for RapidIn"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../../data/RapidIn_output",
        help="Output directory for RapidIn (influence files, etc.)"
    )
    parser.add_argument(
        "--retain_grads",
        type=str,
        default="../../data/retain_set_grads_path/",
        help="Directory for retain gradient cache"
    )
    parser.add_argument(
        "--forget_grads",
        type=str,
        default="../../data/forget_set_grads_path/",
        help="Directory for forget gradient cache"
    )

    # Path to MP_main.py (default: ../MP_main.py relative to this script)
    parser.add_argument(
        "--mp_main_path",
        type=str,
        default=None,
        help="Path to MP_main.py (default: ../MP_main.py relative to this script)"
    )

    # ===== 2) Model / RapidGrad parameters =====
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rapidgrad_k", type=int, default=65536)
    parser.add_argument("--shuffle_lambda", type=int, default=20)
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="RapidIn influence top_k (per test)"
    )

    # Train set sizes (can be inferred automatically from JSONL files)
    parser.add_argument(
        "--n_forget",
        type=int,
        default=None,
        help="Number of forget train samples (default: auto count from forget_path)"
    )
    parser.add_argument(
        "--n_retain",
        type=int,
        default=None,
        help="Number of retain train samples (default: auto count from retain_path)"
    )

    # ===== 3) Mapping script and parameters =====
    parser.add_argument(
        "--mapping_script",
        type=str,
        default="./influence_to_weights.py",
        help="Path to RapidIn→weights mapping script"
    )
    parser.add_argument(
        "--map_topk",
        type=int,
        default=40,
        help="Top-k per test for mapping script (its --topk argument)"
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)

    parser.add_argument("--tau_f", type=float, default=0.7)
    parser.add_argument("--tau_r", type=float, default=1.2)
    parser.add_argument("--wmin_f", type=float, default=0.2)
    parser.add_argument("--wmax_f", type=float, default=3.0)
    parser.add_argument("--wmin_r", type=float, default=0.2)
    parser.add_argument("--wmax_r", type=float, default=3.0)

    parser.add_argument(
        "--out_forget",
        type=str,
        default="../../data/RapidIn_output/forget_weights.jsonl",
        help="Output path for forget weights JSONL"
    )
    parser.add_argument(
        "--out_retain",
        type=str,
        default="../../data/RapidIn_output/retain_weights.jsonl",
        help="Output path for retain weights JSONL"
    )

    # ===== 4) Intermediate file cleanup =====
    parser.add_argument(
        "--cleanup_intermediate",
        action="store_true",
        help="If set, remove gradient directories and raw influence_results_{N}.json after mapping"
    )

    # First parse to obtain the config path
    raw_args, _ = parser.parse_known_args()
    if raw_args.config:
        cfg_path = Path(raw_args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"YAML config must be a mapping (dict), got: {type(cfg)}")
        # Override defaults with YAML; keys must match argparse argument names
        parser.set_defaults(**cfg)

    # Parse again to get the final arguments (after YAML overrides)
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mp_main_path is not None:
        mp_main = Path(args.mp_main_path).resolve()
    else:
        mp_main = (this_dir.parent / "MP_main.py").resolve()

    if not mp_main.exists():
        raise FileNotFoundError(f"MP_main.py not found at: {mp_main}")

    # Infer n_forget / n_retain if not provided
    if args.n_forget is None:
        n_forget = count_jsonl(args.forget_path)
    else:
        n_forget = args.n_forget

    if args.n_retain is None:
        n_retain = count_jsonl(args.retain_path)
    else:
        n_retain = args.n_retain

    print(f"[INFO] n_forget = {n_forget}, n_retain = {n_retain}")

    # ---------------- 1) Caching: forget / retain ----------------
    cfg_cache_forget = build_caching_config(
        train_path=args.forget_path,
        grads_path=args.forget_grads,
        outdir=args.outdir,
        model_path=args.model_path,
        max_length=args.max_length,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        rapidgrad_k=args.rapidgrad_k,
        shuffle_lambda=args.shuffle_lambda,
    )
    cfg_cache_retain = build_caching_config(
        train_path=args.retain_path,
        grads_path=args.retain_grads,
        outdir=args.outdir,
        model_path=args.model_path,
        max_length=args.max_length,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        rapidgrad_k=args.rapidgrad_k,
        shuffle_lambda=args.shuffle_lambda,
    )

    cfg_cache_forget_path = this_dir / "config_cache_forget.json"
    cfg_cache_retain_path = this_dir / "config_cache_retain.json"
    write_json(cfg_cache_forget_path, cfg_cache_forget)
    write_json(cfg_cache_retain_path, cfg_cache_retain)

    print("[STAGE] Caching forget grads...")
    run_mp_main(mp_main, cfg_cache_forget_path)
    print("[STAGE] Caching retain grads...")
    run_mp_main(mp_main, cfg_cache_retain_path)

    # ---------------- 2) Influence: F→F, F→R, R→F, R→R ----------------
    # F→F
    cfg_ftf = build_retrieval_config(
        train_path=args.forget_path,
        test_path=args.forget_path,
        grads_path=args.forget_grads,
        outdir=args.outdir,
        model_path=args.model_path,
        max_length=args.max_length,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        rapidgrad_k=args.rapidgrad_k,
        shuffle_lambda=args.shuffle_lambda,
        top_k=args.top_k,
    )
    cfg_ftf_path = this_dir / "config_retrieval_ftf.json"
    write_json(cfg_ftf_path, cfg_ftf)
    print("[STAGE] Influence F→F (forget test → forget train)...")
    run_mp_main(mp_main, cfg_ftf_path)
    f2f_path = rename_influence_file(outdir, n_forget, "FTF")

    # F→R
    cfg_ftr = build_retrieval_config(
        train_path=args.forget_path,
        test_path=args.retain_path,
        grads_path=args.forget_grads,
        outdir=args.outdir,
        model_path=args.model_path,
        max_length=args.max_length,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        rapidgrad_k=args.rapidgrad_k,
        shuffle_lambda=args.shuffle_lambda,
        top_k=args.top_k,
    )
    cfg_ftr_path = this_dir / "config_retrieval_ftr.json"
    write_json(cfg_ftr_path, cfg_ftr)
    print("[STAGE] Influence F→R (forget test → retain train)...")
    run_mp_main(mp_main, cfg_ftr_path)
    f2r_path = rename_influence_file(outdir, n_forget, "FTR")

    # R→F
    cfg_rtf = build_retrieval_config(
        train_path=args.retain_path,
        test_path=args.forget_path,
        grads_path=args.retain_grads,
        outdir=args.outdir,
        model_path=args.model_path,
        max_length=args.max_length,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        rapidgrad_k=args.rapidgrad_k,
        shuffle_lambda=args.shuffle_lambda,
        top_k=args.top_k,
    )
    cfg_rtf_path = this_dir / "config_retrieval_rtf.json"
    write_json(cfg_rtf_path, cfg_rtf)
    print("[STAGE] Influence R→F (retain test → forget train)...")
    run_mp_main(mp_main, cfg_rtf_path)
    r2f_path = rename_influence_file(outdir, n_retain, "RTF")

    # R→R
    cfg_rtr = build_retrieval_config(
        train_path=args.retain_path,
        test_path=args.retain_path,
        grads_path=args.retain_grads,
        outdir=args.outdir,
        model_path=args.model_path,
        max_length=args.max_length,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        rapidgrad_k=args.rapidgrad_k,
        shuffle_lambda=args.shuffle_lambda,
        top_k=args.top_k,
    )
    cfg_rtr_path = this_dir / "config_retrieval_rtr.json"
    write_json(cfg_rtr_path, cfg_rtr)
    print("[STAGE] Influence R→R (retain test → retain train)...")
    run_mp_main(mp_main, cfg_rtr_path)
    r2r_path = rename_influence_file(outdir, n_retain, "RTR")

    # ---------------- 3) Call mapping script to compute weights ----------------
    mapping_script = Path(args.mapping_script).resolve()
    if not mapping_script.exists():
        raise FileNotFoundError(f"Mapping script not found: {mapping_script}")

    print("[STAGE] Mapping RapidIn influence → per-sample weights...")

    cmd = [
        "python",
        str(mapping_script),
        "--f2f", str(f2f_path),
        "--f2r", str(f2r_path),
        "--r2f", str(r2f_path),
        "--r2r", str(r2r_path),
        "--n_forget", str(n_forget),
        "--n_retain", str(n_retain),
        "--topk", str(args.map_topk),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--gamma", str(args.gamma),
        "--delta", str(args.delta),
        "--no-rank",  # Use value-based scores (disable rank-based mode)
        "--tau_f", str(args.tau_f),
        "--tau_r", str(args.tau_r),
        "--wmin_f", str(args.wmin_f),
        "--wmax_f", str(args.wmax_f),
        "--wmin_r", str(args.wmin_r),
        "--wmax_r", str(args.wmax_r),
        "--out_forget", str(Path(args.out_forget).resolve()),
        "--out_retain", str(Path(args.out_retain).resolve()),
    ]

    subprocess.run(cmd, check=True)

    print("[DONE] RapidIn pipeline finished.")
    print(f"       Forget weights: {Path(args.out_forget).resolve()}")
    print(f"       Retain weights: {Path(args.out_retain).resolve()}")

    # ---------------- 4) Cleanup intermediate files ----------------
    if args.cleanup_intermediate:
        print("[CLEANUP] Removing gradient dirs and raw influence_results_{N}.json ...")
        # Gradient directories
        for d in [args.forget_grads, args.retain_grads]:
            p = Path(d).resolve()
            if p.exists():
                print(f"  - rm -r {p}")
                shutil.rmtree(p, ignore_errors=True)

        # Raw influence_results_{N}.json (without suffix)
        for N in [n_forget, n_retain]:
            raw = outdir / f"influence_results_{N}.json"
            if raw.exists():
                print(f"  - rm {raw.name}")
                raw.unlink()

    print("[ALL DONE] RapidIn pipeline and cleanup complete.")


if __name__ == "__main__":
    main()
