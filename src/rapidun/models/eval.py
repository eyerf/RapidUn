import math
import statistics
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from .data import LoReUnGADataset, DataCollatorForCausal, per_sample_answer_ce


@torch.no_grad()
def compute_loreun_weights(
    model,
    tok,
    rows: List[Dict[str, Any]],
    max_len: int,
    bsz: int,
    label_field_for_forget: Optional[str],
    tau: float,
    w_min: float,
    w_max: float,
    weighting_mode: str,
    accelerator: Accelerator,
) -> List[float]:
    """
    Compute per-sample LoReUn weights from answer losses on the forget set.

    Steps:
      1) Build a LoReUnGADataset with tag="forget" and the specified label field.
      2) Compute per-sample average answer loss with per_sample_answer_ce.
      3) Use a median-based exponential mapping to convert losses to weights:
           - "hard_high": higher-than-median loss → larger weight.
           - otherwise: lower-than-median loss → larger weight.
      4) Normalize weights to have mean 1.0.
      5) Clip weights to [w_min, w_max].

    Args:
        model: Causal LM model.
        tok: Tokenizer.
        rows: List of training examples (forget set).
        max_len: Max sequence length for tokenization.
        bsz: Batch size for evaluation.
        label_field_for_forget: Field name to use as the supervised response
                                for forget examples (e.g., poisoned label).
        tau: Temperature for the exponential mapping.
        w_min: Minimum allowed weight after mapping and normalization.
        w_max: Maximum allowed weight after mapping and normalization.
        weighting_mode: "hard_high" or other string determining mapping direction.
        accelerator: accelerate.Accelerator instance for distributed evaluation.

    Returns:
        List of per-sample weights aligned with `rows`.
    """
    model.eval()
    ds = LoReUnGADataset(
        rows,
        tok,
        max_len=max_len,
        tag="forget",
        label_field=label_field_for_forget,
        weights=None,
    )
    dl = DataLoader(
        ds,
        batch_size=bsz,
        shuffle=False,
        collate_fn=DataCollatorForCausal(tok, max_len),
    )
    dl = accelerator.prepare(dl)

    N = len(rows)
    losses_buf: List[Optional[float]] = [None] * N

    # Collect per-sample losses across all processes
    for batch in dl:
        lps = per_sample_answer_ce(model, batch)
        idx = batch["index"].long()
        gl = accelerator.gather(lps.detach())
        gi = accelerator.gather(idx.detach())
        for L, I in zip(gl.cpu().tolist(), gi.cpu().tolist()):
            if 0 <= I < N:
                losses_buf[I] = float(L)

    accelerator.wait_for_everyone()
    model.train()

    # Fill any missing entries with 0.0 (should not normally occur)
    losses: List[float] = [x if x is not None else 0.0 for x in losses_buf]
    if not losses:
        return [1.0] * N

    med = statistics.median(losses)
    denom = max(1e-6, tau)

    if weighting_mode == "hard_high":
        # Higher-than-median loss → larger weight
        raw_w = [math.exp((L - med) / denom) for L in losses]
    else:
        # Lower-than-median loss → larger weight
        raw_w = [math.exp((med - L) / denom) for L in losses]

    # Normalize to mean 1.0
    mean_w = sum(raw_w) / len(raw_w)
    if mean_w > 1e-8:
        raw_w = [w / mean_w for w in raw_w]

    # Clip to [w_min, w_max]
    weights = [max(w_min, min(w_max, w)) for w in raw_w]
    return weights


@torch.no_grad()
def eval_ppl_on(
    rows: List[Dict[str, Any]],
    model,
    tok,
    accelerator: Accelerator,
    max_len: int = 1024,
    bsz: int = 4,
    tag: str = "retain",
    label_field: Optional[str] = None,
) -> float:
    """
    Evaluate perplexity on a given dataset (retain / forget / arbitrary rows).

    The function:
      1) Builds a LoReUnGADataset with the given tag and optional label field.
      2) Computes per-sample answer losses and averages them over batches.
      3) Returns exp(mean_loss), capped to avoid numerical overflow.

    Args:
        rows: List of examples to evaluate on.
        model: Causal LM model.
        tok: Tokenizer.
        accelerator: accelerate.Accelerator instance for distributed evaluation.
        max_len: Max sequence length for tokenization.
        bsz: Batch size for evaluation.
        tag: "retain", "forget", or any other tag understood by LoReUnGADataset.
        label_field: Optional label field to use for supervision.

    Returns:
        Scalar perplexity estimate on the provided rows.
    """
    model.eval()
    ds = LoReUnGADataset(
        rows,
        tok,
        max_len=max_len,
        tag=tag,
        label_field=label_field,
        weights=None,
    )
    dl = DataLoader(
        ds,
        batch_size=bsz,
        shuffle=False,
        collate_fn=DataCollatorForCausal(tok, max_len),
    )
    dl = accelerator.prepare(dl)

    loss_sum, n_batches = 0.0, 0
    for batch in dl:
        per_sample = per_sample_answer_ce(model, batch)
        # Gather from all processes and average batch loss
        loss_sum += accelerator.gather(per_sample.mean().detach()).mean().item()
        n_batches += 1

    accelerator.wait_for_everyone()
    mean_loss = loss_sum / max(1, n_batches)
    # Cap exponent to keep perplexity in a reasonable numeric range
    ppl = math.exp(min(20, mean_loss))
    model.train()
    return ppl
