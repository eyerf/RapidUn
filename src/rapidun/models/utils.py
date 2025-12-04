import os
import json
import math
import random
import statistics
from pathlib import Path
from typing import List, Optional, Any, Dict

import numpy as np
import torch
from accelerate.utils import set_seed as accel_set_seed

# Baseline time cost for influence estimation (added to training wall time)
INFLUENCE_TIME = 100.0


def seed_worker(worker_id: int):
    """
    Seed worker processes for DataLoader to ensure reproducible shuffling
    and data augmentation.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_all_seeds_and_determinism(seed: int, strict: bool = True):
    """
    Set random seeds for all relevant libraries and configure deterministic
    behavior for CUDA/CuDNN.

    Args:
        seed: Global random seed.
        strict: If True, enforce fully deterministic algorithms where possible.
                If False, allow fallbacks with warnings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        accel_set_seed(seed, device_specific=True)
    except Exception:
        # If accelerate is not fully initialized, ignore the error
        pass

    # Make CuDNN / matmul behavior deterministic and disable TF32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Enforce deterministic algorithms globally
    torch.use_deterministic_algorithms(strict, warn_only=not strict)


def pct(x: List[float], p: float) -> float:
    """
    Approximate p-quantile of a list via sorted index lookup.

    Args:
        x: Input list of floats.
        p: Quantile in [0, 1].

    Returns:
        The value at the corresponding quantile, or NaN if the list is empty.
    """
    if not x:
        return float("nan")
    s = sorted(x)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[k])


def load_weight_jsonl(path: Optional[str], expected_n: int) -> Optional[List[float]]:
    """
    Load per-sample weights from a JSONL file.

    Each line is expected to be a JSON object with a "weight" field.
    The function:
      - Fills missing path with None.
      - Clips or pads to expected_n with default weight 1.0.
      - Ensures all weights are >= 1e-6.
      - Normalizes weights to have mean 1.0.

    Args:
        path: Path to the JSONL file. If None or empty, returns None.
        expected_n: Expected number of weights.

    Returns:
        List of normalized weights, or None if path is empty.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Weight file not found: {p}")
    ws: List[float] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            w = float(obj.get("weight", 1.0))
            ws.append(w)

    if expected_n and len(ws) != expected_n:
        print(f"[WARN] weights length {len(ws)} != expected {expected_n}. clip/pad with 1.0")
        if len(ws) > expected_n:
            ws = ws[:expected_n]
        else:
            ws = ws + [1.0] * (expected_n - len(ws))

    # Avoid zero weights and renormalize to mean 1.0
    ws = [max(1e-6, float(w)) for w in ws]
    m = sum(ws) / max(1, len(ws))
    if m > 0:
        ws = [w / m for w in ws]
    return ws


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file as a list of dictionaries.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects (one per non-empty line).
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_pack_file(packs_dir: Path, p_like: Any) -> str:
    """
    Resolve a pack file path relative to a base directory.

    Resolution strategy:
      1) If p_like is an absolute path that exists, return it.
      2) Try packs_dir, packs_dir.parent, packs_dir.parent.parent,
         this file's directory, and current working directory.
      3) Fallback to packs_dir / p.name.

    Args:
        packs_dir: Base directory containing the pack files.
        p_like: Either a path-like object or string.

    Returns:
        Absolute string path to the resolved file.
    """
    p = Path(p_like)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    for root in [packs_dir, packs_dir.parent, packs_dir.parent.parent, Path(__file__).parent, Path.cwd()]:
        q = (root / p).resolve()
        if q.exists():
            return str(q)
    return str((packs_dir / p.name).resolve())


def reset_accelerate_state():
    """
    Clear environment flags and reset any global Accelerate state.

    This is useful when running multiple experiments in the same Python process
    to avoid stale configuration from previous runs.
    """
    from accelerate.state import AcceleratorState

    os.environ.pop("ACCELERATE_MIXED_PRECISION", None)
    os.environ.pop("ACCELERATE_USE_MPS_DEVICE", None)
    os.environ.pop("ACCELERATE_USE_CPU", None)
    try:
        AcceleratorState._reset_state()
    except Exception:
        # If Accelerate is not initialized or version differs, ignore
        pass
