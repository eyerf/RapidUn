"""
Entry point for running RapidIn multi-process influence computation.

Usage:
  python MP_main.py --config_path path/to/config.yaml
"""

import argparse
import random

import numpy as np
import torch.multiprocessing as mp

import RapidIn as rapidin


# Optional default config path; typically overridden via the --config_path argument
CONFIG_PATH = None


def _redact_sensitive_text(s: str) -> str:
    """
    Best-effort redaction for identity-bearing absolute paths in printed strings.
    """
    if not isinstance(s, str):
        return s
    import re
    s = re.sub(r"/home/[^/\s]+", "/home/USER", s)
    s = re.sub(r"/Users/[^/\s]+", "/Users/USER", s)
    return s


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=CONFIG_PATH,
        type=str,
        help="Path to the RapidIn configuration file (YAML or JSON).",
    )
    args = parser.parse_args()
    config_path = args.config_path

    # Initialize logging and load the full RapidIn config
    rapidin.init_logging()
    config = rapidin.get_config(config_path)

    # Avoid printing identity-bearing absolute paths (best-effort redaction)
    print(_redact_sensitive_text(str(config)))

    # Set random seeds for reproducibility of the influence computation
    seed = int(config.influence.seed)
    random.seed(seed)
    np.random.seed(seed)

    # Launch multi-process influence calculation according to the config
    _ = rapidin.calc_infl_mp(config)
    print("Finished")


if __name__ == "__main__":
    # Use 'spawn' to avoid CUDA / fork-related issues with PyTorch multiprocessing
    mp.set_start_method("spawn")
    # Alternative start methods (kept here for reference):
    # mp.set_start_method("forkserver")
    # torch.multiprocessing.set_sharing_strategy("file_system")

    main()