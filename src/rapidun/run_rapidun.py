import os as _os

# Make hash-based operations reproducible across runs
_os.environ.setdefault("PYTHONHASHSEED", "0")
# Control cuBLAS workspace config for deterministic behavior on CUDA
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import argparse
from models.trainer import run_training


def main():
    """
    CLI entry point for LoReUn-GA training.

    Usage:
        python main.py --config path/to/config.yaml
    """
    parser = argparse.ArgumentParser(description="Run LoReUn-GA training.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()