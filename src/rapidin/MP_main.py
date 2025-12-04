"""
Entry point for running RapidIn multi-process influence computation.

Usage:
  python MP_main.py --config_path path/to/config.yaml
"""

import os
from typing import Dict, Optional, Sequence
from transformers import AutoTokenizer, LlamaForCausalLM
import argparse
import logging
import torch
import json
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformers
import RapidIn as rapidin
from RapidIn import TrainDataset, TestDataset
import torch.multiprocessing as mp
import random
import numpy as np

# Optional default config path; typically overridden via the --config_path argument
CONFIG_PATH = None


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str,
                        help="Path to the RapidIn configuration file (YAML or JSON).")
    args = parser.parse_args()
    config_path = args.config_path

    # Initialize logging and load the full RapidIn config
    rapidin.init_logging()
    config = rapidin.get_config(config_path)
    print(config)

    # Set random seeds for reproducibility of the influence computation
    random.seed(int(config.influence.seed))
    np.random.seed(int(config.influence.seed))

    # Launch multi-process influence calculation according to the config
    infl = rapidin.calc_infl_mp(config)
    print("Finished")


if __name__ == "__main__":
    # Use 'spawn' to avoid CUDA / fork-related issues with PyTorch multiprocessing
    mp.set_start_method('spawn')
    # Alternative start methods (kept here for reference):
    # mp.set_start_method('forkserver')
    # torch.multiprocessing.set_sharing_strategy('file_system')

    main()