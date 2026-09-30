"""
Utility functions for reproducibility, system info, and checkpointing.
"""

import os
import random
import subprocess
from typing import Dict, Any, Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across random, numpy, and torch for strict reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detects and returns optimal available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_commit_hash() -> str:
    """Returns current git commit hash for run provenance and reproducibility."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "./checkpoints",
    filename: str = "checkpoint.pt",
    best_filename: str = "best_model.pt",
) -> str:
    """Saves model checkpoint and updates best model file if flagged."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)
        return best_path
    return filepath
