"""
utils.py — Utility helpers: device selection, seeding, metrics dataclass,
and a lightweight logger.
"""

import random
from dataclasses import dataclass

import numpy as np
import torch


# ── Device ───────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Seeding ──────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Fix all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducible shuffling."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── Metrics ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Metrics:
    loss: float
    acc: float


# ── Logging ──────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    """Simple timestamped print (can be swapped for proper logging later)."""
    print(msg, flush=True)
