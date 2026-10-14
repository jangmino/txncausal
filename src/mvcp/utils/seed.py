"""Deterministic seeding with separated RNG streams.

The original notebook coupled shuffling and negative-sampling into one global RNG and
loaded frozen single-draw embeddings, so multi-seed std understated true variance.
Here every consumer draws from its own numpy ``Generator`` keyed by (seed, stream),
and torch/cuda are put in deterministic mode.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def make_rng(seed: int, stream: str) -> np.random.Generator:
    """Independent numpy Generator for a named stream (e.g. 'shuffle', 'negatives').

    Streams derived from the same seed are statistically independent, so shuffling a
    batch never perturbs how negatives are drawn (and vice versa).
    """
    ss = np.random.SeedSequence([seed, _stream_int(stream)])
    return np.random.default_rng(ss)


def _stream_int(stream: str) -> int:
    return int.from_bytes(stream.encode("utf-8"), "little") % (2**31)
