"""Frozen negative sets for the 1:49 sampled protocol.

Three sets, all keyed by user_idx(str) -> list[49] of item_idx:
  published : the recovered baseline file (git show c360693^:negative_test.json).
              Validated CLEAN: all 3358 test users, 49 each, in [0,NUM_ITEMS),
              excludes each user's target and history items, no random fallback.
  corrected : freshly drawn uniform negatives excluding history+target -- used for
              BOTH val and test under one rule (kills the val/test rule mismatch).
  cb_parity : reproduces ConcurrentBERT's non-excluding draw, so the protocol delta is
              MEASURED, not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.seed import make_rng
from .dataset import UserSeq


def load_published(path: str | Path) -> dict[str, list[int]]:
    with open(path) as f:
        return {str(k): [int(x) for x in v] for k, v in json.load(f).items()}


def validate(neg: dict[str, list[int]], seqs: dict[int, UserSeq], num_items: int,
             n: int = 49) -> dict[str, object]:
    """Assert coverage / length / index-space / no target-or-history leakage."""
    users = set(str(u) for u in seqs)
    keys = set(neg)
    report: dict[str, object] = {
        "coverage_ok": users == keys,
        "missing_users": len(users - keys),
        "all_len_n": all(len(v) == n for v in neg.values()),
        "index_range_ok": all(0 <= x < num_items for v in neg.values() for x in v),
    }
    tgt_in = 0
    hist_in = 0
    for u, s in seqs.items():
        v = neg.get(str(u))
        if v is None:
            continue
        vs = set(v)
        if s.target in vs:
            tgt_in += 1
        if vs & set(s.items):
            hist_in += 1
    report["users_target_in_neg"] = tgt_in
    report["users_history_in_neg"] = hist_in
    report["clean"] = (
        report["coverage_ok"] and report["all_len_n"] and report["index_range_ok"]
        and tgt_in == 0 and hist_in == 0
    )
    return report


def build(seqs: dict[int, UserSeq], num_items: int, seed: int, mode: str,
          n: int = 49) -> dict[str, list[int]]:
    """Draw a frozen negative set. mode='corrected' excludes history+target; mode='cb_parity'
    reproduces the baseline's non-excluding uniform draw."""
    rng = make_rng(seed, f"negatives:{mode}")
    all_items = np.arange(num_items)
    out: dict[str, list[int]] = {}
    for u, s in sorted(seqs.items()):
        if mode == "corrected":
            block = set(s.items if s.history_items is None else s.history_items) | {s.target}
            pool = all_items[~np.isin(all_items, list(block))]
        elif mode == "cb_parity":
            pool = all_items  # baseline sampler did not exclude purchases
        else:
            raise ValueError(mode)
        out[str(u)] = rng.choice(pool, size=n, replace=False).astype(int).tolist()
    return out
