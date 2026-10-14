"""Load the given train/val/test CSV split and build the union vocabulary.

Reproduces the indexing used by the original ``preprocess.ipynb`` so that the recovered
``negative_test.json`` (keyed by user_idx -> item_idx) aligns exactly:

    item2idx = {product_id(int) -> rank}  over sorted(unique product_id) of the UNION
    user2idx = {user_id(int)    -> rank}  over sorted(unique user_id)    of the UNION

Verified: this yields NUM_ITEMS=1608, NUM_USERS=16787 on ``data/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
_ID_COLS = ("user_id", "product_id", "order_id", "initial_paid_at")


@dataclass
class Vocab:
    item2idx: dict[int, int]
    user2idx: dict[int, int]

    @property
    def num_items(self) -> int:
        return len(self.item2idx)

    @property
    def num_users(self) -> int:
        return len(self.user2idx)


def load_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the three split CSVs, coercing ids to int and paid_at to int (epoch ms)."""
    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        df = pd.read_csv(data_dir / f"{split}_data.csv", encoding="utf-8-sig")
        df = df.dropna(subset=list(_ID_COLS))
        df["user_id"] = df["user_id"].astype(float).astype("int64")
        df["product_id"] = df["product_id"].astype(float).astype("int64")
        df["order_id"] = df["order_id"].astype(str)
        df["initial_paid_at"] = df["initial_paid_at"].astype("int64")
        frames[split] = df
    return frames


def build_vocab(frames: dict[str, pd.DataFrame]) -> Vocab:
    all_df = pd.concat(frames.values(), ignore_index=True)
    items = sorted(all_df["product_id"].unique().tolist())
    users = sorted(all_df["user_id"].unique().tolist())
    return Vocab(
        item2idx={int(p): i for i, p in enumerate(items)},
        user2idx={int(u): i for i, u in enumerate(users)},
    )
