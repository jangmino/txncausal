"""Split integrity checks (pre-flight PF1).

The given ``data`` split was verified byte-identical to Park & Oh (2024)'s
ConcurrentBERT ``DataPrepWith.py`` output (GroupShuffleSplit 0.8->0.75, rs=42): same
users per split, same 16,787/1,608 union, 100% identical per-user target. These checks
guard that property so a future re-split cannot silently diverge from the baseline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .vocab import SPLITS


def structural_checks(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    users = {s: set(frames[s]["user_id"].unique().tolist()) for s in SPLITS}
    counts = {s: len(users[s]) for s in SPLITS}
    total = sum(counts.values())
    ratios = {s: round(counts[s] / total, 3) for s in SPLITS}
    overlaps = {
        "train&val": len(users["train"] & users["val"]),
        "train&test": len(users["train"] & users["test"]),
        "val&test": len(users["val"] & users["test"]),
    }
    # each user has exactly one row on their last purchase date (the fixed target)
    multi_last = 0
    for s in SPLITS:
        g = frames[s].groupby("user_id")["initial_paid_at"]
        last = g.transform("max")
        last_rows = frames[s][frames[s]["initial_paid_at"] == last]
        cnt = last_rows.groupby("user_id").size()
        multi_last += int((cnt != 1).sum())
    return {
        "counts": counts,
        "ratios": ratios,
        "disjoint": all(v == 0 for v in overlaps.values()),
        "overlaps": overlaps,
        "users_with_multi_row_last_date": multi_last,
    }


def reproduce_cb_split(raw_path: str | Path) -> dict[str, set[int]]:
    """Reproduce ConcurrentBERT's GroupShuffleSplit(0.8 then 0.75, rs=42) from raw.

    Returns {split: set(user_id)}. Requires scikit-learn and the raw katchers file.
    """
    from sklearn.model_selection import GroupShuffleSplit

    raw = pd.read_csv(raw_path, sep="\t", dtype={"user_id": str, "product_id": str})
    raw = raw[raw["user_id"].notna() & (raw["user_id"] != "user_id")].copy()
    raw["initial_paid_at"] = pd.to_numeric(raw["initial_paid_at"], errors="coerce")
    raw = raw[raw["initial_paid_at"].notna()]

    ud = raw.groupby("user_id")["initial_paid_at"].transform("nunique")
    raw = raw[ud > 1]
    dg = raw.groupby("user_id")["initial_paid_at"].max().reset_index(name="initial_paid_at")
    dmax_all = raw.merge(dg, how="inner", on=["user_id", "initial_paid_at"])
    dmax = dmax_all.groupby("user_id").sample(n=1, random_state=42).reset_index(drop=True)
    joined = raw.merge(dg, how="inner", on="user_id", suffixes=["", "_y"])
    dnot = joined[joined["initial_paid_at"] != joined["initial_paid_at_y"]].drop(
        columns=["initial_paid_at_y"]
    )
    proc = pd.concat([dmax, dnot]).reset_index(drop=True)
    cnt = proc.groupby("user_id")["product_id"].transform("count")
    proc = proc[(cnt > 1) & (cnt < 68)].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    tv, te = next(gss.split(proc, groups=proc["user_id"]))
    tvd, ted = proc.iloc[tv], proc.iloc[te]
    gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
    tr, va = next(gss.split(tvd, groups=tvd["user_id"]))
    trd, vad = tvd.iloc[tr], tvd.iloc[va]
    return {
        "train": set(trd["user_id"].astype(int)),
        "val": set(vad["user_id"].astype(int)),
        "test": set(ted["user_id"].astype(int)),
    }
