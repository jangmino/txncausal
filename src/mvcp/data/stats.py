"""Dataset statistics that set expectations for the transaction-aware study.

Key facts these expose (verified on ``data/``): sequences are SHORT (mean ~3.25
history orders/user) and co-purchase is SPARSE, so transaction-level modeling can only
move the ~17-32% of users that actually have multi-item orders.
"""
from __future__ import annotations

import numpy as np

from .dataset import UserSeq


def split_stats(seqs: dict[int, UserSeq]) -> dict[str, float]:
    n = len(seqs)
    n_txn = np.array([s.n_txn for s in seqs.values()])
    n_items = np.array([len(s.items) for s in seqs.values()])
    repeat = np.array([s.repeat for s in seqs.values()])
    cop = np.array([s.has_copurchase for s in seqs.values()])
    last_txn_multi = np.array([_last_txn_size(s) >= 2 for s in seqs.values()])
    return {
        "users": n,
        "mean_txn_per_user": float(n_txn.mean()),
        "max_txn_per_user": int(n_txn.max(initial=0)),
        "mean_items_per_user": float(n_items.mean()),
        "max_items_per_user": int(n_items.max(initial=0)),
        "pct_target_in_history(leak)": float(100 * repeat.mean()),
        "pct_users_with_copurchase": float(100 * cop.mean()),
        "pct_users_last_txn_multi": float(100 * last_txn_multi.mean()),
    }


def _last_txn_size(s: UserSeq) -> int:
    if not s.txn_ids:
        return 0
    last = max(s.txn_ids)
    return sum(1 for t in s.txn_ids if t == last)


def render_markdown(all_stats: dict[str, dict]) -> str:
    cols = list(next(iter(all_stats.values())).keys())
    lines = ["| split | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for split, st in all_stats.items():
        vals = []
        for c in cols:
            v = st[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {split} | " + " | ".join(vals) + " |")
    return "\n".join(lines)
