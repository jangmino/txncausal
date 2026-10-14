"""Per-user transaction-structured sequences.

A user's HISTORY is an ordered list of transactions (orders); items *inside* one
transaction are an unordered set (co-purchase), transactions *between* each other are
time-ordered. We emit both an item token sequence and a parallel ``txn_pos`` sequence in
which same-order items share one position id -- the substrate for transaction-level
positional encoding.

The target is the single held-out item on the user's last purchase date (already fixed by
the split, ``rs=42``); history excludes that entire last date.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd
import torch

from .vocab import Vocab


@dataclass
class UserSeq:
    user_idx: int
    items: list[int]          # item_idx, oldest -> newest (intra-txn order = as-stored)
    txn_ids: list[int]        # 0..T-1, shared within an order; parallel to `items`
    target: int               # item_idx of the held-out last-date item
    repeat: bool              # target appears in history items (label-in-features leak flag)
    has_copurchase: bool      # any history transaction with >=2 items
    # set when `items` is a recent-orders window of a longer history (Instacart): the FULL
    # history's distinct items -- the basis of `repeat` and of full-catalog masking
    history_items: list[int] | None = None
    extra_strata: tuple[str, ...] = ()   # dataset-specific strata this user belongs to
    # a second target ranked inside the bundle's `aux_candidates` only (Instacart ALL, R5)
    aux_target: int | None = None
    n_txn: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_txn = (max(self.txn_ids) + 1) if self.txn_ids else 0


def build_user_seqs(frame: pd.DataFrame, vocab: Vocab) -> dict[int, UserSeq]:
    """Group one split's rows into per-user transaction sequences."""
    out: dict[int, UserSeq] = {}
    i2 = vocab.item2idx
    u2 = vocab.user2idx
    for uid, udf in frame.groupby("user_id", sort=False):
        udf = udf.sort_values("initial_paid_at", kind="stable")
        last_t = int(udf["initial_paid_at"].max())
        last_rows = udf[udf["initial_paid_at"] == last_t]
        # split fixed exactly one target row per user; take the first defensively
        target = i2[int(last_rows["product_id"].iloc[0])]

        hist = udf[udf["initial_paid_at"] != last_t]
        items: list[int] = []
        txn_ids: list[int] = []
        has_cop = False
        # one order per (order_id); transactions ordered by paid_at then order_id
        order_keys = (
            hist[["order_id", "initial_paid_at"]]
            .drop_duplicates()
            .sort_values(["initial_paid_at", "order_id"], kind="stable")["order_id"]
            .tolist()
        )
        for t, oid in enumerate(order_keys):
            odf = hist[hist["order_id"] == oid]
            # a transaction is a SET of distinct products (dedup option/quantity rows,
            # keeping first-seen order); co-purchase = >=2 distinct products in one order
            seen: set[int] = set()
            oitems: list[int] = []
            for p in odf["product_id"].tolist():
                idx = i2[int(p)]
                if idx not in seen:
                    seen.add(idx)
                    oitems.append(idx)
            if len(oitems) >= 2:
                has_cop = True
            items.extend(oitems)
            txn_ids.extend([t] * len(oitems))

        out[u2[int(uid)]] = UserSeq(
            user_idx=u2[int(uid)],
            items=items,
            txn_ids=txn_ids,
            target=target,
            repeat=target in set(items),
            has_copurchase=has_cop,
        )
    return out


def shuffle_intra_txn(s: UserSeq, rng) -> UserSeq:
    """Permutation-robustness probe: shuffle items WITHIN each transaction; the
    transaction sequence, targets and flags are untouched (doc S10.1)."""
    items = list(s.items)
    groups: dict[int, list[int]] = {}
    for i, tx in enumerate(s.txn_ids):
        groups.setdefault(tx, []).append(i)
    for idx in groups.values():
        vals = [items[i] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            items[i] = v
    return replace(s, items=items, txn_ids=list(s.txn_ids))


def sort_intra_txn(s: UserSeq) -> UserSeq:
    """Items inside each transaction in ascending item-id order (txn_ids are
    non-decreasing, so sorting (txn, item) pairs keeps the transaction sequence)."""
    return replace(s, items=[it for _, it in sorted(zip(s.txn_ids, s.items))])


class SequenceCollator:
    """Left-pad a batch of ``UserSeq`` into aligned tensors (PAD=0, newest at index -1).

    Emits, per user, right-aligned real tokens so that padding never shifts the positional
    ids of real tokens:
      item_tokens : item_idx + 1  (0 = PAD)
      item_pos    : 1..n          (absolute item position; used only when pos_mode='item')
      txn_pos     : 1..T'         (shared within an order; core of transaction-level PE)
      pad_mask    : True where PAD (src_key_padding_mask)
      last_txn    : real tokens belonging to the most recent transaction
    """

    def __init__(self, max_seq_len: int = 64, max_num_txn: int = 52):
        self.max_seq_len = max_seq_len
        self.max_num_txn = max_num_txn

    def __call__(self, batch: list[UserSeq]) -> dict[str, torch.Tensor]:
        L = self.max_seq_len
        B = len(batch)
        item_tokens = torch.zeros(B, L, dtype=torch.long)
        item_pos = torch.zeros(B, L, dtype=torch.long)
        txn_pos = torch.zeros(B, L, dtype=torch.long)
        pad_mask = torch.ones(B, L, dtype=torch.bool)      # True = PAD
        last_txn = torch.zeros(B, L, dtype=torch.bool)
        targets = torch.zeros(B, dtype=torch.long)
        repeat = torch.zeros(B, dtype=torch.bool)
        has_cop = torch.zeros(B, dtype=torch.bool)

        for b, u in enumerate(batch):
            items = u.items[-L:]           # keep most-recent L items
            txns = u.txn_ids[-L:]
            n = len(items)
            if n == 0:
                targets[b] = u.target
                repeat[b] = u.repeat
                has_cop[b] = u.has_copurchase
                continue
            # renumber kept transactions to a contiguous 1..T', clamped to embedding size
            uniq = sorted(set(txns))
            remap = {t: min(k + 1, self.max_num_txn) for k, t in enumerate(uniq)}
            off = L - n
            for j, (it, tx) in enumerate(zip(items, txns)):
                pos = off + j
                item_tokens[b, pos] = it + 1
                item_pos[b, pos] = min(j + 1, L)
                txn_pos[b, pos] = remap[tx]
                pad_mask[b, pos] = False
            last_id = txn_pos[b, L - 1].item()   # newest token's transaction
            last_txn[b] = (txn_pos[b] == last_id) & (~pad_mask[b])
            targets[b] = u.target
            repeat[b] = u.repeat
            has_cop[b] = u.has_copurchase

        return {
            "item_tokens": item_tokens,
            "item_pos": item_pos,
            "txn_pos": txn_pos,
            "pad_mask": pad_mask,
            "last_txn": last_txn,
            "target": targets,
            "repeat": repeat,
            "has_copurchase": has_cop,
        }


class CachedCollator:
    """Memoises a collator's per-user rows (its batch-of-1 outputs) and concatenates them.
    Rows do not depend on batch-mates (fixed length, right-aligned), so the result equals
    ``base(batch)`` -- without re-running the per-token Python loop every epoch. A row is
    reused only for the very same ``UserSeq`` object; a transformed copy is collated anew."""

    def __init__(self, base):
        self.base = base
        self._memo: dict[int, tuple[UserSeq, dict[str, torch.Tensor]]] = {}

    def __call__(self, batch: list[UserSeq]) -> dict[str, torch.Tensor]:
        rows = []
        for u in batch:
            hit = self._memo.get(u.user_idx)
            if hit is None or hit[0] is not u:
                hit = self._memo[u.user_idx] = (u, self.base([u]))
            rows.append(hit[1])
        # flat triplet keys carry the batch row in `*_row` (0 in a batch of one)
        return {k: torch.cat([r[k] + b if k.endswith("_row") else r[k]
                              for b, r in enumerate(rows)]) for k in rows[0]}
