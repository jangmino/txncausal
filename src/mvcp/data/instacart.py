"""Instacart adapter: Kim Jiyeon's preprocessed package -> the Bundle/UserSeq of the
Katchers path (S3, I2/I3/I7).

The package is used as-is (same users, split, targets and frozen 1:49 test negatives):
    split.json              {train,val,test: [user_id]}
    target.json             {user_id: item_idx}             held-out item of the last order
    user_item_history.json  {user_id: [[item_idx, ...], ...]}  orders oldest -> newest
    negative_test.json      {user_id: {pos, negs[49]}}
    item_meta.json          {item_idx: ...}                 only its size is read

item_idx is 0-based here as in UserSeq (the collator adds 1 for PAD=0); user_idx = user_id.
The model input is the most recent whole orders -- at most ``max_orders`` of them and
``max_items`` items (I3); ``repeat`` and full-catalog masking use the FULL history (I7,
``UserSeq.history_items``).

The new-sample packages built by ``mvcp.data.instacart_all`` have the same five files
(the research plan). For them ``strata`` adds the pre-registered
breakdowns to ``extra_strata`` (R1-R4), and an ``r5.json`` next to the files sets
``UserSeq.aux_target`` / ``Bundle.aux_candidates`` (R5).

The REES46 packages built by ``mvcp.data.rees46`` are read
the same way; their ``strata`` are 'k1', 'k2' and 'recency'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..pipeline import Bundle
from .dataset import UserSeq
from .instacart_all import PD_DEPTS
from .vocab import SPLITS, Vocab


def _window(orders: list[list[int]], max_orders: int, max_items: int) -> list[list[int]]:
    """The newest whole orders within both caps (the newest order always stays). Cutting
    at order boundaries -- not mid-order, as the collator's item cap would -- keeps the
    input independent of the stored intra-order order."""
    k = len(orders) - 1
    total = len(orders[k])
    while k > max(0, len(orders) - max_orders) and total + len(orders[k - 1]) <= max_items:
        k -= 1
        total += len(orders[k])
    return orders[k:]


def load_instacart(data_dir: str | Path = "data/instacart", max_orders: int = 50,
                   max_items: int = 512, cold_threshold: int = 3,
                   strata: tuple[str, ...] = ()) -> Bundle:
    """``strata`` (new-sample packages): 'recency' -> r1_in_last / r1_before (a repeat
    target in / not in the newest history order; novel users get neither), 'last_size' ->
    r4_last_t1..t3 (terciles of the newest history order's size, boundaries within the
    split), 'dept' -> r2_high/mid/low (reorder tier of the target's department) and
    r3_pd / r3_other (target in produce or dairy eggs). REES46: 'k1' -> k1_one_item (the
    history is a single item) / k1_single_txns (several transactions, all single-item);
    users with a multi-item transaction are the `copurchase` stratum. 'k2' -> k2_h1 /
    k2_h2 / k2_h3p (1 / 2 / 3 or more history transactions)."""
    d = Path(data_dir)
    split, target, hist, negs, meta = (
        json.loads((d / f"{name}.json").read_text())
        for name in ("split", "target", "user_item_history", "negative_test", "item_meta"))
    r5 = json.loads((d / "r5.json").read_text()) if (d / "r5.json").exists() else None
    seqs: dict[str, dict[int, UserSeq]] = {}
    for sp in SPLITS:
        # Instacart strata (I7): last history order single/multi, and terciles of the
        # user's mean basket size (boundaries taken within the split)
        mean_basket = {u: float(np.mean([len(o) for o in hist[str(u)]])) for u in split[sp]}
        lo, hi = np.quantile(list(mean_basket.values()), [1 / 3, 2 / 3])
        last_lo, last_hi = np.quantile([len(hist[str(u)][-1]) for u in split[sp]],
                                       [1 / 3, 2 / 3])
        seqs[sp] = {}
        for u in split[sp]:
            orders = hist[str(u)]
            win = _window(orders, max_orders, max_items)
            full = sorted({i for o in orders for i in o})
            tgt = int(target[str(u)])
            mb = mean_basket[u]
            names = ("last_multi" if len(orders[-1]) >= 2 else "last_single",
                     "basket_t1" if mb <= lo else "basket_t2" if mb <= hi else "basket_t3")
            if "recency" in strata and tgt in set(full):
                names += ("r1_in_last" if tgt in orders[-1] else "r1_before",)
            if "dept" in strata:
                m = meta[str(tgt)]
                names += (f"r2_{m['tier']}",
                          "r3_pd" if m["department_id"] in PD_DEPTS else "r3_other")
            if "last_size" in strata:
                n = len(orders[-1])
                names += ("r4_last_t1" if n <= last_lo else
                          "r4_last_t2" if n <= last_hi else "r4_last_t3",)
            if "k1" in strata and all(len(o) == 1 for o in orders):
                names += ("k1_one_item" if len(orders) == 1 else "k1_single_txns",)
            if "k2" in strata:
                names += ("k2_h1" if len(orders) == 1 else
                          "k2_h2" if len(orders) == 2 else "k2_h3p",)
            seqs[sp][u] = UserSeq(
                user_idx=u,
                items=[i for o in win for i in o],
                txn_ids=[t for t, o in enumerate(win) for _ in o],
                target=tgt,
                repeat=tgt in set(full),
                has_copurchase=any(len(o) >= 2 for o in orders),
                history_items=full,
                extra_strata=names,
                aux_target=r5["target"].get(str(u)) if r5 else None,
            )
    for u, v in negs.items():
        assert v["pos"] == target[u], f"negative_test.json pos != target for user {u}"
    cold = {u for sp in SPLITS for u, sq in seqs[sp].items() if sq.n_txn <= cold_threshold}
    return Bundle(
        vocab=Vocab(item2idx={i: i for i in range(len(meta))},
                    user2idx={u: u for sp in SPLITS for u in seqs[sp]}),
        seqs=seqs, cold_users=cold,
        test_negatives={u: [int(x) for x in v["negs"]] for u, v in negs.items()},
        aux_candidates=r5["catalog"] if r5 else None,
    )
