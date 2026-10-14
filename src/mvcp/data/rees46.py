"""Preprocessing for the REES46 electronics-store datasets (the research plan
S3): raw Kaggle CSV -> two packages in the format ``load_instacart`` reads.

  main    (data/rees46)         day transactions with re-placed lines removed -- an
          item that was already in the user's previous transaction is dropped.
  logged  (data/rees46_logged)  the same without that removal (robustness check).

The construction follows scripts/rees46_reference.py statement by statement (same draws
from the same generators), so the reference checksums are reproduced. EVERY id is read as
a string and converted to int64 exactly: the ids have 19 digits, and a float64 read merges
distinct users. Per package:
    split.json              {train,val,test: [user_idx]}
    target.json             {user_idx: item_idx}          one item of the held-out last transaction
    user_item_history.json  {user_idx: [[item_idx, ...], ...]}  transactions before the held-out one
    negative_test.json      {user_idx: {pos, negs[49]}}   mvcp's `corrected` rule, seed 0
    item_meta.json          {item_idx: {product_id}}      the raw id as a string
    user_map.json           {user_idx: user_id}           the raw id as a string
    manifest.json           counts and checksums
item_idx = rank of product_id in the catalog, user_idx = rank of user_id among the logged
users (both 0-based and shared by the two packages; the raw ids never reach the harness).

    uv run python -m mvcp.data.rees46
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import negatives as neg
from .dataset import UserSeq
from .instacart_all import _dump, sha

MAX_DAYS, MAX_BASKET, MIN_BUYERS = 20, 20, 5


def _txns(rows: pd.DataFrame, catalog: list[int]) -> dict[int, list[list[int]]]:
    """{user: [[product, ...] oldest transaction first]} for users left with >= 2
    transactions (products outside the catalog dropped, emptied transactions with them)."""
    rows = rows[rows.product_id.isin(set(catalog))]
    data = {}
    for u, g in rows.groupby("user_id", sort=True):
        txns = [t.product_id.tolist() for _, t in g.groupby("rank", sort=True)]
        if len(txns) >= 2:
            data[int(u)] = txns
    return data


def build(raw_csv: str | Path) -> dict:
    """Both datasets in memory, in raw ids (the reference definition)."""
    raw = pd.read_csv(raw_csv, dtype=str)
    raw["ts"] = pd.to_datetime(raw.event_time.str.replace(" UTC", ""), utc=True)
    raw["row"] = np.arange(len(raw))
    d = raw[raw.user_id.notna() & (raw.ts.dt.year >= 2020)].copy()
    for c in ("user_id", "order_id", "product_id"):
        d[c] = d[c].astype("int64")                   # exact: parsed from the digit string
    d["day"] = d.ts.dt.strftime("%Y%m%d")

    # transaction = (user, UTC day); items = distinct products in logged order
    d = d.sort_values(["user_id", "ts", "order_id", "row"]).drop_duplicates(
        ["user_id", "day", "product_id"])
    tx = d.groupby(["user_id", "day"], sort=True).size().rename("n").reset_index()
    tx["rank"] = tx.groupby("user_id").cumcount()
    d = d.merge(tx[["user_id", "day", "rank"]], on=["user_id", "day"])
    # bulk accounts out: more than MAX_DAYS purchase days, or a day with more than MAX_BASKET products
    days, biggest = tx.groupby("user_id").size(), tx.groupby("user_id").n.max()
    u0 = set(days.index[(days >= 2) & (days <= MAX_DAYS)]) & \
        set(biggest.index[biggest <= MAX_BASKET])
    # re-placed line = the item is also in the user's previous transaction (the ORIGINAL
    # adjacency: applied once, before any line or product is removed)
    prev = d[["user_id", "rank", "product_id"]].assign(rank=lambda x: x["rank"] + 1,
                                                       replaced=True)
    d = d.merge(prev, on=["user_id", "rank", "product_id"], how="left")
    d["replaced"] = d.replaced.notna()
    d = d[d.user_id.isin(u0)].sort_values(["user_id", "rank", "ts", "order_id", "row"])
    later = d[d["rank"] > 0]
    # catalog = products bought by >= MIN_BUYERS users of U0 (the same set with or without
    # the re-placed lines: such a line never adds a buyer)
    buyers = d.groupby("product_id").user_id.nunique()
    catalog = sorted(buyers.index[buyers >= MIN_BUYERS].tolist())

    logged, main = _txns(d, catalog), _txns(d[~d.replaced], catalog)
    users = sorted(logged)
    rng = np.random.default_rng(42)
    perm = rng.permutation(np.array(users))                              # draw 1
    n_train, n_val = round(0.6 * len(users)), round(0.2 * len(users))
    split = {"train": sorted(perm[:n_train].tolist()),
             "val": sorted(perm[n_train:n_train + n_val].tolist()),
             "test": sorted(perm[n_train + n_val:].tolist())}
    tgt_logged = {u: logged[u][-1][int(rng.integers(len(logged[u][-1])))]
                  for u in users}                                        # draw 2
    rng_main = np.random.default_rng(43)
    tgt_main = {u: main[u][-1][int(rng_main.integers(len(main[u][-1])))]
                for u in sorted(main)}
    return {"users": users, "split": split, "catalog": catalog, "u0": len(u0),
            "replaced_lines": [int(later.replaced.sum()), len(later)],
            "main": {"data": main, "target": tgt_main},
            "logged": {"data": logged, "target": tgt_logged}}


def write_package(out_dir: str | Path, b: dict, which: str) -> dict:
    """Write one dataset ('main' | 'logged') and return its manifest."""
    d = b[which]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = {p: i for i, p in enumerate(b["catalog"])}
    uidx = {u: i for i, u in enumerate(b["users"])}
    raw_split = {sp: [u for u in us if u in d["data"]] for sp, us in b["split"].items()}
    split = {sp: [uidx[u] for u in us] for sp, us in raw_split.items()}
    hist = {uidx[u]: [[idx[p] for p in o] for o in d["data"][u][:-1]]
            for us in raw_split.values() for u in us}
    target = {uidx[u]: idx[d["target"][u]] for us in raw_split.values() for u in us}

    test = {u: UserSeq(u, [], [], target[u], False, False,
                       history_items=sorted({i for o in hist[u] for i in o}))
            for u in split["test"]}
    negs = neg.build(test, len(idx), 0, mode="corrected")
    _dump(out / "split.json", split)
    _dump(out / "target.json", {str(u): t for u, t in target.items()})
    _dump(out / "user_item_history.json", {str(u): h for u, h in hist.items()})
    _dump(out / "negative_test.json",
          {str(u): {"pos": target[u], "negs": negs[str(u)]} for u in split["test"]})
    _dump(out / "item_meta.json", {str(i): {"product_id": str(p)} for p, i in idx.items()})
    _dump(out / "user_map.json", {str(uidx[u]): str(u) for us in raw_split.values() for u in us})
    manifest = {
        "dataset": which, "items": len(idx), "u0": b["u0"],
        "replaced_lines": b["replaced_lines"],
        "users": {sp: len(us) for sp, us in split.items()},
        "history_transactions": {sp: sum(len(hist[u]) for u in us)
                                 for sp, us in split.items()},
        # users whose held-out transaction is the same in the logged dataset
        "heldout_same_as_logged": {
            sp: sum(d["data"][u][-1] == b["logged"]["data"][u][-1] for u in us)
            for sp, us in raw_split.items()},
        # the reference checksums: raw ids as strings, the full logged split
        "sha_catalog": sha([str(p) for p in b["catalog"]]),
        "sha_split": sha({sp: [str(u) for u in us] for sp, us in b["split"].items()}),
        "sha_target": sha({str(u): str(v) for u, v in d["target"].items()}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/rees46_raw/kz.csv")
    ap.add_argument("--out-main", default="data/rees46")
    ap.add_argument("--out-logged", default="data/rees46_logged")
    a = ap.parse_args()
    b = build(a.raw)
    for which, out in (("main", a.out_main), ("logged", a.out_logged)):
        print(json.dumps(write_package(out, b, which)))


if __name__ == "__main__":
    main()
