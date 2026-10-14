"""Preprocessing for the new-sample Instacart datasets (the research plan
S3): raw Kaggle CSVs -> two packages in the format ``load_instacart`` reads.

  ALL    (data/instacart_all)   a fresh sample of 16,597 users (10-80 orders, not in
         the PD-study package), every department; catalog = products bought by >= 5 of
         the sampled users.
  PD-new (data/instacart_pdnew) the same users restricted to produce + dairy eggs
         (users with >= 5 such orders); same split assignment.

The construction follows scripts/instacart_all_reference.py statement by statement (same
draws from the same generators), so the reference checksums are reproduced. Per package:
    split.json              {train,val,test: [user_id]}
    target.json             {user_id: item_idx}           one item of the held-out last order
    user_item_history.json  {user_id: [[item_idx, ...], ...]}   orders before the held-out one
    negative_test.json      {user_id: {pos, negs[49]}}    mvcp's `corrected` rule, seed 0
    item_meta.json          {item_idx: {product_id, department_id, department, tier}}
    manifest.json           counts and checksums
    r5.json  (ALL only)     {"catalog": the PD-new catalog as ALL item_idx (PD-new item
                             order), "target": {user_id: PD-new target as ALL item_idx}}
                            for the users whose held-out order is the same in both
item_idx = rank of product_id in the package's catalog (0-based).

    uv run python -m mvcp.data.instacart_all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import negatives as neg
from .dataset import UserSeq

N_USERS, N_TRAIN, N_VAL = 16597, 9958, 3319
PD_DEPTS = (4, 16)                                    # produce, dairy eggs
# department reorder-rate tiers over the whole raw data (plan S3): >= 0.60 / 0.50-0.60 / < 0.50
TIER_HIGH = {"dairy eggs", "beverages", "produce", "bakery", "deli", "pets"}
TIER_LOW = {"pantry", "canned goods", "dry goods pasta", "household", "personal care",
            "international", "missing", "other"}


def sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _build(rows: pd.DataFrame, min_orders: int):
    """catalog = products bought by >= 5 of the users; orders by order_number, items by
    add_to_cart_order; users with fewer than ``min_orders`` remaining orders are dropped."""
    sup = rows.groupby("product_id").user_id.nunique()
    catalog = sorted(sup[sup >= 5].index.tolist())
    rows = rows[rows.product_id.isin(set(catalog))].sort_values(
        ["user_id", "order_number", "add_to_cart_order"])
    data, onum = {}, {}
    for u, g in rows.groupby("user_id", sort=True):
        byo = [(n, o.product_id.tolist()) for n, o in g.groupby("order_number", sort=True)]
        if len(byo) >= min_orders:
            data[u] = [o for _, o in byo]
            onum[u] = byo[-1][0]
    return catalog, data, onum


def build(raw_dir: str | Path, pd_split: str | Path, n_users: int = N_USERS,
          n_train: int = N_TRAIN, n_val: int = N_VAL) -> dict:
    """Both datasets in memory, in product ids (the reference definition)."""
    raw = Path(raw_dir)
    cols = ["order_id", "product_id", "add_to_cart_order"]
    dt = {"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16"}
    orders = pd.read_csv(raw / "orders.csv", usecols=["order_id", "user_id", "order_number"])
    products = pd.read_csv(raw / "products.csv", usecols=["product_id", "department_id"])
    op = pd.concat([pd.read_csv(raw / f, usecols=cols, dtype=dt) for f in
                    ("order_products__prior.csv", "order_products__train.csv")],
                   ignore_index=True)
    op = op.merge(orders, on="order_id").merge(products, on="product_id")

    # users: 10-80 orders with product rows, not in the PD-study package
    pkg_users = {u for v in json.loads(Path(pd_split).read_text()).values() for u in v}
    n_ord = op.groupby("user_id").order_id.nunique()
    pool = np.array(sorted(set(n_ord.index[(n_ord >= 10) & (n_ord <= 80)]) - pkg_users))
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(pool, n_users, replace=False))       # draw 1
    perm = rng.permutation(sample)                                   # draw 2
    split = {"train": sorted(perm[:n_train].tolist()),
             "val": sorted(perm[n_train:n_train + n_val].tolist()),
             "test": sorted(perm[n_train + n_val:].tolist())}

    s = op[op.user_id.isin(set(sample.tolist()))]
    cat_all, all_, held_all = _build(s, 2)
    tgt_all = {u: all_[u][-1][int(rng.integers(len(all_[u][-1])))]
               for u in sorted(all_)}                                # draw 3
    cat_pd, pd_, held_pd = _build(s[s.department_id.isin(PD_DEPTS)], 5)
    rng_pd = np.random.default_rng(43)
    tgt_pd = {u: pd_[u][-1][int(rng_pd.integers(len(pd_[u][-1])))] for u in sorted(pd_)}

    dep_name = pd.read_csv(raw / "departments.csv").set_index("department_id").department
    dep_of = products.set_index("product_id").department_id
    return {"pool": len(pool), "split": split, "dep_name": dep_name, "dep_of": dep_of,
            "all": {"catalog": cat_all, "data": all_, "target": tgt_all, "held": held_all},
            "pdnew": {"catalog": cat_pd, "data": pd_, "target": tgt_pd, "held": held_pd}}


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")))


def write_package(out_dir: str | Path, b: dict, which: str) -> dict:
    """Write one dataset ('all' | 'pdnew') and return its manifest."""
    d = b[which]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = {p: i for i, p in enumerate(d["catalog"])}
    split = {sp: [u for u in us if u in d["data"]] for sp, us in b["split"].items()}
    users = [u for sp in split for u in split[sp]]
    hist = {u: [[idx[p] for p in o] for o in d["data"][u][:-1]] for u in users}
    target = {u: idx[d["target"][u]] for u in users}

    def tier(p):
        name = b["dep_name"][b["dep_of"][p]]
        return "high" if name in TIER_HIGH else "low" if name in TIER_LOW else "mid"

    test = {u: UserSeq(u, [], [], target[u], False, False,
                       history_items=sorted({i for o in hist[u] for i in o}))
            for u in split["test"]}
    negs = neg.build(test, len(idx), 0, mode="corrected")
    _dump(out / "split.json", split)
    _dump(out / "target.json", {str(u): t for u, t in target.items()})
    _dump(out / "user_item_history.json", {str(u): h for u, h in hist.items()})
    _dump(out / "negative_test.json",
          {str(u): {"pos": target[u], "negs": negs[str(u)]} for u in split["test"]})
    _dump(out / "item_meta.json", {str(i): {
        "product_id": int(p), "department_id": int(b["dep_of"][p]),
        "department": str(b["dep_name"][b["dep_of"][p]]), "tier": tier(p)}
        for p, i in idx.items()})
    manifest = {
        "dataset": which, "pool": b["pool"], "items": len(idx),
        "users": {sp: len(us) for sp, us in split.items()},
        "dropped_users": sorted(set(u for us in b["split"].values() for u in us) - set(users)),
        "history_orders": {sp: sum(len(hist[u]) for u in us) for sp, us in split.items()},
        # the reference checksums: the full sampled split, targets as product ids
        "sha_split": sha(b["split"]),
        "sha_target": sha({str(u): int(v) for u, v in d["target"].items()}),
    }
    if which == "all":
        pdn = b["pdnew"]
        aligned = {sp: [u for u in us if u in pdn["data"] and pdn["held"][u] == d["held"][u]]
                   for sp, us in split.items()}
        _dump(out / "r5.json", {
            "catalog": [idx[p] for p in pdn["catalog"]],
            "target": {str(u): idx[pdn["target"][u]] for sp in aligned for u in aligned[sp]}})
        manifest["r5_aligned_users"] = {sp: len(us) for sp, us in aligned.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/instacart_raw")
    ap.add_argument("--pd-split", default="data/instacart/split.json",
                    help="the PD-study package split (its users are excluded)")
    ap.add_argument("--out-all", default="data/instacart_all")
    ap.add_argument("--out-pdnew", default="data/instacart_pdnew")
    a = ap.parse_args()
    b = build(a.raw, a.pd_split)
    for which, out in (("all", a.out_all), ("pdnew", a.out_pdnew)):
        m = write_package(out, b, which)
        print(json.dumps({**m, "dropped_users": len(m["dropped_users"])}))


if __name__ == "__main__":
    main()
