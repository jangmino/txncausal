"""Reference definition of the new-sample Instacart datasets
(S3). Planning artifact: it builds the two datasets in
memory from the raw Kaggle CSVs and PRINTS the gate values and checksums the plan quotes.
It writes nothing. Phase 0-B turns this definition into the real preprocessing + adapter
and pins the same values in tests.

  ALL  : a fresh sample of 16,597 users (10-80 orders, not in the earlier package), every
         department; catalog = products bought by >= 5 of the sampled users.
  PD   : the same users restricted to produce + dairy eggs (users with >= 5 such orders).

Random draws, in this order, from np.random.default_rng(42): the user sample, the split
permutation, the ALL targets (ascending user id). PD targets use default_rng(43).

    uv run python scripts/instacart_all_reference.py
"""
import hashlib, json
from collections import Counter
import numpy as np, pandas as pd
from mvcp.data.instacart import _window

R = "data/instacart_raw/"
N_USERS, N_TRAIN, N_VAL = 16597, 9958, 3319
PD_DEPTS = [4, 16]                                    # produce, dairy eggs
dt = {"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16"}
cols = ["order_id", "product_id", "add_to_cart_order"]
orders = pd.read_csv(R + "orders.csv", usecols=["order_id", "user_id", "order_number"])
op = pd.concat([pd.read_csv(R + f, usecols=cols, dtype=dt) for f in ("order_products__prior.csv", "order_products__train.csv")], ignore_index=True)
op = op.merge(orders, on="order_id").merge(pd.read_csv(R + "products.csv", usecols=["product_id", "department_id"]), on="product_id")

# 1. users: 10-80 orders with product rows, not in the earlier (PD study) package
pkg = json.load(open("data/instacart/split.json")); pkg_users = {u for v in pkg.values() for u in v}
n_ord = op.groupby("user_id").order_id.nunique()
pool = np.array(sorted(set(n_ord.index[(n_ord >= 10) & (n_ord <= 80)]) - pkg_users))
rng = np.random.default_rng(42)
sample = np.sort(rng.choice(pool, N_USERS, replace=False))       # draw 1
perm = rng.permutation(sample)                                   # draw 2
split = {"train": sorted(perm[:N_TRAIN].tolist()), "val": sorted(perm[N_TRAIN:N_TRAIN + N_VAL].tolist()),
         "test": sorted(perm[N_TRAIN + N_VAL:].tolist())}

def build(rows, min_orders):
    """catalog = products bought by >= 5 of the users; orders by order_number, items by add_to_cart_order."""
    sup = rows.groupby("product_id").user_id.nunique()
    catalog = sorted(sup[sup >= 5].index.tolist())
    rows = rows[rows.product_id.isin(set(catalog))].sort_values(["user_id", "order_number", "add_to_cart_order"])
    data, onum = {}, {}
    for u, g in rows.groupby("user_id", sort=True):
        byo = [(n, o.product_id.tolist()) for n, o in g.groupby("order_number", sort=True)]
        if len(byo) >= min_orders:
            data[u] = [o for _, o in byo]; onum[u] = byo[-1][0]
    return catalog, data, onum

s = op[op.user_id.isin(set(sample.tolist()))]
cat_all, all_, held_no_all = build(s, 2)
tgt_all = {u: all_[u][-1][int(rng.integers(len(all_[u][-1])))] for u in sorted(all_)}        # draw 3 (ascending user id)
cat_pd, pd_, held_no_pd = build(s[s.department_id.isin(PD_DEPTS)], 5)
rng_pd = np.random.default_rng(43)
tgt_pd = {u: pd_[u][-1][int(rng_pd.integers(len(pd_[u][-1])))] for u in sorted(pd_)}

sha = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
print("pool", len(pool), "| sample", len(sample), "| split", {k: len(v) for k, v in split.items()}, "| sha split", sha(split))
print("sha target ALL", sha({str(u): int(v) for u, v in tgt_all.items()}), "| sha target PD", sha({str(u): int(v) for u, v in tgt_pd.items()}))

deps = pd.read_csv(R + "departments.csv").set_index("department_id").department
pdep = pd.read_csv(R + "products.csv", usecols=["product_id", "department_id"]).set_index("product_id").department_id
HIGH = {"dairy eggs", "beverages", "produce", "bakery", "deli", "pets"}; LOW = {"pantry", "canned goods", "dry goods pasta", "household", "personal care", "international", "missing", "other"}
tier = lambda p: "high" if deps[pdep[p]] in HIGH else "low" if deps[pdep[p]] in LOW else "mid"

def report(name, catalog, data, tgt, is_all):
    print(f"\n=== {name}: users {len(data)} | catalog {len(catalog)}" + (f" | tiers {dict(Counter(tier(p) for p in catalog))}" if is_all else ""))
    for sp in ("train", "val", "test"):
        us = [u for u in split[sp] if u in data]
        sizes = np.array([len(o) for u in us for o in data[u][:-1]]); nh = np.array([len(data[u]) - 1 for u in us])
        toks, sas, txn, intra, cut = [], 0, 0, 0, 0
        rep = inl = nov = pdt = 0; tiers = Counter(); lastsz = []; mb = []
        for u in us:
            hist = data[u][:-1]; win = _window(hist, 50, 512); n = sum(len(o) for o in win); toks.append(n)
            cut += sum(len(o) for o in hist[-50:]) > 512; sas += n; txn += n - len(win[0]) + 1; intra += sum(len(o) - 1 for o in win)
            full = {i for o in hist for i in o}; t = tgt[u]; lastsz.append(len(hist[-1])); mb.append(np.mean([len(o) for o in hist]))
            rep += t in full; inl += t in set(hist[-1]); nov += t not in full; pdt += int(pdep[t]) in PD_DEPTS; tiers[tier(t)] += 1
        print(f"[{sp}] users {len(us)} | history orders {int(nh.sum()):,} (per user mean {nh.mean():.1f} min {nh.min()} max {nh.max()}) | items/order {sizes.mean():.2f} | multi-item {np.mean(sizes >= 2):.1%}")
        print(f"   window(50, 512): tokens/user mean {np.mean(toks):.0f} p50 {np.median(toks):.0f} p90 {np.percentile(toks, 90):.0f} max {max(toks)} | cut by 512 cap: {cut} ({cut / len(us):.1%}) | labels/epoch SASRec {sas:,} (intra {intra / sas:.1%}) TxnCausal {txn:,}")
        print(f"   target: repeat {rep} ({rep / len(us):.1%}) | in last order {inl} ({inl / len(us):.1%}) | novel {nov} | produce/dairy {pdt}" + (f" | tier {dict(tiers)}" if is_all else ""))
        print(f"   last history order: single-item {int(np.sum(np.array(lastsz) == 1))} | size terciles {np.quantile(lastsz, [1 / 3, 2 / 3])} | mean-basket terciles {np.round(np.quantile(mb, [1 / 3, 2 / 3]), 2)}")
    tr = {i for u in split["train"] if u in data for o in data[u] for i in o}
    print("   catalog items never seen by train users:", len(set(catalog) - tr), "| val+test targets not in train:", sum(tgt[u] not in tr for sp in ("val", "test") for u in split[sp] if u in data))

report("ALL-new (21 departments)", cat_all, all_, tgt_all, True)
report("PD-new (produce + dairy eggs, same users with >= 5 such orders)", cat_pd, pd_, tgt_pd, False)
print("\nPD catalog subset of ALL catalog:", set(cat_pd) <= set(cat_all))
al = {sp: [u for u in split[sp] if u in pd_ and held_no_pd[u] == held_no_all[u]] for sp in split}
print("aligned users (PD-new held-out order == ALL-new held-out order):", {k: len(v) for k, v in al.items()}, "| of PD-new users", {sp: sum(u in pd_ for u in split[sp]) for sp in split})
same = sum(tgt_pd[u] == tgt_all[u] for u in al["test"]); print("test aligned users whose two targets coincide:", same)

# --- extra checks for the plan
missing = sorted(set(sample.tolist()) - set(all_)); raw_n = s.groupby("user_id").order_id.nunique()
left = s[s.product_id.isin(set(cat_all))].groupby("user_id").order_id.nunique()
print("\nusers dropped from ALL-new:", missing, "| raw orders:", [int(raw_n[u]) for u in missing], "| orders left after catalog filter:", [int(left.get(u, 0)) for u in missing])
nleft = np.array([len(all_[u]) for u in all_]); print("ALL-new users by remaining orders: <5:", int((nleft < 5).sum()), "| <10:", int((nleft < 10).sum()), "| min", nleft.min())
print("raw purchases covered by ALL catalog:", f"{s.product_id.isin(set(cat_all)).mean():.1%}", "| PD share of ALL-new purchases:", f"{s[s.product_id.isin(set(cat_all))].department_id.isin(PD_DEPTS).mean():.1%}")
print("ALL-new total history orders:", sum(len(v) - 1 for v in all_.values()), "| PD-new:", sum(len(v) - 1 for v in pd_.values()))
