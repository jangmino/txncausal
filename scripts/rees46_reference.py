"""Reference definition of the REES46 electronics-store datasets
(S3). Planning artifact: it builds the two datasets in memory
from the raw Kaggle CSV and PRINTS the gate values and checksums the plan quotes. It writes
nothing. Phase 0-B turns this definition into the real preprocessing + adapter and pins the
same values in tests.

  main    (`rees46`)         day transactions with re-placed lines removed -- an item that
                             was already in the user's previous transaction is dropped.
  logged  (`rees46_logged`)  the same without that removal (robustness check).

EVERY id is read as a string and converted to int64 exactly: the ids have 19 digits, and a
float64 read merges distinct users (233,575 -> 98,262).

Random draws from np.random.default_rng(42), in this order: the split permutation of the
logged users (ascending id), then the logged targets (ascending id). Main targets use
default_rng(43).

    uv run python scripts/rees46_reference.py
"""
import hashlib, json
import numpy as np, pandas as pd
from mvcp.data.instacart import _window

RAW = "data/rees46_raw/kz.csv"
MAX_DAYS, MAX_BASKET, MIN_BUYERS = 20, 20, 5
raw = pd.read_csv(RAW, dtype=str)
print("raw rows", len(raw), "| user_id missing", f"{raw.user_id.isna().mean():.1%}", "| orders", raw.order_id.nunique())
raw["ts"] = pd.to_datetime(raw.event_time.str.replace(" UTC", ""), utc=True)
raw["row"] = np.arange(len(raw))
print("rows stamped before 2020:", int((raw.ts.dt.year < 2020).sum()))
d = raw[raw.user_id.notna() & (raw.ts.dt.year >= 2020)].copy()
for c in ("user_id", "order_id", "product_id"):
    d[c] = d[c].astype("int64")                       # exact: parsed from the digit string
d["day"] = d.ts.dt.strftime("%Y%m%d")
print("rows with a user id:", len(d), "| users", d.user_id.nunique(), "| orders", d.order_id.nunique(), "| products", d.product_id.nunique(), "| days", d.day.min(), "..", d.day.max())

# 1. transaction = (user, UTC day); items = distinct products in logged order
d = d.sort_values(["user_id", "ts", "order_id", "row"]).drop_duplicates(["user_id", "day", "product_id"])
tx = d.groupby(["user_id", "day"], sort=True).size().rename("n").reset_index()
tx["rank"] = tx.groupby("user_id").cumcount()
d = d.merge(tx[["user_id", "day", "rank"]], on=["user_id", "day"])
# 2. bulk accounts out: more than 20 purchase days, or a day with more than 20 products
days, biggest = tx.groupby("user_id").size(), tx.groupby("user_id").n.max()
u0 = set(days.index[(days >= 2) & (days <= MAX_DAYS)]) & set(biggest.index[biggest <= MAX_BASKET])
print("users with >= 2 purchase days:", int((days >= 2).sum()), "| > 20 days:", int((days > MAX_DAYS).sum()), "| a day with > 20 products:", int((biggest > MAX_BASKET).sum()), "| U0:", len(u0))
# 3. re-placed line = the item is also in the user's previous transaction (original adjacency)
prev = d[["user_id", "rank", "product_id"]].assign(rank=lambda x: x["rank"] + 1, replaced=True)
d = d.merge(prev, on=["user_id", "rank", "product_id"], how="left")
d["replaced"] = d.replaced.notna()
d = d[d.user_id.isin(u0)].sort_values(["user_id", "rank", "ts", "order_id", "row"])
later = d[d["rank"] > 0]
print("items of non-first transactions (U0):", len(later), "| re-placed lines:", int(later.replaced.sum()), f"({later.replaced.mean():.1%})")
# 4. catalog = products bought by >= 5 users of U0 (the same set with or without step 3)
buyers = d.groupby("product_id").user_id.nunique()
catalog = sorted(buyers.index[buyers >= MIN_BUYERS].tolist())
assert catalog == sorted((b := d[~d.replaced].groupby("product_id").user_id.nunique()).index[b >= MIN_BUYERS].tolist())


def build(rows):
    """{user: [[product, ...] oldest transaction first]} for users left with >= 2 transactions."""
    rows = rows[rows.product_id.isin(set(catalog))]
    data = {}
    for u, g in rows.groupby("user_id", sort=True):
        txns = [t.product_id.tolist() for _, t in g.groupby("rank", sort=True)]
        if len(txns) >= 2:
            data[int(u)] = txns
    return data


logged, main = build(d), build(d[~d.replaced])
assert set(main) <= set(logged)
users = sorted(logged)
rng = np.random.default_rng(42)
perm = rng.permutation(np.array(users))                              # draw 1
n_train, n_val = round(0.6 * len(users)), round(0.2 * len(users))
split = {"train": sorted(perm[:n_train].tolist()), "val": sorted(perm[n_train:n_train + n_val].tolist()),
         "test": sorted(perm[n_train + n_val:].tolist())}
tgt_logged = {u: logged[u][-1][int(rng.integers(len(logged[u][-1])))] for u in users}          # draw 2
rng_main = np.random.default_rng(43)
tgt_main = {u: main[u][-1][int(rng_main.integers(len(main[u][-1])))] for u in sorted(main)}

sha = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
print("catalog", len(catalog), "| sha catalog", sha([str(p) for p in catalog]))
print("sha split", sha({k: [str(u) for u in v] for k, v in split.items()}),
      "| sha target logged", sha({str(u): str(v) for u, v in tgt_logged.items()}),
      "| sha target main", sha({str(u): str(v) for u, v in tgt_main.items()}))


def report(name, data, tgt):
    print(f"\n=== {name}: users {len(data)} | history transactions {sum(len(v) - 1 for v in data.values()):,}")
    for sp in ("train", "val", "test"):
        us = [u for u in split[sp] if u in data]
        nh = np.array([len(data[u]) - 1 for u in us]); sizes = np.array([len(o) for u in us for o in data[u][:-1]])
        toks, sas, txn, cut = [], 0, 0, 0
        rep = inl = cop = single = one = 0
        for u in us:
            hist = data[u][:-1]; win = _window(hist, 20, 64); n = sum(len(o) for o in win); toks.append(n)
            cut += sum(len(o) for o in hist) > n; sas += n; txn += n - len(win[0]) + 1
            full = {i for o in hist for i in o}; t = tgt[u]
            rep += t in full; inl += t in set(hist[-1]); cop += any(len(o) >= 2 for o in hist); single += len(hist[-1]) == 1
            one += len(hist) == 1 and len(hist[0]) == 1
        print(f"[{sp}] users {len(us)} | history txns {int(nh.sum()):,} mean {nh.mean():.2f} (1: {np.mean(nh == 1):.1%} 2: {np.mean(nh == 2):.1%} 3+: {np.mean(nh >= 3):.1%}) max {nh.max()} | items/txn {sizes.mean():.2f} | multi-item txns {np.mean(sizes >= 2):.1%}")
        print(f"   window(20, 64): tokens/user mean {np.mean(toks):.2f} p99 {np.percentile(toks, 99):.0f} max {max(toks)} | cut by the 64 cap: {cut} | labels/epoch SASRec {sas:,} TxnCausal {txn:,}")
        print(f"   target: repeat {rep} ({rep / len(us):.1%}) | in last transaction {inl} ({inl / len(us):.1%}) | novel {len(us) - rep} | users with a multi-item transaction in history {cop} ({cop / len(us):.1%}) | last history txn single-item {single} | history is ONE item {one} ({one / len(us):.1%})")
    seen = {i for u in split["train"] if u in data for o in data[u][:-1] for i in o} | {tgt[u] for u in split["train"] if u in data}
    print("   catalog items the train split never shows (history + target):", len(set(catalog) - seen), "| val+test targets among them:", sum(tgt[u] not in seen for sp in ("val", "test") for u in split[sp] if u in data))


report("main (re-placed lines removed)", main, tgt_main)
report("logged (as logged)", logged, tgt_logged)
same = {sp: sum(u in main and main[u][-1] == logged[u][-1] for u in split[sp]) for sp in split}
print("\nusers of main whose held-out transaction is identical in logged:", same, "| of main users", {sp: sum(u in main for u in split[sp]) for sp in split})
top = d.category_code.fillna("(none)").str.split(".").str[0].value_counts(normalize=True).head(6)
print("top-level categories of the rows (U0):", {k: round(v, 3) for k, v in top.items()})
