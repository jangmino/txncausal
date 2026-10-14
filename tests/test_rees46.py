"""REES46 electronics-store datasets: the construction rules
against a plain-Python re-implementation on a synthetic raw file (19-digit ids included),
determinism, the K1/K2/K4 strata, the S3 gate values and checksums of the built packages,
TxnCausal invariance at the new settings, and the CLI / report paths for the two new
--data names."""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from mvcp import cli
from mvcp.data import rees46 as rz
from mvcp.data.dataset import shuffle_intra_txn
from mvcp.data.instacart import load_instacart
from mvcp.data.instacart_all import sha
from mvcp.models.registry import build_model
from mvcp.utils.seed import make_rng

from conftest import ROOT

SPLITS = ("train", "val", "test")
STRATA = ("k1", "k2", "recency")
# 19-digit ids as in the raw file: float64 cannot tell neighbours apart up there
UB, PB, OB = 1515915625440000000, 1515966223509000000, 2388440981134000000
A, B, C, D = PB + 1, PB + 2, PB + 3, PB + 4


# ---------------------------------------------------------------- synthetic raw file

def _write_raw(tmp, seed=0):
    """A small raw dump in the Kaggle layout, rows in shuffled file order."""
    rng = np.random.default_rng(seed)
    rows, oid = [], [0]

    def buy(u, day, products, sec=None, order=None):
        """One order on `day` (days since 2020-03-01); returns its order id."""
        if order is None:
            oid[0] += 1
            order = OB + oid[0]
        sec = int(rng.integers(0, 86400)) if sec is None else sec
        t = datetime(2020, 3, 1) + timedelta(days=int(day), seconds=sec)
        for p in products:
            rows.append({"event_time": t.strftime("%Y-%m-%d %H:%M:%S UTC"), "order_id": order,
                         "product_id": p, "category_id": 1, "category_code": "", "brand": "",
                         "price": "1.0", "user_id": "" if u is None else UB + u})
        return order

    pop = np.arange(120, 0, -1) / np.arange(120, 0, -1).sum()
    for u in range(400):
        days = sorted(rng.choice(200, size=int(rng.integers(1, 7)), replace=False).tolist())
        prev = []
        for day in days:
            today = []
            for _ in range(1 + int(rng.random() < 0.2)):      # 1-2 orders that day
                items = (PB + 1 + rng.choice(120, size=int(rng.choice([1, 1, 1, 2, 3])),
                                             replace=False, p=pop)).tolist()
                if prev and rng.random() < 0.35:              # a re-placed line
                    items.append(prev[int(rng.integers(len(prev)))])
                items += items[:1] * int(rng.integers(0, 2))  # a duplicated row
                sec = int(rng.integers(0, 4)) * 20000         # orders often share a timestamp
                buy(u, day, items, sec=sec)
                today += items
            prev = today
    # products bought by fewer than 5 users of U0 (4 ordinary users + the two bulk accounts)
    rare = PB + 150
    for u in (0, 1, 2, 3):
        buy(u, 210, [rare, A])
        buy(u, 211, [B])
    for d in range(21):
        buy(500, d, [rare, PB + 1 + d])                       # 21 purchase days
    buy(501, 0, [rare] + [PB + 1 + j for j in range(20)])     # a day with 21 products
    buy(501, 1, [A])
    buy(502, 0, [A]); buy(502, 1, [A])                        # noqa: E702  only a re-placed line
    buy(503, 0, [A]); buy(503, 1, [PB + 160]); buy(503, 2, [A])     # noqa: E702
    buy(504, 0, [A, B], sec=9 * 3600); buy(504, 0, [B, C], sec=15 * 3600)    # noqa: E702
    buy(504, 5, [D])
    buy(505, 0, [A], sec=86399); buy(505, 1, [B], sec=1)      # noqa: E702  two UTC days
    o = buy(506, 3, [C, A], sec=100); buy(506, 3, [B], sec=100, order=o - 10**6)   # noqa: E702
    buy(506, 9, [D])
    buy(None, 0, [A, B]); buy(None, 1, [C])                   # noqa: E702  no user id
    rows.append({**rows[-1], "event_time": "1970-01-01 00:33:40 UTC", "user_id": UB + 505,
                 "product_id": C})                             # stamped before 2020
    rows = [rows[i] for i in rng.permutation(len(rows))]
    path = tmp / "kz.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _expected(path):
    """The plan's S3 rules in plain Python, from the file's rows."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    days = defaultdict(lambda: defaultdict(list))             # user -> day -> rows
    for i, r in enumerate(rows):
        if r["user_id"] and r["event_time"] >= "2020":
            days[int(r["user_id"])][r["event_time"][:10]].append(
                (r["event_time"], int(r["order_id"]), i, int(r["product_id"])))
    orig = {u: [list(dict.fromkeys(p for *_, p in sorted(v[day]))) for day in sorted(v)]
            for u, v in days.items()}
    u0 = [u for u, t in orig.items() if 2 <= len(t) <= 20 and max(map(len, t)) <= 20]
    buyers = Counter(p for u in u0 for p in {p for t in orig[u] for p in t})
    catalog = sorted(p for p, n in buyers.items() if n >= 5)

    def data(drop_replaced):
        out = {}
        for u in u0:
            txns = [[p for p in t if p in catalog
                     and not (drop_replaced and k > 0 and p in orig[u][k - 1])]
                    for k, t in enumerate(orig[u])]
            txns = [t for t in txns if t]
            if len(txns) >= 2:
                out[u] = txns
        return out
    return catalog, data(False), data(True), orig, u0


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rz")
    raw = _write_raw(tmp)
    b = rz.build(raw)
    man = {w: rz.write_package(tmp / w, b, w) for w in ("main", "logged")}
    return {"tmp": tmp, "raw": raw, "b": b, "manifest": man}


def test_preprocess_rules_on_synthetic_raw(synth):
    b = synth["b"]
    catalog, logged, main, orig, u0 = _expected(synth["raw"])
    assert b["catalog"] == catalog and b["u0"] == len(u0)
    assert b["logged"]["data"] == logged and b["main"]["data"] == main
    assert set(main) < set(logged) and b["users"] == sorted(logged)
    u = lambda k: UB + k                                        # noqa: E731
    assert PB + 150 not in catalog and PB + 160 not in catalog    # < 5 buyers among U0
    assert not {u(500), u(501)} & set(orig) & set(u0)           # bulk accounts
    assert logged[u(502)] == [[A], [A]] and u(502) not in main  # left with one transaction
    # re-placed = in the ORIGINAL previous transaction: the day between is dropped later
    assert orig[u(503)] == [[A], [PB + 160], [A]] and main[u(503)] == [[A], [A]]
    assert logged[u(504)] == [[A, B, C], [D]]                   # same day = one transaction
    assert logged[u(505)] == [[A], [B]]                         # UTC date; the 1970 row is out
    assert logged[u(506)] == [[B, C, A], [D]]                   # time -> order id -> file row
    n_rep = sum(p in orig[x][k - 1] for x in u0 for k, t in enumerate(orig[x]) if k
                for p in t)
    assert b["replaced_lines"] == [n_rep, sum(len(t) for x in u0 for t in orig[x][1:])]
    assert n_rep > 20 and any(len(main[x]) < len(logged[x]) for x in main)
    # the draws: split permutation and logged targets from default_rng(42), main from (43)
    users = sorted(logged)
    rng = np.random.default_rng(42)
    perm = rng.permutation(np.array(users))
    n_tr, n_va = round(0.6 * len(users)), round(0.2 * len(users))
    assert b["split"] == {"train": sorted(perm[:n_tr].tolist()),
                          "val": sorted(perm[n_tr:n_tr + n_va].tolist()),
                          "test": sorted(perm[n_tr + n_va:].tolist())}
    assert b["logged"]["target"] == {x: logged[x][-1][int(rng.integers(len(logged[x][-1])))]
                                     for x in users}
    rng = np.random.default_rng(43)
    assert b["main"]["target"] == {x: main[x][-1][int(rng.integers(len(main[x][-1])))]
                                   for x in sorted(main)}


def test_19_digit_ids_are_not_merged(synth):
    """Every synthetic user id collapses onto a handful of float64 values; an id column that
    went through float64 anywhere would merge users (the survey's 233,575 -> 98,262)."""
    b = synth["b"]
    assert len({float(x) for x in b["users"]}) < len(b["users"]) / 10
    assert len({float(p) for p in b["catalog"]}) < len(b["catalog"]) / 10
    for which in ("main", "logged"):
        d = synth["tmp"] / which
        umap = json.loads((d / "user_map.json").read_text())
        meta = json.loads((d / "item_meta.json").read_text())
        split = json.loads((d / "split.json").read_text())
        ids = [int(v) for v in umap.values()]
        assert len(set(ids)) == len(ids) == len(b[which]["data"]) == \
            sum(len(v) for v in split.values())
        assert [int(meta[str(i)]["product_id"]) for i in range(len(meta))] == b["catalog"]
        # packages carry small re-numbered ids only: the rank among the logged users
        assert all(b["users"][int(k)] == int(v) for k, v in umap.items())
        assert max(x for v in split.values() for x in v) < len(b["users"])


def test_preprocess_is_byte_deterministic(synth, tmp_path):
    b2 = rz.build(synth["raw"])
    for which in ("main", "logged"):
        rz.write_package(tmp_path / which, b2, which)
        names = sorted(p.name for p in (synth["tmp"] / which).iterdir())
        assert names == sorted(p.name for p in (tmp_path / which).iterdir()) and len(names) == 7
        assert all((synth["tmp"] / which / n).read_bytes() ==
                   (tmp_path / which / n).read_bytes() for n in names)


def test_package_round_trip_split_sharing_and_strata(synth):
    b, tmp = synth["b"], synth["tmp"]
    pk = {w: load_instacart(tmp / w, 20, 64, strata=STRATA) for w in ("main", "logged")}
    for which, bundle in pk.items():
        d = tmp / which
        meta = json.loads((d / "item_meta.json").read_text())
        umap = json.loads((d / "user_map.json").read_text())
        hist = json.loads((d / "user_item_history.json").read_text())
        negs = json.loads((d / "negative_test.json").read_text())
        pid = lambda i: int(meta[str(i)]["product_id"])        # noqa: E731
        assert bundle.num_items == len(b["catalog"])
        for sp in SPLITS:
            assert set(bundle.seqs[sp]) <= set(pk["logged"].seqs[sp])     # shared assignment
            for x, q in bundle.seqs[sp].items():
                raw_u = int(umap[str(x)])
                data = b[which]["data"][raw_u]
                assert raw_u in b["split"][sp]
                assert [[pid(i) for i in o] for o in hist[str(x)]] == data[:-1]
                assert pid(q.target) == b[which]["target"][raw_u] and q.target in \
                    [i for i in range(bundle.num_items) if pid(i) in data[-1]]
                h = data[:-1]
                k1 = {n for n in q.extra_strata if n.startswith("k1_")}
                assert k1 == (set() if any(len(o) >= 2 for o in h) else
                              {"k1_one_item"} if len(h) == 1 else {"k1_single_txns"})
                assert q.has_copurchase == (not k1)
                assert {n for n in q.extra_strata if n.startswith("k2_")} == \
                    {"k2_h1" if len(h) == 1 else "k2_h2" if len(h) == 2 else "k2_h3p"}
                tgt = b[which]["target"][raw_u]
                assert {n for n in q.extra_strata if n.startswith("r1_")} == (
                    {"r1_in_last"} if tgt in h[-1] else
                    {"r1_before"} if any(tgt in o for o in h) else set())
                assert ("last_single" in q.extra_strata) == (len(h[-1]) == 1)
        for x, q in bundle.seqs["test"].items():
            v = negs[str(x)]["negs"]
            assert len(set(v)) == 49 and not set(v) & (set(q.history_items) | {q.target})
        same = synth["manifest"][which]["heldout_same_as_logged"]
        assert same == {sp: sum(b[which]["data"][int(umap[str(x)])][-1] ==
                                b["logged"]["data"][int(umap[str(x)])][-1]
                                for x in bundle.seqs[sp]) for sp in SPLITS}
    assert any(n.startswith("r1_in_last") for q in pk["logged"].seqs["train"].values()
               for n in q.extra_strata)
    # without `strata` the records keep the Instacart defaults only
    plain = load_instacart(tmp / "main", 20, 64)
    assert all(len(q.extra_strata) == 2 for q in plain.seqs["test"].values())


# ---------------------------------------------------------------- CLI + reports end to end

def _script(name, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_cli_rees46_end_to_end(synth, monkeypatch, tmp_path):
    """--data rees46 / rees46_logged on the synthetic packages: the K strata reach the
    result files, and the masked-primary options of the report scripts run."""
    real_load = cli.load_config

    def fake_load(p):
        cfg = real_load(p)
        if "configs/data/rees46" in p:
            cfg["data_dir"] = str(synth["tmp"] / ("logged" if "logged" in p else "main"))
        if "train" in p:
            cfg.update(nhead=2, num_layers=1, epochs=2, batch_size=16, device="cpu")
        return cfg
    monkeypatch.setattr(cli, "load_config", fake_load)
    out = {"rees46": tmp_path / "main", "rees46_logged": tmp_path / "logged"}
    hp = ["--dim", "32", "--dropout", "0.1", "--lr", "0.001"]

    def train(data, model, seed, *extra):
        monkeypatch.setattr(sys, "argv", ["mvcp-train", "--model", model, "--data", data,
                                          "--seed", str(seed), "--out", str(out[data]),
                                          *hp, *extra])
        cli.train()
    for data in out:
        for seed in (0, 1):
            for model in ("sasrec_final2", "txf_txn_causal_final2"):
                train(data, model, seed)
            for model in ("sasrec", "txf_txn_causal"):
                train(data, model, seed, "--eval-split", "val")
    name = "_lr0.001_d32_do0.1"
    for data, which in (("rees46", "main"), ("rees46_logged", "logged")):
        n_test = synth["manifest"][which]["users"]["test"]
        res = json.loads((out[data] / f"sasrec_final2{name}_seed0.json").read_text())
        st = res["full_catalog"]
        n = lambda k: st.get(k, {"n": 0})["n"]       # noqa: E731  (an empty stratum has no key)
        assert res["data"] == data and st["overall"]["n"] == n_test
        assert n("k1_one_item") + n("k1_single_txns") + n("copurchase") == n_test == \
            n("k2_h1") + n("k2_h2") + n("k2_h3p") == n("last_single") + n("last_multi")
        assert n("r1_in_last") + n("r1_before") == n("repeat")
    assert json.loads((out["rees46_logged"] / f"sasrec_final2{name}_seed0.json")
                      .read_text())["full_catalog"]["r1_in_last"]["n"] > 0

    pairs = f"P1=sasrec_final2{name}:txf_txn_causal_final2{name}"
    shown = "overall,k1_one_item,copurchase,k2_h1"
    # (the logged package: the tiny main test split has no repeat user for the value table)
    logged = str(out["rees46_logged"])
    txt = _script("report_ic.py", "--seeds", "0-1", "--tag", "t", "--n-boot", "50", "--dir",
                  logged, "--primary", "full", "--strata", shown, "--detail", "--pairs", pairs)
    assert "#### Primary metric — masked full-catalog NDCG@10 overall (B − A)" in txt
    assert "#### By stratum — masked NDCG@10 (B − A)" in txt and "| P1 | k1_one_item | " in txt
    res = json.loads((out["rees46_logged"] / "compare" / "t_P1.json").read_text())
    assert res["primary"] == "full"
    head = next(r for r in res["rows"] if (r["protocol"], r["stratum"]) == ("full", "overall"))
    assert f"**{head['delta']:+.4f}**" in txt
    txt = _script("report_ic.py", "--seeds", "0-1", "--tag", "t2", "--n-boot", "50", "--dir",
                  logged, "--pairs", pairs)                          # the default is unchanged
    assert "#### Primary metric — unmasked full-catalog NDCG@10 overall (B − A)" in txt

    def masked(data):
        return np.mean([json.loads((out[data] / f"sasrec{name}_val_seed{s}.json").read_text())
                        ["full_catalog"]["overall"]["ndcg@10"] for s in (0, 1)])
    grid = _script("report_ic_grid.py", "--dir", logged, "--seeds", "0-1",
                   "--protocol", "full_catalog")
    assert "| val masked overall | novel | repeat | nomask overall |" in grid
    assert f"- best sasrec: dim 32, dropout 0.1, lr 0.001 → {masked('rees46_logged'):.5f} " \
           "(n=2)" in grid
    assert "val nomask overall | novel | repeat | masked overall" in _script(
        "report_ic_grid.py", "--dir", logged, "--seeds", "0-1")

    queue = _script("rz_expand_queue.py", "--dir", str(out["rees46"]), "--seeds", "0-1")
    jobs = [ln for ln in queue.splitlines() if ln and not ln.startswith("#")]
    # dim 32 is the bottom of its ladder: 5 unseen neighbours per model and seed
    want = {(64, "0.1", "0.001"), (32, "0", "0.001"), (32, "0.3", "0.001"),
            (32, "0.1", "0.0005"), (32, "0.1", "0.002")}
    assert sorted(jobs) == sorted(
        f"{m} {s} --data rees46 --dim {d} --dropout {p} --lr {lr} --eval-split val"
        for m in ("sasrec", "txf_txn_causal") for s in (0, 1) for d, p, lr in want)
    assert f"# sasrec current best dim 32 / dropout 0.1 / lr 0.001 = {masked('rees46'):.5f}" in queue


def test_baselines_script_on_rees46(synth, monkeypatch, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "instacart_baselines", ROOT / "scripts" / "instacart_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_config", lambda p: {
        "data_dir": str(synth["tmp"] / "logged"), "max_orders": 20, "max_seq_len": 64,
        "strata": list(STRATA)})
    monkeypatch.setattr(sys, "argv", [
        "x", "--data", "rees46_logged", "--out", str(tmp_path / "b.json"), "--primary",
        "full_catalog", "--strata", "k1_one_item,copurchase,r1_in_last"])
    mod.main()
    rows = json.loads((tmp_path / "b.json").read_text())
    txt = capsys.readouterr().out
    assert "| masked k1_one_item | masked copurchase | masked r1_in_last |" in txt
    r = rows["test/lastfirst/window"]
    assert f" {r['full_catalog']['k1_one_item']['ndcg@10']:.4f} |" \
        f" {r['full_catalog']['copurchase']['ndcg@10']:.4f} |" in txt
    # masking hands a repeat target to any history-first rule; unmasked, a re-placed line
    # (the target is in the last transaction) is what `lastfirst` ranks first
    assert rows["test/histfirst/full"]["full_catalog"]["repeat"]["ndcg@10"] == 1.0
    assert r["full_nomask"]["r1_in_last"]["ndcg@10"] > r["full_nomask"]["novel"]["ndcg@10"]


# ---------------------------------------------------------------- the built packages (plan S3)

def _pkg(name):
    d = ROOT / "data" / name
    if not (d / "user_item_history.json").exists():
        pytest.skip(f"data/{name} not built (python -m mvcp.data.rees46)")
    return d


@pytest.fixture(scope="module")
def rz_main():
    return load_instacart(_pkg("rees46"), 20, 64, strata=STRATA)


@pytest.fixture(scope="module")
def rz_logged():
    return load_instacart(_pkg("rees46_logged"), 20, 64, strata=STRATA)


def test_gate_checksums():
    """The reference checksums (scripts/rees46_reference.py) and the package bytes."""
    m, lg = _pkg("rees46"), _pkg("rees46_logged")
    mm, ml = (json.loads((d / "manifest.json").read_text()) for d in (m, lg))
    assert mm["sha_catalog"] == ml["sha_catalog"] == "f4955af5cd6f2d37"
    assert mm["sha_split"] == ml["sha_split"] == "f7c3e7e7ff40cbf9"
    assert (mm["sha_target"], ml["sha_target"]) == ("9180393e7b58d8e0", "3879dfa880e6b275")
    assert mm["u0"] == 58845 and mm["replaced_lines"] == [21364, 175592]
    assert mm["heldout_same_as_logged"] == {"train": 28767, "val": 9593, "test": 9525}
    for d, sha_target in ((m, "9180393e7b58d8e0"), (lg, "3879dfa880e6b275")):
        meta = json.loads((d / "item_meta.json").read_text())       # back to the raw ids
        umap = json.loads((d / "user_map.json").read_text())
        tgt = json.loads((d / "target.json").read_text())
        assert sha({umap[u]: meta[str(i)]["product_id"] for u, i in tgt.items()}) == sha_target
        assert sha([meta[str(i)]["product_id"] for i in range(len(meta))]) == "f4955af5cd6f2d37"
    lmap = json.loads((lg / "user_map.json").read_text())
    assert sha({sp: [lmap[str(u)] for u in us] for sp, us in
                json.loads((lg / "split.json").read_text()).items()}) == "f7c3e7e7ff40cbf9"
    assert all(len(v) == 19 for v in lmap.values()) and len(set(lmap.values())) == 55768
    file_sha = lambda d, n: hashlib.sha256((d / f"{n}.json").read_bytes()).hexdigest()[:16]  # noqa: E731
    names = ("split", "target", "user_item_history", "negative_test", "item_meta", "user_map")
    assert {n: file_sha(m, n) for n in names} == {
        "split": "f86493a3268e482e", "target": "0d7266abbfb55616",
        "user_item_history": "23b0d0eb16bb2f09", "negative_test": "e1276f756dcb3787",
        "item_meta": "a10240677fb1e69e", "user_map": "351b0a98b7ea5faa"}
    assert {n: file_sha(lg, n) for n in names} == {
        "split": "346e598127f8245b", "target": "d7a87dc4315830dd",
        "user_item_history": "90cef1eaccfc557e", "negative_test": "9e23470214e742e2",
        "item_meta": "a10240677fb1e69e", "user_map": "907fed8a52522566"}


def _counts(b, pred):
    return tuple(sum(pred(q) for q in b.seqs[sp].values()) for sp in SPLITS)


def _has(name):
    return lambda q: name in q.extra_strata


def _gates(b, data_dir):
    """The S3 table rows, per split (train / val / test) unless noted."""
    hist = json.loads((data_dir / "user_item_history.json").read_text())
    full = load_instacart(data_dir, 20, 10**9)                  # the 20-transaction window only
    tr = b.seqs["train"].values()
    n_hist = [len(hist[str(u)]) for u in b.seqs["train"]]
    sizes = [len(o) for u in b.seqs["train"] for o in hist[str(u)]]
    seen = {i for q in tr for i in q.history_items} | {q.target for q in tr}
    return {
        "users": _counts(b, lambda q: True),
        "history_txns": tuple(sum(len(hist[str(u)]) for u in b.seqs[sp]) for sp in SPLITS),
        "train_history_1_2_3p": (n_hist.count(1), n_hist.count(2), sum(n >= 3 for n in n_hist)),
        "max_history": max(len(h) for h in hist.values()),
        "train_items_per_txn": round(float(np.mean(sizes)), 2),
        "train_multi_item_txns": sum(s >= 2 for s in sizes),
        "train_tokens_mean_p99": (round(float(np.mean([len(q.items) for q in tr])), 2),
                                  float(np.percentile([len(q.items) for q in tr], 99))),
        "max_tokens": _counts(b, lambda q: 0)[:0] + tuple(
            max(len(q.items) for q in b.seqs[sp].values()) for sp in SPLITS),
        "cut_by_64": tuple(sum(len(q.items) < len(w.items) for q, w in
                               zip(b.seqs[sp].values(), full.seqs[sp].values()))
                           for sp in SPLITS),
        "labels": (sum(len(q.items) for q in tr),
                   sum(len(q.items) - q.txn_ids.count(0) + 1 for q in tr)),
        "repeat": _counts(b, lambda q: q.repeat),
        "in_last": _counts(b, _has("r1_in_last")),
        "k1_copurchase": _counts(b, lambda q: q.has_copurchase),
        "k1_one_item": _counts(b, _has("k1_one_item")),
        "k1_single_txns": _counts(b, _has("k1_single_txns")),
        "k2_test": tuple(_counts(b, _has(n))[2] for n in ("k2_h1", "k2_h2", "k2_h3p")),
        "last_single": _counts(b, _has("last_single")),
        "catalog_unseen_by_train": b.num_items - len(seen),
        "heldout_targets_unseen": sum(q.target not in seen for sp in ("val", "test")
                                      for q in b.seqs[sp].values()),
    }


def test_gate_values_main(rz_main):
    assert rz_main.num_items == 7563
    assert _gates(rz_main, _pkg("rees46")) == {
        "users": (30128, 10039, 10003), "history_txns": (53868, 17900, 18082),
        "train_history_1_2_3p": (18847, 5927, 5354), "max_history": 19,
        "train_items_per_txn": 1.60, "train_multi_item_txns": 17443,
        "train_tokens_mean_p99": (2.86, 16.0), "max_tokens": (63, 38, 50),
        "cut_by_64": (1, 0, 0), "labels": (86080, 68641),
        "repeat": (473, 151, 169), "in_last": (50, 13, 13),
        "k1_copurchase": (12815, 4263, 4283), "k1_one_item": (13098, 4383, 4277),
        "k1_single_txns": (4215, 1393, 1443), "k2_test": (6219, 1948, 1836),
        "last_single": (20801, 6916, 6875), "catalog_unseen_by_train": 27,
        "heldout_targets_unseen": 30}
    allseq = [q for sp in SPLITS for q in rz_main.seqs[sp].values()]
    assert all(q.repeat == (q.target in q.history_items) for q in allseq)
    assert all(0 <= q.target < 7563 and q.n_txn >= 1 for q in allseq)


def test_gate_values_logged(rz_logged, rz_main):
    assert rz_logged.num_items == 7563
    assert _gates(rz_logged, _pkg("rees46_logged")) == {
        "users": (33461, 11154, 11153), "history_txns": (61246, 20392, 20713),
        "train_history_1_2_3p": (20583, 6661, 6217), "max_history": 19,
        "train_items_per_txn": 1.59, "train_multi_item_txns": 19499,
        "train_tokens_mean_p99": (2.90, 16.0), "max_tokens": (63, 59, 64),
        "cut_by_64": (1, 0, 1), "labels": (97172, 78664),
        "repeat": (5373, 1775, 1870), "in_last": (4995, 1648, 1739),
        "k1_copurchase": (13818, 4582, 4626), "k1_one_item": (14533, 4851, 4723),
        "k1_single_txns": (5110, 1721, 1804), "k2_test": (6770, 2219, 2164),
        "last_single": (23363, 7796, 7755), "catalog_unseen_by_train": 36,
        "heldout_targets_unseen": 42}
    for sp in SPLITS:                           # the same users keep their split and their id
        assert set(rz_main.seqs[sp]) <= set(rz_logged.seqs[sp])


def test_gate_test_negatives(rz_main, rz_logged):
    for b in (rz_main, rz_logged):
        te, neg = b.seqs["test"], b.test_negatives
        assert set(neg) == {str(u) for u in te}
        for u, q in te.items():
            v = neg[str(u)]
            assert len(v) == 49 == len(set(v)) and all(0 <= x < b.num_items for x in v)
            assert not set(v) & (set(q.history_items) | {q.target})


def test_txn_causal_invariant_sasrec_not_at_rees46_settings(rz_main, rz_logged):
    """Plan S2: TxnCausal scores and loss ignore intra-transaction order at the new
    settings -- the longest users (63 and 64 tokens) included."""
    for b in (rz_main, rz_logged):
        tr = {**b.seqs["train"], **b.seqs["test"]}
        users = sorted(tr.values(), key=lambda q: -len(q.items))[:4] + \
            [q for q in b.seqs["train"].values() if q.has_copurchase][:4]
        assert len(users[0].items) >= 63
        rng = make_rng(0, "t")
        perm = [shuffle_intra_txn(q, rng) for q in users]
        assert any(p.items != q.items for p, q in zip(perm, users))
        for name, invariant in (("txf_txn_causal_final2", True), ("sasrec_final2", False)):
            torch.manual_seed(0)
            model, col = build_model(name, b.num_items, dim=32, max_seq_len=64)
            model.eval()
            with torch.no_grad():
                x, y = model(col(users)), model(col(perm))
                lx, ly = model.train_loss(col(users)), model.train_loss(col(perm))
            assert torch.allclose(x, y, atol=1e-5) == invariant
            assert torch.allclose(lx, ly, atol=1e-5) == invariant
