"""New-sample Instacart datasets: preprocessing rules
and determinism, the S3 gate values and checksums of the built packages, the R1-R4 strata,
the R5 second-target pass, and the CLI / report paths for the two new --data names."""
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
import torch

from mvcp import cli
from mvcp.compare import compare, matrix_gates
from mvcp.data import instacart_all as ia
from mvcp.data.dataset import SequenceCollator, UserSeq, shuffle_intra_txn
from mvcp.data.instacart import load_instacart
from mvcp.eval.harness import evaluate
from mvcp.records import _load_per_user
from mvcp.models.registry import build_model
from mvcp.utils.seed import make_rng

from conftest import ROOT

SPLITS = ("train", "val", "test")
ALL_STRATA, PD_STRATA = ("recency", "last_size", "dept"), ("recency", "last_size")


# ---------------------------------------------------------------- synthetic raw data

def _write_raw(tmp, seed=0):
    """A small raw dump in the Kaggle layout. Returns {user: [(order_number, [pid, ...])]}
    (items in add_to_cart order), the PD-study split path and the product departments."""
    rng = np.random.default_rng(seed)
    dep_of = {p: (4, 16, 13, 19)[p % 4] for p in range(1, 241)}
    dep_of.update({p: 13 for p in range(241, 247)})             # rare: one buyer each
    n_orders = {u: int(rng.integers(10, 17)) for u in range(1, 71)}
    n_orders.update({71: 5, 72: 9, 73: 85})                     # too few / too many orders
    truth, orders, prior, train = {}, [], [], []
    oid = 0
    for u, n in n_orders.items():
        truth[u] = []
        for k in range(1, n + 1):
            oid += 1
            items = [int(x) for x in rng.choice(240, size=int(rng.integers(1, 7)),
                                                replace=False) + 1]
            if u <= 6 and k == 3:
                items.append(240 + u)
            truth[u].append((k, items))
            orders.append((oid, u, "train" if k == n else "prior", k))
            rows = [(oid, p, j + 1, 0) for j, p in enumerate(items)]
            (train if k == n else prior).extend(rows[::-1])     # stored out of cart order
    orders.append((oid + 1, 72, "test", 10))                    # an order without rows
    raw = tmp / "raw"
    raw.mkdir()

    def dump(name, header, rows):
        with open(raw / name, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)
    dump("orders.csv", ["order_id", "user_id", "eval_set", "order_number"], orders)
    dump("order_products__prior.csv",
         ["order_id", "product_id", "add_to_cart_order", "reordered"], prior)
    dump("order_products__train.csv",
         ["order_id", "product_id", "add_to_cart_order", "reordered"], train)
    dump("products.csv", ["product_id", "product_name", "aisle_id", "department_id"],
         [(p, f"p{p}", 1, d) for p, d in dep_of.items()])
    dump("departments.csv", ["department_id", "department"],
         [(4, "produce"), (16, "dairy eggs"), (13, "pantry"), (19, "snacks")])
    (tmp / "pd_split.json").write_text(json.dumps({"train": [1, 2], "val": [3], "test": []}))
    return raw, tmp / "pd_split.json", truth, dep_of


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ia")
    raw, pd_split, truth, dep_of = _write_raw(tmp)
    b = ia.build(raw, pd_split, n_users=40, n_train=24, n_val=8)
    man = {w: ia.write_package(tmp / w, b, w) for w in ("all", "pdnew")}
    return {"tmp": tmp, "raw": raw, "pd_split": pd_split, "truth": truth, "dep_of": dep_of,
            "b": b, "manifest": man}


def _expected(truth, users, keep, min_orders):
    """The plan's S3 rule in plain Python: catalog, {user: orders} and held-out number."""
    buyers = Counter(p for u in users for p in {p for _, o in truth[u] for p in o if keep(p)})
    catalog = sorted(p for p, n in buyers.items() if n >= 5)
    data, held = {}, {}
    for u in users:
        os_ = [(k, [p for p in o if p in set(catalog)]) for k, o in truth[u]]
        os_ = [(k, o) for k, o in os_ if o]
        if len(os_) >= min_orders:
            data[u], held[u] = [o for _, o in os_], os_[-1][0]
    return catalog, data, held


def test_preprocess_rules_on_synthetic_raw(synth):
    b, truth, dep_of = synth["b"], synth["truth"], synth["dep_of"]
    users = sorted(u for us in b["split"].values() for u in us)
    assert len(users) == 40 and {sp: len(v) for sp, v in b["split"].items()} == \
        {"train": 24, "val": 8, "test": 8}
    assert not set(users) & {1, 2, 3, 71, 72, 73}       # PD-study users, <10 or >80 orders
    assert b["pool"] == 67
    for which, keep, min_orders in (("all", lambda p: True, 2),
                                    ("pdnew", lambda p: dep_of[p] in ia.PD_DEPTS, 5)):
        catalog, data, held = _expected(truth, users, keep, min_orders)
        d = b[which]
        assert d["catalog"] == catalog and d["data"] == data and d["held"] == held
        assert all(d["target"][u] in data[u][-1] for u in data)
    assert not set(b["all"]["catalog"]) & set(range(241, 247))          # < 5 buyers
    assert set(b["pdnew"]["catalog"]) <= set(b["all"]["catalog"])


def test_preprocess_is_byte_deterministic(synth, tmp_path):
    b2 = ia.build(synth["raw"], synth["pd_split"], n_users=40, n_train=24, n_val=8)
    for which in ("all", "pdnew"):
        ia.write_package(tmp_path / which, b2, which)
        names = sorted(p.name for p in (synth["tmp"] / which).iterdir())
        assert names == sorted(p.name for p in (tmp_path / which).iterdir())
        assert all((synth["tmp"] / which / n).read_bytes() ==
                   (tmp_path / which / n).read_bytes() for n in names)
    assert "r5.json" in names or (synth["tmp"] / "all" / "r5.json").exists()


def test_package_round_trip_strata_and_r5(synth):
    b, tmp = synth["b"], synth["tmp"]
    al = load_instacart(tmp / "all", strata=ALL_STRATA)
    pdn = load_instacart(tmp / "pdnew", strata=PD_STRATA)
    meta_a = json.loads((tmp / "all" / "item_meta.json").read_text())
    meta_p = json.loads((tmp / "pdnew" / "item_meta.json").read_text())
    pid_a = lambda i: meta_a[str(i)]["product_id"]              # noqa: E731
    pid_p = lambda i: meta_p[str(i)]["product_id"]              # noqa: E731
    assert [pid_a(i) for i in range(al.num_items)] == b["all"]["catalog"]
    # R5: candidates are the PD-new catalog in PD-new item order; the second target is the
    # PD-new target, for exactly the users whose held-out order is the same in both
    assert [pid_a(i) for i in al.aux_candidates] == [pid_p(i) for i in range(pdn.num_items)]
    assert pdn.aux_candidates is None
    for sp in SPLITS:
        for u, q in al.seqs[sp].items():
            data = b["all"]["data"][u]
            assert [pid_a(i) for i in q.history_items] == \
                sorted({p for o in data[:-1] for p in o})
            assert pid_a(q.target) == b["all"]["target"][u]
            aligned = u in b["pdnew"]["data"] and b["pdnew"]["held"][u] == b["all"]["held"][u]
            assert (q.aux_target is not None) == aligned
            if aligned:
                assert pid_a(q.aux_target) == pid_p(pdn.seqs[sp][u].target) == \
                    b["pdnew"]["target"][u]
            # strata: R1 from the newest order, R2/R3 from the target's department
            last = data[-2]
            r1 = {n for n in q.extra_strata if n.startswith("r1_")}
            assert r1 == (set() if not q.repeat else
                          {"r1_in_last"} if b["all"]["target"][u] in last else {"r1_before"})
            m = meta_a[str(q.target)]
            assert f"r2_{m['tier']}" in q.extra_strata
            assert ("r3_pd" in q.extra_strata) == (m["department_id"] in ia.PD_DEPTS)
            assert sum(n.startswith("r4_last_t") for n in q.extra_strata) == 1
        assert all(not n.startswith(("r2_", "r3_")) for q in pdn.seqs[sp].values()
                   for n in q.extra_strata)
    # without `strata` (the PD-study config) the records keep the PD-study strata only
    plain = load_instacart(tmp / "all")
    assert all(len(q.extra_strata) == 2 for q in plain.seqs["test"].values())
    negs = json.loads((tmp / "all" / "negative_test.json").read_text())
    for u, q in al.seqs["test"].items():
        v = negs[str(u)]["negs"]
        assert len(set(v)) == 49 and not set(v) & (set(q.history_items) | {q.target})


# ---------------------------------------------------------------- harness: second target

def _seq(u, txns, target, aux=None):
    items = [i for g in txns for i in g]
    return UserSeq(u, items, [t for t, g in enumerate(txns) for _ in g], target,
                   target in set(items), any(len(g) >= 2 for g in txns), aux_target=aux)


def test_aux_target_ranked_inside_candidates_only():
    seqs = {0: _seq(0, [[1]], target=5, aux=2), 1: _seq(1, [[1, 3]], target=5, aux=4),
            2: _seq(2, [[0]], target=5)}
    #                               item:  0    1    2    3    4    5
    scorer = lambda b: torch.tensor([[9.0, 8.0, 5.0, 7.0, 5.0, 1.0]]).repeat(   # noqa: E731
        b["item_tokens"].shape[0], 1)
    negs = {str(u): [0, 3] for u in seqs}
    res = evaluate(scorer, seqs, SequenceCollator(8, 4), negs, 6, ks=(1, 2),
                   aux_candidates=[1, 2, 4])
    by = {r.user_idx: r for r in res["per_user"]}
    # among {1, 2, 4}: item 1 is ahead, items 2 and 4 tie -> mid-rank 1.5 (history unmasked)
    assert by[0].aux == by[1].aux == {"mrr": 1 / 2.5, "recall@1": 0.0, "ndcg@1": 0.0,
                                      "recall@2": 1.0, "ndcg@2": 1 / np.log2(3.5)}
    assert by[2].aux == {} and res["aux"]["n"] == 2
    assert res["aux"]["recall@2"] == 1.0
    plain = evaluate(scorer, seqs, SequenceCollator(8, 4), negs, 6, ks=(1, 2))
    assert "aux" not in plain and all(r.aux == {} for r in plain["per_user"])
    for a, b in zip(res["per_user"], plain["per_user"]):        # the primary metrics unchanged
        assert (a.full, a.full_nomask, a.sampled) == (b.full, b.full_nomask, b.sampled)


def test_matrix_gates_match_compare(tmp_path):
    from test_instacart import _write
    rng = np.random.default_rng(1)
    A = rng.uniform(0, 0.5, size=(4, 150))
    B = A + rng.normal(0.01, 0.05, size=A.shape)
    for s in range(4):
        _write(tmp_path, "a", s, A[s], ["x"])
        _write(tmp_path, "b", s, B[s], ["x"])
    row = next(r for r in compare(tmp_path, "a", "b", list(range(4)), n_boot=300)["rows"]
               if r["protocol"] == "full" and r["stratum"] == "overall")
    got = matrix_gates(A, B, n_boot=300)
    for k in ("delta", "w_p", "a_mean", "b_mean"):
        assert abs(got[k] - row[k]) < 1e-12
    assert np.allclose(got["boot_ci"], row["boot_ci"]) and np.allclose(got["two_ci"],
                                                                      row["two_ci"])
    assert got["seeds_pos"] == row["seeds_pos"] and got["n_users"] == 150


# ---------------------------------------------------------------- CLI + reports end to end

def test_cli_new_datasets_end_to_end(synth, monkeypatch, tmp_path):
    """--data instacart_all / instacart_pdnew on the synthetic packages: strata and the R5
    field reach the result files, records load back, and both report scripts run."""
    real_load = cli.load_config

    def fake_load(p):
        cfg = real_load(p)
        if "configs/data/instacart_" in p:
            cfg["data_dir"] = str(synth["tmp"] / p.split("instacart_")[1].split(".")[0])
            cfg["max_seq_len"] = 64
        if "train" in p:
            cfg.update(dim=16, nhead=2, num_layers=1, epochs=2, batch_size=16, device="cpu")
        return cfg
    monkeypatch.setattr(cli, "load_config", fake_load)
    out = {"instacart_all": tmp_path / "all", "instacart_pdnew": tmp_path / "pdnew"}
    for data in out:
        for model in ("sasrec_final2", "txf_txn_causal_final2"):
            for seed in (0, 1):
                monkeypatch.setattr(sys, "argv", [
                    "mvcp-train", "--model", model, "--data", data, "--seed", str(seed),
                    "--dim", "16", "--out", str(out[data])])
                cli.train()
    al = load_instacart(synth["tmp"] / "all", strata=ALL_STRATA)
    aligned = {u for u, q in al.seqs["test"].items() if q.aux_target is not None}
    assert aligned and aligned != set(al.seqs["test"])
    res = json.loads((out["instacart_all"] / "sasrec_final2_d16_seed0.json").read_text())
    assert res["data"] == "instacart_all" and res["r5"]["n"] == len(aligned)
    strata = res["full_nomask"]
    n = lambda k: strata.get(k, {"n": 0})["n"]       # noqa: E731  (an empty stratum has no key)
    assert n("r1_in_last") + n("r1_before") == n("repeat") > 0
    assert n("r3_pd") + n("r3_other") == 8 == sum(n(f"r4_last_t{k}") for k in (1, 2, 3)) \
        == sum(n(f"r2_{t}") for t in ("high", "mid", "low"))
    shown = ["overall"] + [k for k in strata if k.startswith(("r1_", "r3_"))][:2]
    recs = _load_per_user(out["instacart_all"], "sasrec_final2_d16", 0)
    assert {r.user_idx for r in recs if r.r5} == aligned
    assert all(set(r.r5) == set(r.full_nomask) for r in recs if r.r5)
    pd_res = json.loads((out["instacart_pdnew"] / "sasrec_final2_d16_seed0.json").read_text())
    assert "r5" not in pd_res and "r2_high" not in pd_res["full_nomask"]
    with gzip.open(out["instacart_pdnew"] / "per_user" / "sasrec_final2_d16_seed0.json.gz",
                   "rt") as f:
        assert all("r5" not in r for r in json.load(f))

    def script(name, *args):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout
    txt = script("report_ic.py", "--seeds", "0-1", "--tag", "t", "--n-boot", "50",
                 "--dir", str(out["instacart_all"]), "--strata", ",".join(shown), "--detail",
                 "--pairs", "P1=sasrec_final2_d16:txf_txn_causal_final2_d16")
    assert f"| comparison | protocol | {' | '.join(shown)} |" in txt
    assert all(f"| P1 | {st} | " in txt for st in shown)


# ---------------------------------------------------------------- the built packages (plan S3)

def _pkg(name):
    d = ROOT / "data" / name
    if not (d / "user_item_history.json").exists():
        pytest.skip(f"data/{name} not built (python -m mvcp.data.instacart_all)")
    return d


@pytest.fixture(scope="module")
def ia_all():
    return load_instacart(_pkg("instacart_all"), strata=ALL_STRATA)


@pytest.fixture(scope="module")
def ia_pd():
    return load_instacart(_pkg("instacart_pdnew"), strata=PD_STRATA)


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_gate_checksums():
    """The reference checksums (scripts/instacart_all_reference.py) and the package bytes."""
    a, p = _pkg("instacart_all"), _pkg("instacart_pdnew")
    ma, mp = (json.loads((d / "manifest.json").read_text()) for d in (a, p))
    assert ma["sha_split"] == mp["sha_split"] == "39f61508b03318be"
    assert (ma["sha_target"], mp["sha_target"]) == ("fa7e5d62acc3287f", "1731d3fe6a7db207")
    assert ma["pool"] == 88181 and ma["dropped_users"] == [106143]
    assert len(mp["dropped_users"]) == 753
    for d, sha_target in ((a, "fa7e5d62acc3287f"), (p, "1731d3fe6a7db207")):
        meta = json.loads((d / "item_meta.json").read_text())      # targets back to product ids
        tgt = json.loads((d / "target.json").read_text())
        assert ia.sha({u: meta[str(i)]["product_id"] for u, i in tgt.items()}) == sha_target
    assert {n: _file_sha(a / f"{n}.json") for n in
            ("split", "target", "user_item_history", "negative_test", "item_meta", "r5")} == {
        "split": "9136ded979bd477e", "target": "da7146114a66a122",
        "user_item_history": "fafc5661bca4e7e5", "negative_test": "b4672a282b7f4920",
        "item_meta": "6a764f4f14277311", "r5": "d75d88916f55f6cf"}
    assert {n: _file_sha(p / f"{n}.json") for n in
            ("split", "target", "user_item_history", "negative_test", "item_meta")} == {
        "split": "f42b04671f1fdeb6", "target": "07764ac1d01cd9a7",
        "user_item_history": "bf3f20d333469c13", "negative_test": "cf6a8e03bd54d99e",
        "item_meta": "fadbd462761de9dd"}


def _counts(b, pred):
    return tuple(sum(pred(q) for q in b.seqs[sp].values()) for sp in SPLITS)


def _has(name):
    return lambda q: name in q.extra_strata


def _last_size(q):
    return q.txn_ids.count(q.n_txn - 1)


def _gates(b, data_dir):
    """The S3 table rows that are counts over users, per split."""
    hist = json.loads((data_dir / "user_item_history.json").read_text())
    full = load_instacart(data_dir, max_items=10**9)            # the 50-order window only
    tr = b.seqs["train"].values()
    # what training touches: history + target. (The plan's 24 / 2 and 3 / 0 also count the
    # other items of the train users' held-out orders, which no model sees.)
    seen = {i for q in tr for i in q.history_items} | {q.target for q in tr}
    return {
        "users": _counts(b, lambda q: True),
        "history_orders": tuple(sum(len(hist[str(u)]) for u in b.seqs[sp]) for sp in SPLITS),
        "min_max_history": (min(len(h) for h in hist.values()),
                            max(len(h) for h in hist.values())),
        "repeat": _counts(b, lambda q: q.repeat),
        "in_last": _counts(b, _has("r1_in_last")),
        "novel": _counts(b, lambda q: not q.repeat),
        "last_single": _counts(b, _has("last_single")),
        "cut_by_512": _counts(full, lambda q: len(q.items) > 512),
        "labels": (sum(len(q.items) for q in tr),
                   sum(len(q.items) - q.txn_ids.count(0) + 1 for q in tr)),
        "last_size_terciles": tuple(np.quantile(
            [_last_size(q) for q in b.seqs["test"].values()], [1 / 3, 2 / 3]).tolist()),
        "basket_terciles": tuple(np.round(np.quantile(
            [np.mean([len(o) for o in hist[str(u)]]) for u in b.seqs["test"]],
            [1 / 3, 2 / 3]), 2).tolist()),
        "catalog_unseen_by_train": b.num_items - len(seen),
        "heldout_targets_unseen": sum(q.target not in seen for sp in ("val", "test")
                                      for q in b.seqs[sp].values()),
    }


def test_gate_values_all(ia_all):
    d = _pkg("instacart_all")
    assert ia_all.num_items == 23653
    meta = json.loads((d / "item_meta.json").read_text())
    assert Counter(m["tier"] for m in meta.values()) == {"high": 7504, "mid": 7453,
                                                         "low": 8696}
    assert _gates(ia_all, d) == {
        "users": (9958, 3318, 3320), "history_orders": (225073, 76114, 75441),
        "min_max_history": (3, 79), "repeat": (6882, 2336, 2311),
        "in_last": (2871, 1021, 973), "novel": (3076, 982, 1009),
        "last_single": (472, 141, 149), "cut_by_512": (797, 285, 280),
        "labels": (2014824, 1925082), "last_size_terciles": (6.0, 12.0),
        "basket_terciles": (6.74, 11.13), "catalog_unseen_by_train": 36,
        "heldout_targets_unseen": 3}
    assert _counts(ia_all, _has("r3_pd")) == (4471, 1425, 1450)
    assert tuple(_counts(ia_all, _has(f"r2_{t}"))[2] for t in ("high", "mid", "low")) == \
        (2071, 711, 538)
    # R4 last-order terciles use the split's boundaries: <= 6 / <= 12 / > 12
    te = ia_all.seqs["test"].values()
    assert all(("r4_last_t1" in q.extra_strata) == (_last_size(q) <= 6) and
               ("r4_last_t3" in q.extra_strata) == (_last_size(q) > 12) for q in te)
    allseq = [q for sp in SPLITS for q in ia_all.seqs[sp].values()]
    assert max(q.n_txn for q in allseq) == 50 and max(len(q.items) for q in allseq) <= 512
    assert all(q.repeat == (q.target in q.history_items) for q in allseq)


def test_gate_values_pdnew(ia_pd):
    d = _pkg("instacart_pdnew")
    assert ia_pd.num_items == 3468 and ia_pd.aux_candidates is None
    assert _gates(ia_pd, d) == {
        "users": (9520, 3159, 3165), "history_orders": (195567, 66023, 65281),
        "min_max_history": (4, 79), "repeat": (7100, 2354, 2367),
        "in_last": (3263, 1102, 1106), "novel": (2420, 805, 798),
        "last_single": (1606, 509, 535), "cut_by_512": (71, 32, 24),
        "labels": (1005446, 967464), "last_size_terciles": (3.0, 6.0),
        "basket_terciles": (3.25, 5.60), "catalog_unseen_by_train": 4,
        "heldout_targets_unseen": 1}
    meta = json.loads((d / "item_meta.json").read_text())
    assert {m["department_id"] for m in meta.values()} == set(ia.PD_DEPTS)


def test_gate_test_negatives(ia_all, ia_pd):
    for b in (ia_all, ia_pd):
        te, neg = b.seqs["test"], b.test_negatives
        assert set(neg) == {str(u) for u in te}
        for u, q in te.items():
            v = neg[str(u)]
            assert len(v) == 49 == len(set(v)) and all(0 <= x < b.num_items for x in v)
            assert not set(v) & (set(q.history_items) | {q.target})


def test_r5_same_users_target_and_candidates(ia_all, ia_pd):
    """The R5 field of an ALL run and the primary metric of a PD-new run rank the same
    target among the same candidates for the same users (items matched by product id)."""
    ma, mp = (json.loads((_pkg(n) / "item_meta.json").read_text())
              for n in ("instacart_all", "instacart_pdnew"))
    cand = ia_all.aux_candidates
    assert [ma[str(i)]["product_id"] for i in cand] == \
        [mp[str(j)]["product_id"] for j in range(ia_pd.num_items)]
    assert _counts(ia_all, lambda q: q.aux_target is not None) == (8597, 2850, 2849)
    te_a, te_p = ia_all.seqs["test"], ia_pd.seqs["test"]
    aligned = [u for u, q in te_a.items() if q.aux_target is not None]
    assert set(aligned) <= set(te_p)
    assert all(cand[te_p[u].target] == te_a[u].aux_target for u in aligned)
    assert sum(te_a[u].aux_target == te_a[u].target for u in aligned) == 428
    # any score vector gives identical per-user metrics through the two paths
    score = torch.from_numpy(np.random.default_rng(0).permutation(ia_all.num_items)
                             .astype(np.float32) // 7)          # with ties
    users = aligned[:300]
    col = SequenceCollator(8, 4)
    ra = evaluate(lambda b: score.repeat(b["item_tokens"].shape[0], 1),
                  {u: te_a[u] for u in users}, col, ia_all.test_negatives, ia_all.num_items,
                  aux_candidates=cand)
    rp = evaluate(lambda b: score[cand].repeat(b["item_tokens"].shape[0], 1),
                  {u: te_p[u] for u in users}, col, ia_pd.test_negatives, ia_pd.num_items)
    assert [r.aux for r in ra["per_user"]] == [r.full_nomask for r in rp["per_user"]]
    assert ra["aux"]["n"] == 300


def test_txn_causal_invariant_sasrec_not_at_all_settings(ia_all):
    """Plan S2: TxnCausal scores and loss ignore intra-order order on the longest ALL users."""
    tr = ia_all.seqs["train"]
    users = sorted(tr.values(), key=lambda q: -len(q.items))[:4] + list(tr.values())[:4]
    assert len(users[0].items) == 512
    rng = make_rng(0, "t")
    perm = [shuffle_intra_txn(q, rng) for q in users]
    for name, invariant in (("txf_txn_causal_final2", True), ("sasrec_final2", False)):
        torch.manual_seed(0)
        model, col = build_model(name, ia_all.num_items, dim=32, max_seq_len=512,
                                 compact_loss=True, len_split=True)
        model.eval()
        with torch.no_grad():
            a, b = model(col(users)), model(col(perm))
            la, lb = model.train_loss(col(users)), model.train_loss(col(perm))
        assert torch.allclose(a, b, atol=1e-5) == invariant
        assert torch.allclose(la, lb, atol=1e-5) == invariant
