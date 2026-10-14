"""Instacart external validation: adapter gate values,
full-history masking, dataset strata, long-sequence efficiency paths (tensor-identical
collation cache, compact SASRec loss), intra-order serializations, compare extensions."""
import gzip
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mvcp import cli
from mvcp.compare import compare, render
from mvcp.data.dataset import (CachedCollator, SequenceCollator, UserSeq, shuffle_intra_txn,
                               sort_intra_txn)
from mvcp.eval.harness import evaluate
from mvcp.models.registry import build_model
from mvcp.train.loop import TrainConfig, train_model
from mvcp.utils.seed import make_rng

def _seq(u, txns, target, full=None, strata=()):
    items = [i for g in txns for i in g]
    tids = [t for t, g in enumerate(txns) for _ in g]
    return UserSeq(u, items, tids, target, target in set(full or items),
                   any(len(g) >= 2 for g in txns), full, strata)


def _users(n=12, num_items=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for u in range(n):
        txns = [[int(x) for x in rng.choice(num_items, size=int(rng.integers(1, 5)),
                                            replace=False)]
                for _ in range(int(rng.integers(1, 7)))]
        out.append(_seq(u, txns, int(rng.integers(0, num_items))))
    return out


@pytest.mark.parametrize("name", ["sasrec", "txf_txn_causal", "txf_txn_causal_k2"])
def test_cached_collator_is_tensor_identical(name):
    _, col = build_model(name, 40, max_seq_len=8, max_num_txn=4)   # truncating window
    users = _users()
    cached = CachedCollator(col)
    for _ in range(2):                                   # second pass reads the memo
        for chunk in (users[:5], users[5:], users[3:9]):
            want, got = col(chunk), cached(chunk)
            assert want.keys() == got.keys()
            assert all(torch.equal(want[k], got[k]) and want[k].dtype == got[k].dtype
                       for k in want)
    # a transformed copy of a cached user is collated anew, never served from the memo
    moved = [sort_intra_txn(u) for u in users[:5]]
    assert all(torch.equal(col(moved)[k], cached(moved)[k]) for k in col(moved))


@pytest.mark.parametrize("name", ["sasrec", "sasrec_final2"])
def test_compact_sasrec_loss_equals_dense(name):
    users = _users() + [UserSeq(99, [], [], 3, False, False)]   # + an empty history
    torch.manual_seed(0)
    dense, col = build_model(name, 40, dim=16, max_seq_len=8, max_num_txn=4)
    torch.manual_seed(0)
    compact, _ = build_model(name, 40, dim=16, max_seq_len=8, max_num_txn=4,
                             compact_loss=True)
    dense.eval(), compact.eval()
    batch = col(users)
    a, b = dense.train_loss(batch), compact.train_loss(batch)
    assert torch.allclose(a, b, atol=1e-6)
    ga = torch.autograd.grad(a, dense.item_embedding.weight)[0]
    gb = torch.autograd.grad(b, compact.item_embedding.weight)[0]
    assert torch.allclose(ga, gb, atol=1e-6)


@pytest.mark.parametrize("name", ["sasrec_final2", "txf_txn_causal_final2"])
def test_len_split_matches_single_pass(name):
    """Encoding the longest rows and the rest separately gives the padded pass's states,
    scores, loss and gradients (float rounding only)."""
    users = _users(n=13, seed=3) + [UserSeq(99, [], [], 3, False, False)]
    kw = dict(dim=16, max_seq_len=24, max_num_txn=8, compact_loss=True)
    torch.manual_seed(0)
    whole, col = build_model(name, 40, **kw)
    torch.manual_seed(0)
    split, _ = build_model(name, 40, len_split=True, **kw)
    whole.eval(), split.eval()
    for chunk in (users, users[:1], users[3:5], [users[2]] * 4):   # incl. B=1, equal lengths
        batch = col(chunk)
        with torch.no_grad():
            h1 = whole.encoder.hidden_states(whole.item_embedding, batch)
            h2 = split.encoder.hidden_states(split.item_embedding, batch)
            real = ~batch["pad_mask"]
            assert torch.allclose(h1[real], h2[real], atol=1e-6)
            assert torch.allclose(whole(batch), split(batch), atol=1e-6)
        a, b = whole.train_loss(batch), split.train_loss(batch)
        assert torch.allclose(a, b, atol=1e-6)
        ga = torch.autograd.grad(a, whole.item_embedding.weight)[0]
        gb = torch.autograd.grad(b, split.item_embedding.weight)[0]
        assert torch.allclose(ga, gb, atol=1e-6)


def test_intra_order_helpers_keep_structure():
    s = _seq(0, [[12, 10, 11], [20], [31, 30]], 40, full=[5, 10, 11, 12, 20, 30, 31],
             strata=("last_multi",))
    t = sort_intra_txn(s)
    assert t.items == [10, 11, 12, 20, 30, 31] and t.txn_ids == s.txn_ids
    p = shuffle_intra_txn(s, make_rng(0, "x"))
    assert [sorted(p.items[:3]), p.items[3:4], sorted(p.items[4:])] == \
        [[10, 11, 12], [20], [30, 31]]
    for r in (t, p):       # window-independent fields survive the copy
        assert (r.history_items, r.extra_strata, r.target, r.n_txn) == \
            (s.history_items, s.extra_strata, 40, 3)


def test_masking_uses_full_history_and_extra_strata():
    # window = [[1]]; item 0 was bought before the window and outranks the target
    seqs = {0: _seq(0, [[1]], target=2, full=[0, 1], strata=("last_single", "basket_t1")),
            1: _seq(1, [[1, 3]], target=2, full=[1, 3], strata=("last_multi", "basket_t1"))}
    scorer = lambda b: torch.tensor([[9.0, 1.0, 5.0, 0.0, 0.0]]).repeat(      # noqa: E731
        b["item_tokens"].shape[0], 1)
    res = evaluate(scorer, seqs, SequenceCollator(8, 4), {"0": [3, 4], "1": [0, 4]},
                   num_items=5, ks=(1,))
    by_user = {r.user_idx: r for r in res["per_user"]}
    assert by_user[0].full["recall@1"] == 1.0          # item 0 masked via history_items
    assert by_user[0].full_nomask["recall@1"] == 0.0
    assert by_user[1].full["recall@1"] == 0.0          # item 0 is not in user 1's history
    st = res["full_catalog"]
    assert (st["last_single"]["n"], st["last_multi"]["n"], st["basket_t1"]["n"]) == (1, 1, 2)
    # Katchers-style records (no extra strata) keep the old key set
    plain = {0: _seq(0, [[0], [1]], target=2)}
    keys = evaluate(scorer, plain, SequenceCollator(8, 4), {"0": [3, 4]}, 5, ks=(1,))
    assert not [k for k in keys["full_catalog"] if k.startswith(("last_", "basket_"))]


def test_select_protocol_nomask():
    from test_train_smoke import _synth

    from mvcp.data import negatives as neg
    tr, va = _synth(60, 30, seed=1), _synth(20, 30, seed=2)
    vneg = neg.build(va, 30, 0, "corrected", n=10)
    scores = {}
    for proto in ("full_catalog", "full_nomask"):
        torch.manual_seed(0)
        model, col = build_model("txf_txn_causal", 30, dim=16, max_seq_len=16, max_num_txn=8)
        cfg = TrainConfig(lr=1e-2, epochs=2, batch_size=16, eval_every=1, patience=3,
                          seed=0, select_protocol=proto)
        info = train_model(model, col, tr, va, vneg, 30, cfg)
        res = evaluate(model, va, col, vneg, 30)
        scores[proto] = info["best_val_score"]
        assert info["best_val_score"] == res[proto]["overall"]["ndcg@10"]
    assert scores["full_catalog"] != scores["full_nomask"]


def _name(argv):
    return cli._result_name(cli._train_parser().parse_args(argv))


def _write(out, name, seed, vals, strata):
    recs = []
    for u, v in enumerate(vals):
        mets = {f"{m}@{k}": float(v) for m in ("ndcg", "recall") for k in (1, 5, 10, 20)}
        mets["mrr"] = float(v)
        recs.append({"user_idx": u, "repeat": u % 4 == 0, "has_copurchase": True,
                     "n_txn": 9, "cold": False, "full": mets,
                     "sampled": {k: x / 4 for k, x in mets.items()},
                     "full_nomask": {k: x / 2 for k, x in mets.items()},
                     "extra_strata": [strata[u % len(strata)]]})
    pu = out / "per_user"
    pu.mkdir(parents=True, exist_ok=True)
    with gzip.open(pu / f"{name}_seed{seed}.json.gz", "wt") as f:
        json.dump(recs, f)


def test_compare_extra_strata_primary_and_protocols(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(5):
        base = rng.uniform(0, 0.5, size=200)
        gain = np.where(np.arange(200) % 2 == 0, 0.08, -0.02)   # only `last_multi` gains
        _write(tmp_path, "a", s, base, ["last_multi", "last_single"])
        _write(tmp_path, "b", s, base + gain, ["last_multi", "last_single"])
    res = compare(tmp_path, "a", "b", list(range(5)), n_boot=200, primary="full_nomask",
                  protocols=["full", "full_nomask", "sampled"])
    row = {(r["protocol"], r["stratum"]): r for r in res["rows"]}
    assert abs(row["full_nomask", "last_multi"]["delta"] - 0.04) < 1e-9
    assert abs(row["full_nomask", "last_single"]["delta"] + 0.01) < 1e-9
    assert abs(row["sampled", "overall"]["delta"] - 0.0075) < 1e-9
    assert row["full", "last_multi"]["n_users"] == 100
    assert res["primary"] == "full_nomask" and res["gates"]["pass"]
    assert "overall, full_nomask" in render(res)
