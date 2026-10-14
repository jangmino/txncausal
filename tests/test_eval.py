"""Phase 1: evaluation harness golden values (masking, tie-break, NDCG)."""
import numpy as np
import torch

from mvcp.data.dataset import SequenceCollator, UserSeq
from mvcp.eval.harness import evaluate


def _const_scorer(vec):
    v = torch.tensor(vec, dtype=torch.float32)

    def scorer(batch):
        b = batch["item_tokens"].shape[0]
        return v.unsqueeze(0).repeat(b, 1)
    return scorer


def _seq(u, txns, target):
    items, tids = [], []
    for t, g in enumerate(txns):
        items += g
        tids += [t] * len(g)
    return UserSeq(u, items, tids, target, target in {i for g in txns for i in g},
                   any(len(g) >= 2 for g in txns))


def test_ndcg_and_recall_topranked():
    seqs = {0: _seq(0, [[0], [1]], target=2)}
    negs = {"0": [1, 3, 4]}
    col = SequenceCollator(8, 4)
    res = evaluate(_const_scorer([0.5, 0.1, 0.9, 0.2, 0.0]), seqs, col, negs,
                   num_items=5, ks=(1, 2, 3))
    # target=2 has the highest score among unmasked items -> rank 0
    assert res["full_catalog"]["overall"]["recall@1"] == 1.0
    assert abs(res["full_catalog"]["overall"]["ndcg@1"] - 1.0) < 1e-9
    assert res["sampled_1to49"]["overall"]["recall@1"] == 1.0


def test_history_masking_applied():
    # item 0 (a history item) has the top score; masking must exclude it so target wins
    seqs = {0: _seq(0, [[0], [1]], target=2)}
    negs = {"0": [3, 4]}
    res = evaluate(_const_scorer([9.0, 1.0, 5.0, 0.0, 0.0]), seqs, SequenceCollator(8, 4),
                   negs, num_items=5, ks=(1,))
    assert res["full_catalog"]["overall"]["recall@1"] == 1.0  # rank 0 only if item0 masked


def test_midrank_tiebreak_neutral():
    # all-equal logits: target must land at the neutral mid-rank, not first
    seqs = {0: _seq(0, [[0]], target=2)}   # history item 0 (masked in full-catalog)
    negs = {"0": [1, 3, 4]}
    res = evaluate(_const_scorer([0.0, 0.0, 0.0, 0.0, 0.0]), seqs, SequenceCollator(8, 4),
                   negs, num_items=5, ks=(1, 3))
    # sampled: 4 tied candidates -> mid-rank = 1.5 -> mrr = 1/2.5 = 0.4
    assert abs(res["sampled_1to49"]["overall"]["mrr"] - 0.4) < 1e-6
    # full-catalog: item0 masked -> 4 tied items {1,2,3,4} -> mid-rank 1.5 -> mrr 0.4
    assert abs(res["full_catalog"]["overall"]["mrr"] - 0.4) < 1e-6


def test_strata_present():
    seqs = {0: _seq(0, [[0], [1]], target=2), 1: _seq(1, [[0, 1]], target=3)}
    negs = {"0": [3, 4], "1": [2, 4]}
    res = evaluate(_const_scorer([0.1, 0.2, 0.3, 0.4, 0.0]), seqs, SequenceCollator(8, 4),
                   negs, num_items=5, ks=(1, 2))
    for name in ("overall", "novel", "repeat", "copurchase", "single_only"):
        assert name in res["full_catalog"]


def test_full_nomask_golden():
    # novel user: history item 0 outscores the target -> masked rank 0, unmasked rank 1
    # repeat user: target 1 is in history; unmasked, history item 0 (9.0) still outranks it
    seqs = {0: _seq(0, [[0], [1]], target=2), 1: _seq(1, [[0], [1]], target=1)}
    negs = {"0": [3, 4], "1": [3, 4]}
    res = evaluate(_const_scorer([9.0, 1.0, 5.0, 0.0, 0.0]), seqs, SequenceCollator(8, 4),
                   negs, num_items=5, ks=(1, 2))
    by_u = {r.user_idx: r for r in res["per_user"]}
    assert by_u[0].full["mrr"] == 1.0 and by_u[0].full_nomask["mrr"] == 0.5
    # repeat user: masked -> {1,2,3,4} with target 1.0 < item2 5.0 -> rank 1;
    # unmasked -> items 0 (9.0) and 2 (5.0) above -> rank 2
    assert by_u[1].full["mrr"] == 0.5 and abs(by_u[1].full_nomask["mrr"] - 1 / 3) < 1e-9
    assert res["full_nomask"]["overall"]["recall@1"] == 0.0
    assert res["full_nomask"]["repeat"]["n"] == 1


def test_full_nomask_midrank_ties():
    # all-equal logits, nothing masked: 5 tied items -> mid-rank 2 -> mrr 1/3
    seqs = {0: _seq(0, [[0]], target=2)}
    res = evaluate(_const_scorer([0.0] * 5), seqs, SequenceCollator(8, 4), {"0": [1, 3]},
                   num_items=5, ks=(1,))
    assert abs(res["full_nomask"]["overall"]["mrr"] - 1 / 3) < 1e-9
    assert abs(res["full_catalog"]["overall"]["mrr"] - 0.4) < 1e-9   # unchanged protocol
