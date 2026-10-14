"""Component ablation: position (item / order) x
mask-and-targets (item-causal / block-causal).

Each variant changes ONE thing of its parent -- where the positional embedding comes
from -- and keeps the parent's mask, targets and loss. The existing names are untouched.
"""
import numpy as np
import pytest
import torch

from mvcp.data import negatives as neg
from mvcp.data.dataset import UserSeq
from mvcp.eval.harness import evaluate
from mvcp.models.registry import RUNGS, build_model
from mvcp.train.loop import TrainConfig, train_model

EMPTY = UserSeq(99, [], [], 3, False, False)
TXNS = [[10, 11, 12], [20], [30, 31]]
VARIANTS = ["sasrec_txnpos", "sasrec_txnpos_final2", "txf_txn_causal_itempos",
            "txf_txn_causal_itempos_final2", "bert4rec_txnpos", "bert4rec_txnpos_final2"]
PARENT = {"sasrec_txnpos": "sasrec", "sasrec_txnpos_final2": "sasrec_final2",
          "txf_txn_causal_itempos": "txf_txn_causal",
          "txf_txn_causal_itempos_final2": "txf_txn_causal_final2",
          "bert4rec_txnpos": "bert4rec", "bert4rec_txnpos_final2": "bert4rec_final2"}


def _build(name, L=16):
    torch.manual_seed(0)
    model, col = build_model(name, num_items=100, dim=32, max_seq_len=L, max_num_txn=8)
    model.eval()
    return model, col


def _hidden(model, batch):
    with torch.no_grad():
        return model.encoder.hidden_states(model.item_embedding, batch)


def _user(model, batch):
    with torch.no_grad():
        return model.user_embedding(batch)


# ---------------------------------------------------------------- definitions

@pytest.mark.parametrize("name", VARIANTS)
def test_variant_differs_from_its_parent_in_the_position_table_only(name):
    """Same spec as the parent plus ``pos``; same parameters except the position table,
    whose rows count orders instead of items (or the reverse)."""
    spec, parent = RUNGS[name], RUNGS[PARENT[name]]
    assert {k: v for k, v in spec.items() if k != "pos"} == parent
    a, b = _build(name)[0].state_dict(), _build(PARENT[name])[0].state_dict()
    assert a.keys() == b.keys()
    assert {k for k in a if a[k].shape != b[k].shape} == {"encoder.pos_emb.weight"}
    rows = a["encoder.pos_emb.weight"].shape[0]
    if name.startswith("txf_txn_causal"):
        assert rows == 16 + 1                      # one per item position + PAD
    else:
        assert rows == 8 + 2                       # one per order + PAD + the next order


@pytest.mark.parametrize("name,used,unused", [
    ("sasrec_txnpos", "txn_pos", "item_pos"),
    ("bert4rec_txnpos", "txn_pos", "item_pos"),
    ("txf_txn_causal_itempos", "item_pos", None),   # txn_pos still drives mask and pooling
])
def test_position_source(name, used, unused, toy_user):
    model, col = _build(name)
    batch = col([toy_user])
    real = ~batch["pad_mask"]
    base = _user(model, batch)
    moved = dict(batch, **{used: batch[used] + real.long() * (batch[used] < 8)})
    assert not torch.allclose(base, _user(model, moved), atol=1e-4)
    if unused:
        scrambled = dict(batch, **{unused: batch[unused].flip(1) * real})
        assert torch.equal(base, _user(model, scrambled))


# ---------------------------------------------------------------- SASRec + order positions

def test_sasrec_txnpos_keeps_the_item_causal_mask_and_labels(toy_user, make_seq):
    model, col = _build("sasrec_txnpos")
    a = col([make_seq(0, TXNS, 40)])
    b = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])
    ha, hb = _hidden(model, a), _hidden(model, b)
    assert torch.allclose(ha[:, :-1], hb[:, :-1], atol=1e-6)     # earlier positions
    assert not torch.allclose(ha[:, -1], hb[:, -1], atol=1e-4)
    parent_col = _build("sasrec")[1]
    assert torch.equal(a["seq_labels"], parent_col([toy_user])["seq_labels"])


def test_sasrec_txnpos_still_orders_the_items_of_an_order(make_seq):
    """Shared positions alone do not give invariance: the causal mask and the next-item
    readout still read the listing order."""
    model, col = _build("sasrec_txnpos")
    a = _user(model, col([make_seq(0, TXNS, 40)]))
    b = _user(model, col([make_seq(0, [[10, 11, 12], [20], [31, 30]], 40)]))
    assert not torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------- TxnCausal + item positions

def test_txn_causal_itempos_is_block_causal(make_seq):
    model, col = _build("txf_txn_causal_itempos")
    b = col([make_seq(0, TXNS, 40)])
    h1 = _hidden(model, b)
    h2 = _hidden(model, col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)]))
    early = (b["txn_pos"][0] < 3) & ~b["pad_mask"][0]
    assert torch.allclose(h1[0, early], h2[0, early], atol=1e-6)
    assert not torch.allclose(h1[0, -1], h2[0, -1], atol=1e-4)
    # symmetric inside an order: the first item of the newest order sees the second
    first = b["txn_pos"][0] == 3
    assert not torch.allclose(h1[0, first][0], h2[0, first][0], atol=1e-4)


def test_txn_causal_itempos_keeps_the_next_txn_targets(toy_user):
    col, parent_col = _build("txf_txn_causal_itempos")[1], _build("txf_txn_causal")[1]
    a, b = col([toy_user]), parent_col([toy_user])
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_txn_causal_itempos_loses_the_invariance(make_seq):
    variant, col = _build("txf_txn_causal_itempos")
    parent, _ = _build("txf_txn_causal")
    a = col([make_seq(0, TXNS, 40)])
    b = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
    assert torch.allclose(_user(parent, a), _user(parent, b), atol=1e-5)
    assert not torch.allclose(_user(variant, a), _user(variant, b), atol=1e-4)


# ---------------------------------------------------------------- BERT4Rec + order positions

def _captured_pos(model, batch):
    seen = {}
    real = model.encoder._encode
    model.encoder._encode = lambda ext: seen.update(ext) or real(ext)
    try:
        _hidden(model, batch)
    finally:
        del model.encoder._encode
    return seen["pos"]


def test_bert_txnpos_mask_slot_takes_the_next_order_position(toy_user):
    model, col = _build("bert4rec_txnpos")
    batch = col([toy_user, EMPTY])
    pos = _captured_pos(model, batch)
    assert pos.shape == (2, 17)
    assert pos[0, -7:].tolist() == [1, 1, 1, 2, 3, 3, 4]      # orders 1..3, then the next
    assert pos[1].tolist() == [0] * 16 + [1]                   # empty history
    assert torch.isfinite(_user(model, batch)).all()


def test_bert_txnpos_fits_the_longest_history(make_seq):
    """max_num_txn orders: the appended slot uses the last row of the table."""
    model, col = _build("bert4rec_txnpos")
    batch = col([make_seq(0, [[i + 1] for i in range(8)], 40)])
    pos = _captured_pos(model, batch)
    assert pos[0, -1] == 9 == model.encoder.pos_emb.num_embeddings - 1
    assert torch.isfinite(_user(model, batch)).all()


def test_bert_txnpos_is_invariant_within_an_order_and_blind_to_the_target(make_seq):
    """Shared positions + bidirectional attention: the listing order inside an order
    cannot matter. The parent (item positions) has no such property."""
    variant, col = _build("bert4rec_txnpos")
    parent, _ = _build("bert4rec")
    a = col([make_seq(0, TXNS, 40)])
    b = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
    assert torch.allclose(_user(variant, a), _user(variant, b), atol=1e-5)
    assert not torch.allclose(_user(parent, a), _user(parent, b), atol=1e-4)
    with torch.no_grad():
        assert torch.equal(variant(a), variant(col([make_seq(0, TXNS, 77)])))


# ---------------------------------------------------------------- all: PAD, loss, train loop

@pytest.mark.parametrize("name", VARIANTS)
def test_left_padding_does_not_change_scores_or_loss(name, toy_user):
    short, long_ = _build(name, L=8), _build(name, L=16)
    long_[0].load_state_dict({k: v for k, v in short[0].state_dict().items()
                              if k != "encoder.pos_emb.weight"}, strict=False)
    table = short[0].encoder.pos_emb.weight
    with torch.no_grad():
        long_[0].encoder.pos_emb.weight[:table.shape[0]] = table
    ua, ub = _user(short[0], short[1]([toy_user])), _user(long_[0], long_[1]([toy_user]))
    assert torch.allclose(ua, ub, atol=1e-5)
    if not name.startswith("bert4rec"):            # BERT4Rec draws a mask per slot
        la = short[0].train_loss(short[1]([toy_user]))
        lb = long_[0].train_loss(long_[1]([toy_user]))
        assert torch.allclose(la, lb, atol=1e-5)


@pytest.mark.parametrize("name", VARIANTS)
def test_train_eval_smoke(name):
    from test_train_smoke import _synth
    num_items = 30
    tr, va, te = (_synth(n, num_items, seed=s) for n, s in ((60, 1), (20, 2), (20, 3)))
    cfg = TrainConfig(lr=1e-2, epochs=4, batch_size=16, eval_every=2, patience=3, seed=0)
    infos = []
    for _ in range(2):                                 # the same seed: the same run
        torch.manual_seed(0)
        model, col = build_model(name, num_items, dim=16, max_seq_len=16, max_num_txn=8)
        infos.append(train_model(model, col, tr, va,
                                 neg.build(va, num_items, 0, "corrected", n=10),
                                 num_items, cfg))
    assert np.isfinite(infos[0]["best_val_score"]) and infos[0] == infos[1]
    model.eval()
    res = evaluate(model, te, col, neg.build(te, num_items, 0, "corrected", n=10),
                   num_items, ks=(1, 5, 10))
    assert res["n_users"] == 20
