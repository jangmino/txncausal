"""SASRec baseline: next-item labels, causality, PAD isolation, L3-identical init."""
import numpy as np
import torch

from mvcp.data import negatives as neg
from mvcp.eval.harness import evaluate
from mvcp.models.registry import build_model
from mvcp.train.loop import TrainConfig, train_model


def _build(name="sasrec", L=16):
    torch.manual_seed(0)
    model, col = build_model(name, num_items=100, dim=32, max_seq_len=L, max_num_txn=8)
    model.eval()
    return model, col


def _hidden(model, batch):
    with torch.no_grad():
        return model.encoder.hidden_states(model.item_embedding, batch)


def test_labels_are_next_items_ending_in_target(toy_user):
    _, col = _build(L=16)
    lab = col([toy_user])["seq_labels"][0]
    # history [10,11,12 | 20 | 30,31] -> target 40
    assert lab[-6:].tolist() == [11, 12, 20, 30, 31, 40]
    assert (lab[:-6] == -100).all()


def test_labels_truncated_with_inputs(toy_user):
    _, col = _build(L=4)
    b = col([toy_user])
    assert (b["item_tokens"][0] - 1).tolist() == [12, 20, 30, 31]
    assert b["seq_labels"][0].tolist() == [20, 30, 31, 40]


def test_causal_future_item_does_not_leak(make_seq):
    model, col = _build()
    a = _hidden(model, col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
    b = _hidden(model, col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)]))
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)   # all earlier positions
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-4)  # newest one does change


def test_pad_tokens_invisible_to_real_positions(toy_user):
    model, col = _build()
    batch = col([toy_user])
    h = _hidden(model, batch)
    garbage = dict(batch, item_tokens=batch["item_tokens"].masked_fill(batch["pad_mask"], 77))
    real = ~batch["pad_mask"][0]
    assert torch.isfinite(h).all()
    assert torch.allclose(h[0, real], _hidden(model, garbage)[0, real], atol=1e-6)


def test_train_eval_smoke():
    from test_train_smoke import _synth
    num_items = 30
    tr, va, te = (_synth(n, num_items, seed=s) for n, s in ((60, 1), (20, 2), (20, 3)))
    model, col = build_model("sasrec", num_items, dim=16, max_seq_len=16, max_num_txn=8)
    cfg = TrainConfig(lr=1e-2, epochs=4, batch_size=16, eval_every=2, patience=3, seed=0)
    info = train_model(model, col, tr, va, neg.build(va, num_items, 0, "corrected", n=10),
                       num_items, cfg)
    assert np.isfinite(info["best_val_score"])
    model.eval()
    res = evaluate(model, te, col, neg.build(te, num_items, 0, "corrected", n=10),
                   num_items, ks=(1, 5, 10))
    assert res["n_users"] == 20


def test_train_frac_subsample_seeded(make_seq):
    from mvcp.cli import _subsample
    seqs = {u: make_seq(u, [[1], [2]], 3) for u in range(200)}
    a, b = _subsample(seqs, 0.25, 0), _subsample(seqs, 0.25, 0)
    assert list(a) == list(b) and len(a) == 50 and set(a) <= set(seqs)
    assert set(_subsample(seqs, 0.25, 1)) != set(a)


def test_final2_weights_newest_position(toy_user, make_seq):
    """sasrec_final2 (symmetric H4b control): newest position's CE x2, weighted mean;
    eval path and init identical to sasrec."""
    import torch.nn.functional as F
    base, col = _build()
    m, _ = _build("sasrec_final2")
    b = col([toy_user, make_seq(1, [], 5)])            # empty-history row: no labels at all
    with torch.no_grad():
        assert torch.equal(m.user_embedding(b), base.user_embedding(b))
        h = _hidden(m, b)
        lab = b["seq_labels"]
        ce = F.cross_entropy((h @ m.item_embedding.weight[1:].t()).flatten(0, 1),
                             lab.flatten(), ignore_index=-100, reduction="none").view_as(lab)
        real = ce[0, -6:]                               # toy_user: 6 labelled positions
        expect = (real[:-1].sum() + 2 * real[-1]) / 7
        assert torch.allclose(m.train_loss(b), expect, atol=1e-6)
        assert torch.allclose(base.train_loss(b), real.mean(), atol=1e-6)


def test_blendemb_symmetric_control(toy_user, make_seq):
    """sasrec_final2_blendemb (symmetric control for TxnCausal H7a): same shared init,
    causal pooling over positions <= t, readout = newest state, finite on empty history."""
    base, col = _build("sasrec_final2")
    m, _ = _build("sasrec_final2_blendemb")
    sa, sb = m.state_dict(), base.state_dict()
    assert set(sa) - set(sb) == {"encoder.blend_query", "encoder.blend_gate",
                                 "encoder.blend_norm.weight", "encoder.blend_norm.bias"}
    assert all(torch.equal(sa[k], sb[k]) for k in sb)
    b1 = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    b2 = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])   # only the newest item
    with torch.no_grad():
        m.encoder.blend_query.normal_()
        m.encoder.blend_gate.normal_()
        s1 = m.encoder.states(_hidden(m, b1), b1, m.item_embedding)
        s2 = m.encoder.states(_hidden(m, b2), b2, m.item_embedding)
        assert torch.allclose(s1[0, :-1], s2[0, :-1], atol=1e-6)          # causal
        assert not torch.allclose(s1[0, -1], s2[0, -1], atol=1e-4)
        assert torch.allclose(m.user_embedding(b1), s1[:, -1], atol=1e-6)
        e = col([make_seq(0, [], 5), toy_user])
        assert torch.isfinite(m.user_embedding(e)).all()
    m.train()
    loss = m.train_loss(col([toy_user]))
    loss.backward()
    assert torch.isfinite(loss) and m.encoder.blend_query.grad.abs().sum() > 0
    from mvcp.train.loop import TrainConfig, _param_groups
    assert len(_param_groups(m, TrainConfig(lr=1e-3, struct_lr_mult=20))[1]["params"]) == 2
