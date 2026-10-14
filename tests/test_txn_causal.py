"""Transaction-causal L5: next-txn targets, block causality, L5 invariance kept."""
import torch

from mvcp.models.pooling import masked_mean
from mvcp.models.registry import build_model


def _build(name="txf_txn_causal"):
    torch.manual_seed(0)
    model, col = build_model(name, num_items=100, dim=32, max_seq_len=16, max_num_txn=8)
    model.eval()
    return model, col


def _hidden(model, batch):
    with torch.no_grad():
        return model.encoder.hidden_states(model.item_embedding, batch)


def test_next_txn_targets(toy_user):
    _, col = _build()
    b = col([toy_user])
    # [10,11,12] -> {20} ; [20] -> {30,31} ; [30,31] -> target 40
    assert list(zip(b["ntx_k"].tolist(), b["ntx_item"].tolist())) == \
        [(0, 20), (1, 30), (1, 31), (2, 40)]
    assert (b["ntx_row"] == 0).all()


def test_block_causal(make_seq):
    model, col = _build()
    b = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    h1 = _hidden(model, b)
    h2 = _hidden(model, col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)]))
    early = (b["txn_pos"][0] < 3) & ~b["pad_mask"][0]
    assert torch.allclose(h1[0, early], h2[0, early], atol=1e-6)
    assert not torch.allclose(h1[0, -1], h2[0, -1], atol=1e-4)


def test_intra_txn_permutation_invariant(make_seq):
    model, col = _build()
    with torch.no_grad():
        a = model.user_embedding(col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
        b = model.user_embedding(col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)]))
    assert torch.allclose(a, b, atol=1e-5)


def test_pad_isolated_and_train_state_matches_readout(toy_user):
    model, col = _build()
    b = col([toy_user])
    h = _hidden(model, b)
    g = dict(b, item_tokens=b["item_tokens"].masked_fill(b["pad_mask"], 77))
    real = ~b["pad_mask"][0]
    assert torch.isfinite(h).all()
    assert torch.allclose(h[0, real], _hidden(model, g)[0, real], atol=1e-6)
    last = masked_mean(h[:, b["last_txn"][0]], torch.ones(1, int(b["last_txn"].sum()),
                                                          dtype=torch.bool))
    with torch.no_grad():
        assert torch.allclose(last, model.user_embedding(b), atol=1e-6)
        assert torch.isfinite(model.train_loss(b))


def test_train_eval_smoke():
    import numpy as np
    from test_train_smoke import _synth

    from mvcp.data import negatives as neg
    from mvcp.eval.harness import evaluate
    from mvcp.train.loop import TrainConfig, train_model
    num_items = 30
    tr, va, te = (_synth(n, num_items, seed=s) for n, s in ((60, 1), (20, 2), (20, 3)))
    model, col = build_model("txf_txn_causal", num_items, dim=16, max_seq_len=16,
                             max_num_txn=8)
    cfg = TrainConfig(lr=1e-2, epochs=4, batch_size=16, eval_every=2, patience=3, seed=0)
    info = train_model(model, col, tr, va, neg.build(va, num_items, 0, "corrected", n=10),
                       num_items, cfg)
    assert np.isfinite(info["best_val_score"])
    model.eval()
    res = evaluate(model, te, col, neg.build(te, num_items, 0, "corrected", n=10),
                   num_items, ks=(1, 5, 10))
    assert res["n_users"] == 20


# ---- H1: blend readout (txf_txn_causal_blend) -------------------------------------------

def test_blend_same_init_as_txn_causal():
    a, _ = _build("txf_txn_causal_blend")
    b, _ = _build()
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) - set(sb) == {"encoder.blend_query", "encoder.blend_gate"}
    assert all(torch.equal(sa[k], sb[k]) for k in sb)
    assert not sa["encoder.blend_query"].any() and not sa["encoder.blend_gate"].any()


def test_blend_i2_intra_txn_permutation_invariant(make_seq):
    model, col = _build("txf_txn_causal_blend")
    with torch.no_grad():
        model.encoder.blend_query.normal_()         # non-uniform attention
        model.encoder.blend_gate.normal_()
        ba = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
        bb = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
        a, b = model.user_embedding(ba), model.user_embedding(bb)
        sa = model.encoder.txn_states(_hidden(model, ba), ba)[0, :3]
        sb = model.encoder.txn_states(_hidden(model, bb), bb)[0, :3]
        la, lb = model.train_loss(ba), model.train_loss(bb)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(sa, sb, atol=1e-5)
    assert torch.allclose(la, lb, atol=1e-5)


def test_blend_i1_uses_order_membership(make_seq):
    """Same items in the same flattened order, different order grouping -> different user."""
    model, col = _build("txf_txn_causal_blend")
    with torch.no_grad():
        a = model.user_embedding(col([make_seq(0, [[10, 11], [12], [30]], 40)]))
        b = model.user_embedding(col([make_seq(0, [[10], [11, 12], [30]], 40)]))
    assert not torch.allclose(a, b, atol=1e-4)


def test_blend_block_causal_states(make_seq):
    model, col = _build("txf_txn_causal_blend")
    with torch.no_grad():
        model.encoder.blend_query.normal_()
        b1 = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
        b2 = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])
        s1 = model.encoder.txn_states(_hidden(model, b1), b1)[0]
        s2 = model.encoder.txn_states(_hidden(model, b2), b2)[0]
    assert torch.allclose(s1[:2], s2[:2], atol=1e-6)      # txns 0,1 never see txn 2
    assert not torch.allclose(s1[2], s2[2], atol=1e-4)


def test_blend_readout_is_last_train_state_and_reduces_to_mean(toy_user, make_seq):
    model, col = _build("txf_txn_causal_blend")
    b = col([toy_user])
    with torch.no_grad():
        model.encoder.blend_query.normal_()
        states = model.encoder.txn_states(_hidden(model, b), b)
        assert torch.allclose(model.user_embedding(b), states[:, 2], atol=1e-6)
        # g -> 1 recovers the txf_txn_causal readout (last-transaction mean)
        model.encoder.blend_gate.fill_(50.0)
        mean_model, _ = _build()
        assert torch.allclose(model.user_embedding(b), mean_model.user_embedding(b),
                              atol=1e-5)


def test_blend_empty_history_and_finite_grads(make_seq):
    model, col = _build("txf_txn_causal_blend")
    model.train()
    b = col([make_seq(0, [], 5), make_seq(1, [[10, 11], [20]], 40)])
    with torch.no_grad():
        model.eval()
        u = model.user_embedding(b)
    assert torch.equal(u[0], torch.zeros_like(u[0])) and torch.isfinite(u).all()
    model.train()
    loss = model.train_loss(b)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.blend_query.grad.abs().sum() > 0


# ---- H2: txn-distance relative attention bias (txf_txn_causal_relb[_noabs]) --------------

def test_relb_buckets():
    model, _ = build_model("txf_txn_causal_relb", num_items=100, dim=32, max_seq_len=64,
                           max_num_txn=52)
    got = model.encoder.rel_bucket.tolist()
    assert len(got) == 53 and got[:11] == [0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5]
    assert set(got[9:]) == {5}
    assert tuple(model.encoder.rel_bias.shape) == (4, 6) and not model.encoder.rel_bias.any()


def test_relb_eval_matches_train_path(toy_user):
    """No-grad eval must not take the fast path that misreads float masks."""
    model, col = _build("txf_txn_causal_relb")
    b = col([toy_user])
    with torch.no_grad():
        model.encoder.rel_bias.normal_()
        ev = model.user_embedding(b)
    model.encoder.dropout.p = 0.0
    for layer in model.encoder.transformer.layers:
        layer.dropout.p = layer.dropout1.p = layer.dropout2.p = 0.0
        layer.self_attn.dropout = 0.0
    model.train()
    tr = model.user_embedding(b)                    # grad-enabled training path
    assert torch.isfinite(ev).all() and torch.allclose(ev, tr, atol=1e-5)
    assert torch.backends.mha.get_fastpath_enabled()  # global flag restored


def test_relb_zero_bias_equals_txn_causal(toy_user):
    a, col = _build("txf_txn_causal_relb")
    b, _ = _build()
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) - set(sb) == {"encoder.rel_bias"} and all(torch.equal(sa[k], sb[k]) for k in sb)
    bt = col([toy_user])
    with torch.no_grad():
        assert torch.allclose(a.user_embedding(bt), b.user_embedding(bt), atol=1e-5)


def test_relb_i2_intra_txn_permutation_invariant(make_seq):
    for name in ("txf_txn_causal_relb", "txf_txn_causal_relb_noabs"):
        model, col = _build(name)
        with torch.no_grad():
            model.encoder.rel_bias.normal_()
            a = model.user_embedding(col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
            b = model.user_embedding(col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)]))
        assert torch.allclose(a, b, atol=1e-5), name


def test_relb_block_causal_and_bias_is_used(make_seq):
    model, col = _build("txf_txn_causal_relb")
    b1 = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    b2 = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])
    with torch.no_grad():
        model.encoder.rel_bias.normal_()
        h1, h2 = _hidden(model, b1), _hidden(model, b2)
        early = (b1["txn_pos"][0] < 3) & ~b1["pad_mask"][0]
        assert torch.allclose(h1[0, early], h2[0, early], atol=1e-6)
        u1 = model.user_embedding(b1)
        model.encoder.rel_bias[:, 1] += 3.0          # re-weight distance-1 keys only
        assert not torch.allclose(u1, model.user_embedding(b1), atol=1e-4)


def test_relb_grad_reaches_bias(toy_user):
    model, col = _build("txf_txn_causal_relb")
    model.train()
    loss = model.train_loss(col([toy_user]))
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.rel_bias.grad.abs().sum() > 0


def test_relb_noabs_ignores_absolute_position(toy_user):
    model, col = _build("txf_txn_causal_relb_noabs")
    b = col([toy_user])
    with torch.no_grad():
        u1 = model.user_embedding(b)
        model.encoder.pos_emb.weight.normal_()
        assert torch.allclose(u1, model.user_embedding(b), atol=1e-6)


# ---- H3: intra-order cloze (txf_txn_causal_cloze*) ---------------------------------------

def test_cloze_eval_unchanged_and_same_init(toy_user, make_seq):
    a, col = _build("txf_txn_causal_cloze0.3")
    b, _ = _build()
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) - set(sb) == {"mask_emb"} and all(torch.equal(sa[k], sb[k]) for k in sb)
    with torch.no_grad():
        bt = col([toy_user])
        assert torch.equal(a.user_embedding(bt), b.user_embedding(bt))
        # I2 carries over (eval path is the champion's)
        u1 = a.user_embedding(col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
        u2 = a.user_embedding(col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)]))
    assert torch.allclose(u1, u2, atol=1e-5)


def test_cloze_one_masked_item_per_multi_item_order(make_seq):
    model, col = _build("txf_txn_causal_cloze0.3")
    b = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40), make_seq(1, [[5], [6]], 7)])
    torch.manual_seed(1)
    rows, pos = model.cloze_positions(b)
    assert rows.tolist() == [0, 0]                      # user 1 has no multi-item order
    txns = b["txn_pos"][rows, pos].tolist()
    assert txns == [1, 3]                               # one pick in txn 1 and one in txn 3
    assert not b["pad_mask"][rows, pos].any()


def test_cloze_masked_item_is_hidden_and_loss_trains(make_seq):
    model, col = _build("txf_txn_causal_cloze0.3")
    b1 = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    b2 = col([make_seq(0, [[10, 11, 99], [20], [30, 31]], 40)])   # item 12 -> 99
    where = b1["item_tokens"] == 13                                  # token id = item + 1
    with torch.no_grad():
        model.mask_emb.normal_()
        h1 = model.encoder.hidden_states(model.item_embedding, b1, replace=(where, model.mask_emb))
        h2 = model.encoder.hidden_states(model.item_embedding, b2, replace=(where, model.mask_emb))
    assert torch.allclose(h1, h2, atol=1e-6)            # the masked item never leaks
    model.train()
    torch.manual_seed(0)
    loss = model.train_loss(b1)
    loss.backward()
    assert torch.isfinite(loss) and model.mask_emb.grad.abs().sum() > 0


# ---- H4: objective weighting (txf_txn_causal_{bnorm,final2,final4,k2}) --------------------

def _manual_terms(model, b):
    import torch.nn.functional as F
    with torch.no_grad():
        states = model.encoder.txn_states(_hidden(model, b), b)
        s = states[b["ntx_row"], b["ntx_k"]]
        return F.cross_entropy(s @ model.item_embedding.weight[1:].t(), b["ntx_item"],
                               reduction="none")


def test_h4_weights_match_definition(toy_user):
    # toy transitions: k0 -> {20}, k1 -> {30, 31}, k2 -> target 40 (the final term)
    base, col = _build()
    b = col([toy_user])
    ce = _manual_terms(base, b)
    for name, w in (("txf_txn_causal_bnorm", [1, .5, .5, 1]),
                    ("txf_txn_causal_final2", [1, 1, 1, 2]),
                    ("txf_txn_causal_final4", [1, 1, 1, 4])):
        m, _ = _build(name)
        w = torch.tensor(w, dtype=torch.float)
        with torch.no_grad():
            assert torch.allclose(m.train_loss(b), (w * ce).sum() / w.sum(), atol=1e-6), name
            assert torch.equal(m.user_embedding(b), base.user_embedding(b))   # eval unchanged
    with torch.no_grad():
        assert torch.allclose(base.train_loss(b), ce.mean(), atol=1e-6)


def test_h4_k2_targets_and_loss(toy_user):
    import torch.nn.functional as F
    m, col = _build("txf_txn_causal_k2")
    b = col([toy_user])
    # state0 -> txn2 {30, 31}; state1 -> target 40; the last state has no k+2
    assert list(zip(b["ntx2_k"].tolist(), b["ntx2_item"].tolist())) == [(0, 30), (0, 31), (1, 40)]
    with torch.no_grad():
        states = m.encoder.txn_states(_hidden(m, b), b)
        s2 = states[b["ntx2_row"], b["ntx2_k"]]
        k2 = F.cross_entropy(s2 @ m.item_embedding.weight[1:].t(), b["ntx2_item"])
        assert torch.allclose(m.train_loss(b), _manual_terms(m, b).mean() + 0.5 * k2, atol=1e-6)
    one = col([__import__("conftest")._seq(0, [[10, 11]], 40)])    # single txn: no k+2 term
    assert one["ntx2_row"].numel() == 0
    with torch.no_grad():
        assert torch.isfinite(m.train_loss(one))


def test_h4_i2_loss_permutation_invariant(make_seq):
    for name in ("txf_txn_causal_bnorm", "txf_txn_causal_final2", "txf_txn_causal_k2"):
        m, col = _build(name)
        with torch.no_grad():
            la = m.train_loss(col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
            lb = m.train_loss(col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)]))
        assert torch.allclose(la, lb, atol=1e-5), name


# ---- round 2: H1 retry (blend over LayerNorm'd raw item embeddings) + combos --------------

def test_blendemb_invariants_and_scale(make_seq, toy_user):
    model, col = _build("txf_txn_causal_final2_blendemb")
    base, _ = _build("txf_txn_causal_final2")
    sa, sb = model.state_dict(), base.state_dict()
    assert set(sa) - set(sb) == {"encoder.blend_query", "encoder.blend_gate",
                                 "encoder.blend_norm.weight", "encoder.blend_norm.bias"}
    assert all(torch.equal(sa[k], sb[k]) for k in sb)          # shared init unchanged
    with torch.no_grad():
        model.encoder.blend_query.normal_()
        ba = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
        bb = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
        assert torch.allclose(model.user_embedding(ba), model.user_embedding(bb), atol=1e-5)
        assert torch.allclose(model.train_loss(ba), model.train_loss(bb), atol=1e-5)   # I2
        a = model.user_embedding(col([make_seq(0, [[10, 11], [12], [30]], 40)]))
        b = model.user_embedding(col([make_seq(0, [[10], [11, 12], [30]], 40)]))
        assert not torch.allclose(a, b, atol=1e-4)                                      # I1
        # the pooled channel is LayerNorm'd: per-token norm ~ sqrt(dim), not ~ |item emb|
        e = model.encoder.blend_norm(model.item_embedding(ba["item_tokens"]))
        assert abs(e[~ba["pad_mask"]].norm(dim=-1).mean() - 32 ** 0.5) < 0.5
        states = model.encoder.txn_states(_hidden(model, ba), ba, model.item_embedding)
        assert torch.allclose(model.user_embedding(ba), states[:, 2], atol=1e-6)
        b1 = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])
        s1 = model.encoder.txn_states(_hidden(model, b1), b1, model.item_embedding)
        assert torch.allclose(states[0, :2], s1[0, :2], atol=1e-6)                      # causal


def test_round2_combos_match_flags():
    m, _ = _build("txf_txn_causal_final2_bnorm")
    assert m.final_weight == 2.0 and m.basket_norm
    m, _ = _build("txf_txn_causal_final2_cloze1")
    assert m.final_weight == 2.0 and m.cloze == 1.0


# ---- structural-effect round: H8 order-size embedding, H9 order-local first layer --------

def test_h8_osize_invariants(make_seq, toy_user):
    m, col = _build("txf_txn_causal_final2_osize")
    base, _ = _build("txf_txn_causal_final2")
    sa, sb = m.state_dict(), base.state_dict()
    assert set(sa) - set(sb) == {"encoder.size_table"} and all(torch.equal(sa[k], sb[k]) for k in sb)
    b = col([toy_user])
    with torch.no_grad():
        assert torch.equal(m.user_embedding(b), base.user_embedding(b))   # zero init = champion
        m.encoder.size_table.normal_()
        ba = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
        bb = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
        assert torch.allclose(m.user_embedding(ba), m.user_embedding(bb), atol=1e-5)    # I2
        assert torch.allclose(m.train_loss(ba), m.train_loss(bb), atol=1e-5)
        m.encoder.size_table[1:] = m.encoder.size_table[0]    # sizes 2,3,4+ look like size 1
        flat = m.user_embedding(ba)
        delta = torch.randn(32)                 # not a constant shift (LayerNorm removes it)
        m.encoder.size_table[2] += delta                        # only size-3 orders change
        assert not torch.allclose(flat, m.user_embedding(ba), atol=1e-4)                # I1
        one = col([make_seq(0, [[10], [20], [30]], 40)])        # no size-3 order
        u1 = m.user_embedding(one)
        m.encoder.size_table[2] -= delta
        assert torch.allclose(u1, m.user_embedding(one), atol=1e-6)
    m.train()
    loss = m.train_loss(b)
    loss.backward()
    assert m.encoder.size_table.grad.abs().sum() > 0


def test_h9_local_first_invariants(make_seq):
    m, col = _build("txf_txn_causal_final2_local1")
    ba = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    bb = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
    with torch.no_grad():
        assert torch.allclose(m.user_embedding(ba), m.user_embedding(bb), atol=1e-5)    # I2
        assert torch.allclose(m.train_loss(ba), m.train_loss(bb), atol=1e-5)
        # order-level causality across both layers: txns 0,1 never see txn 2
        b2 = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])
        h1, h2 = _hidden(m, ba), _hidden(m, b2)
        early = (ba["txn_pos"][0] < 3) & ~ba["pad_mask"][0]
        assert torch.allclose(h1[0, early], h2[0, early], atol=1e-6)
    # first layer is order-local: a 1-layer model's txn-2 tokens ignore txn 0/1 items
    torch.manual_seed(0)
    one, col1 = build_model("txf_txn_causal_final2_local1", num_items=100, dim=32,
                            max_seq_len=16, max_num_txn=8, num_layers=1)
    one.eval()
    c1 = col1([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    c2 = col1([make_seq(0, [[13, 14, 15], [21], [30, 31]], 40)])
    with torch.no_grad():
        g1, g2 = _hidden(one, c1), _hidden(one, c2)
    last = c1["last_txn"][0]
    assert torch.allclose(g1[0, last], g2[0, last], atol=1e-6)
    base, _ = _build("txf_txn_causal_final2")          # 2-layer local1 != block-causal champion
    with torch.no_grad():
        assert not torch.allclose(m.user_embedding(ba), base.user_embedding(ba), atol=1e-4)


def test_h7_struct_param_groups():
    from mvcp.train.loop import TrainConfig, _param_groups
    m, _ = _build("txf_txn_causal_final2_relb")
    assert not isinstance(_param_groups(m, TrainConfig(lr=1e-3)), list)   # default: unchanged
    groups = _param_groups(m, TrainConfig(lr=1e-3, struct_lr_mult=20))
    assert groups[1]["lr"] == 0.02 and [p.shape for p in groups[1]["params"]] == [(4, 6)]
    assert len(groups[0]["params"]) + 1 == len(list(m.parameters()))
    b, _ = _build("txf_txn_causal_final2_blendemb")
    g = _param_groups(b, TrainConfig(lr=1e-3, struct_lr_mult=20))
    assert len(g[1]["params"]) == 2                     # query + gate (LayerNorm stays base lr)
    import pytest
    with pytest.raises(ValueError):
        _param_groups(_build()[0], TrainConfig(lr=1e-3, struct_lr_mult=20))


def test_round3_combo_blendemb_relb(make_seq):
    from mvcp.train.loop import TrainConfig, _param_groups
    m, col = _build("txf_txn_causal_final2_blendemb_relb")
    assert m.encoder.readout == "blend_emb" and m.encoder.rel_bias is not None
    assert len(_param_groups(m, TrainConfig(lr=1e-3, struct_lr_mult=20))[1]["params"]) == 3
    with torch.no_grad():
        m.encoder.rel_bias.normal_()
        m.encoder.blend_query.normal_()
        a = m.user_embedding(col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
        b = m.user_embedding(col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)]))
    assert torch.allclose(a, b, atol=1e-5)


def test_h10_token_level_next_order_loss(toy_user, make_seq):
    import torch.nn.functional as F
    m, col = _build("txf_txn_causal_final2_blendemb_tok1")
    base, _ = _build("txf_txn_causal_final2_blendemb")
    b = col([toy_user])
    with torch.no_grad():
        assert torch.equal(m.user_embedding(b), base.user_embedding(b))     # eval unchanged
        h = _hidden(m, b)
        logp = F.log_softmax(h[0] @ m.item_embedding.weight[1:].t(), -1)   # (L,N)
        tp = b["txn_pos"][0]
        # order1 tokens {10,11,12} -> 20 ; order2 token {20} -> 30, 31 ; order3 {30,31} -> 40
        pairs = [(i, 20) for i in (tp == 1).nonzero().flatten()] \
            + [(i, t) for i in (tp == 2).nonzero().flatten() for t in (30, 31)] \
            + [(i, 40) for i in (tp == 3).nonzero().flatten()]
        expect = -torch.stack([logp[i, t] for i, t in pairs]).mean()
        assert len(pairs) == 7
        assert torch.allclose(m.token_loss(h, b), expect, atol=1e-6)
        assert torch.allclose(m.train_loss(b), base.train_loss(b) + expect, atol=1e-5)
        la = m.train_loss(col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
        lb = m.train_loss(col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)]))
    assert torch.allclose(la, lb, atol=1e-5)                                  # I2
