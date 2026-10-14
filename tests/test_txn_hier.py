"""H6 hierarchical set -> sequence encoder (txf_txn_hier): invariants I1/I2, order-level
causality, readout = last training state, empty-history safety."""
import torch

from mvcp.models.registry import build_model


def _build():
    torch.manual_seed(0)
    model, col = build_model("txf_txn_hier", num_items=100, dim=32, max_seq_len=16,
                             max_num_txn=8)
    model.eval()
    return model, col


def _states(model, b):
    with torch.no_grad():
        return model.encoder.hidden_states(model.item_embedding, b)


def test_i2_intra_order_permutation_invariant(make_seq):
    model, col = _build()
    ba = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    bb = col([make_seq(0, [[12, 10, 11], [20], [31, 30]], 40)])
    with torch.no_grad():
        assert torch.allclose(model.user_embedding(ba), model.user_embedding(bb), atol=1e-5)
        assert torch.allclose(model.train_loss(ba), model.train_loss(bb), atol=1e-5)


def test_i1_uses_order_membership(make_seq):
    model, col = _build()
    with torch.no_grad():
        a = model.user_embedding(col([make_seq(0, [[10, 11], [12], [30]], 40)]))
        b = model.user_embedding(col([make_seq(0, [[10], [11, 12], [30]], 40)]))
    assert not torch.allclose(a, b, atol=1e-4)


def test_order_causal(make_seq):
    model, col = _build()
    s1 = _states(model, col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))[0]
    s2 = _states(model, col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)]))[0]
    assert torch.allclose(s1[:2], s2[:2], atol=1e-6)
    assert not torch.allclose(s1[2], s2[2], atol=1e-4)


def test_readout_is_last_state_and_targets(toy_user):
    model, col = _build()
    b = col([toy_user])
    with torch.no_grad():
        assert torch.allclose(model.user_embedding(b), _states(model, b)[:, 2], atol=1e-6)
    assert list(zip(b["ntx_k"].tolist(), b["ntx_item"].tolist())) == \
        [(0, 20), (1, 30), (1, 31), (2, 40)]


def test_empty_history_and_finite_grads(make_seq):
    model, col = _build()
    b = col([make_seq(0, [], 5), make_seq(1, [[10, 11], [20]], 40)])
    with torch.no_grad():
        u = model.user_embedding(b)
    assert torch.equal(u[0], torch.zeros_like(u[0])) and torch.isfinite(u).all()
    model.train()
    loss = model.train_loss(b)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
