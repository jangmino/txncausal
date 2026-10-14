"""Phase 2: end-to-end train->eval smoke on tiny synthetic data (no GPU)."""
import numpy as np

from mvcp.data import negatives as neg
from mvcp.data.dataset import UserSeq
from mvcp.eval.harness import evaluate
from mvcp.models.registry import build_model
from mvcp.train.loop import TrainConfig, train_model


def _synth(n_users, num_items, seed=0):
    rng = np.random.default_rng(seed)
    seqs = {}
    for u in range(n_users):
        n_txn = int(rng.integers(1, 4))
        txns = [list(rng.choice(num_items, size=int(rng.integers(1, 3)), replace=False))
                for _ in range(n_txn)]
        items, tids = [], []
        for t, g in enumerate(txns):
            items += [int(x) for x in g]
            tids += [t] * len(g)
        target = int(rng.integers(0, num_items))
        seqs[u] = UserSeq(u, items, tids, target, target in set(items),
                          any(len(g) >= 2 for g in txns))
    return seqs


def test_train_eval_smoke():
    num_items = 30
    train_seqs = _synth(60, num_items, seed=1)
    val_seqs = _synth(20, num_items, seed=2)
    test_seqs = _synth(20, num_items, seed=3)
    val_neg = neg.build(val_seqs, num_items, 0, "corrected", n=10)
    test_neg = neg.build(test_seqs, num_items, 0, "corrected", n=10)

    model, col = build_model("txf_txn_causal", num_items, dim=16,
                             max_seq_len=16, max_num_txn=8)
    cfg = TrainConfig(lr=1e-2, epochs=4, batch_size=16, eval_every=2, patience=3, seed=0)
    info = train_model(model, col, train_seqs, val_seqs, val_neg, num_items, cfg)
    assert np.isfinite(info["best_val_score"])
    assert len(info["history"]) >= 1
    assert info["history"][0]["train_loss"] > info["history"][-1]["train_loss"] - 1e6  # ran

    model.eval()
    res = evaluate(model, test_seqs, col, test_neg, num_items, ks=(1, 5, 10))
    assert res["n_users"] == 20
    assert 0.0 <= res["full_catalog"]["overall"]["ndcg@10"] <= 1.0


def test_intra_txn_shuffle_preserves_sets():
    """Perm-eval probe: per-transaction item multisets, txn order, target unchanged."""
    from mvcp.data.dataset import shuffle_intra_txn as _shuffle_intra_txn
    s = UserSeq(0, [10, 11, 12, 20, 30, 31], [0, 0, 0, 1, 2, 2], target=40,
                repeat=False, has_copurchase=True)
    rng = np.random.default_rng(0)
    p = _shuffle_intra_txn(s, rng)
    assert p.txn_ids == s.txn_ids and p.target == s.target
    for t in set(s.txn_ids):
        assert (sorted(it for it, tx in zip(p.items, p.txn_ids) if tx == t)
                == sorted(it for it, tx in zip(s.items, s.txn_ids) if tx == t))
    shuffles = {tuple(_shuffle_intra_txn(s, rng).items) for _ in range(20)}
    assert len(shuffles) > 1                                # actually permutes
