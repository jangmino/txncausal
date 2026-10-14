"""H5 order-structure augmentation (train/loop._augment)."""
import numpy as np

from mvcp.train.loop import _augment


def _groups(s):
    g = {}
    for it, tx in zip(s.items, s.txn_ids):
        g.setdefault(tx, []).append(it)
    return [g[t] for t in sorted(g)]


def test_zero_prob_is_identity(make_seq):
    s = make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)
    a = _augment(s, np.random.default_rng(0), 0.0, 0.0)
    assert (a.items, a.txn_ids, a.target) == (s.items, s.txn_ids, s.target)


def test_txn_drop_keeps_newest_and_renumbers(make_seq):
    s = make_seq(0, [[10, 11, 12], [20], [21], [30, 31]], 40)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        a = _augment(s, rng, 0.5, 0.0)
        gs = _groups(a)
        assert gs[-1] == [30, 31] and a.target == 40          # newest order always kept
        assert sorted(set(a.txn_ids)) == list(range(len(gs)))   # contiguous 0..T'-1
        assert all(g in ([10, 11, 12], [20], [21], [30, 31]) for g in gs)   # orders whole
        seen.add(len(gs))
    assert seen == {1, 2, 3, 4}
    assert _augment(s, rng, 1.0, 0.0).items == [30, 31]


def test_item_drop_keeps_one_per_order(make_seq):
    s = make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = _augment(s, rng, 0.0, 0.9)
        gs = _groups(a)
        assert len(gs) == 3 and gs[1] == [20]
        assert set(gs[0]) <= {10, 11, 12} and set(gs[2]) <= {30, 31}
        assert all(len(g) >= 1 for g in gs)
        assert a.has_copurchase == any(len(g) >= 2 for g in gs)


def test_empty_history(make_seq):
    s = make_seq(0, [], 5)
    a = _augment(s, np.random.default_rng(0), 0.5, 0.5)
    assert a.items == [] and a.target == 5
