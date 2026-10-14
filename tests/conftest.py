import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from mvcp.data.dataset import UserSeq


def _seq(user_idx, txns, target):
    """Build a UserSeq from a list of transactions (each a list of item_idx)."""
    items, txn_ids = [], []
    for t, grp in enumerate(txns):
        items.extend(grp)
        txn_ids.extend([t] * len(grp))
    has_cop = any(len(g) >= 2 for g in txns)
    return UserSeq(user_idx=user_idx, items=items, txn_ids=txn_ids, target=target,
                   repeat=target in set(items), has_copurchase=has_cop)


@pytest.fixture
def toy_user():
    # transactions [10,11,12] -> [20] -> [30,31], target 40 (novel)
    return _seq(0, [[10, 11, 12], [20], [30, 31]], 40)


@pytest.fixture
def make_seq():
    return _seq


@pytest.fixture(scope="session")
def real_bundle():
    from mvcp.pipeline import load_bundle
    data = ROOT / "data"
    if not (data / "test_data.csv").exists():
        pytest.skip("data not present")
    return load_bundle(data)
