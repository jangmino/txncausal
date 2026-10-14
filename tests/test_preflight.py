"""Pre-flight gates PF1-PF7 on the real data (skips if data absent)."""
import numpy as np

from mvcp.data import negatives as neg
from mvcp.data.split import structural_checks
from mvcp.data.vocab import load_frames


def test_p1_split_structure(real_bundle):
    sc = structural_checks(load_frames("data"))
    assert sc["counts"] == {"train": 10071, "val": 3358, "test": 3358}
    assert sc["disjoint"]
    assert sc["users_with_multi_row_last_date"] == 0
    assert sc["ratios"]["train"] == 0.6


def test_vocab_dims(real_bundle):
    assert real_bundle.num_items == 1608
    assert real_bundle.vocab.num_users == 16787


def test_p4_target_in_history_leak(real_bundle):
    # reproduce the confirmed leak rate: test 12.06% (405/3358)
    st = real_bundle.seqs["test"]
    leak = sum(s.repeat for s in st.values())
    assert leak == 405
    assert abs(100 * leak / len(st) - 12.06) < 0.1


def test_p2_p3_p7_published_negatives_clean(real_bundle):
    pub = neg.load_published("data/negatives/published.json")
    rep = neg.validate(pub, real_bundle.seqs["test"], real_bundle.num_items)
    assert rep["coverage_ok"]                 # all 3358 test users -> no random fallback
    assert rep["all_len_n"]                   # exactly 49 each
    assert rep["index_range_ok"]              # valid item_idx space
    assert rep["users_target_in_neg"] == 0    # negatives exclude the target
    assert rep["users_history_in_neg"] == 0   # negatives exclude history
    assert rep["clean"]


def test_corrected_negatives_exclude_history_and_target(real_bundle):
    corrected = neg.build(real_bundle.seqs["test"], real_bundle.num_items, seed=0,
                          mode="corrected")
    rep = neg.validate(corrected, real_bundle.seqs["test"], real_bundle.num_items)
    assert rep["clean"]
