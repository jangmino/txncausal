"""Phase 0: sequence construction invariants."""
from mvcp.data.dataset import SequenceCollator


def test_txn_pos_shared_within_order(toy_user):
    b = SequenceCollator(max_seq_len=16, max_num_txn=8)([toy_user])
    tp = b["txn_pos"][0]
    tok = b["item_tokens"][0]
    real = tok != 0
    # 3 distinct transactions among real tokens
    assert set(tp[real].tolist()) == {1, 2, 3}
    # first transaction's three items share one txn position
    first = tp[real][:3].tolist()
    assert len(set(first)) == 1


def test_left_padding_and_newest_last(toy_user):
    b = SequenceCollator(max_seq_len=16, max_num_txn=8)([toy_user])
    tok = b["item_tokens"][0]
    # 6 real items -> 10 pads on the left, newest (31 -> token 32) at index -1
    assert int((tok == 0).sum()) == 16 - 6
    assert int(tok[-1]) == 31 + 1
    assert bool(b["pad_mask"][0][0]) and not bool(b["pad_mask"][0][-1])


def test_last_txn_mask_is_final_transaction(toy_user):
    b = SequenceCollator(max_seq_len=16, max_num_txn=8)([toy_user])
    last = b["last_txn"][0]
    toks = b["item_tokens"][0][last].tolist()
    assert sorted(toks) == sorted([30 + 1, 31 + 1])  # the last order [30,31]


def test_repeat_flag(make_seq):
    leaky = make_seq(1, [[10, 11], [12]], target=10)   # 10 is in history
    clean = make_seq(2, [[10, 11], [12]], target=99)
    assert leaky.repeat and not clean.repeat


def test_intra_order_dedup_on_real_data(real_bundle):
    # co-purchase counts distinct products; a user cannot have has_copurchase from a
    # single repeated product row
    st = real_bundle.seqs["test"]
    assert 0 < sum(s.has_copurchase for s in st.values()) < len(st)
