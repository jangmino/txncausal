"""Text protocol of the text baseline: the released history builder and the builder of
the retrained version. The first test pins the history handling of the released
builder (merged on the overflow path only); the second pins the retrained version."""
import pandas as pd

from mvcp.baselines._concurrentbert_text import compile_order_history
from mvcp.baselines.concurrentbert import compile_order_history_repaired


class _StubTok:
    """len(input_ids) == len(text): lets tests set the 512 'token' budget in chars."""
    def __call__(self, text, text_pair=None, **kw):
        class R:  # noqa: N801
            input_ids = list(range(len(text) + len(text_pair or "")))
        return R()


def _hist():
    return pd.DataFrame({
        "initial_paid_at": [1, 1, 2],
        "category_name":   ["양말", "양말", "식품"],
        "product_name":    ["페이크삭스", "장목양말", "꿀고구마"],
        "attribute_values": [None, None, "2kg"],
    })


def test_aspublished_returns_bare_header_when_fits():
    out = compile_order_history(_hist(), _StubTok(), item_max="x" * 10)
    assert out == "## 주문 히스토리"          # released builder: header only when the history fits


def test_repaired_merges_all_orders():
    out = compile_order_history_repaired(_hist(), _StubTok(), item_max="x" * 10)
    assert out.startswith("## 주문 히스토리")
    assert "[동시구매]" in out                 # first order has 2 items
    assert "페이크삭스" in out and "장목양말" in out and "꿀고구마 2kg" in out
    assert out.count("의 주문 내역") == 2      # a dated line per order


def test_repaired_respects_token_budget():
    long_hist = pd.concat([_hist()] * 60, ignore_index=True)
    long_hist["initial_paid_at"] = range(len(long_hist))
    out = compile_order_history_repaired(long_hist, _StubTok(), item_max="x" * 10)
    assert len(out) + 10 < 512                 # stub: 1 char == 1 token
    assert out.count("의 주문 내역") >= 1      # kept a chronological prefix


def test_canonical_cand_map_option_free_and_uniform():
    """Symmetric variant: every candidate text is option-free canonical -- no
    own-row/representative-row distinction left for the model to exploit."""
    from mvcp.baselines.concurrentbert import canonical_cand_map

    class _V:
        item2idx = {100: 0, 200: 1}
        num_items = 2

    df = pd.DataFrame({
        "product_id":    [100, 100, 200],
        "product_name":  ["[더블딜] 아기물티슈 100매", "아기물티슈 100매", "꿀고구마 2kg"],
        "category_name": ["유아물티슈", "유아물티슈", "꿀고구마"],
        "attribute_values": ["캡형,5팩", None, "소,2kg"],
    })
    m = canonical_cand_map({"train": df}, _V())
    assert m[0] == "## 다음 구매 상품 예측\n유아물티슈 - 아기물티슈 100매"
    assert m[1] == "## 다음 구매 상품 예측\n꿀고구마 - 꿀고구마 2kg"
    for t in m.values():  # attribute strings must never leak into candidates
        assert "캡형" not in t and "소," not in t
