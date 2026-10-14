"""One display name per item: the modal product name with leading [promo] brackets
stripped. Used for the candidate texts of the retrained text baseline."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_BRACKET = re.compile(r"^\s*(\[[^\]]*\]\s*)+")


def canonical_names(frames: dict[str, pd.DataFrame],
                    item2idx: dict[int, int]) -> list[str]:
    """One display name per item_idx: modal name, promo brackets stripped."""
    all_df = pd.concat([f[["product_id", "product_name"]] for f in frames.values()],
                       ignore_index=True)
    names: dict[int, str] = {}
    for pid, g in all_df.groupby("product_id"):
        cnt = Counter(str(n) for n in g["product_name"].dropna())
        raw = cnt.most_common(1)[0][0] if cnt else str(pid)
        names[item2idx[int(pid)]] = _BRACKET.sub("", raw).strip() or raw
    return [names[i] for i in range(len(item2idx))]
