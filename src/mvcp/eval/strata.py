"""Stratified aggregation + paired significance.

The crux of the honest evaluation: the ~12% target-in-history (repeat) subset is a
trivially-easy stratum that must never be blended into the headline, and the
transaction-aware effect should concentrate in the co-purchase stratum. So every metric
is reported overall and by stratum.
"""
from __future__ import annotations

import numpy as np


def _len_bucket(n_txn: int) -> str:
    if n_txn <= 2:
        return "len<=2"
    if n_txn <= 5:
        return "len3-5"
    return "len6+"


def stratify(results, which: str, ks) -> dict:
    keys = [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["mrr"]

    def agg(subset):
        if not subset:
            return {"n": 0}
        out = {"n": len(subset)}
        for key in keys:
            out[key] = float(np.mean([getattr(r, which)[key] for r in subset]))
        return out

    strata = {
        "overall": results,
        "novel": [r for r in results if not r.repeat],          # headline should use this
        "repeat": [r for r in results if r.repeat],
        "copurchase": [r for r in results if r.has_copurchase],
        "single_only": [r for r in results if not r.has_copurchase],
        "cold": [r for r in results if r.cold],
    }
    buckets: dict[str, list] = {}
    for r in results:
        buckets.setdefault(_len_bucket(r.n_txn), []).append(r)
    extra: dict[str, list] = {}          # dataset-specific strata (none on Katchers)
    for r in results:
        for name in r.extra_strata or ():
            extra.setdefault(name, []).append(r)
    return {name: agg(sub) for name, sub in
            {**strata, **buckets, **dict(sorted(extra.items()))}.items()}


def paired_delta(res_a, res_b, which: str, metric: str, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Per-user paired bootstrap of (model_b - model_a) on one metric.

    res_a/res_b: lists of UserResult aligned by user_idx.
    Returns mean delta and a 95% CI; CI excluding 0 => significant at ~0.05.
    """
    a = {r.user_idx: getattr(r, which)[metric] for r in res_a}
    b = {r.user_idx: getattr(r, which)[metric] for r in res_b}
    common = sorted(set(a) & set(b))
    d = np.array([b[u] - a[u] for u in common])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(d),
        "mean_delta": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
    }
