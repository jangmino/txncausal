"""Per-user evaluation records as written by ``mvcp-train`` (``OUT/per_user/*.json.gz``):
loading, and averaging over seeds. Every paired statistic is computed from these."""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

KS = (1, 5, 10, 20)                 # must match harness default


@dataclass
class _Rec:
    """Light stand-in for harness.UserResult, rebuilt from per_user json.gz."""
    user_idx: int
    repeat: bool
    has_copurchase: bool
    n_txn: int
    cold: bool
    full: dict
    sampled: dict
    full_nomask: dict | None = None   # absent in records written before Phase 0-B
    extra_strata: list | None = None  # dataset-specific strata (Instacart); absent on Katchers
    r5: dict | None = None            # second-target metrics (Instacart ALL); else absent


def _load_per_user(out: Path, rung: str, seed: int) -> list[_Rec] | None:
    p = out / "per_user" / f"{rung}_seed{seed}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        return [_Rec(**d) for d in json.load(f)]


def _mean_over_seeds(per_seed: dict[int, list[_Rec]]) -> list[_Rec]:
    """One record per user; full/sampled metrics averaged over seeds."""
    seeds = sorted(per_seed)
    by_seed = {s: {r.user_idx: r for r in per_seed[s]} for s in seeds}
    common = sorted(set.intersection(*(set(by_seed[s]) for s in seeds)))
    out = []
    for u in common:
        rs = [by_seed[s][u] for s in seeds]
        r0 = rs[0]
        out.append(_Rec(
            u, r0.repeat, r0.has_copurchase, r0.n_txn, r0.cold,
            {k: sum(r.full[k] for r in rs) / len(rs) for k in r0.full},
            {k: sum(r.sampled[k] for r in rs) / len(rs) for k in r0.sampled},
            ({k: sum(r.full_nomask[k] for r in rs) / len(rs) for k in r0.full_nomask}
             if all(r.full_nomask is not None for r in rs) else None),
            r0.extra_strata))
    return out
