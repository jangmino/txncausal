"""Four-cell split of the component-ablation comparisons (the research plan;
the split of report/baselines_results.md S2.2): users by history (has a multi-item order /
single-item orders only) x target (first-time / repeat), paired delta B - A in every cell
with the three gates of mvcp.compare (matrix_gates). Masked full-catalog NDCG@10 by
default. Descriptive -- outside the pre-registered verdicts.

    uv run python scripts/report_abl_cells.py --seeds 100-119 \
        --pairs "P1=sasrec_final2_lr0.00025_d128_do0.5:<sasrec_txnpos_final2 name>,..."
    (--split val appends _val to every name)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mvcp.compare import METRIC, matrix_gates, parse_seeds
from mvcp.records import _load_per_user

CELLS = {"co-purchase history · first-time": lambda r: r.has_copurchase and not r.repeat,
         "co-purchase history · repeat": lambda r: r.has_copurchase and r.repeat,
         "single-item only · first-time": lambda r: not r.has_copurchase and not r.repeat,
         "single-item only · repeat": lambda r: not r.has_copurchase and r.repeat}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pairs", required=True, help="csv of LABEL=a_name:b_name")
ap.add_argument("--seeds", required=True)
ap.add_argument("--split", choices=("test", "val"), default="test")
ap.add_argument("--dir", default="results/katchers")
ap.add_argument("--protocol", default="full", help="full / full_nomask / sampled")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--json", default=None, help="also write the cells here")
x = ap.parse_args()
out, seeds = Path(x.dir), parse_seeds(x.seeds)
suf = "_val" if x.split == "val" else ""

cells = []
print("| comparison | cell | n | A | B | Δ | user-boot CI95 | W p | two-level CI95 | seeds+ | gates |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for lab, names in (item.split("=") for item in x.pairs.split(",")):
    pa, pb = ({s: {r.user_idx: r for r in _load_per_user(out, n + suf, s)} for s in seeds}
              for n in names.split(":"))
    users, rec = sorted(pa[seeds[0]]), pa[seeds[0]]
    A, B = (np.array([[getattr(p[s][u], x.protocol)[METRIC] for u in users] for s in seeds])
            for p in (pa, pb))
    for cell, keep in CELLS.items():
        ix = [j for j, u in enumerate(users) if keep(rec[u])]
        g = matrix_gates(A[:, ix], B[:, ix], n_boot=x.n_boot)
        ok = [not (g["boot_ci"][0] <= 0 <= g["boot_ci"][1]), g["w_p"] < 0.05,
              not (g["two_ci"][0] <= 0 <= g["two_ci"][1])]
        cells.append({"pair": lab, "cell": cell, **g, "gates": int(sum(ok))})
        print(f"| {lab} | {cell} | {len(ix):,} | {g['a_mean']:.4f} | {g['b_mean']:.4f} "
              f"| {g['delta']:+.4f} | [{g['boot_ci'][0]:+.4f}, {g['boot_ci'][1]:+.4f}] "
              f"| {g['w_p']:.2g} | [{g['two_ci'][0]:+.4f}, {g['two_ci'][1]:+.4f}] "
              f"| {g['seeds_pos']}/{len(seeds)} |{' ✓' if all(ok) else ''} |")
if x.json:
    Path(x.json).parent.mkdir(parents=True, exist_ok=True)
    Path(x.json).write_text(json.dumps({"pairs": x.pairs, "seeds": seeds, "split": x.split,
                                        "protocol": x.protocol, "cells": cells}, indent=2))
