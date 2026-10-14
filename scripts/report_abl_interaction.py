"""Interaction of the component-ablation 2x2 (S1):
(P2 - S2) = (S1 - P1) = (block-causal, order pos) - (item-causal, order pos)
                        - (block-causal, item pos) + (item-causal, item pos),
paired by seed and user, per protocol x stratum, with mvcp.compare.matrix_gates'
resampling. Descriptive -- outside the pre-registered verdicts.

    uv run python scripts/report_abl_interaction.py --seeds 100-119 \
        --ii sasrec_final2_lr0.00025_d128_do0.5 --io <sasrec_txnpos_final2 name> \
        --bi <txf_txn_causal_itempos_final2 name> --bo txf_txn_causal_final2_lr0.00025_d128_do0.3
    (--split val appends _val to every name)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mvcp.compare import METRIC, STRATA, matrix_gates, parse_seeds
from mvcp.records import _load_per_user

LABEL = {"full": "masked", "full_nomask": "unmasked", "sampled": "1:49"}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ii", required=True, help="item-causal, item positions (SASRec)")
ap.add_argument("--io", required=True, help="item-causal, order positions")
ap.add_argument("--bi", required=True, help="block-causal, item positions")
ap.add_argument("--bo", required=True, help="block-causal, order positions (TxnCausal)")
ap.add_argument("--seeds", required=True)
ap.add_argument("--split", choices=("test", "val"), default="test")
ap.add_argument("--dir", default="results/katchers")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--json", default=None, help="also write the rows here")
x = ap.parse_args()
out, seeds = Path(x.dir), parse_seeds(x.seeds)
suf = "_val" if x.split == "val" else ""

per = {k: {s: {r.user_idx: r for r in _load_per_user(out, getattr(x, k) + suf, s)}
           for s in seeds} for k in ("ii", "io", "bi", "bo")}
users = sorted(per["ii"][seeds[0]])
rec = per["ii"][seeds[0]]
rows = []
print("| protocol | stratum | n | interaction | user-boot CI95 | W p | two-level CI95 | seeds+ |")
print("|---|---|---|---|---|---|---|---|")
for which in LABEL:
    M = {k: np.array([[getattr(p[s][u], which)[METRIC] for u in users] for s in seeds])
         for k, p in per.items()}
    for st, keep in STRATA.items():
        ix = [j for j, u in enumerate(users) if keep(rec[u])]
        g = matrix_gates((M["io"] + M["bi"])[:, ix], (M["bo"] + M["ii"])[:, ix], x.n_boot)
        rows.append({"protocol": which, "stratum": st,
                     **{k: g[k] for k in ("n_users", "delta", "boot_ci", "w_p", "two_ci",
                                          "seeds_pos", "per_seed_delta")}})
        print(f"| {LABEL[which]} | {st} | {len(ix):,} | {g['delta']:+.4f} "
              f"| [{g['boot_ci'][0]:+.4f}, {g['boot_ci'][1]:+.4f}] | {g['w_p']:.2g} "
              f"| [{g['two_ci'][0]:+.4f}, {g['two_ci'][1]:+.4f}] "
              f"| {g['seeds_pos']}/{len(seeds)} |")
if x.json:
    Path(x.json).parent.mkdir(parents=True, exist_ok=True)
    Path(x.json).write_text(json.dumps({k: getattr(x, k) for k in ("ii", "io", "bi", "bo")}
                                       | {"seeds": seeds, "split": x.split, "rows": rows},
                                       indent=2))
