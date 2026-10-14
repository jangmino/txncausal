"""Paired comparison of mvcp models against the symmetric BERT-NSP retraining (Park & Oh
2024's model, report/phase4_bert.md) on Katchers test users, NDCG@10:
  --protocol full     the primary protocol (full catalog, history masked); BERT-NSP has ONE
                      such run (bert_nsp_sym_full, seed 0)
  --protocol sampled  the 2024 paper's 1:49 protocol; BERT-NSP has seeds 0-2 (bert_nsp_sym)
                      on the PUBLISHED negatives, so --dir must hold runs made with
                      `--negatives published`
BERT-NSP's per-user values (averaged over its seeds) are repeated across the other model's
seeds: the user bootstrap and Wilcoxon use seed-averaged values as in mvcp.compare, and the
seed x user bootstrap carries the training variance of the mvcp model only.

    uv run python scripts/bl_bertnsp_paired.py --dir results/katchers --seeds 100-119 \
        --models txf_txn_causal_final2_lr0.00025_d128_do0.3,sasrec_final2_lr0.00025_d128_do0.5
    uv run python scripts/bl_bertnsp_paired.py --dir results/katchers_pub --protocol sampled \
        --bert bert_nsp_sym --bert-seeds 0-2 --models txf_txn_causal_final2_lr0.00025_d128_do0.3
"""
import argparse
from pathlib import Path

import numpy as np

from mvcp.compare import METRIC, STRATA, matrix_gates, parse_seeds
from mvcp.records import _load_per_user

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/katchers")
ap.add_argument("--models", required=True)
ap.add_argument("--seeds", default="100-119")
ap.add_argument("--bert-dir", default="results/bert_nsp")
ap.add_argument("--bert", default="bert_nsp_sym_full")
ap.add_argument("--bert-seeds", default="0")
ap.add_argument("--protocol", choices=("full", "sampled"), default="full")
a = ap.parse_args()
seeds = parse_seeds(a.seeds)

berts = [{r.user_idx: r for r in _load_per_user(Path(a.bert_dir), a.bert, s)}
         for s in parse_seeds(a.bert_seeds)]
bert = berts[0]
a_val = {u: float(np.mean([getattr(b[u], a.protocol)[METRIC] for b in berts])) for u in bert}
print(f"B − A, A = {a.bert}(seed {a.bert_seeds}), {a.protocol} {METRIC}, seed {a.seeds}\n")
print("| B | stratum | n | A | B | Δ | user-boot CI95 | W p | two-level CI95 | seeds+ | gates |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name in a.models.split(","):
    per = [{r.user_idx: r for r in _load_per_user(Path(a.dir), name, s)} for s in seeds]
    assert all(set(p) == set(bert) for p in per), "user sets differ"
    assert all(per[0][u].repeat == bert[u].repeat and
               per[0][u].has_copurchase == bert[u].has_copurchase for u in bert)
    for st, keep in STRATA.items():
        users = sorted(u for u in bert if keep(bert[u]))
        B = np.array([[getattr(p[u], a.protocol)[METRIC] for u in users] for p in per])
        A = np.tile([a_val[u] for u in users], (len(seeds), 1))
        g = matrix_gates(A, B)
        ok = all(lo * hi > 0 for lo, hi in (g["boot_ci"], g["two_ci"])) and g["w_p"] < 0.05
        print(f"| `{name}` | {st} | {g['n_users']:,} | {g['a_mean']:.4f} | {g['b_mean']:.4f} "
              f"| {g['delta']:+.4f} | [{g['boot_ci'][0]:+.4f}, {g['boot_ci'][1]:+.4f}] "
              f"| {g['w_p']:.2g} | [{g['two_ci'][0]:+.4f}, {g['two_ci'][1]:+.4f}] "
              f"| {g['seeds_pos']}/{len(seeds)} | {'✓' if ok else ''} |")
