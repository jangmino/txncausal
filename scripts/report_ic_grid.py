"""Instacart HP grid table (Phase 0-D / 1): per model x
config, val UNMASKED full-catalog NDCG@10 (the selection metric, I6) mean ± std over
seeds, from results/instacart/*_val_seed*.json.

    uv run python scripts/report_ic_grid.py [--models sasrec,txf_txn_causal] [--seeds 0-4]
    REES46 (selection = the MASKED metric): --dir results/rees46 --protocol full_catalog
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from mvcp.compare import parse_seeds

PAT = re.compile(r"^(?P<model>.+?)_lr(?P<lr>[^_]+)_d(?P<dim>\d+)_do(?P<do>[^_]+)"
                 r"(?P<rest>(_[^_]+)*?)_val_seed(?P<seed>\d+)\.json$")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/instacart_all")
ap.add_argument("--models", default="sasrec,txf_txn_causal")
ap.add_argument("--seeds", default="0-4")
ap.add_argument("--protocol", choices=("full_nomask", "full_catalog"), default="full_nomask",
                help="the selection metric's protocol (ranks the configs)")
a = ap.parse_args()
models, seeds = a.models.split(","), set(parse_seeds(a.seeds))

cells = defaultdict(dict)                    # (model, dim, do, lr, rest) -> {seed: result}
for f in Path(a.dir).glob("*_val_seed*.json"):
    m = PAT.match(f.name)
    if not m or m["model"] not in models or int(m["seed"]) not in seeds:
        continue
    key = (m["model"], int(m["dim"]), float(m["do"]), float(m["lr"]), m["rest"])
    cells[key][int(m["seed"])] = json.loads(f.read_text())

sel, other = a.protocol, {"full_nomask": "full_catalog", "full_catalog": "full_nomask"}[a.protocol]
LAB = {"full_nomask": "nomask", "full_catalog": "masked"}
print(f"| model | dim | dropout | lr | n | val {LAB[sel]} overall | novel | repeat "
      f"| {LAB[other]} overall | 1:49 | best epoch (median) | epochs run (median) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
best = {}
for key in sorted(cells, key=lambda k: (models.index(k[0]), k[4], k[1], k[2], -k[3])):
    runs = cells[key]
    nm = lambda st: [r[sel][st]["ndcg@10"] for r in runs.values()]  # noqa: E731
    mean = float(np.mean(nm("overall")))
    if key[0] not in best or mean > best[key[0]][1]:
        best[key[0]] = (key, mean, len(runs))
    ep = [max(r["history"], key=lambda e: e["val_score"])["epoch"] for r in runs.values()]
    print(f"| {key[0]}{key[4]} | {key[1]} | {key[2]:g} | {key[3]:g} | {len(runs)} | "
          + " | ".join(f"{np.mean(nm(st)):.4f} ± {np.std(nm(st)):.4f}"
                       for st in ("overall", "novel", "repeat"))
          + f" | {np.mean([r[other]['overall']['ndcg@10'] for r in runs.values()]):.4f}"
          + f" | {np.mean([r['sampled_1to49']['overall']['ndcg@10'] for r in runs.values()]):.4f}"
          + f" | {int(np.median(ep))} "
          + f"| {int(np.median([r['history'][-1]['epoch'] for r in runs.values()]))} |")
print()
for mdl, (key, mean, n) in best.items():
    print(f"- best {mdl}{key[4]}: dim {key[1]}, dropout {key[2]:g}, lr {key[3]:g} "
          f"→ {mean:.5f} (n={n})")
capped = sorted(f"{k[0]}{k[4]} {k[1]}/{k[2]:g}/{k[3]:g} seed {s}" for k, runs in cells.items()
                for s, r in runs.items() if r["history"][-1]["epoch"] >= 500)
if capped:
    print(f"- runs that reached the 500-epoch cap: {len(capped)} ({'; '.join(capped)})")
