"""Phase 0-D HP grid table: per model x config,
val full-catalog NDCG@10 mean ± std over seeds, from results/katchers/*_val_seed*.json.

    uv run python scripts/report_beat_grid.py [--dir results/katchers] [--models sasrec,txf_txn_causal]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(?P<model>.+?)_lr(?P<lr>[^_]+)_d(?P<dim>\d+)_do(?P<do>[^_]+)_val_seed(?P<seed>\d+)\.json$")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/katchers")
ap.add_argument("--models", default="sasrec,txf_txn_causal")
a = ap.parse_args()
models = a.models.split(",")

cells = defaultdict(dict)                          # (model, dim, do, lr) -> {seed: result}
for f in Path(a.dir).glob("*_val_seed*.json"):
    m = PAT.match(f.name)
    if not m or m["model"] not in models:
        continue
    cells[(m["model"], int(m["dim"]), float(m["do"]), float(m["lr"]))][int(m["seed"])] = \
        json.loads(f.read_text())

print("| model | dim | dropout | lr | n | val overall | novel | repeat | copurchase "
      "| overall nomask | best epoch (median) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
best = {}
for key in sorted(cells, key=lambda k: (models.index(k[0]), k[1], k[2], -k[3])):
    runs = cells[key]
    fc = lambda st: [r["full_catalog"][st]["ndcg@10"] for r in runs.values()]  # noqa: E731
    nm = [r["full_nomask"]["overall"]["ndcg@10"] for r in runs.values()]
    ep = [max(r["history"], key=lambda e: e["val_score"])["epoch"] for r in runs.values()]
    mean = float(np.mean(fc("overall")))
    if key[0] not in best or mean > best[key[0]][1]:
        best[key[0]] = (key, mean, len(runs))
    print(f"| {key[0]} | {key[1]} | {key[2]:g} | {key[3]:g} | {len(runs)} | "
          + " | ".join(f"{np.mean(fc(st)):.4f} ± {np.std(fc(st)):.4f}"
                       for st in ("overall", "novel", "repeat", "copurchase"))
          + f" | {np.mean(nm):.4f} | {int(np.median(ep))} |")
print()
for mdl, (key, mean, n) in best.items():
    print(f"- best {mdl}: dim {key[1]}, dropout {key[2]:g}, lr {key[3]:g} → {mean:.4f} (n={n})")
