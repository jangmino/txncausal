"""No-training reference rows for Instacart (I10, Phase 0-C),
scored through the same harness as the models (mid-rank ties, fixed denominator).

  pop        global popularity: purchase counts of the train users (history + target)
  persfreq   personal frequency: #orders containing the item -> recency (latest order it
             appears in) -> popularity; items never bought fall back to popularity
  histfirst  history items first (score 1), everything else tied at 0 -- no model at all
  lastfirst  items of the most recent order first -> popularity
  random     uniform random scores (1:49 chance level; expected NDCG@10 = 0.0909)

`window` reads the model input (the I3 window), `full` the user's whole history.

    uv run python scripts/instacart_baselines.py            # -> results/instacart/baselines.json
    uv run python scripts/instacart_baselines.py --data instacart_all \
        --strata r1_in_last,r1_before,novel,r2_high,r2_mid,r2_low   # -> results/instacart_all/
    uv run python scripts/instacart_baselines.py --data rees46 --primary full_catalog \
        --strata k1_one_item,k1_single_txns,copurchase              # -> results/rees46/
    uv run python scripts/instacart_baselines.py --data katchers --primary full_catalog \
        --strata copurchase --out results/katchers/baselines.json
    (Katchers: the CSV bundle; its window is the whole history, test negatives = seed 0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mvcp.data import negatives as neg
from mvcp.data.instacart import load_instacart
from mvcp.eval.harness import evaluate
from mvcp.pipeline import load_bundle
from mvcp.utils.config import load_config


def _dense_rank(*keys: np.ndarray) -> np.ndarray:
    """Lexicographic dense rank (first key most significant); equal key tuples tie."""
    return np.unique(np.stack(keys, 1), axis=0, return_inverse=True)[1].ravel()


def scores(kind: str, orders: list[list[int]], pop: np.ndarray, rng) -> np.ndarray:
    n = len(pop)
    if kind == "pop":
        return pop
    if kind == "random":
        return rng.random(n)
    count, last = np.zeros(n), np.full(n, -1.0)
    for t, o in enumerate(orders):
        count[o] += 1
        last[o] = t
    if kind == "histfirst":
        return (count > 0).astype(float)
    if kind == "lastfirst":
        return _dense_rank((last == len(orders) - 1).astype(float), pop).astype(float)
    return _dense_rank(count, last, pop).astype(float)          # persfreq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="instacart_all", help="configs/data/<name>.yaml")
    ap.add_argument("--out", default=None, help="default results/<data>/baselines.json")
    ap.add_argument("--strata", default="overall,novel,repeat",
                    help="csv of strata to store (and print for the --primary protocol)")
    ap.add_argument("--primary", choices=("full_nomask", "full_catalog"),
                    default="full_nomask", help="protocol of the extra --strata columns")
    a = ap.parse_args()
    a.out = a.out or f"results/{a.data}/baselines.json"
    strata = a.strata.split(",")
    dcfg = load_config(f"configs/data/{a.data}.yaml")
    DATA = dcfg["data_dir"]
    if a.data == "katchers":
        b = load_bundle(DATA)
        hist = {str(u): [[it for it, tx in zip(s.items, s.txn_ids) if tx == t]
                         for t in range(s.n_txn)]
                for seqs in b.seqs.values() for u, s in seqs.items()}
    else:
        b = load_instacart(DATA, dcfg["max_orders"], dcfg["max_seq_len"],
                           strata=tuple(dcfg.get("strata", ())))
        hist = json.loads(Path(DATA, "user_item_history.json").read_text())
    pop = np.zeros(b.num_items)
    for u, s in b.seqs["train"].items():
        for o in hist[str(u)]:
            pop[o] += 1
        pop[s.target] += 1

    def orders_of(s, history):
        if history == "full":
            return hist[str(s.user_idx)]
        return [[it for it, tx in zip(s.items, s.txn_ids) if tx == t] for t in range(s.n_txn)]

    rows = {}
    for split in ("val", "test"):
        seqs = b.seqs[split]
        negs = (b.test_negatives if split == "test" and b.test_negatives is not None
                else neg.build(seqs, b.num_items, 0, mode="corrected"))
        for kind, history in (("pop", "-"), ("persfreq", "window"), ("persfreq", "full"),
                              ("histfirst", "window"), ("histfirst", "full"),
                              ("lastfirst", "window"), ("random", "-")):
            rng = np.random.default_rng(0)
            col = lambda batch: {"scores": torch.from_numpy(np.stack(      # noqa: E731
                [scores(kind, orders_of(s, history), pop, rng) for s in batch]))}
            res = evaluate(lambda bt: bt["scores"], seqs, col, negs, b.num_items)
            rows[f"{split}/{kind}/{history}"] = {
                p: {st: {m: res[p][st][m] for m in ("ndcg@5", "ndcg@10", "recall@10")}
                    for st in dict.fromkeys(["overall", "novel", "repeat"] + strata)}
                for p in ("full_nomask", "full_catalog", "sampled_1to49")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))

    extra = [st for st in strata if st not in ("overall", "novel", "repeat")]
    print("| split | baseline | history | nomask overall | nomask novel | nomask repeat "
          "| masked overall | masked novel | masked repeat | 1:49 N@10 | 1:49 N@5 |"
          + "".join(f" {'nomask' if a.primary == 'full_nomask' else 'masked'} {st} |"
                    for st in extra))
    print("|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(extra))
    for key, r in rows.items():
        cells = [f"{r[p][st]['ndcg@10']:.4f}" for p in ("full_nomask", "full_catalog")
                 for st in ("overall", "novel", "repeat")]
        s = r["sampled_1to49"]["overall"]
        print(f"| {' | '.join(key.split('/'))} | {' | '.join(cells)} "
              f"| {s['ndcg@10']:.4f} | {s['ndcg@5']:.4f} |"
              + "".join(f" {r[a.primary][st]['ndcg@10']:.4f} |" for st in extra))


if __name__ == "__main__":
    main()
