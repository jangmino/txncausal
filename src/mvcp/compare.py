"""Paired comparison A -> B of two result names over seeds.

For full-catalog NDCG@10, per protocol {full, full_nomask} x stratum {overall, novel,
repeat, copurchase}, Delta = B - A with
  * user-boot : paired bootstrap over users on seed-averaged per-user metrics
                (eval.strata.paired_delta -- the phase7 rule),
  * W p       : Wilcoxon signed-rank on the same seed-averaged per-user deltas,
  * 2-level   : seed x user two-level bootstrap CI95 (also carries training variance;
                generalizes scripts/report_learning_curve.py::two_level),
  * seeds+    : seeds whose per-seed mean delta is > 0.
Verdicts on overall/full: the section-1 confirmation gates (all three exclude 0) and the
section-7 val screening rule (promote / keep / reject).

    uv run mvcp-compare --a sasrec --b txf_txn_causal --seeds 100-119 --dir results/katchers
    uv run mvcp-compare --a sasrec --b txf_txn_causal --seeds 0-9 --split val
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import wilcoxon

from .eval.strata import paired_delta, stratify
from .records import KS, _load_per_user, _mean_over_seeds

METRIC = "ndcg@10"
PROTOCOLS = ("full", "full_nomask")
STRATA = {"overall": lambda r: True, "novel": lambda r: not r.repeat,
          "repeat": lambda r: r.repeat, "copurchase": lambda r: r.has_copurchase}


def parse_seeds(spec: str) -> list[int]:
    """'0-9' | '100-119' | '0,1,5'."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def two_level(per_a, per_b, which: str, keep, metric: str = METRIC, n_boot: int = 2000,
              seed: int = 0):
    """CI95 of mean(b - a) resampling seeds AND users (paired by seed and user)."""
    seeds = sorted(set(per_a) & set(per_b))
    D = []
    for s in seeds:
        a = {r.user_idx: getattr(r, which)[metric] for r in per_a[s] if keep(r)}
        b = {r.user_idx: getattr(r, which)[metric] for r in per_b[s] if keep(r)}
        D.append([b[u] - a[u] for u in sorted(a)])
    D = np.array(D)                                   # (S, U)
    rng = np.random.default_rng(seed)
    S, U = D.shape
    boots = np.array([D[rng.integers(0, S, S)][:, rng.integers(0, U, U)].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), D.mean(1)


def matrix_gates(A: np.ndarray, B: np.ndarray, n_boot: int = 2000) -> dict:
    """The three gates of mean(B - A) for (seeds, users) matrices paired by seed and user,
    with compare()'s resampling (user bootstrap and Wilcoxon on seed-averaged values,
    seed x user two-level bootstrap) -- for quantities that are not one run's metric."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    rec = lambda v, u: SimpleNamespace(user_idx=u, x={"v": float(v)})   # noqa: E731
    rows = lambda M: {s: [rec(v, u) for u, v in enumerate(r)] for s, r in enumerate(M)}  # noqa: E731
    d = paired_delta(rows([A.mean(0)])[0], rows([B.mean(0)])[0], "x", "v", n_boot=n_boot)
    diff = B.mean(0) - A.mean(0)
    lo, hi, ps = two_level(rows(A), rows(B), "x", lambda r: True, metric="v", n_boot=n_boot)
    return {"n_users": d["n"], "a_mean": float(A.mean()), "b_mean": float(B.mean()),
            "delta": d["mean_delta"], "boot_ci": d["ci95"],
            "w_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
            "two_ci": [lo, hi], "seeds_pos": int((ps > 0).sum()),
            "per_seed_delta": [float(x) for x in ps]}


def compare(out: Path, a: str, b: str, seeds: list[int], n_boot: int = 2000,
            primary: str = "full", protocols=PROTOCOLS, metric: str = METRIC) -> dict:
    pa = {s: r for s in seeds if (r := _load_per_user(out, a, s)) is not None}
    pb = {s: r for s in seeds if (r := _load_per_user(out, b, s)) is not None}
    common = sorted(set(pa) & set(pb))
    if not common:
        raise SystemExit(f"no common seeds with per-user results for {a} / {b} in {out}")
    pa, pb = {s: pa[s] for s in common}, {s: pb[s] for s in common}
    ra, rb = _mean_over_seeds(pa), _mean_over_seeds(pb)
    strata = dict(STRATA)
    for name in sorted({n for r in ra for n in r.extra_strata or ()}):   # dataset-specific
        strata[name] = lambda r, name=name: name in (r.extra_strata or ())
    rows = []
    for which in protocols:
        if getattr(ra[0], which) is None or getattr(rb[0], which) is None:
            continue                                  # records written before full_nomask
        for st, keep in strata.items():
            xa, xb = [r for r in ra if keep(r)], [r for r in rb if keep(r)]
            assert [r.user_idx for r in xa] == [r.user_idx for r in xb]
            d = paired_delta(xa, xb, which, metric, n_boot=n_boot)
            diff = np.array([getattr(y, which)[metric] - getattr(x, which)[metric]
                             for x, y in zip(xa, xb)])
            p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            lo, hi, ps = two_level(pa, pb, which, keep, metric=metric, n_boot=n_boot)
            mean_a = [stratify(pa[s], which, KS)[st][metric] for s in common]
            mean_b = [stratify(pb[s], which, KS)[st][metric] for s in common]
            rows.append({
                "protocol": which, "stratum": st, "n_users": d["n"],
                "a_mean": float(np.mean(mean_a)), "a_std": float(np.std(mean_a)),
                "b_mean": float(np.mean(mean_b)), "b_std": float(np.std(mean_b)),
                "delta": d["mean_delta"], "boot_ci": d["ci95"], "w_p": p,
                "two_ci": [lo, hi], "seeds_pos": int((ps > 0).sum()),
                "per_seed_delta": [float(x) for x in ps],
            })
    head = next(r for r in rows if r["protocol"] == primary and r["stratum"] == "overall")
    return {"a": a, "b": b, "dir": str(out), "seeds": common, "primary": primary,
            "metric": metric,
            "missing_seeds": sorted(set(seeds) - set(common)), "rows": rows,
            "gates": gates(head), "screen": screen(head)}


def gates(row: dict) -> dict:
    """Section-1 confirmation: B beats A iff all three pass in the positive direction."""
    g = {"user_boot": row["boot_ci"][0] > 0,
         "wilcoxon": row["w_p"] < 0.05 and row["delta"] > 0,
         "two_level": row["two_ci"][0] > 0}
    g["pass"] = all(g.values())
    return g


def screen(row: dict) -> str:
    """Section-7 val screening of B (variant) against A (champion)."""
    if row["two_ci"][0] > 0:
        return "promote"
    n = len(row["per_seed_delta"])
    if row["delta"] > 0 and row["seeds_pos"] >= math.ceil(0.7 * n):
        return "keep"
    return "reject"


def render(res: dict) -> str:
    lines = [f"### {res['a']} → {res['b']}  (dir {res['dir']}, seeds {res['seeds'][0]}–"
             f"{res['seeds'][-1]}, n={len(res['seeds'])})", ""]
    if res["missing_seeds"]:
        lines += [f"- ⚠️ missing seeds (not in both): {res['missing_seeds']}", ""]
    lines += ["| protocol | stratum | A | B | Δ | user-boot CI95 | W p | 2-level CI95 | seeds+ |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        star = lambda ci: " ✓" if ci[0] > 0 or ci[1] < 0 else ""   # noqa: E731
        lines.append(
            f"| {r['protocol']} | {r['stratum']} | {r['a_mean']:.4f} ± {r['a_std']:.4f} "
            f"| {r['b_mean']:.4f} ± {r['b_std']:.4f} | {r['delta']:+.4f} "
            f"| [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}]{star(r['boot_ci'])} "
            f"| {r['w_p']:.2g} "
            f"| [{r['two_ci'][0]:+.4f}, {r['two_ci'][1]:+.4f}]{star(r['two_ci'])} "
            f"| {r['seeds_pos']}/{len(r['per_seed_delta'])} |")
    g = res["gates"]
    where = f"overall, {res['primary']}" + ("" if res["metric"] == METRIC
                                            else f", {res['metric']}")
    lines += ["", f"- three gates ({where}): user-boot {'✓' if g['user_boot'] else '✗'}"
                  f" · Wilcoxon {'✓' if g['wilcoxon'] else '✗'}"
                  f" · 2-level {'✓' if g['two_level'] else '✗'} → "
                  f"**{'PASS' if g['pass'] else 'FAIL'}**",
              f"- screening verdict ({where}; B = variant vs A = champion): **{res['screen']}**"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline result name (e.g. sasrec)")
    ap.add_argument("--b", required=True, help="candidate result name")
    ap.add_argument("--seeds", required=True, help="'0-9', '100-119' or csv")
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="val appends the _val suffix mvcp-train --eval-split val adds")
    ap.add_argument("--dir", default="results/katchers")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None, help="also write the full result here")
    ap.add_argument("--primary", default="full",
                    help="protocol whose overall row feeds the gates (Instacart: full_nomask)")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS),
                    help="csv of full / full_nomask / sampled")
    ap.add_argument("--metric", default=METRIC, help="e.g. ndcg@5")
    x = ap.parse_args()
    suf = "_val" if x.split == "val" else ""
    res = compare(Path(x.dir), x.a + suf, x.b + suf, parse_seeds(x.seeds), x.n_boot,
                  primary=x.primary, protocols=x.protocols.split(","), metric=x.metric)
    print(render(res))
    if x.json:
        Path(x.json).parent.mkdir(parents=True, exist_ok=True)
        Path(x.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
