"""Instacart comparison tables (S1): absolute values of
the models involved, then each paired comparison B - A with the three gates on the primary
metric (UNMASKED full-catalog NDCG@10 overall) and its per-stratum / per-protocol
breakdown. Every comparison's full mvcp.compare result goes to <dir>/compare/<tag>_<label>.json.

    uv run python scripts/report_ic.py --seeds 100-119 --tag c1 \
        --pairs "P1=sasrec_final2_lr0.001_d64_do0.1:txf_txn_causal_final2_lr0.001_d64_do0.1"
    (--split val appends _val to every name)
    New-sample datasets: --dir results/instacart_all
    and --strata to print the R1-R4 strata of the per-user records.
    REES46: --dir results/rees46 --primary full (the primary
    metric there is the MASKED one) and --strata for K1-K4.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mvcp.compare import compare, parse_seeds

PROTOCOLS = ("full_nomask", "full", "sampled")
KEY = {"full_nomask": "full_nomask", "full": "full_catalog", "sampled": "sampled_1to49"}
LABEL = {"full_nomask": "unmasked", "full": "masked", "sampled": "1:49"}
STRATA = ("overall", "novel", "repeat", "last_multi", "last_single",
          "basket_t1", "basket_t2", "basket_t3")


def gate_flags(r: dict) -> list[bool]:
    """user-boot CI, Wilcoxon, seed x user CI -- each in the direction of the delta."""
    out = [not (r["boot_ci"][0] <= 0 <= r["boot_ci"][1]), r["w_p"] < 0.05,
           not (r["two_ci"][0] <= 0 <= r["two_ci"][1])]
    return [g and r["delta"] != 0 for g in out]


def verdict(r: dict) -> str:
    """Plan section 1: all three gates = confirmed (either sign), some = trend."""
    g = gate_flags(r)
    if all(g):
        return "confirmed (B higher)" if r["delta"] > 0 else "confirmed (A higher)"
    return "trend" if any(g) else "n.s."


def mark(r: dict) -> str:
    return " ✓" if all(gate_flags(r)) else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="csv of LABEL=a_name:b_name")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--dir", default="results/instacart_all")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--strata", default=",".join(STRATA),
                    help="csv of strata for the breakdown table")
    ap.add_argument("--primary", choices=PROTOCOLS, default="full_nomask",
                    help="protocol of the gates and of the --detail table")
    ap.add_argument("--detail", action="store_true",
                    help="also print one row per stratum (primary metric) with all gates")
    a = ap.parse_args()
    strata = a.strata.split(",")
    out, seeds = Path(a.dir), parse_seeds(a.seeds)
    suf = "_val" if a.split == "val" else ""
    (out / "compare").mkdir(parents=True, exist_ok=True)
    pairs = [(lab, *names.split(":")) for lab, names in
             (item.split("=") for item in a.pairs.split(","))]

    # ---- absolute values (mean over seeds of the per-run stratum means)
    names = list(dict.fromkeys(n for _, x, y in pairs for n in (x, y)))
    print(f"#### Absolute values ({a.split}, seed {a.seeds}, NDCG@10; also N@5 for 1:49)\n")
    print("| model | n | " + " | ".join(f"{LABEL[p]} {s}" for p in PROTOCOLS
                                      for s in ("overall", "novel", "repeat"))
          + " | 1:49 N@5 overall | novel | repeat | best epoch |")
    print("|---|---|" + "---|" * 13)
    for n in names:
        runs = [json.loads(f.read_text()) for s in seeds
                if (f := out / f"{n}{suf}_seed{s}.json").exists()]
        cell = lambda p, s, m="ndcg@10": f"{np.mean([r[KEY[p]][s][m] for r in runs]):.4f}"  # noqa: E731
        ep = [max(r["history"], key=lambda e: e["val_score"])["epoch"] for r in runs]
        print(f"| `{n}` | {len(runs)} | "
              + " | ".join(cell(p, s) for p in PROTOCOLS for s in ("overall", "novel", "repeat"))
              + " | " + " | ".join(cell("sampled", s, "ndcg@5")
                                   for s in ("overall", "novel", "repeat"))
              + f" | {int(np.median(ep))} |")

    # ---- paired comparisons
    results = {}
    for lab, x, y in pairs:
        res = compare(out, x + suf, y + suf, seeds, a.n_boot, primary=a.primary,
                      protocols=PROTOCOLS)
        (out / "compare" / f"{a.tag}_{lab}.json").write_text(json.dumps(res, indent=2))
        results[lab] = {(r["protocol"], r["stratum"]): r for r in res["rows"]}
        if res["missing_seeds"]:
            print(f"\n⚠️ {lab}: missing seeds {res['missing_seeds']}")
    print(f"\n#### Primary metric — {LABEL[a.primary]} full-catalog NDCG@10 overall (B − A)\n")
    print("| comparison | A | B | Δ | user-boot CI95 | W p | two-level CI95 | seeds+ | grade |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lab, x, y in pairs:
        r = results[lab][a.primary, "overall"]
        g = gate_flags(r)
        print(f"| {lab} `{x}` → `{y}` | {r['a_mean']:.4f} ± {r['a_std']:.4f} "
              f"| {r['b_mean']:.4f} ± {r['b_std']:.4f} | **{r['delta']:+.4f}** "
              f"| [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}]{' ✓' if g[0] else ' ✗'} "
              f"| {r['w_p']:.2g}{' ✓' if g[1] else ' ✗'} "
              f"| [{r['two_ci'][0]:+.4f}, {r['two_ci'][1]:+.4f}]{' ✓' if g[2] else ' ✗'} "
              f"| {r['seeds_pos']}/{len(r['per_seed_delta'])} | **{verdict(r)}** |")
    print("\n#### Δ by stratum and protocol (B − A; ✓ = all three gates pass in that cell)\n")
    print("| comparison | protocol | " + " | ".join(strata) + " |")
    print("|---|---|" + "---|" * len(strata))
    for lab, _, _ in pairs:
        for p in PROTOCOLS:
            print(f"| {lab} | {LABEL[p]} | " + " | ".join(
                f"{results[lab][p, s]['delta']:+.4f}{mark(results[lab][p, s])}"
                for s in strata) + " |")
    n = {s: results[pairs[0][0]][a.primary, s]["n_users"] for s in strata}
    print("\n(users per stratum: " + ", ".join(f"{s} {n[s]:,}" for s in strata) + ")")
    if a.detail:
        print(f"\n#### By stratum — {LABEL[a.primary]} NDCG@10 (B − A)\n")
        print("| comparison | stratum | n | A | B | Δ | user-boot CI95 | W p | two-level CI95 | seeds+ | gates |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for lab, _, _ in pairs:
            for s in strata:
                r = results[lab][a.primary, s]
                print(f"| {lab} | {s} | {r['n_users']:,} | {r['a_mean']:.4f} | {r['b_mean']:.4f} "
                      f"| {r['delta']:+.4f} | [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}] "
                      f"| {r['w_p']:.2g} | [{r['two_ci'][0]:+.4f}, {r['two_ci'][1]:+.4f}] "
                      f"| {r['seeds_pos']}/{len(r['per_seed_delta'])} |{mark(r)} |")


if __name__ == "__main__":
    main()
