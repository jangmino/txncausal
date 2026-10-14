"""Phase 1 screening table (§7): each variant vs the
champion (§7 verdict) and vs SASRec*, on val, via mvcp.compare. Writes each comparison's
JSON to <dir>/compare/ and prints scoreboard-ready markdown rows.

    uv run python scripts/screen_round.py --champion txf_txn_causal_lr0.0005_d128_do0.3 \
        --sasrec sasrec_lr0.0005_d64_do0.3 --variants H1=txf_txn_causal_blend_lr0.0005_d128_do0.3,...
"""
import argparse
import json
from pathlib import Path

from mvcp.compare import compare, parse_seeds

ap = argparse.ArgumentParser()
ap.add_argument("--champion", required=True, help="result name without the _val suffix")
ap.add_argument("--sasrec", required=True)
ap.add_argument("--variants", required=True, help="csv of LABEL=result_name")
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--dir", default="results/katchers")
ap.add_argument("--tag", default="round")
a = ap.parse_args()
out, seeds = Path(a.dir), parse_seeds(a.seeds)
(out / "compare").mkdir(parents=True, exist_ok=True)


def row(res, protocol, stratum):
    return next(r for r in res["rows"] if r["protocol"] == protocol and r["stratum"] == stratum)


print("| H | variant | Δ overall [two-level CI] | seeds+ | Δ novel | Δ repeat | Δ copurchase "
      "| Δ overall nomask | grade | − SASRec\\* Δ overall [two-level CI] |")
print("|---|---|---|---|---|---|---|---|---|---|")
for item in a.variants.split(","):
    label, name = item.split("=")
    res = compare(out, a.champion + "_val", name + "_val", seeds)
    ref = compare(out, a.sasrec + "_val", name + "_val", seeds)
    for tag, r in (("champ", res), ("sasrec", ref)):
        (out / "compare" / f"{a.tag}_{label}_{tag}.json").write_text(json.dumps(r, indent=2))
    h, hs = row(res, "full", "overall"), row(ref, "full", "overall")
    d = lambda p, s: f"{row(res, p, s)['delta']:+.4f}"  # noqa: E731
    miss = f" ⚠️ seeds {res['missing_seeds']}" if res["missing_seeds"] else ""
    print(f"| {label} | `{name}` | {h['delta']:+.4f} [{h['two_ci'][0]:+.4f}, {h['two_ci'][1]:+.4f}] "
          f"| {h['seeds_pos']}/{len(h['per_seed_delta'])} | {d('full', 'novel')} "
          f"| {d('full', 'repeat')} | {d('full', 'copurchase')} | {d('full_nomask', 'overall')} "
          f"| **{res['screen']}**{miss} | {hs['delta']:+.4f} [{hs['two_ci'][0]:+.4f}, "
          f"{hs['two_ci'][1]:+.4f}] |")
