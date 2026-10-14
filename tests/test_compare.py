"""mvcp.compare: paired A->B comparison, section-1 gates and section-7 screening verdicts."""
import gzip
import json

import numpy as np

from mvcp.compare import compare, parse_seeds, render, screen


def _write(out, name, seed, per_user_val, nomask=True):
    recs = []
    for u, v in enumerate(per_user_val):
        mets = {f"{m}@{k}": v for m in ("ndcg", "recall") for k in (1, 5, 10, 20)}
        mets["mrr"] = v
        r = {"user_idx": u, "repeat": u % 4 == 0, "has_copurchase": u % 3 == 0,
             "n_txn": 2, "cold": False, "full": mets, "sampled": dict(mets)}
        if nomask:
            r["full_nomask"] = {k: x / 2 for k, x in mets.items()}
        recs.append(r)
    pu = out / "per_user"
    pu.mkdir(parents=True, exist_ok=True)
    with gzip.open(pu / f"{name}_seed{seed}.json.gz", "wt") as f:
        json.dump(recs, f)


def test_parse_seeds():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("100-101,5") == [100, 101, 5]


def test_constant_gain_passes_gates_and_promotes(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(5):
        base = rng.uniform(0, 0.5, size=200)
        _write(tmp_path, "a", s, base)
        _write(tmp_path, "b", s, base + 0.05)          # +0.05 for every user and seed
    res = compare(tmp_path, "a", "b", list(range(5)), n_boot=200)
    head = res["rows"][0]
    assert (head["protocol"], head["stratum"]) == ("full", "overall")
    assert abs(head["delta"] - 0.05) < 1e-9 and head["seeds_pos"] == 5
    assert res["gates"]["pass"] and res["screen"] == "promote"
    nomask = [r for r in res["rows"] if r["protocol"] == "full_nomask"]
    assert len(nomask) == 4 and abs(nomask[0]["delta"] - 0.025) < 1e-9
    assert "PASS" in render(res)


def test_noise_rejects_and_old_records_skip_nomask(tmp_path):
    rng = np.random.default_rng(1)
    for s in range(10):
        _write(tmp_path, "a", s, rng.uniform(0, 0.5, size=200), nomask=False)
        _write(tmp_path, "b", s, rng.uniform(0, 0.5, size=200), nomask=False)
    res = compare(tmp_path, "a", "b", list(range(12)), n_boot=200)
    assert res["missing_seeds"] == [10, 11]
    assert {r["protocol"] for r in res["rows"]} == {"full"}
    assert not res["gates"]["pass"]
    assert res["screen"] in ("reject", "keep")        # noise: never promote
    assert res["screen"] != "promote"


def test_screen_rule():
    def row(lo, delta, pos, n=10):
        return {"two_ci": [lo, lo + 0.01], "delta": delta, "seeds_pos": pos,
                "per_seed_delta": [0.0] * n}
    assert screen(row(+0.001, 0.004, 6)) == "promote"     # 2-level CI excludes 0
    assert screen(row(-0.001, 0.004, 7)) == "keep"        # mean > 0 and >= 7/10 seeds
    assert screen(row(-0.001, 0.004, 6)) == "reject"
    assert screen(row(-0.001, -0.001, 8)) == "reject"
    assert screen(row(-0.001, 0.004, 14, n=20)) == "keep"  # ceil(0.7 * 20) = 14
