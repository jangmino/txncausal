"""mvcp-train CLI: result naming (back-compat + HP overrides), --eval-split val hygiene
(test is never evaluated), full_nomask in outputs, --skip-existing."""
import gzip
import json
import sys
from types import SimpleNamespace

import pytest

from mvcp import cli


def _name(argv):
    return cli._result_name(cli._train_parser().parse_args(argv))


def test_result_name_backcompat_and_overrides():
    assert _name(["--model", "sasrec"]) == "sasrec"
    assert _name(["--model", "sasrec", "--lr", "0.0005", "--train-frac", "0.1"]) == \
        "sasrec_lr0.0005_frac0.1"
    assert _name(["--model", "txf_txn_causal", "--num-layers", "1"]) == "txf_txn_causal_nl1"
    assert _name(["--model", "sasrec", "--lr", "0.001", "--dim", "128", "--dropout", "0.3",
                  "--weight-decay", "1e-4", "--eval-split", "val"]) == \
        "sasrec_lr0.001_d128_do0.3_wd0.0001_val"


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    from test_train_smoke import _synth
    seqs = {"train": _synth(60, 100, seed=1), "val": _synth(20, 100, seed=2),
            "test": _synth(20, 100, seed=3)}
    bundle = SimpleNamespace(seqs=seqs, num_items=100, cold_users=set())
    monkeypatch.setattr(cli, "_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(cli, "load_config", lambda *a: {
        "dim": 16, "nhead": 2, "num_layers": 1, "dropout": 0.1, "lr": 1e-2,
        "weight_decay": 0.0, "epochs": 2, "batch_size": 16, "eval_every": 1,
        "patience": 3, "device": "cpu"})
    seen = []
    real_eval = cli.evaluate

    def spy(model, s, *a, **k):
        seen.append("test" if s is seqs["test"] else "val" if s is seqs["val"] else "?")
        return real_eval(model, s, *a, **k)
    monkeypatch.setattr(cli, "evaluate", spy)
    monkeypatch.setattr(cli, "_negatives",
                        lambda *a: pytest.fail("test negatives built in val mode"))
    return SimpleNamespace(out=tmp_path / "r", seen=seen)


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mvcp-train"] + argv)
    cli.train()


def test_val_split_never_touches_test(monkeypatch, tiny):
    _run(monkeypatch, ["--model", "txf_txn_causal", "--eval-split", "val", "--dim", "16",
                       "--out", str(tiny.out)])
    # train-time selection evals + the final eval are all on val
    assert tiny.seen and set(tiny.seen) == {"val"}
    res = json.loads((tiny.out / "txf_txn_causal_d16_val_seed0.json").read_text())
    assert res["eval_split"] == "val"
    # final val eval of the restored best state reproduces the selection score exactly
    assert res["full_catalog"]["overall"]["ndcg@10"] == res["best_val"]
    assert "full_nomask" in res
    with gzip.open(tiny.out / "per_user" / "txf_txn_causal_d16_val_seed0.json.gz", "rt") as f:
        recs = json.load(f)
    assert len(recs) == 20 and "full_nomask" in recs[0]


def test_skip_existing_exits_before_loading(monkeypatch, tiny):
    _run(monkeypatch, ["--model", "sasrec", "--eval-split", "val", "--out", str(tiny.out)])
    monkeypatch.setattr(cli, "_bundle", lambda *a, **k: pytest.fail("data loaded"))
    _run(monkeypatch, ["--model", "sasrec", "--eval-split", "val", "--out", str(tiny.out),
                       "--skip-existing"])


def test_val_split_rejects_test_probes(monkeypatch, tiny):
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["--model", "sasrec", "--eval-split", "val", "--perm-eval",
                           "--out", str(tiny.out)])


def test_result_name_augmentation():
    assert _name(["--model", "txf_txn_causal", "--dim", "128", "--txn-drop", "0.1",
                  "--eval-split", "val"]) == "txf_txn_causal_d128_td0.1_val"
    assert _name(["--model", "txf_txn_causal", "--item-drop", "0.2"]) == \
        "txf_txn_causal_id0.2"


def test_result_name_struct_lr():
    assert _name(["--model", "txf_txn_causal_final2_relb", "--dim", "128",
                  "--struct-lr-mult", "20", "--eval-split", "val"]) == \
        "txf_txn_causal_final2_relb_d128_slr20_val"
