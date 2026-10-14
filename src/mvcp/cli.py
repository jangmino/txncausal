"""Console entry points: mvcp-prep / mvcp-stats / mvcp-train.

`prep`  : split integrity + validation of the released negatives -> PREFLIGHT.md.
`stats` : DATASTATS.md.
`train` : train ONE model x seed, evaluate, write OUT/<name>_seed<seed>.json and the
          per-user records OUT/per_user/<name>_seed<seed>.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .data import negatives as neg
from .data.dataset import CachedCollator
from .eval.harness import evaluate
from .models.registry import build_model
from .pipeline import load_bundle
from .train.loop import TrainConfig, train_model
from .utils.config import load_config
from .utils.seed import make_rng, seed_everything


def _bundle(data_dir="data", cold_threshold=3):
    return load_bundle(data_dir, cold_threshold)


def stats() -> None:
    from .data.stats import render_markdown, split_stats
    b = _bundle()
    allst = {s: split_stats(b.seqs[s]) for s in ("train", "val", "test")}
    md = "# DATASTATS\n\n" + render_markdown(allst) + "\n"
    Path("DATASTATS.md").write_text(md)
    print(md)


def prep() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", default="data/negatives/published.json")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    b = _bundle(a.data_dir)
    from .data.split import structural_checks
    from .data.vocab import load_frames
    lines = ["# PREFLIGHT\n"]
    sc = structural_checks(load_frames(a.data_dir))
    lines.append(f"- split counts: {sc['counts']} ratios {sc['ratios']} "
                 f"disjoint={sc['disjoint']} multi_last={sc['users_with_multi_row_last_date']}")
    lines.append(f"- vocab: num_items={b.num_items} num_users={b.vocab.num_users}")
    if Path(a.published).exists():
        pub = neg.load_published(a.published)
        rep = neg.validate(pub, b.seqs["test"], b.num_items)
        lines.append(f"- published negatives: {rep}")
    else:
        lines.append(f"- published negatives: MISSING at {a.published}")
    md = "\n".join(lines) + "\n"
    Path("PREFLIGHT.md").write_text(md)
    print(md)


def _negatives(kind, b, seed):
    if kind == "published":
        return neg.load_published("data/negatives/published.json")
    return neg.build(b.seqs["test"], b.num_items, seed, mode=kind)


def _subsample(seqs, frac, seed):
    """Seeded random fraction of users -- every model sees the same subset at a seed."""
    users = list(seqs)
    k = max(1, round(frac * len(users)))
    keep = set(make_rng(seed, "trainfrac").choice(users, size=k, replace=False).tolist())
    return {u: s for u, s in seqs.items() if u in keep}


def _train_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-users", type=int, default=None, help="smoke: cap train users")
    ap.add_argument("--negatives", default="corrected")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lr", type=float, default=None,
                    help="override train yaml lr (closing experiments; suffixes name)")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="override train yaml num_layers (closing; suffixes name)")
    ap.add_argument("--device", default=None,
                    help="override train yaml device (e.g. cpu/mps on the MacBook)")
    ap.add_argument("--train-frac", type=float, default=None,
                    help="learning curve: seeded random fraction of train users (suffixes name)")
    ap.add_argument("--dim", type=int, default=None,
                    help="override train yaml dim (HP tuning; suffixes name)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="override train yaml dropout (HP tuning; suffixes name)")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="override train yaml weight_decay (HP tuning; suffixes name)")
    ap.add_argument("--struct-lr-mult", type=float, default=None,
                    help="H7: lr multiplier for zero-init structural params (suffixes name)")
    ap.add_argument("--txn-drop", type=float, default=None,
                    help="H5a: training-time prob of dropping each past order (suffixes name)")
    ap.add_argument("--item-drop", type=float, default=None,
                    help="H5b: training-time prob of dropping an item inside a multi-item "
                         "order, >=1 kept (suffixes name)")
    ap.add_argument("--eval-split", choices=("test", "val"), default="test",
                    help="val: evaluate the best state on val only -- test is never "
                         "touched (development runs; suffixes name with _val)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="exit before loading data if this run's results already exist")
    ap.add_argument("--data", choices=("katchers", "instacart_all", "instacart_pdnew",
                                       "rees46", "rees46_logged"),
                    default="katchers",
                    help="dataset (configs/data/<name>.yaml); keep one --out dir per dataset")
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="override the data yaml item window (suffixes name)")
    ap.add_argument("--max-num-txn", type=int, default=None,
                    help="override the data yaml order-position table size (suffixes name)")
    ap.add_argument("--mask-prob", type=float, default=None,
                    help="bert4rec: prob of hiding each history slot in training "
                         "(suffixes name)")
    return ap


def _result_name(a) -> str:
    return a.model \
        + (f"_lr{a.lr:g}" if a.lr is not None else "") \
        + (f"_nl{a.num_layers}" if a.num_layers is not None else "") \
        + (f"_d{a.dim}" if a.dim is not None else "") \
        + (f"_do{a.dropout:g}" if a.dropout is not None else "") \
        + (f"_wd{a.weight_decay:g}" if a.weight_decay is not None else "") \
        + (f"_slr{a.struct_lr_mult:g}" if a.struct_lr_mult is not None else "") \
        + (f"_td{a.txn_drop:g}" if a.txn_drop is not None else "") \
        + (f"_id{a.item_drop:g}" if a.item_drop is not None else "") \
        + (f"_frac{a.train_frac:g}" if a.train_frac is not None else "") \
        + (f"_L{a.max_seq_len}" if a.max_seq_len is not None else "") \
        + (f"_T{a.max_num_txn}" if a.max_num_txn is not None else "") \
        + (f"_mp{a.mask_prob:g}" if a.mask_prob is not None else "") \
        + ("_val" if a.eval_split == "val" else "")


def train() -> None:
    ap = _train_parser()
    a = ap.parse_args()
    if a.mask_prob is not None and not a.model.startswith("bert4rec"):
        ap.error("--mask-prob is a bert4rec option")
    name = _result_name(a)
    if a.skip_existing and Path(a.out, f"{name}_seed{a.seed}.json").exists() \
            and Path(a.out, "per_user", f"{name}_seed{a.seed}.json.gz").exists():
        print(f"[skip] {name}_seed{a.seed} (results exist in {a.out})")
        return
    tcfg = load_config("configs/train/default.yaml")
    dcfg = load_config(f"configs/data/{a.data}.yaml")
    if a.lr is not None:
        tcfg["lr"] = a.lr
    if a.num_layers is not None:
        tcfg["num_layers"] = a.num_layers
    if a.dim is not None:
        tcfg["dim"] = a.dim
    if a.dropout is not None:
        tcfg["dropout"] = a.dropout
    if a.weight_decay is not None:
        tcfg["weight_decay"] = a.weight_decay
    if a.device is not None:
        tcfg["device"] = a.device
    max_seq_len = a.max_seq_len or dcfg.get("max_seq_len", 64)
    if a.data != "katchers":
        from .data.instacart import load_instacart
        b = load_instacart(dcfg["data_dir"], dcfg["max_orders"], max_seq_len,
                           strata=tuple(dcfg.get("strata", ())))
    else:
        b = _bundle()

    train_seqs = b.seqs["train"]
    if a.max_users:
        train_seqs = {u: train_seqs[u] for u in list(train_seqs)[:a.max_users]}
    if a.train_frac is not None:
        train_seqs = _subsample(train_seqs, a.train_frac, a.seed)
    val_neg = neg.build(b.seqs["val"], b.num_items, a.seed, mode="corrected")
    if a.eval_split != "test":
        test_neg = None
    elif getattr(b, "test_negatives", None) is not None:   # the dataset's own frozen set
        test_neg = b.test_negatives
    else:
        test_neg = _negatives(a.negatives, b, a.seed)

    if not dcfg.get("sdp_mem_efficient", True):
        # its backward is non-deterministic on variable-length batches; the plain kernel is not
        import torch
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    seed_everything(a.seed)  # model init follows --seed (torch's default seed is per-process random)
    model, collator = build_model(a.model, b.num_items, dim=tcfg["dim"],
                                  max_seq_len=max_seq_len,
                                  max_num_txn=a.max_num_txn or dcfg.get("max_num_txn", 52),
                                  nhead=tcfg["nhead"],
                                  num_layers=tcfg["num_layers"], dropout=tcfg["dropout"],
                                  compact_loss=dcfg.get("compact_loss", False),
                                  len_split=dcfg.get("len_split", False),
                                  **({} if a.mask_prob is None
                                     else {"mask_prob": a.mask_prob}))
    if dcfg.get("cache_collate", False):
        collator = CachedCollator(collator)
    cfg = TrainConfig(lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
                      epochs=a.epochs or dcfg.get("epochs") or tcfg["epochs"],
                      batch_size=tcfg["batch_size"], seed=a.seed, device=tcfg["device"],
                      eval_every=tcfg["eval_every"], patience=tcfg["patience"],
                      txn_drop=a.txn_drop or 0.0,
                      item_drop=a.item_drop or 0.0,
                      struct_lr_mult=a.struct_lr_mult or 1.0,
                      select_protocol=dcfg.get("select_protocol", "full_catalog"),
                      eval_batch_size=dcfg.get("eval_batch_size", 256))
    info = train_model(model, collator, train_seqs, b.seqs["val"], val_neg, b.num_items, cfg)
    model.eval()
    aux = {"aux_candidates": c} if (c := getattr(b, "aux_candidates", None)) else {}
    if a.eval_split == "val":   # best state on val; test inputs/negatives never built
        res = evaluate(model, b.seqs["val"], collator, val_neg, b.num_items,
                       device=tcfg["device"], cold_users=b.cold_users,
                       batch_size=cfg.eval_batch_size, **aux)
    else:
        res = evaluate(model, b.seqs["test"], collator, test_neg, b.num_items,
                       device=tcfg["device"], cold_users=b.cold_users,
                       batch_size=cfg.eval_batch_size, **aux)
    out = {
        "model": name, "seed": a.seed, "best_val": info["best_val_score"],
        "full_catalog": res["full_catalog"], "sampled_1to49": res["sampled_1to49"],
        "history": info["history"],
        "eval_split": a.eval_split, "full_nomask": res["full_nomask"],
    }
    if a.data != "katchers":
        out["data"] = a.data
    if "aux" in res:   # R5: the second target ranked inside the candidate subset
        out["r5"] = res["aux"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pu_dir = Path(a.out, "per_user")
    pu_dir.mkdir(parents=True, exist_ok=True)

    def _dump_per_user(tag, records):
        recs = [{"user_idx": r.user_idx, "repeat": r.repeat,
                 "has_copurchase": r.has_copurchase, "n_txn": r.n_txn, "cold": r.cold,
                 "full": r.full, "sampled": r.sampled, "full_nomask": r.full_nomask,
                 **({"extra_strata": list(r.extra_strata)} if r.extra_strata else {}),
                 **({"r5": r.aux} if r.aux else {})}
                for r in records]
        with gzip.open(pu_dir / f"{tag}_seed{a.seed}.json.gz", "wt") as f:
            json.dump(recs, f)

    # per-user records: every paired statistic is computed from these
    _dump_per_user(name, res["per_user"])
    Path(a.out, f"{name}_seed{a.seed}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("model", "seed", "best_val")}, indent=2))
    print("full-catalog overall:", out["full_catalog"]["overall"])
