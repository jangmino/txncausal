"""ConcurrentBERT-NSP (Park and Oh, 2024) inside the evaluation harness of this repository.

Follows scripts/BERT/{DataPrepWith,Train}.py of the repository released with that study
(msparkDev/ConcurrentBERT-Recommender):
  * text protocol: the released text-pair construction (_concurrentbert_text.py) --
    '## 주문 히스토리' prompt with '[동시구매]' markers, '## 다음 구매 상품 예측'
    candidates, 512-token add-if-fits truncation against the longest candidate;
  * recipe: google-bert/bert-base-multilingual-cased, BertForNextSentencePrediction,
    10 epochs, lr 2e-5, wd 0.01, best-by-eval-loss;
  * train/val pairs: 1 positive + 1 negative per user, the negative drawn uniformly with
    a seed (the effective draw of the released sampler).

Scoring follows the harness, so that the result can be paired with the other models:
each test user is scored on the 1:49 negatives released with the study, with mid-rank
tie-break, denominator 3358, and per-user records. Output matches the mvcp results layout
(per_user gz has full={} since the full catalog is not ranked in this mode).

``prep --text`` and ``prep --cand`` select the two versions of Appendix A of the paper:
the released code as is (aspublished, asis), and the retrained version in which the
history is in every prompt and one text per product is used for all candidates
(repaired, canonical).

Requires the `bert` extra IN THE PROJECT VENV:  uv pip install -e ".[bert]"
then:  uv run python -m mvcp.baselines.concurrentbert <prep|train|eval> ...
Do NOT use `uv run --with transformers,...`: the ephemeral overlay resolves its own
torch without CUDA (silent CPU fallback).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import negatives as neg
from ..data.vocab import build_vocab, load_frames
from ..eval.harness import _metrics, _rank_of_target, _rank_of_target_full
from ..pipeline import load_bundle
from ..utils.seed import make_rng

from ._concurrentbert_text import (add_text_if_fits, compile_order_history,
                                   format_next_purchase, get_longer_text)

MODEL_CKPT = "google-bert/bert-base-multilingual-cased"
KS = (1, 5, 10, 20)


def compile_order_history_repaired(user_prompt, tokenizer, item_max):
    """History builder of the retrained version: every order line is merged into the
    prompt -- chronological, dates on every line, '[동시구매]' marker, items
    comma-joined -- with the same greedy stop at the 512-token pair budget. (The
    released builder merges the history on its overflow path only; Appendix A of the
    paper.)"""
    user_text = "## 주문 히스토리"
    for date, group in user_prompt.groupby("initial_paid_at", sort=False):
        items = []
        for _, row in group.iterrows():
            t = f'{row["category_name"]} - {row["product_name"]}'
            if pd.notna(row["attribute_values"]):
                t += f' {row["attribute_values"]}'
            items.append(t)
        line = (f'\n{date}의 주문 내역' + (' [동시구매]' if len(group) > 1 else '')
                + ': ' + ', '.join(items))
        new = add_text_if_fits(user_text, line, tokenizer, item_max)
        if new is None:
            break
        user_text = new
    return user_text


COMPILE_FNS = {"aspublished": compile_order_history,
               "repaired": compile_order_history_repaired}


# ---------------------------------------------------------------- text assembly

def item_row_map(frames: dict[str, pd.DataFrame]) -> dict[int, pd.Series]:
    """item_idx -> representative row (the released recipe: groupby('product_id').last())."""
    vocab = build_vocab(frames)
    all_df = pd.concat(frames.values(), ignore_index=True)
    last = all_df.groupby("product_id", as_index=False).last()
    return {vocab.item2idx[int(r["product_id"])]: r for _, r in last.iterrows()}


def canonical_cand_map(frames: dict[str, pd.DataFrame], vocab) -> dict[int, str]:
    """SYMMETRIC candidate texts: one canonical, OPTION-FREE text per item, applied to
    every candidate (target and negatives) in both training and eval. Options/attribute
    strings are merchandising snapshots, not item identity (they vary per sale row);
    with them the text of a target (its own sale row) and the texts of the negatives
    (a representative row) come from different sources. Name = modal product_name
    with [promo] brackets stripped
    (``_canonical.canonical_names``); category = modal category_name."""
    from ._canonical import canonical_names
    names = canonical_names(frames, vocab.item2idx)
    all_df = pd.concat([f[["product_id", "category_name"]] for f in frames.values()],
                       ignore_index=True)
    cats: dict[int, str] = {}
    for pid, g in all_df.groupby("product_id"):
        m = g["category_name"].dropna()
        cats[vocab.item2idx[int(pid)]] = str(m.mode().iloc[0]) if len(m) else ""
    return {i: f'## 다음 구매 상품 예측\n{cats.get(i, "")} - {names[i]}'
            for i in range(vocab.num_items)}


def user_frames(frame: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """user_id -> (history rows sorted by paid_at, target row). Split CSVs keep exactly
    one row on each user's last date (verified: structural_checks multi_last=0)."""
    out = {}
    for uid, udf in frame.groupby("user_id", sort=False):
        last_t = int(udf["initial_paid_at"].max())
        target_row = udf[udf["initial_paid_at"] == last_t].iloc[0]
        hist = udf[udf["initial_paid_at"] != last_t].sort_values("initial_paid_at")
        out[int(uid)] = (hist, target_row)
    return out


def build_pairs_split(frame, vocab, tokenizer, seed: int, split: str,
                      item_rows: dict[int, pd.Series],
                      compile_fn=compile_order_history,
                      cand_map: dict[int, str] | None = None) -> pd.DataFrame:
    """Train/val: (prompt, positive) + (prompt, 1 uniform negative) per user."""
    rng = make_rng(seed, f"cb_pairs:{split}")
    rows = []
    for uid, (hist, tgt_row) in user_frames(frame).items():
        tgt_idx = vocab.item2idx[int(tgt_row["product_id"])]
        neg_idx = int(rng.integers(0, vocab.num_items))
        if cand_map is not None:
            item_pos, item_neg = cand_map[tgt_idx], cand_map[neg_idx]
        else:
            item_pos = format_next_purchase(tgt_row)
            item_neg = format_next_purchase(item_rows[neg_idx])
        item_max = get_longer_text(item_pos, item_neg, tokenizer)
        prompt = compile_fn(hist, tokenizer, item_max)
        rows += [{"prev": prompt, "next": item_pos, "label": 1},
                 {"prev": prompt, "next": item_neg, "label": 0}]
    return pd.DataFrame(rows)


def build_test_manifest(frame, vocab, tokenizer, published: dict[str, list[int]],
                        item_rows: dict[int, pd.Series],
                        compile_fn=compile_order_history,
                        cand_map: dict[int, str] | None = None):
    """Per test user: prompt (truncated vs the longest of ALL 50 candidates, like their
    test mode) + ordered candidate list [target] + 49 published negatives."""
    prompts, cands = [], []
    for uid, (hist, tgt_row) in user_frames(frame).items():
        u_idx = vocab.user2idx[uid]
        tgt_idx = vocab.item2idx[int(tgt_row["product_id"])]
        cand_idx = [tgt_idx] + [int(x) for x in published[str(u_idx)]]
        if cand_map is not None:
            texts = [cand_map[i] for i in cand_idx]
        else:
            texts = [format_next_purchase(tgt_row)] + [
                format_next_purchase(item_rows[i]) for i in cand_idx[1:]]
        item_max = texts[0]
        for t in texts[1:]:
            item_max = get_longer_text(item_max, t, tokenizer)
        prompt = compile_fn(hist, tokenizer, item_max)
        prompts.append({"user_idx": u_idx, "prev": prompt})
        cands += [{"user_idx": u_idx, "pos": p, "item_idx": i, "next": t,
                   "label": int(p == 0)} for p, (i, t) in enumerate(zip(cand_idx, texts))]
    return pd.DataFrame(prompts), pd.DataFrame(cands)


# ---------------------------------------------------------------- subcommands

def prep(a) -> None:
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_CKPT)
    compile_fn = COMPILE_FNS[a.text]
    frames = load_frames(a.data_dir)
    vocab = build_vocab(frames)
    item_rows = item_row_map(frames)
    cand_map = canonical_cand_map(frames, vocab) if a.cand == "canonical" else None
    out = Path(a.workdir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        frame = frames[split]
        if a.max_users:
            keep = frame["user_id"].drop_duplicates().head(a.max_users)
            frame = frame[frame["user_id"].isin(keep)]
        df = build_pairs_split(frame, vocab, tokenizer, a.seed, split, item_rows,
                               compile_fn, cand_map)
        df.to_csv(out / f"{split}.csv", index=False)
        print(f"[prep] {split}: {len(df)} pairs")
    frame = frames["test"]
    if a.max_users:
        keep = frame["user_id"].drop_duplicates().head(a.max_users)
        frame = frame[frame["user_id"].isin(keep)]
    published = neg.load_published(a.published)
    prompts, cands = build_test_manifest(frame, vocab, tokenizer, published,
                                         item_rows, compile_fn, cand_map)
    prompts.to_csv(out / "test_prompts.csv", index=False)
    cands.to_csv(out / "test_cands.csv", index=False)
    print(f"[prep] test: {len(prompts)} users, {len(cands)} pairs")


def train(a) -> None:
    import torch
    from datasets import Dataset
    from transformers import (BertForNextSentencePrediction, BertTokenizer,
                              Trainer, TrainingArguments)
    tokenizer = BertTokenizer.from_pretrained(MODEL_CKPT)
    wd = Path(a.workdir)

    def tok(batch):
        return tokenizer(batch["prev"], batch["next"], padding="max_length",
                         truncation=True, max_length=512)

    ds = {s: Dataset.from_pandas(pd.read_csv(wd / f"{s}.csv")).map(tok, batched=True)
          for s in ("train", "val")}
    model = BertForNextSentencePrediction.from_pretrained(MODEL_CKPT)
    args = TrainingArguments(
        output_dir=str(wd / f"ckpt_seed{a.seed}"), num_train_epochs=a.epochs,
        learning_rate=2e-5, weight_decay=0.01,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch * 4,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, seed=a.seed, log_level="error",
        disable_tqdm=True, report_to=[], fp16=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["val"])
    trainer.train()
    final = wd / f"model_seed{a.seed}"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[train] saved {final}")


def evaluate(a) -> None:
    import torch
    from transformers import BertForNextSentencePrediction, BertTokenizer
    wd = Path(a.workdir)
    mdir = wd / f"model_seed{a.seed}"
    tokenizer = BertTokenizer.from_pretrained(str(mdir))
    model = BertForNextSentencePrediction.from_pretrained(str(mdir)).to(a.device).eval()
    prompts = pd.read_csv(wd / "test_prompts.csv").set_index("user_idx")["prev"]
    cands = pd.read_csv(wd / "test_cands.csv")
    if getattr(a, "swap_target_rep", False):
        # source check: replace the target's own-row text with its REPRESENTATIVE-row
        # text (the same source the negatives use), keeping prompts fixed (63.3% of
        # targets differ)
        frames = load_frames(a.data_dir)
        item_rows = item_row_map(frames)
        tgt = cands["pos"] == 0
        swapped = cands.loc[tgt, "item_idx"].map(
            lambda i: format_next_purchase(item_rows[int(i)]))
        n_changed = int((swapped.values != cands.loc[tgt, "next"].values).sum())
        cands.loc[tgt, "next"] = swapped.values
        print(f"[swap] target texts replaced; changed for {n_changed}/{tgt.sum()} users")
    b = load_bundle(a.data_dir)

    recs, sampled_overall = [], []
    users = sorted(prompts.index)
    with torch.no_grad():
        for n, u in enumerate(users):
            g = cands[cands.user_idx == u].sort_values("pos")
            scores = []
            texts = g["next"].tolist()
            for i in range(0, len(texts), a.batch):
                enc = tokenizer([prompts[u]] * len(texts[i:i + a.batch]),
                                texts[i:i + a.batch], padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(a.device)
                lg = model(**enc).logits
                scores += (lg[:, 1] - lg[:, 0]).cpu().tolist()   # the released score direction
            rank = _rank_of_target(np.array(scores), 0)          # harness mid-rank rule
            m = _metrics(rank, KS)
            s = b.seqs["test"][u]
            recs.append({"user_idx": u, "repeat": s.repeat,
                         "has_copurchase": s.has_copurchase, "n_txn": s.n_txn,
                         "cold": u in b.cold_users, "full": {}, "sampled": m})
            sampled_overall.append(m)
            if (n + 1) % 200 == 0:
                print(f"[eval] {n + 1}/{len(users)}", flush=True)

    agg = {k: float(np.mean([m[k] for m in sampled_overall]))
           for k in sampled_overall[0]}
    out = Path(a.out)
    (out / "per_user").mkdir(parents=True, exist_ok=True)
    tag = f"{a.tag}_seed{a.seed}"
    (out / f"{tag}.json").write_text(json.dumps(
        {"model": a.tag, "seed": a.seed, "n_users": len(recs),
         "sampled_1to49_overall": agg}, indent=2))
    with gzip.open(out / "per_user" / f"{tag}.json.gz", "wt") as f:
        json.dump(recs, f)
    print(json.dumps(agg, indent=2))


def evaluate_full(a) -> None:
    """Full-catalog protocol for the SYM recipe: repaired prompts + canonical texts for
    ALL 1,608 candidates, harness rules verbatim (history masked to -inf, target never
    masked, mid-rank tie-break, denominator 3358). Prompt truncation budget = longest
    canonical text over the FULL catalog (the released test-mode rule extended -- prompts can
    differ slightly from the 1:49 manifest, so sampled metrics here are a consistency
    check, not a replacement). Resumable: per-user records are appended to
    <out>/<tag>_full_seed<seed>.partial.jsonl and completed users are skipped."""
    import torch
    from transformers import BertForNextSentencePrediction, BertTokenizer
    wd = Path(a.workdir)
    mdir = wd / f"model_seed{a.seed}"
    tokenizer = BertTokenizer.from_pretrained(str(mdir))
    model = BertForNextSentencePrediction.from_pretrained(str(mdir)).to(a.device).eval()

    frames = load_frames(a.data_dir)
    vocab = build_vocab(frames)
    cand_map = canonical_cand_map(frames, vocab)
    texts = [cand_map[i] for i in range(vocab.num_items)]
    item_max = texts[0]
    for t in texts[1:]:
        item_max = get_longer_text(item_max, t, tokenizer)
    published = neg.load_published(a.published)
    b = load_bundle(a.data_dir)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.tag}_full_seed{a.seed}"
    partial = out / f"{tag}.partial.jsonl"
    done: dict[int, dict] = {}
    if partial.exists():
        with open(partial) as f:
            for line in f:
                r = json.loads(line)
                done[r["user_idx"]] = r
        print(f"[full] resuming: {len(done)} users already scored")

    frame = frames["test"]
    users = user_frames(frame)
    if a.max_users:
        users = dict(list(users.items())[:a.max_users])
    recs = []
    with torch.no_grad(), open(partial, "a") as pf:
        for n, (uid, (hist, tgt_row)) in enumerate(sorted(
                users.items(), key=lambda kv: vocab.user2idx[kv[0]])):
            u = vocab.user2idx[uid]
            if u in done:
                recs.append(done[u])
                continue
            s = b.seqs["test"][u]
            assert s.target == vocab.item2idx[int(tgt_row["product_id"])]
            prompt = compile_order_history_repaired(hist, tokenizer, item_max)
            scores = []
            for i in range(0, len(texts), a.batch):
                enc = tokenizer([prompt] * len(texts[i:i + a.batch]),
                                texts[i:i + a.batch], padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(a.device)
                lg = model(**enc).logits
                scores += (lg[:, 1] - lg[:, 0]).cpu().tolist()
            scores = np.array(scores)

            full_scores = scores.copy()
            if s.items:
                full_scores[list(set(s.items))] = -np.inf
            full_scores[s.target] = scores[s.target]  # never mask the target itself
            full_m = _metrics(_rank_of_target_full(full_scores, s.target), KS)
            cand = np.array([s.target] + [int(x) for x in published[str(u)]])
            samp_m = _metrics(_rank_of_target(scores[cand], 0), KS)

            rec = {"user_idx": u, "repeat": s.repeat,
                   "has_copurchase": s.has_copurchase, "n_txn": s.n_txn,
                   "cold": u in b.cold_users, "full": full_m, "sampled": samp_m}
            recs.append(rec)
            pf.write(json.dumps(rec) + "\n")
            pf.flush()
            if (n + 1) % 20 == 0:
                print(f"[full] {n + 1}/{len(users)}", flush=True)

    agg = {p: {k: float(np.mean([r[p][k] for r in recs])) for k in recs[0][p]}
           for p in ("full", "sampled")}
    (out / "per_user").mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(
        {"model": f"{a.tag}_full", "seed": a.seed, "n_users": len(recs),
         "full_overall": agg["full"], "sampled_1to49_overall": agg["sampled"]},
        indent=2))
    with gzip.open(out / "per_user" / f"{tag}.json.gz", "wt") as f:
        json.dump(recs, f)
    print(json.dumps(agg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--published", default="data/negatives/published.json")
    p.add_argument("--workdir", default="artifacts/bert_nsp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-users", type=int, default=None, help="smoke")
    p.add_argument("--text", choices=sorted(COMPILE_FNS), default="aspublished")
    p.add_argument("--cand", choices=("asis", "canonical"), default="asis",
                   help="candidate texts: asis (own-row target vs representative-row "
                        "negatives, as released) or canonical "
                        "(option-free modal text for ALL candidates, symmetric)")
    t = sub.add_parser("train")
    t.add_argument("--workdir", default="artifacts/bert_nsp")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=16)
    e = sub.add_parser("eval")
    e.add_argument("--workdir", default="artifacts/bert_nsp")
    e.add_argument("--data-dir", default="data")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--device", default="cuda")
    e.add_argument("--out", default="results/bert_nsp")
    e.add_argument("--tag", default="bert_nsp")
    e.add_argument("--swap-target-rep", action="store_true",
                   help="source check: target scored with representative-row text")
    e.add_argument("--full", action="store_true",
                   help="full-catalog protocol (sym recipe: repaired prompts + "
                        "canonical texts for all items, history masked)")
    e.add_argument("--published", default="data/negatives/published.json")
    e.add_argument("--max-users", type=int, default=None, help="smoke")
    a = ap.parse_args()
    if a.cmd == "eval" and a.full:
        evaluate_full(a)
    else:
        {"prep": prep, "train": train, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
