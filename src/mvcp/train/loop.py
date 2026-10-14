"""Model-agnostic trainer. Every model reuses this exact loop and hyperparameters, so a
metric delta reflects the model, not the training recipe.

Objective: catalog-wide Cross-Entropy over the tied item table (doc S5.3).
Selection: best full-catalog masked val NDCG@10 -- a STATIONARY estimator (no random
negatives redrawn each eval), fixing the original moving-target early stopping.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import UserSeq
from ..eval.harness import evaluate
from ..utils.seed import make_rng, seed_everything


def _augment(s: UserSeq, rng: np.random.Generator, txn_drop: float,
             item_drop: float) -> UserSeq:
    """H5 order-structure augmentation (training only): each PAST order is dropped whole
    with prob ``txn_drop`` (the newest order always stays), and inside multi-item orders
    each item is dropped with prob ``item_drop`` (>= 1 item per order stays). Kept orders
    are renumbered 0..T'-1; the held-out target is untouched."""
    groups: dict[int, list[int]] = {}
    for it, tx in zip(s.items, s.txn_ids):
        groups.setdefault(tx, []).append(it)
    order = sorted(groups)
    kept = [t for t in order[:-1] if rng.random() >= txn_drop] + order[-1:]
    items: list[int] = []
    tids: list[int] = []
    for new_t, t in enumerate(kept):
        g = groups[t]
        if item_drop and len(g) > 1:
            keep = rng.random(len(g)) >= item_drop
            if not keep.any():
                keep[rng.integers(len(g))] = True
            g = [it for it, k in zip(g, keep) if k]
        items += g
        tids += [new_t] * len(g)
    has_cop = any(v >= 2 for v in Counter(tids).values())
    return UserSeq(s.user_idx, items, tids, s.target, s.target in set(items), has_cop)


# zero-initialised structural parameters (H1 gate/query, H2 bias): with Adam at lr 2.5e-4
# they barely move (H1 gate stayed 0.49-0.52), so H7 gives them their own lr multiplier
STRUCT_PARAM_KEYS = ("blend_query", "blend_gate", "rel_bias")


def _param_groups(model, cfg: "TrainConfig"):
    if cfg.struct_lr_mult == 1.0:
        return model.parameters()
    struct = [p for n, p in model.named_parameters()
              if any(k in n for k in STRUCT_PARAM_KEYS)]
    if not struct:
        raise ValueError("struct_lr_mult set but the model has no structural parameters")
    ids = {id(p) for p in struct}
    return [{"params": [p for p in model.parameters() if id(p) not in ids]},
            {"params": struct, "lr": cfg.lr * cfg.struct_lr_mult}]


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    batch_size: int = 64
    eval_every: int = 5
    patience: int = 6
    device: str = "cpu"
    seed: int = 0
    select_metric: str = "ndcg@10"       # 'overall' stratum of `select_protocol`
    select_protocol: str = "full_catalog"   # or 'full_nomask' (history not masked)
    txn_drop: float = 0.0                # H5a: drop whole past orders (training only)
    item_drop: float = 0.0               # H5b: drop items inside multi-item orders
    struct_lr_mult: float = 1.0          # H7: lr multiplier for STRUCT_PARAM_KEYS params
    eval_batch_size: int = 256           # selection evals (lower for long sequences)


def train_model(model, collator, train_seqs: dict[int, UserSeq],
                val_seqs: dict[int, UserSeq], val_negatives: dict[str, list[int]],
                num_items: int, cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    dev = cfg.device
    model.to(dev)
    opt = torch.optim.Adam(_param_groups(model, cfg), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    rng = make_rng(cfg.seed, "shuffle")
    rng_aug = make_rng(cfg.seed, "augment") if (cfg.txn_drop or cfg.item_drop) else None

    users = list(train_seqs)
    best_score, best_state, patience = -1.0, None, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = users.copy()
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            chunk = order[i:i + cfg.batch_size]
            seqs_batch = [train_seqs[u] for u in chunk]
            if rng_aug is not None:
                seqs_batch = [_augment(s, rng_aug, cfg.txn_drop, cfg.item_drop)
                              for s in seqs_batch]
            batch = collator(seqs_batch)
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model.train_loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(chunk)
        train_loss = total / len(order)

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            model.eval()
            res = evaluate(model, val_seqs, collator, val_negatives, num_items,
                           device=dev, batch_size=cfg.eval_batch_size)
            score = res[cfg.select_protocol]["overall"][cfg.select_metric]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_score": score})
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_score": best_score, "history": history}
