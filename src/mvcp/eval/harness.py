"""The single evaluation harness. Every reported number comes from here.

Two protocols on the SAME model outputs:
  full_catalog : rank the target against all NUM_ITEMS, history items masked to -inf.
                 Primary metric; removes the 1:49 easy-task ceiling. (1,608 items -> cheap)
  sampled_1to49: rank the target against 49 frozen negatives (Park & Oh parity).
  full_nomask  : auxiliary -- full-catalog rank with history items NOT masked (guards
                 against wins that only exist because masking hands repeat users the answer).

Fairness rules baked in:
  * mid-rank neutral tie-break (target is NOT placed first on ties);
  * fixed denominator = all evaluated users, no silent skip;
  * per-user records retained so adjacent models can be compared with paired tests;
  * every metric is emitted overall AND per stratum {repeat/novel, copurchase, len, cold}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.dataset import SequenceCollator, UserSeq
from .strata import stratify


@dataclass
class UserResult:
    user_idx: int
    repeat: bool
    has_copurchase: bool
    n_txn: int
    cold: bool
    full: dict[str, float] = field(default_factory=dict)
    sampled: dict[str, float] = field(default_factory=dict)
    full_nomask: dict[str, float] = field(default_factory=dict)
    extra_strata: tuple[str, ...] = ()
    aux: dict[str, float] = field(default_factory=dict)   # empty unless aux_target is set


def _rank_of_target(scores: np.ndarray, target_pos: int) -> float:
    """Mid-rank of the target (0-based): #strictly-greater + 0.5 * #ties(excl. target)."""
    s_t = scores[target_pos]
    greater = int(np.sum(scores > s_t))
    ties = int(np.sum(scores == s_t)) - 1  # exclude target itself
    return greater + 0.5 * ties


def _metrics(rank: float, ks) -> dict[str, float]:
    out: dict[str, float] = {"mrr": 1.0 / (rank + 1.0)}
    for k in ks:
        hit = rank < k
        out[f"recall@{k}"] = float(hit)
        out[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 2.0)) if hit else 0.0
    return out


@torch.no_grad()
def evaluate(scorer, seqs: dict[int, UserSeq], collator: SequenceCollator,
             negatives: dict[str, list[int]], num_items: int, device: str = "cpu",
             cold_users: set[int] | None = None, ks=(1, 5, 10, 20),
             batch_size: int = 256, aux_candidates: list[int] | None = None) -> dict:
    """`scorer(batch) -> logits (B, num_items)` (a model or any callable). Returns overall +
    per-stratum aggregates for both protocols, plus per-user records.
    ``aux_candidates``: users with an ``aux_target`` also get its unmasked mid-rank among
    these items only (``UserResult.aux``; the result gains an ``aux`` aggregate)."""
    cold_users = cold_users or set()
    if aux_candidates is not None:
        aux_cand = np.asarray(aux_candidates)
        aux_pos = {int(it): j for j, it in enumerate(aux_candidates)}
    users = list(seqs)
    results: list[UserResult] = []

    for i in range(0, len(users), batch_size):
        chunk = users[i:i + batch_size]
        batch = collator([seqs[u] for u in chunk])
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        logits = scorer(batch_dev).detach().cpu().numpy()  # (B, num_items)

        for row, u in enumerate(chunk):
            s = seqs[u]
            lg = logits[row]
            # full-catalog: mask history items (already-purchased) to -inf
            full_scores = lg.copy()
            seen = s.items if s.history_items is None else s.history_items
            if seen:
                full_scores[list(set(seen))] = -np.inf
            full_scores[s.target] = lg[s.target]  # never mask the target itself
            full_rank = _rank_of_target_full(full_scores, s.target)
            nomask_rank = _rank_of_target_full(lg, s.target)

            # sampled 1:49
            negs = negatives[str(u)]
            cand = np.array([s.target] + list(negs))
            cand_scores = lg[cand]
            samp_rank = _rank_of_target(cand_scores, 0)
            aux = {}
            if aux_candidates is not None and s.aux_target is not None:
                aux = _metrics(_rank_of_target(lg[aux_cand], aux_pos[s.aux_target]), ks)

            results.append(UserResult(
                user_idx=u, repeat=s.repeat, has_copurchase=s.has_copurchase,
                n_txn=s.n_txn, cold=u in cold_users,
                full=_metrics(full_rank, ks), sampled=_metrics(samp_rank, ks),
                full_nomask=_metrics(nomask_rank, ks), extra_strata=s.extra_strata,
                aux=aux,
            ))

    out_aux = {}
    if aux_candidates is not None:
        sub = [r.aux for r in results if r.aux]
        out_aux = {"aux": {"n": len(sub), **{k: float(np.mean([a[k] for a in sub]))
                                            for k in (sub[0] if sub else {})}}}
    return {
        **out_aux,
        "n_users": len(results),
        "full_catalog": stratify(results, "full", ks),
        "sampled_1to49": stratify(results, "sampled", ks),
        "full_nomask": stratify(results, "full_nomask", ks),
        "per_user": results,
    }


def _rank_of_target_full(scores: np.ndarray, target: int) -> float:
    s_t = scores[target]
    greater = int(np.sum(scores > s_t))
    ties = int(np.sum(scores == s_t)) - 1
    return greater + 0.5 * ties
