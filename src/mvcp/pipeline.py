"""Assemble the data objects every entry point needs (vocab, per-split sequences,
cold-start set), so CLI and tests share one construction path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data.dataset import UserSeq, build_user_seqs
from .data.vocab import SPLITS, Vocab, build_vocab, load_frames


@dataclass
class Bundle:
    vocab: Vocab
    seqs: dict[str, dict[int, UserSeq]]
    cold_users: set[int]
    test_negatives: dict[str, list[int]] | None = None   # a dataset's own frozen 1:49 set
    aux_candidates: list[int] | None = None   # candidate subset of `UserSeq.aux_target`

    @property
    def num_items(self) -> int:
        return self.vocab.num_items


def load_bundle(data_dir: str | Path, cold_threshold: int = 3) -> Bundle:
    frames = load_frames(data_dir)
    vocab = build_vocab(frames)
    seqs = {s: build_user_seqs(frames[s], vocab) for s in SPLITS}
    cold = {
        u for split in SPLITS for u, sq in seqs[split].items()
        if sq.n_txn <= cold_threshold
    }
    return Bundle(vocab=vocab, seqs=seqs, cold_users=cold)
