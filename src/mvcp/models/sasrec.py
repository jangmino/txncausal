"""SASRec baseline (Kang & McAuley, ICDM 2018) on the shared backbone.

Canonical SASRec = CAUSAL self-attention over the flattened item sequence with item-level
absolute positions (a total order, even inside one order), trained to predict the next
item at EVERY position. Everything else is held equal to L3 (txf_item_pos): same pre-LN
blocks, dim/heads/layers/dropout, tied item table, newest-token readout, same init draws.
So L3 -> SASRec isolates exactly the SASRec recipe (causal mask + all-position targets).

Loss: catalog-wide softmax CE per position -- the project's objective, and the stronger
SASRec variant (Klenitskiy & Vasilev, RecSys 2023) -- instead of the original BCE with
one sampled negative.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import SequenceCollator
from .base import RecModel
from .transformer import TxnTransformerEncoder, by_length_split


class SASRecCollator(SequenceCollator):
    """SequenceCollator + ``seq_labels`` (B,L): at each real position, the next item of
    the flattened history+target sequence (the newest position's is the target);
    -100 (ignored) at PAD. Inputs are unchanged, so eval batches are L3's.
    ``intra_labels=False`` also ignores labels whose next item is in the SAME order."""

    def __init__(self, max_seq_len: int = 64, max_num_txn: int = 52,
                 intra_labels: bool = True):
        super().__init__(max_seq_len, max_num_txn)
        self.intra_labels = intra_labels

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        out = super().__call__(batch)
        L = self.max_seq_len
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        for b, u in enumerate(batch):
            n = min(len(u.items), L)
            if n:
                labels[b, L - n:] = torch.tensor((u.items + [u.target])[-n:])
                if not self.intra_labels:
                    cur = torch.tensor(u.txn_ids[-n:])
                    nxt = torch.tensor((u.txn_ids + [-1])[-n:])   # target: never same order
                    labels[b, L - n:][cur == nxt] = -100
        out["seq_labels"] = labels
        return out


class SASRecEncoder(TxnTransformerEncoder):
    def __init__(self, dim: int, max_seq_len: int, max_num_txn: int, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, readout: str = "last",
                 len_split: bool = False, pos: str = "item"):
        # pos='txn' (component ablation, the research plan): the items of an
        # order share one positional embedding; the causal mask and labels stay item-level
        assert pos in ("item", "txn")
        super().__init__(dim, max_seq_len, max_num_txn, pos_mode=pos,
                         pool_mode="last_item", nhead=nhead, num_layers=num_layers,
                         dropout=dropout)
        assert readout in ("last", "blend_emb")
        self.nhead = nhead
        self.len_split = len_split        # encode long and short rows separately
        self.readout = readout
        if readout == "blend_emb":
            # symmetric control for TxnCausal H7a: same zero-init gate/query + LayerNorm
            # over raw item embeddings, pooled causally over positions <= t (no RNG draws)
            self.blend_query = nn.Parameter(torch.zeros(dim))
            self.blend_gate = nn.Parameter(torch.zeros(dim))
            self.blend_norm = nn.LayerNorm(dim)

    def hidden_states(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B,L,D); position i attends only to real positions <= i."""
        if self.len_split:
            return by_length_split(lambda sub: self._hidden_states(item_embedding, sub),
                                   batch)
        return self._hidden_states(item_embedding, batch)

    def _hidden_states(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pos = batch["txn_pos" if self.pos_mode == "txn" else "item_pos"]
        x = item_embedding(batch["item_tokens"]) + self.pos_emb(pos)
        x = self.dropout(self.input_norm(x))
        L = x.size(1)
        eye = torch.eye(L, dtype=torch.bool, device=x.device)
        future = torch.ones(L, L, dtype=torch.bool, device=x.device).triu(1)
        blocked = future | batch["pad_mask"][:, None, :]          # (B,L,L) True = masked
        # the diagonal stays open so all-PAD query rows stay finite (never read: their
        # labels are -100 and the readout is the newest, always-real token)
        blocked = blocked & ~eye
        return self.transformer(x, mask=blocked.repeat_interleave(self.nhead, dim=0))

    def states(self, h: torch.Tensor, batch: dict[str, torch.Tensor],
               item_embedding) -> torch.Tensor:
        """(B,L,D) per-position states: ``h``, or g*h + (1-g)*attnpool(LN(e_<=t))."""
        if self.readout == "last":
            return h
        src = self.blend_norm(item_embedding(batch["item_tokens"]))              # (B,L,D)
        L = h.size(1)
        upto = ~torch.ones(L, L, dtype=torch.bool, device=h.device).triu(1)       # key <= query
        upto = upto[None] & ~batch["pad_mask"][:, None, :]                         # (B,L,L)
        score = (src @ self.blend_query)[:, None, :].masked_fill(~upto, float("-inf"))
        score = score.masked_fill(~upto.any(-1, keepdim=True), 0.0)  # PAD rows: finite
        pooled = (torch.softmax(score, -1) * upto) @ src
        g = torch.sigmoid(self.blend_gate)
        return g * h + (1 - g) * pooled

    def forward(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.readout == "last":
            return self.hidden_states(item_embedding, batch)[:, -1, :]
        h = self.hidden_states(item_embedding, batch)
        return self.states(h, batch, item_embedding)[:, -1, :]


class SASRecModel(RecModel):
    def __init__(self, num_items: int, dim: int, encoder, final_weight: float = 1.0,
                 compact_loss: bool = False):
        super().__init__(num_items, dim, encoder)
        self.final_weight = final_weight
        self.compact_loss = compact_loss

    def train_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Next-item CE at every real position (RecModel: newest position only).
        ``final_weight`` multiplies the newest position's term (it predicts the held-out
        target) -- the symmetric control for TxnCausal's H4b final-term weighting."""
        h = self.encoder.hidden_states(self.item_embedding, batch)    # (B,L,D)
        h = self.encoder.states(h, batch, self.item_embedding)
        if self.compact_loss:
            # long sequences: score the labelled positions only -- the same loss without
            # (B,L,N) logits over PAD positions
            labels = batch["seq_labels"]
            real = labels != -100
            ce = F.cross_entropy(h[real] @ self.item_embedding.weight[1:].t(), labels[real],
                                 reduction="none")
            last = torch.zeros_like(real)
            last[:, -1] = True
            w = 1.0 + (self.final_weight - 1.0) * last[real].float()
            return (w * ce).sum() / w.sum()
        logits = h @ self.item_embedding.weight[1:].t()                # (B,L,N)
        if self.final_weight == 1.0:
            return F.cross_entropy(logits.flatten(0, 1), batch["seq_labels"].flatten(),
                                   ignore_index=-100)
        labels = batch["seq_labels"]
        ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100,
                             reduction="none").view_as(labels)
        w = (labels != -100).float()
        w[:, -1] *= self.final_weight
        return (w * ce).sum() / w.sum()
