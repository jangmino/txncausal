"""Configurable Transformer encoder: the base class of the SASRec, BERT4Rec and TxnCausal
encoders, which set its two switches and add their own attention mask.

    pos_mode : none  -> no positional embedding
               item  -> absolute item-level position (a total order)
               txn   -> transaction-level position (shared within a transaction)
    pool_mode: last_item      -> pick the newest token
               last_txn_mean  -> mean over the last transaction's tokens
               last_txn_attn  -> attention over the last transaction's tokens

Key property (unit-tested): pos_mode='txn' + a permutation-invariant pool (last_txn_mean/
attn) is exactly invariant to intra-transaction reordering; pos_mode='item' is not.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import AttentionPooling, masked_mean


def by_length_split(fn, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """``fn(sub_batch) -> (b,l,D)`` on two groups of rows -- the longest sequences and the
    rest, cut where the attention-map area b1*l1^2 + b2*l2^2 is smallest -- each trimmed to
    its own longest sequence (inputs are right-aligned), scattered back to (B,L,D). Real
    tokens get the states of one padded pass (PAD keys are masked either way) without most
    of the attention over padding; the states at PAD positions differ and are never read."""
    pad = batch["pad_mask"]
    B, L = pad.shape
    n_real, order = (~pad).sum(1).clamp_min(1).sort(descending=True, stable=True)
    cut = B
    if B > 1:
        i = torch.arange(1, B, device=pad.device)
        cut = 1 + int((i * n_real[0] ** 2 + (B - i) * n_real[1:] ** 2).argmin())
    out = None
    for lo, hi in ((0, cut), (cut, B)):
        if lo < hi:
            rows, keep = order[lo:hi], int(n_real[lo])
            h = fn({k: v[rows, L - keep:] for k, v in batch.items() if v.shape[:2] == (B, L)})
            if out is None:
                out = h.new_zeros(B, L, h.size(-1))
            out[rows, L - keep:] = h
    return out


class TxnTransformerEncoder(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, max_num_txn: int,
                 pos_mode: str = "txn", pool_mode: str = "last_txn_mean",
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert pos_mode in ("none", "item", "txn")
        assert pool_mode in ("last_item", "last_txn_mean", "last_txn_attn")
        self.pos_mode = pos_mode
        self.pool_mode = pool_mode

        if pos_mode == "item":
            self.pos_emb = nn.Embedding(max_seq_len + 1, dim, padding_idx=0)
        elif pos_mode == "txn":
            self.pos_emb = nn.Embedding(max_num_txn + 2, dim, padding_idx=0)
        else:
            self.pos_emb = None
        if self.pos_emb is not None:
            # match the item-embedding scale (base.py std=0.02); the PyTorch default
            # N(0,1) drowns item content 50x at the encoder input
            nn.init.normal_(self.pos_emb.weight, std=0.02)
            with torch.no_grad():
                self.pos_emb.weight[0].zero_()

        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        self.pool_attn = AttentionPooling(dim) if pool_mode == "last_txn_attn" else None

    def forward(self, item_embedding: nn.Embedding,
                batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = item_embedding(batch["item_tokens"])                # (B,L,D)
        if self.pos_mode == "item":
            x = x + self.pos_emb(batch["item_pos"])
        elif self.pos_mode == "txn":
            x = x + self.pos_emb(batch["txn_pos"])
        x = self.dropout(self.input_norm(x))

        pad = batch["pad_mask"]                                  # True = PAD
        h = self.transformer(x, src_key_padding_mask=pad)       # (B,L,D)

        if self.pool_mode == "last_item":
            return h[:, -1, :]                                   # newest token (right-aligned)
        last = batch["last_txn"]                                 # (B,L) bool
        if self.pool_mode == "last_txn_mean":
            return masked_mean(h, last)
        return self.pool_attn(h, last)
