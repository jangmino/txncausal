"""Hierarchical set -> sequence encoder, one of the screened variants (Appendix B of the paper).

  * order encoder: one intra-order self-attention layer (a token sees only its own order,
    no positions) + mean pooling -> one vector per order, permutation-invariant;
  * order sequence: order vectors + order positions -> CAUSAL transformer over orders;
  * order k's output state predicts every item of order k+1 (the last predicts the
    held-out target) -- TxnCausal's dense next-txn supervision, reused through
    TxnCausalModel (``hidden_states`` returns order states, ``txn_states`` is identity).

L6 (a [TXN] token per order) failed with only the final target supervised; this retries
the hierarchy with dense next-txn supervision.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TxnHierEncoder(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, max_num_txn: int, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.max_num_txn = max_num_txn
        self.set_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.pos_emb = nn.Embedding(max_num_txn + 1, dim, padding_idx=0)   # 0 = no order
        nn.init.normal_(self.pos_emb.weight, std=0.02)        # item-embedding scale
        with torch.no_grad():
            self.pos_emb.weight[0].zero_()
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)

    def hidden_states(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B,K,D): causal state of each order slot k (0-based = txn_pos - 1)."""
        tp, pad = batch["txn_pos"], batch["pad_mask"]
        x = item_embedding(batch["item_tokens"])                               # (B,L,D)
        eye = torch.eye(x.size(1), dtype=torch.bool, device=x.device)
        other = (tp[:, :, None] != tp[:, None, :]) | pad[:, None, :]   # other order / PAD
        x = self.set_layer(x, src_mask=(other & ~eye).repeat_interleave(self.nhead, dim=0))
        ks = torch.arange(1, self.max_num_txn + 1, device=x.device)
        member = (tp[:, None, :] == ks[None, :, None]) & ~pad[:, None, :]      # (B,K,L)
        present = member.any(-1)                                                # (B,K)
        z = (member.float() @ x) / member.sum(-1, keepdim=True).clamp_min(1)   # (B,K,D)
        z = z + self.pos_emb(ks[None, :] * present)
        z = self.dropout(self.input_norm(z))
        K = z.size(1)
        eye_k = torch.eye(K, dtype=torch.bool, device=z.device)
        future = torch.ones(K, K, dtype=torch.bool, device=z.device).triu(1)
        blocked = (future[None] | ~present[:, None, :]) & ~eye_k   # empty rows stay finite
        return self.transformer(z, mask=blocked.repeat_interleave(self.nhead, dim=0))

    def txn_states(self, h: torch.Tensor, batch: dict[str, torch.Tensor],
                   item_embedding=None) -> torch.Tensor:
        return h

    def forward(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.hidden_states(item_embedding, batch)
        last = (batch["txn_pos"][:, -1] - 1).clamp_min(0)     # newest token's order
        out = h[torch.arange(h.size(0), device=h.device), last]
        return out * (~batch["pad_mask"]).any(1, keepdim=True)
