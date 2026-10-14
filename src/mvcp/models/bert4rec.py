"""BERT4Rec baseline (Sun et al., CIKM 2019) on the shared recipe.

L3's backbone (txf_item_pos: bidirectional self-attention + item-level absolute
positions; same pre-LN blocks, dim/heads/layers/dropout, tied item table) with the cloze
objective and a [MASK]-slot readout:

  * the encoded sequence is the history window (the L items every model sees) plus ONE
    appended slot -- the held-out target's place -- that always holds [MASK]; the position
    table has one more row for it;
  * training: every real history slot is also replaced by [MASK] with prob ``mask_prob``
    (redrawn each step, torch RNG); loss = catalog softmax CE averaged over the hidden
    slots. The appended slot is the one "final term" per user and epoch, the counterpart
    of the other models' held-out-target term (``final_weight`` multiplies it);
  * eval: the user representation is the appended slot's state. The target is never an
    input, in training or in eval.

Unlike the original: no output transform (linear + GELU + LayerNorm) and no output bias --
scores are user-state x item-embedding dot products like every model here; and the target
slot is always hidden instead of mixing in extra "only the last item masked" samples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RecModel
from .transformer import TxnTransformerEncoder, by_length_split


class BERT4RecEncoder(TxnTransformerEncoder):
    def __init__(self, dim: int, max_seq_len: int, max_num_txn: int, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, len_split: bool = False,
                 pos: str = "item"):
        # pos='txn' (component ablation, the research plan): the items of an
        # order share one positional embedding and the appended [MASK] slot takes the
        # position of the next order (the positional scheme of BTBR, Li et al. 2023);
        # the transaction table already has the extra row
        assert pos in ("item", "txn")
        super().__init__(dim, max_seq_len + 1, max_num_txn, pos_mode=pos,
                         pool_mode="last_item", nhead=nhead, num_layers=num_layers,
                         dropout=dropout)
        self.len_split = len_split        # encode long and short rows separately
        self.mask_emb = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.mask_emb, std=0.02)       # the item-embedding scale

    def hidden_states(self, item_embedding, batch: dict[str, torch.Tensor],
                      masked: torch.Tensor | None = None) -> torch.Tensor:
        """(B,L+1,D): the history slots, then the appended [MASK] slot (index -1).
        ``masked`` (B,L) bool: history slots whose input is replaced by [MASK]."""
        x = item_embedding(batch["item_tokens"])
        if masked is not None:
            x = torch.where(masked[..., None], self.mask_emb, x)
        pad = batch["pad_mask"]
        B = x.size(0)
        if self.pos_mode == "txn":
            pos = torch.cat([batch["txn_pos"], batch["txn_pos"][:, -1:] + 1], 1)
        else:
            pos = torch.cat([batch["item_pos"], (~pad).sum(1, keepdim=True) + 1], 1)
        ext = {"x": torch.cat([x, self.mask_emb.expand(B, 1, -1)], 1),
               "pos": pos,
               "pad_mask": torch.cat([pad, pad.new_zeros(B, 1)], 1)}
        if self.len_split:
            return by_length_split(self._encode, ext)
        return self._encode(ext)

    def _encode(self, ext: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.dropout(self.input_norm(ext["x"] + self.pos_emb(ext["pos"])))
        return self.transformer(x, src_key_padding_mask=ext["pad_mask"])

    def forward(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.hidden_states(item_embedding, batch)[:, -1, :]


class BERT4RecModel(RecModel):
    def __init__(self, num_items: int, dim: int, encoder: nn.Module, mask_prob: float = 0.4,
                 final_weight: float = 1.0):
        super().__init__(num_items, dim, encoder)
        self.mask_prob = mask_prob
        self.final_weight = final_weight

    def draw_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B,L) bool: each real history slot with prob ``mask_prob``. Drawn here, not in
        the collator -- cached collate rows must not carry a frozen mask."""
        pad = batch["pad_mask"]
        return (torch.rand(pad.shape, device=pad.device) < self.mask_prob) & ~pad

    def train_loss(self, batch: dict[str, torch.Tensor],
                   masked: torch.Tensor | None = None) -> torch.Tensor:
        """CE of the hidden item at every [MASK] slot: the drawn history slots and the
        appended slot (label = the held-out target, weighted by ``final_weight``)."""
        if masked is None:
            masked = self.draw_mask(batch)
        h = self.encoder.hidden_states(self.item_embedding, batch, masked)   # (B,L+1,D)
        where = torch.cat([masked, torch.ones_like(masked[:, :1])], 1)
        labels = torch.cat([batch["item_tokens"] - 1, batch["target"][:, None]], 1)
        ce = F.cross_entropy(h[where] @ self.item_embedding.weight[1:].t(), labels[where],
                             reduction="none")
        w = torch.ones(where.shape, device=ce.device)
        w[:, -1] = self.final_weight
        w = w[where]
        return (w * ce).sum() / w.sum()
