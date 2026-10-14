"""GRU4Rec baseline (Hidasi et al., ICLR 2016) on the shared recipe.

SASRec with ONLY the sequence encoder swapped: item embedding -> LayerNorm -> dropout ->
one GRU layer (hidden = dim). No position table (the recurrence carries the order). The
collator (next-item label at every position), the loss code (``SASRecModel.train_loss``:
catalog softmax CE, compact loss, final-term weight), the tied item table and the
newest-position readout are SASRec's, so SASRec -> GRU4Rec isolates self-attention vs
recurrence.

Unlike the original: catalog softmax CE instead of BPR/TOP1 over sampled negatives, and
user-level batches instead of session-parallel mini-batches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU4RecEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)

    def hidden_states(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B,L,D); the state at a real position has read the real tokens up to it and
        nothing else. Inputs are left-padded, so a plain pass would run the recurrence
        through the PADs first: rows are rolled to the left edge and packed (PAD never
        enters the GRU), and the states are rolled back. PAD positions hold zeros."""
        x = self.dropout(self.input_norm(item_embedding(batch["item_tokens"])))
        L = x.size(1)
        n = (~batch["pad_mask"]).sum(1)                                     # (B,)
        ar = torch.arange(L, device=x.device)[None, :]
        left = x.gather(1, ((ar - n[:, None]) % L)[..., None].expand_as(x))
        # an empty history is packed as one PAD step; its state lands on a PAD position
        packed = pack_padded_sequence(left, n.clamp_min(1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=L)
        return out.gather(1, ((ar + n[:, None]) % L)[..., None].expand_as(out))

    def states(self, h: torch.Tensor, batch: dict[str, torch.Tensor],
               item_embedding) -> torch.Tensor:
        return h

    def forward(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.hidden_states(item_embedding, batch)[:, -1, :]
