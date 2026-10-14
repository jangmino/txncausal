"""RecModel: owns the item-embedding table and a tied Cross-Entropy head (doc S5.3).

    logits = user_emb @ item_embedding.weight[1:].T          # weight[0] is PAD

The encoder (a set/transformer/txn-token module) receives the shared embedding table and
the batch and returns the user representation. Tying the head to the input table is the
learned-item-ID dot-product objective from the advisor's proposal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecModel(nn.Module):
    def __init__(self, num_items: int, dim: int, encoder: nn.Module):
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.encoder = encoder
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def user_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(self.item_embedding, batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user = self.user_embedding(batch)                       # (B, D)
        return user @ self.item_embedding.weight[1:].t()        # (B, num_items)

    def train_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(self(batch), batch["target"])
