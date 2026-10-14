"""Masked pooling primitives shared across rungs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """x:(B,L,D), valid:(B,L) bool (True=keep). Rows with no valid token -> zeros."""
    m = valid.unsqueeze(-1).float()
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


class AttentionPooling(nn.Module):
    """Additive single-query attention pooling (the student's original design), masked.

    scores_i = w^T e_i ; alpha = softmax over valid tokens ; out = sum alpha_i e_i.
    Permutation-invariant over the pooled set -> reused for last-transaction pooling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)                 # (B,L)
        scores = scores.masked_fill(~valid, float("-inf"))
        # rows with no valid token would be all -inf -> softmax nan; guard them
        empty = (~valid).all(dim=1, keepdim=True)
        scores = scores.masked_fill(empty, 0.0)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)       # (B,L,1)
        out = (weights * x).sum(dim=1)
        return out.masked_fill(empty, 0.0)
