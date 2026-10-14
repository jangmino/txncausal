"""TxnCausal: SASRec-level supervision without a total order inside a transaction.

SASRec gets dense supervision by predicting the next item at every position -- but a
third of those labels are "next item inside the same order", an order the data does
not have. TxnCausal uses transaction-level positions and a last-transaction mean
readout, with:

  * BLOCK-causal attention: a token sees its own transaction and earlier ones, never
    later ones (items inside a transaction still see each other symmetrically);
  * next-transaction targets at EVERY transaction boundary: the mean state of
    transaction k predicts each item of transaction k+1 (one CE term per item), and the
    last transaction predicts the held-out target.

The model is invariant to the order of the items within a transaction (Section IV-D of
the paper; tests/test_txn_causal.py).

The variants screened on validation data (Appendix B of the paper) are flags of the
same encoder; the model of the paper uses none of them except final_weight (FW):
  readout='blend' (H1): transaction k's state = g * mean(txn k tokens)
      + (1 - g) * attnpool(tokens of txns <= k), g = sigmoid(learned vector) -- a
      global-preference channel next to the last-order mean; both pools are
      permutation-invariant, and training and eval read the SAME per-transaction states.
      readout='blend_emb' (H1 retry): the pooled channel reads the raw item embeddings of
      txns <= k (L1's input) through its own LayerNorm, so its scale matches the states.
  rel_bias=True (H2): a learned per-head attention bias b[h, bucket(txn_q - txn_k)],
      buckets {0 (same order), 1, 2, 3-4, 5-8, 9+} (T5-style, shared across layers) --
      the co-purchase relation made explicit in attention; same-order pairs share bucket 0,
      so intra-order invariance holds. abs_pos=False drops the absolute txn position.
  order_size=True (H8): a learned embedding of the order's size (1/2/3/4+) is added to
      every token of that order -- identical within an order, so invariance holds.
  local_first=True (H9): the first layer attends only WITHIN each order; later layers
      are block-causal (an order-level co-purchase representation before mixing orders).
  token_weight=lambda (H10, model flag): every TOKEN of order k also predicts each item of
      order k+1 (partial-order-consistent per-position labels, cf. SASRec's per-position
      next-item labels); mean over (token, item) pairs, added with weight lambda.
  cloze=lambda (H3, model flag): a second training forward where one item of every
      multi-item order is replaced by a learned [MASK] vector; its hidden state (same-order
      peers + earlier orders, block-causal) predicts the item by CE. Symmetric co-purchase
      supervision; loss = next-txn + lambda * cloze. Training only -- eval is unchanged.
  objective weighting (H4, model flags; eval unchanged): basket_norm weighs each next-txn
      CE term by 1/|next basket| (every transition counts once); final_weight multiplies
      the held-out-target term; k2_weight adds CE of txn k+2's items from state k
      (collator k2=True emits those triplets).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import SequenceCollator
from .base import RecModel
from .pooling import masked_mean
from .transformer import TxnTransformerEncoder, by_length_split


class TxnCausalCollator(SequenceCollator):
    """SequenceCollator + next-transaction targets as flat triplets
    (``ntx_row``, ``ntx_k``, ``ntx_item``): the state of kept transaction k (0-based,
    = txn_pos - 1) predicts ``ntx_item``. ``k2=True`` adds ``ntx2_*`` triplets: state k
    predicts the items of transaction k+2 (the held-out target counts as the txn after
    the last)."""

    def __init__(self, max_seq_len: int = 64, max_num_txn: int = 52, k2: bool = False):
        super().__init__(max_seq_len, max_num_txn)
        self.k2 = k2

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        out = super().__call__(batch)
        L = self.max_seq_len
        rows, ks, items = [], [], []
        for b, u in enumerate(batch):
            its, txs = u.items[-L:], u.txn_ids[-L:]
            uniq = sorted(set(txs))
            for k, t in enumerate(uniq):
                nxt = ([it for it, tx in zip(its, txs) if tx == uniq[k + 1]]
                       if k + 1 < len(uniq) else [u.target])
                rows += [b] * len(nxt)
                ks += [min(k + 1, self.max_num_txn) - 1] * len(nxt)
                items += nxt
        out["ntx_row"] = torch.tensor(rows, dtype=torch.long)
        out["ntx_k"] = torch.tensor(ks, dtype=torch.long)
        out["ntx_item"] = torch.tensor(items, dtype=torch.long)
        if self.k2:
            rows, ks, items = [], [], []
            for b, u in enumerate(batch):
                its, txs = u.items[-L:], u.txn_ids[-L:]
                uniq = sorted(set(txs))
                for k in range(len(uniq) - 1):
                    nxt = ([it for it, tx in zip(its, txs) if tx == uniq[k + 2]]
                           if k + 2 < len(uniq) else [u.target])
                    rows += [b] * len(nxt)
                    ks += [min(k + 1, self.max_num_txn) - 1] * len(nxt)
                    items += nxt
            out["ntx2_row"] = torch.tensor(rows, dtype=torch.long)
            out["ntx2_k"] = torch.tensor(ks, dtype=torch.long)
            out["ntx2_item"] = torch.tensor(items, dtype=torch.long)
        return out


class TxnCausalEncoder(TxnTransformerEncoder):
    def __init__(self, dim: int, max_seq_len: int, max_num_txn: int, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, readout: str = "mean",
                 rel_bias: bool = False, abs_pos: bool = True, order_size: bool = False,
                 local_first: bool = False, len_split: bool = False, pos: str = "txn"):
        # pos='item' (component ablation, the research plan): item-level
        # positions under the block-causal mask and next-transaction targets; the
        # intra-order invariance is lost by design
        assert pos in ("txn", "item")
        super().__init__(dim, max_seq_len, max_num_txn, pos_mode=pos,
                         pool_mode="last_txn_mean", nhead=nhead, num_layers=num_layers,
                         dropout=dropout)
        assert readout in ("mean", "blend", "blend_emb")
        self.nhead = nhead
        self.len_split = len_split        # encode long and short rows separately
        self.max_num_txn = max_num_txn
        self.readout = readout
        self.abs_pos = abs_pos        # pos_emb is still built (same init draws) when False
        if rel_bias:
            edges = [0, 1, 2, 3, 5, 9]                 # bucket lower bounds (distance)
            d = torch.arange(max_num_txn + 1)
            self.register_buffer("rel_bucket", sum((d >= e).long() for e in edges) - 1,
                                 persistent=False)
            self.rel_bias = nn.Parameter(torch.zeros(nhead, len(edges)))   # zero init
        else:
            self.rel_bias = None
        # order size 1/2/3/4+; zero init draws no RNG (shared init stays the champion's)
        self.size_table = nn.Parameter(torch.zeros(4, dim)) if order_size else None
        assert not (local_first and rel_bias), "local_first + rel_bias not implemented"
        self.local_first = local_first
        if readout in ("blend", "blend_emb"):
            # zero init = uniform attention over txns <= k and g = 0.5; torch.zeros draws
            # no RNG, so every shared parameter keeps txf_txn_causal's init at a seed
            self.blend_query = nn.Parameter(torch.zeros(dim))
            self.blend_gate = nn.Parameter(torch.zeros(dim))
        if readout == "blend_emb":
            self.blend_norm = nn.LayerNorm(dim)      # ones/zeros init: no RNG either

    def hidden_states(self, item_embedding, batch: dict[str, torch.Tensor],
                      replace: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """(B,L,D); a token attends to real tokens of its own and earlier transactions.
        ``replace=(where (B,L) bool, vec (D,))`` swaps those input embeddings for ``vec``."""
        if self.len_split and replace is None:
            return by_length_split(lambda sub: self._hidden_states(item_embedding, sub),
                                   batch)
        return self._hidden_states(item_embedding, batch, replace)

    def _hidden_states(self, item_embedding, batch: dict[str, torch.Tensor],
                       replace: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        x = item_embedding(batch["item_tokens"])
        if replace is not None:
            x = torch.where(replace[0][..., None], replace[1], x)
        if self.size_table is not None:
            real = ~batch["pad_mask"]
            same = (batch["txn_pos"][:, :, None] == batch["txn_pos"][:, None, :]) & real[:, None, :]
            x = x + self.size_table[same.sum(-1).clamp(1, 4) - 1] * real[..., None]
        if self.abs_pos:
            x = x + self.pos_emb(batch["item_pos" if self.pos_mode == "item" else "txn_pos"])
        x = self.dropout(self.input_norm(x))
        tp = batch["txn_pos"]
        eye = torch.eye(x.size(1), dtype=torch.bool, device=x.device)
        blocked = (tp[:, None, :] > tp[:, :, None]) | batch["pad_mask"][:, None, :]
        blocked = blocked & ~eye      # keeps all-PAD query rows finite (never read)
        mask = blocked.repeat_interleave(self.nhead, dim=0)
        if self.rel_bias is not None:
            dist = (tp[:, :, None] - tp[:, None, :]).clamp(0, self.max_num_txn)  # q - k
            bias = self.rel_bias[:, self.rel_bucket[dist]]                        # (H,B,L,L)
            # MHA's 3-D mask rows are batch-major (b * nhead + h), like repeat_interleave
            mask = bias.transpose(0, 1).flatten(0, 1).masked_fill(mask, float("-inf"))
            # the no-grad eval fast path (_transformer_encoder_layer_fwd) reads a float mask
            # as boolean (NaN outputs) -> run the regular path, as in training
            prev = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                return self.transformer(x, mask=mask)
            finally:
                torch.backends.mha.set_fastpath_enabled(prev)
        if self.local_first:          # layer 1: own order only; later layers: block-causal
            local = ((tp[:, None, :] != tp[:, :, None]) | batch["pad_mask"][:, None, :]) & ~eye
            masks = [local.repeat_interleave(self.nhead, dim=0)] + \
                [mask] * (len(self.transformer.layers) - 1)
            for layer, m in zip(self.transformer.layers, masks):
                x = layer(x, src_mask=m)
            return x
        return self.transformer(x, mask=mask)

    def txn_states(self, h: torch.Tensor, batch: dict[str, torch.Tensor],
                   item_embedding=None) -> torch.Tensor:
        """(B,K,D): the state of each transaction k (0-based = txn_pos - 1)."""
        ks = torch.arange(1, self.max_num_txn + 1, device=h.device)
        member = ((batch["txn_pos"][:, None, :] == ks[None, :, None])
                  & ~batch["pad_mask"][:, None, :]).float()                     # (B,K,L)
        states = (member @ h) / member.sum(-1, keepdim=True).clamp_min(1.0)    # (B,K,D)
        if self.readout == "mean":
            return states
        upto = ((batch["txn_pos"][:, None, :] <= ks[None, :, None])
                & ~batch["pad_mask"][:, None, :])                                # (B,K,L)
        src = (h if self.readout == "blend"
               else self.blend_norm(item_embedding(batch["item_tokens"])))      # (B,L,D)
        score = (src @ self.blend_query)[:, None, :].masked_fill(~upto, float("-inf"))
        score = score.masked_fill(~upto.any(-1, keepdim=True), 0.0)  # empty k: finite
        pooled = (torch.softmax(score, -1) * upto) @ src                        # (B,K,D)
        g = torch.sigmoid(self.blend_gate)
        return g * states + (1 - g) * pooled

    def forward(self, item_embedding, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.hidden_states(item_embedding, batch)
        if self.readout == "mean":
            return masked_mean(h, batch["last_txn"])
        # newest token's transaction; users without history keep the zero vector
        last = (batch["txn_pos"][:, -1] - 1).clamp_min(0)
        out = self.txn_states(h, batch, item_embedding)[
            torch.arange(h.size(0), device=h.device), last]
        return out * (~batch["pad_mask"]).any(1, keepdim=True)


class TxnCausalModel(RecModel):
    def __init__(self, num_items: int, dim: int, encoder: nn.Module, cloze: float = 0.0,
                 basket_norm: bool = False, final_weight: float = 1.0,
                 k2_weight: float = 0.0, token_weight: float = 0.0):
        super().__init__(num_items, dim, encoder)
        self.token_weight = token_weight
        self.cloze = cloze
        self.basket_norm = basket_norm
        self.final_weight = final_weight
        self.k2_weight = k2_weight
        if cloze:
            self.mask_emb = nn.Parameter(torch.zeros(dim))   # zero init: draws no RNG

    def train_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """CE of every next-transaction item from its predecessor transaction's state."""
        h = self.encoder.hidden_states(self.item_embedding, batch)             # (B,L,D)
        states = self.encoder.txn_states(h, batch, self.item_embedding)         # (B,K,D)
        rows, ks = batch["ntx_row"], batch["ntx_k"]
        s = states[rows, ks]                                                     # (P,D)
        if not self.basket_norm and self.final_weight == 1.0:
            loss = F.cross_entropy(s @ self.item_embedding.weight[1:].t(), batch["ntx_item"])
        else:
            ce = F.cross_entropy(s @ self.item_embedding.weight[1:].t(), batch["ntx_item"],
                                 reduction="none")
            w = torch.ones_like(ce)
            if self.basket_norm:
                key = rows * states.size(1) + ks
                w = w / torch.bincount(key)[key].float()
            if self.final_weight != 1.0:   # the last kept txn's term = held-out target
                last = (batch["txn_pos"][:, -1] - 1).clamp_min(0)
                w = torch.where(ks == last[rows], w * self.final_weight, w)
            loss = (w * ce).sum() / w.sum()
        if self.k2_weight and batch["ntx2_row"].numel():
            s2 = states[batch["ntx2_row"], batch["ntx2_k"]]
            loss = loss + self.k2_weight * F.cross_entropy(
                s2 @ self.item_embedding.weight[1:].t(), batch["ntx2_item"])
        if self.token_weight:
            loss = loss + self.token_weight * self.token_loss(h, batch)
        if self.cloze:
            loss = loss + self.cloze * self.cloze_loss(batch)
        return loss

    def token_loss(self, h: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """H10: CE of each next-order item from EVERY token state of the preceding order."""
        logp = F.log_softmax(h @ self.item_embedding.weight[1:].t(), dim=-1)    # (B,L,N)
        rows, ks = batch["ntx_row"], batch["ntx_k"]
        pair = ((batch["txn_pos"][rows] == ks[:, None] + 1)
                & ~batch["pad_mask"][rows])                                     # (P,L)
        nll = -logp[rows].gather(-1, batch["ntx_item"][:, None, None].expand(
            -1, logp.size(1), 1)).squeeze(-1)                                  # (P,L)
        return (nll * pair).sum() / pair.sum().clamp_min(1)

    def cloze_positions(self, batch: dict[str, torch.Tensor]):
        """(rows, positions): one uniformly drawn token of every multi-item order."""
        tp = batch["txn_pos"]
        ks = torch.arange(1, self.encoder.max_num_txn + 1, device=tp.device)
        member = (tp[:, None, :] == ks[None, :, None]) & ~batch["pad_mask"][:, None, :]
        rows, k = (member.sum(-1) >= 2).nonzero(as_tuple=True)
        draw = torch.rand(tp.shape, device=tp.device)[:, None, :].masked_fill(~member, -1.0)
        return rows, draw.argmax(-1)[rows, k]

    def cloze_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rows, pos = self.cloze_positions(batch)
        if rows.numel() == 0:
            return self.mask_emb.sum() * 0.0
        where = torch.zeros_like(batch["pad_mask"])
        where[rows, pos] = True
        h = self.encoder.hidden_states(self.item_embedding, batch,
                                       replace=(where, self.mask_emb))
        logits = h[rows, pos] @ self.item_embedding.weight[1:].t()
        return F.cross_entropy(logits, batch["item_tokens"][rows, pos] - 1)
