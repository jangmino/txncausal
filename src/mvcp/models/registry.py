"""Model registry: name -> (model, collator).

The names are the ones the queue files and the result files use; README.md maps them
to the names of the paper.
"""
from __future__ import annotations

from ..data.dataset import SequenceCollator
from .bert4rec import BERT4RecEncoder, BERT4RecModel
from .gru4rec import GRU4RecEncoder
from .sasrec import SASRecCollator, SASRecEncoder, SASRecModel
from .txn_causal import TxnCausalCollator, TxnCausalEncoder, TxnCausalModel
from .txn_hier import TxnHierEncoder

# name -> spec. 'kind' selects the encoder family.
RUNGS: dict[str, dict] = {
    "sasrec":                     {"kind": "sasrec"},       # baseline (Kang & McAuley 2018)
    "sasrec_final2":              {"kind": "sasrec", "final_weight": 2.0},  # symmetric control for H4b
    "sasrec_final2_blendemb":     {"kind": "sasrec", "final_weight": 2.0,
                                   "readout": "blend_emb"},  # symmetric control for H7a
    "txf_txn_causal":             {"kind": "txn_causal"},   # TxnCausal: block-causal, next-txn targets
    "txf_txn_causal_blend":       {"kind": "txn_causal", "readout": "blend"},  # H1 global-pref channel
    "txf_txn_causal_relb":        {"kind": "txn_causal", "rel_bias": True},    # H2a txn-distance bias
    "txf_txn_causal_relb_noabs":  {"kind": "txn_causal", "rel_bias": True, "abs_pos": False},  # H2b
    "txf_txn_causal_cloze0.1":    {"kind": "txn_causal", "cloze": 0.1},        # H3 intra-order cloze
    "txf_txn_causal_cloze0.3":    {"kind": "txn_causal", "cloze": 0.3},
    "txf_txn_causal_cloze1":      {"kind": "txn_causal", "cloze": 1.0},
    "txf_txn_causal_bnorm":       {"kind": "txn_causal", "basket_norm": True},  # H4a
    "txf_txn_causal_final2":      {"kind": "txn_causal", "final_weight": 2.0},  # H4b
    "txf_txn_causal_final4":      {"kind": "txn_causal", "final_weight": 4.0},
    "txf_txn_causal_k2":          {"kind": "txn_causal", "k2_weight": 0.5},     # H4c
    "txf_txn_hier":               {"kind": "txn_hier"},  # H6 order-set encoder -> causal over orders
    # round 2 (champion = final2): combinations and retries on top of it
    "txf_txn_causal_final2_bnorm":    {"kind": "txn_causal", "final_weight": 2.0, "basket_norm": True},
    "txf_txn_causal_final2_cloze1":   {"kind": "txn_causal", "final_weight": 2.0, "cloze": 1.0},
    "txf_txn_causal_final2_blendemb": {"kind": "txn_causal", "final_weight": 2.0,
                                       "readout": "blend_emb"},   # H1 retry (H7a w/ --struct-lr-mult)
    # structural-effect round (champion = final2)
    "txf_txn_causal_final2_relb":   {"kind": "txn_causal", "final_weight": 2.0, "rel_bias": True},  # H7b
    "txf_txn_causal_final2_osize":  {"kind": "txn_causal", "final_weight": 2.0, "order_size": True},  # H8
    "txf_txn_causal_final2_local1": {"kind": "txn_causal", "final_weight": 2.0, "local_first": True},  # H9
    "txf_txn_causal_final2_blendemb_relb": {"kind": "txn_causal", "final_weight": 2.0,
                                            "readout": "blend_emb", "rel_bias": True},  # C'+H7b
    "txf_txn_causal_final2_blendemb_tok1": {"kind": "txn_causal", "final_weight": 2.0,
                                            "readout": "blend_emb", "token_weight": 1.0},  # C'+H10
    # more sequential baselines on the shared recipe
    "gru4rec":                    {"kind": "gru4rec"},      # Hidasi et al. 2016: SASRec w/ a GRU encoder
    "gru4rec_final2":             {"kind": "gru4rec", "final_weight": 2.0},
    "bert4rec":                   {"kind": "bert4rec"},     # Sun et al. 2019: cloze + [MASK] readout
    "bert4rec_final2":            {"kind": "bert4rec", "final_weight": 2.0},
    # position variants of the decomposition: position (item / transaction) x
    # mask-and-targets (item-causal / block-causal) -- the two cells between SASRec and
    # TxnCausal, and BERT4Rec with transaction-level positions
    "sasrec_txnpos":              {"kind": "sasrec", "pos": "txn"},
    "sasrec_txnpos_final2":       {"kind": "sasrec", "pos": "txn", "final_weight": 2.0},
    "txf_txn_causal_itempos":         {"kind": "txn_causal", "pos": "item"},
    "txf_txn_causal_itempos_final2":  {"kind": "txn_causal", "pos": "item", "final_weight": 2.0},
    "bert4rec_txnpos":            {"kind": "bert4rec", "pos": "txn"},
    "bert4rec_txnpos_final2":     {"kind": "bert4rec", "pos": "txn", "final_weight": 2.0},
}
# TxnCausalModel flags (the rest of a txn_causal spec configures the encoder)
_TXN_MODEL_KEYS = ("cloze", "basket_norm", "final_weight", "k2_weight", "token_weight")


def build_model(name: str, num_items: int, dim: int = 64, max_seq_len: int = 64,
                max_num_txn: int = 52, nhead: int = 4, num_layers: int = 2,
                dropout: float = 0.1, compact_loss: bool = False, len_split: bool = False,
                mask_prob: float = 0.4):
    if name not in RUNGS:
        raise KeyError(f"unknown model '{name}'. options: {list(RUNGS)}")
    spec = RUNGS[name]
    kind = spec["kind"]
    if kind == "sasrec":
        enc = SASRecEncoder(dim, max_seq_len, max_num_txn, nhead=nhead,
                            num_layers=num_layers, dropout=dropout,
                            readout=spec.get("readout", "last"), len_split=len_split,
                            pos=spec.get("pos", "item"))
        collator = SASRecCollator(max_seq_len, max_num_txn,
                                  intra_labels=spec.get("intra_labels", True))
    elif kind == "gru4rec":
        enc = GRU4RecEncoder(dim, dropout=dropout)
        collator = SASRecCollator(max_seq_len, max_num_txn)
    elif kind == "bert4rec":
        enc = BERT4RecEncoder(dim, max_seq_len, max_num_txn, nhead=nhead,
                              num_layers=num_layers, dropout=dropout, len_split=len_split,
                              pos=spec.get("pos", "item"))
        collator = SequenceCollator(max_seq_len, max_num_txn)
    elif kind == "txn_causal":
        enc = TxnCausalEncoder(dim, max_seq_len, max_num_txn, nhead=nhead,
                               num_layers=num_layers, dropout=dropout,
                               readout=spec.get("readout", "mean"),
                               rel_bias=spec.get("rel_bias", False),
                               abs_pos=spec.get("abs_pos", True),
                               order_size=spec.get("order_size", False),
                               local_first=spec.get("local_first", False),
                               len_split=len_split, pos=spec.get("pos", "txn"))
        collator = TxnCausalCollator(max_seq_len, max_num_txn,
                                     k2=spec.get("k2_weight", 0.0) > 0)
    elif kind == "txn_hier":
        enc = TxnHierEncoder(dim, max_seq_len, max_num_txn, nhead=nhead,
                             num_layers=num_layers, dropout=dropout)
        collator = TxnCausalCollator(max_seq_len, max_num_txn)
    else:
        raise ValueError(kind)
    if kind in ("txn_causal", "txn_hier"):
        return TxnCausalModel(num_items, dim, enc,
                              **{k: spec[k] for k in _TXN_MODEL_KEYS if k in spec}), collator
    if kind in ("sasrec", "gru4rec"):
        return SASRecModel(num_items, dim, enc, final_weight=spec.get("final_weight", 1.0),
                           compact_loss=compact_loss), collator
    if kind == "bert4rec":
        return BERT4RecModel(num_items, dim, enc, mask_prob=mask_prob,
                             final_weight=spec.get("final_weight", 1.0)), collator
    raise ValueError(kind)
