# TxnCausal

Code and data for the paper

> Jiyeon Kim and Jangmin Oh, "A Transaction-Causal Transformer for Next-Item
> Recommendation in E-Commerce with Co-Purchases," submitted to *IEEE Access*, 2026.

TxnCausal is a transformer for next-item recommendation on purchase histories in which
several items are bought in one transaction. The items of a transaction share one
position, attention is block-causal across transactions, and every transaction boundary
predicts the items of the next transaction. The repository contains the model, the three
sequential baselines and the text-based baseline of the paper, the evaluation harness with
its three significance gates, the Katcher's dataset, and one queue file for every run
behind a number in the paper.

## Contents

```
src/mvcp/
  models/        txn_causal.py (TxnCausal), sasrec.py, gru4rec.py, bert4rec.py,
                 transformer.py (shared encoder), registry.py (model names)
  data/          vocab.py, dataset.py, split.py, negatives.py   Katcher's
                 instacart_all.py, rees46.py                    builders for the public datasets
                 instacart.py                                   loader of the built packages
  train/loop.py  one training loop for every model
  eval/          harness.py (ranking protocols), strata.py
  compare.py     paired comparison with the three gates (mvcp-compare)
  baselines/     concurrentbert.py (ConcurrentBERT-NSP)
  cli.py         mvcp-prep, mvcp-stats, mvcp-train
configs/
  train/default.yaml      default hyperparameters
  data/*.yaml             per-dataset settings
  queue/*.txt             every run: <model> <seed> [options], one per line
scripts/
  run_queue.sh            runs a queue file with N concurrent jobs
  report_*.py             tables from the result files
  *_expand_queue.py       the expansion rounds of the tuning procedure
  instacart_baselines.py  the no-training ranking rule
tests/                    unit tests, among them those named in Appendix C of the paper
data/                     the Katcher's dataset; see "Data" for Instacart and REES46
```

## Setup

Python 3.10 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # creates .venv from uv.lock (PyTorch 2.8.0)
uv run pytest -q           # 164 passed, 13 skipped
uv run mvcp-prep           # checks the split and the released negatives -> PREFLIGHT.md
```

The skipped tests need the two public datasets (11 tests; they run once the packages of
"Data" are built) or a GPU (2 tests). On Linux, `uv sync` installs the CUDA 12.8 build of
PyTorch; all runs of the paper were made with it on one NVIDIA RTX 4090. On macOS the CPU
build is installed, which is enough for the tests and for the quick check below.

A quick check on a CPU (about two minutes on a laptop):

```bash
uv run mvcp-train --model txf_txn_causal_final2 --seed 0 --dim 128 --dropout 0.3 \
    --lr 0.00025 --eval-split val --epochs 3 --device cpu --out results/quick
```

It trains TxnCausal with final-target weighting for three epochs and prints the
validation score, about 0.19 in full-catalog NDCG@10 (the tuned model reaches 0.207 after
early stopping).

## Data

### Katcher's (included)

`data/train_data.csv`, `data/val_data.csv`, and `data/test_data.csv` are the purchase log
of Katcher's, a Korean e-commerce platform, from April to July 2023: 16,787 users and
1,608 items, split by user into 10,071 / 3,358 / 3,358. The files are the split of
Park and Oh (2024), whose raw log is public in the repository of that study
([msparkDev/ConcurrentBERT-Recommender](https://github.com/msparkDev/ConcurrentBERT-Recommender));
`data/negatives/published.json` holds the 49 negatives per test user released with it.
`data/README.md` describes the columns and the terms of use.

An item is a `product_id`, and a transaction is the set of distinct products that share an
`order_id`. For each user, one item bought at the last purchase time is the held-out
target, and everything before that time is the input history.

### Instacart (download)

The Instacart data are not redistributed here. The six CSV files are public; the copy we
used is a Kaggle mirror that downloads without an account.

```bash
mkdir -p data/instacart_raw
curl -L -o data/instacart_raw/instacart.zip \
  https://www.kaggle.com/api/v1/datasets/download/yasserh/instacart-online-grocery-basket-analysis-dataset
unzip data/instacart_raw/instacart.zip -d data/instacart_raw
uv run python -m mvcp.data.instacart_all      # -> data/instacart_all/ (about 40 seconds)
```

| File | sha256, first 16 hex digits |
|---|---|
| `orders.csv` | `92a60ced67ac33bd` |
| `order_products__prior.csv` | `27b59dc9cf0d485f` |
| `order_products__train.csv` | `d27c4034883051c8` |
| `products.csv` | `abc61fdd748003d2` |
| `aisles.csv` / `departments.csv` | `2c5f4677761907d3` / `e6829cea3da95ec2` |

The builder draws the sample of the paper (16,596 users, all 21 departments, 23,653
items; 9,958 / 3,318 / 3,320) and writes `manifest.json` with the checksums
`sha_split = 39f61508b03318be` and `sha_target = fa7e5d62acc3287f`. `data/instacart/split.json`
lists the users of an earlier two-department sample, which the builder excludes. The
builder also writes `data/instacart_pdnew/`, a two-department subset of the same users
that the paper does not use.

### REES46 (download)

Used for one sentence of Section VIII. Public on Kaggle
(`mkechinov/ecommerce-purchase-history-from-electronics-store`, data provided by the
REES46 Marketing Platform).

```bash
mkdir -p data/rees46_raw
curl -L -o data/rees46_raw/kz.zip \
  https://www.kaggle.com/api/v1/datasets/download/mkechinov/ecommerce-purchase-history-from-electronics-store
unzip data/rees46_raw/kz.zip -d data/rees46_raw
uv run python -m mvcp.data.rees46             # -> data/rees46/
```

`kz.zip` has the checksum `88dbf392f4663d1a` and `kz.csv` `5ac5727e68300848`. The manifest
of `data/rees46/` gives `sha_split = f7c3e7e7ff40cbf9` and `sha_target = 9180393e7b58d8e0`
(50,170 users, 7,563 items). The builder also writes `data/rees46_logged/`, which the
paper does not use.

## Model names

The queue files and the result files use the names of the registry
(`src/mvcp/models/registry.py`). A result name is the model name followed by its options,
for example `txf_txn_causal_final2_lr0.00025_d128_do0.3`.

| Paper | Registry name | Katcher's (lr, d, dropout) | Instacart |
|---|---|---|---|
| SASRec-T | `sasrec` | 2.5e-4, 128, 0.5 | 1e-3, 512, 0.3 |
| SASRec-T+FW | `sasrec_final2` | same | same |
| TxnCausal-T | `txf_txn_causal` | 2.5e-4, 128, 0.3 | 2e-3, 512, 0.1 |
| TxnCausal-T+FW (proposed) | `txf_txn_causal_final2` | same | same |
| GRU4Rec-T+FW | `gru4rec_final2` | 5e-4, 256, 0.3 | 2e-3, 1024, 0.5 |
| BERT4Rec-T | `bert4rec` | 1e-3, 256, 0.5, masking 0.4 | 2e-3, 256, 0.3, masking 0.2 |
| SASRec variant | `sasrec_txnpos_final2` | 2.5e-4, 128, 0.3 | |
| TxnCausal variant | `txf_txn_causal_itempos_final2` | 2.5e-4, 128, 0.5 | |
| BERT4Rec variant | `bert4rec_txnpos` | 5e-4, 128, 0.3, masking 0.4 | |

The other entries of the registry are the variants of the screening on validation data
(Appendix B and Section VIII of the paper).

## Reproducing the results

Every run is a line of a queue file. `scripts/run_queue.sh -P N -o OUT QUEUE` runs the
file with N concurrent jobs and skips finished jobs, so the same command resumes. A run
writes `OUT/<name>_seed<seed>.json` (scores by protocol and stratum, training history)
and `OUT/per_user/<name>_seed<seed>.json.gz` (the metric and the stratum flags of every
user); all paired statistics are computed from the per-user records.

The command blocks below share shell variables (`R`, `T`, `S`, `G`, `B`, ...); run them in
one shell, in order.

Runs are seeded and regenerate bit for bit on the same hardware and software. The
commands below use one directory per dataset. Times are for one RTX 4090.

### Katcher's: TxnCausal and SASRec (Tables 3, 4 S1-S5, and 6)

```bash
R=results/katchers
scripts/run_queue.sh -P 12 -o $R configs/queue/beat_c2.txt     # 60 test runs, about 27 minutes
scripts/run_queue.sh -P 12 -o $R configs/queue/beat_post.txt   # symmetric control, training-set size
T=txf_txn_causal_final2_lr0.00025_d128_do0.3
uv run mvcp-compare --dir $R --seeds 100-119 --a sasrec_final2_lr0.00025_d128_do0.5 --b $T   # S1: +0.0050
uv run mvcp-compare --dir $R --seeds 100-119 --a sasrec_lr0.00025_d128_do0.5 --b $T          # +0.0095
uv run mvcp-compare --dir $R --seeds 100-119 --a sasrec_lr0.00025_d128_do0.5 \
    --b txf_txn_causal_lr0.00025_d128_do0.3                                                  # +0.0046
uv run mvcp-compare --dir $R --seeds 100-109 --a sasrec_lr0.00025_d128_do0.5_frac0.1 \
    --b ${T}_frac0.1                                              # 10% of the training users (Section VIII)
```

`mvcp-compare` prints, for the masked (`full`) and the unmasked (`full_nomask`) ranking
and for the strata overall, novel, repeat, and copurchase, the two means, the difference
B - A, the user-level bootstrap interval, the Wilcoxon p-value, the seed-by-user two-level
interval, and the number of seeds with a positive difference, followed by the verdict of
the three gates on the overall score.

### Katcher's: GRU4Rec and BERT4Rec (Tables 3, 4 G and B, and 5)

```bash
scripts/run_queue.sh -P 12 -o $R configs/queue/bl_k_c1.txt     # 80 test runs, about 65 minutes
G=gru4rec_final2_lr0.0005_d256_do0.3; B=bert4rec_lr0.001_d256_do0.5_mp0.4
S=sasrec_final2_lr0.00025_d128_do0.5
uv run python scripts/report_ic.py --dir $R --seeds 100-119 --tag baselines --primary full \
    --strata overall,novel,repeat,copurchase --detail --pairs "G=$G:$T,B=$B:$T,SG=$G:$S,SB=$B:$S"
uv run python scripts/report_abl_cells.py --dir $R --seeds 100-119 \
    --pairs "GRU4Rec=$G:$T,BERT4Rec=$B:$T,SASRec=$S:$T"         # the four cells of Table 5
uv run python scripts/instacart_baselines.py --data katchers --primary full_catalog \
    --strata copurchase --out $R/baselines.json                 # the no-training rule
```

### Katcher's: decomposition (Tables 4 P1-P3 and 7)

```bash
scripts/run_queue.sh -P 12 -o $R configs/queue/abl_k_c1.txt    # 120 test runs, about 59 minutes
SO=sasrec_txnpos_final2_lr0.00025_d128_do0.3; TI=txf_txn_causal_itempos_final2_lr0.00025_d128_do0.5
BO=bert4rec_txnpos_lr0.0005_d128_do0.3_mp0.4
uv run python scripts/report_ic.py --dir $R --seeds 100-119 --tag decomp --primary full \
    --strata overall,novel,repeat,copurchase --detail \
    --pairs "P1=$S:$SO,P2=$SO:$T,P3=$BO:$T,pos_block=$TI:$T,mask_item=$S:$TI,bert=$B:$BO"
uv run python scripts/report_abl_interaction.py --dir $R --seeds 100-119 --ii $S --io $SO --bi $TI --bo $T
uv run python scripts/report_abl_cells.py --dir $R --seeds 100-119 --pairs "P2=$SO:$T,P3=$BO:$T"
```

### Katcher's: ConcurrentBERT-NSP (Tables 3, 4 N1-N7, and 9)

Needs the extra dependencies and a GPU. These runs were made before seeding was fixed in
the training code and regenerate statistically, not bit for bit.

```bash
uv pip install -e ".[bert]"
C="uv run python -m mvcp.baselines.concurrentbert"
# the released code as is (0.6496 under the sampled 1:49 protocol)
$C prep  --workdir artifacts/bert_nsp
$C train --workdir artifacts/bert_nsp
$C eval  --workdir artifacts/bert_nsp --tag bert_nsp
$C eval  --workdir artifacts/bert_nsp --tag bert_nsp_swap --swap-target-rep   # target text from the source of the negatives
# the retrained version (0.5574 +- 0.0037 over three runs; full catalog 0.1589)
$C prep  --workdir artifacts/bert_nsp_sym --text repaired --cand canonical
for s in 0 1 2; do
  $C train --workdir artifacts/bert_nsp_sym --seed $s            # about 40 minutes each
  $C eval  --workdir artifacts/bert_nsp_sym --seed $s --tag bert_nsp_sym
done
$C eval  --workdir artifacts/bert_nsp_sym --seed 0 --tag bert_nsp_sym --full   # about 2.4 hours
# paired with the ID-based models
uv run python scripts/bl_bertnsp_paired.py --dir $R --models $T,$S,$G,$B
scripts/run_queue.sh -P 12 -o results/katchers_pub configs/queue/bl_k_pubdraw.txt
uv run python scripts/bl_bertnsp_paired.py --dir results/katchers_pub --protocol sampled \
    --bert bert_nsp_sym --bert-seeds 0-2 --models $T
```

### Instacart (Table 8)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=results/instacart_all
scripts/run_queue.sh -P 2 -o $I configs/queue/ia_c1.txt        # 40 test runs, about 5 hours
scripts/run_queue.sh -P 3 -o $I configs/queue/bl_ia_c1.txt     # GRU4Rec, BERT4Rec
scripts/run_queue.sh -P 3 -o $I configs/queue/bl_ia_c2.txt     # GRU4Rec with its width capped (footnote)
TA=txf_txn_causal_final2_lr0.002_d512_do0.1
uv run python scripts/report_ic.py --dir $I --seeds 100-119 --tag table8 --detail \
    --strata overall,r1_in_last,r1_before,novel,repeat \
    --pairs "S=sasrec_final2_lr0.001_d512_do0.3:$TA,G=gru4rec_final2_lr0.002_d1024_do0.5:$TA,B=bert4rec_lr0.002_d256_do0.3_mp0.2:$TA,Gcap=gru4rec_lr0.008_d512_do0.3:$TA"
uv run python scripts/instacart_baselines.py --data instacart_all --strata r1_in_last,r1_before
```

The primary metric on Instacart is the unmasked ranking, the default of `report_ic.py`.
The strata `r1_in_last`, `r1_before`, and `novel` are the columns "previous", "earlier",
and "first-time" of Table 8.

### REES46 (Section VIII)

```bash
scripts/run_queue.sh -P 12 -o results/rees46 configs/queue/rz_c1.txt
uv run python scripts/report_ic.py --dir results/rees46 --seeds 100-119 --tag c1 --primary full \
    --pairs "P1=sasrec_final2_lr0.00025_d64_do0.7:txf_txn_causal_final2_lr0.00025_d64_do0.7"
```

### Tuning (Appendix B, Table 10)

Tuning used the validation split only (`--eval-split val`). The queue files below hold the
runs in the order in which they were made; the expansion rounds were generated by the
`*_expand_queue.py` scripts from the results of the round before.

| Models | Queue files (`configs/queue/`) | Table |
|---|---|---|
| Katcher's: SASRec, TxnCausal | `beat_0d`, `beat_0d_ext`, `beat_r1`, `beat_r1_sasfill` | `scripts/report_beat_grid.py --dir $R` |
| Katcher's: screening of variants | `beat_r1`, `beat_r2`, `beat_r2a`, `beat_r2b`, `beat_r3`, `beat_r4` | `scripts/screen_round.py` |
| Katcher's: GRU4Rec, BERT4Rec | `bl_k_0c`, `bl_k_mp`, `bl_k_grid`, `bl_k_r1`-`bl_k_r3`, `bl_k_mp2`, `bl_k_val10_gru`, `bl_k_val10_bert` | `scripts/report_ic_grid.py --dir $R --models gru4rec,bert4rec --protocol full_catalog` |
| Katcher's: position variants | `abl_k_0c`, `abl_k_mp`, `abl_k_grid_2x2`, `abl_k_grid_bert`, `abl_k_r1_2x2`, `abl_k_r1_bert`, `abl_k_probe_2x2`, `abl_k_mp2`, `abl_k_val10_2x2`, `abl_k_val10_bert` | `scripts/report_ic_grid.py --dir $R --models sasrec_txnpos_final2,txf_txn_causal_itempos_final2,bert4rec_txnpos --protocol full_catalog` |
| Instacart: SASRec, TxnCausal | `ia_0c`, `ia_p1_grid`, `ia_p1_r1`, `ia_p1_val10`; after the test runs `ia_p1_r2`, `ia_p1_r2_val10` | `scripts/report_ic_grid.py --dir $I --seeds 0-2` |
| Instacart: GRU4Rec, BERT4Rec | `bl_ia_0c`, `bl_ia_mp`, `bl_ia_grid_gru`, `bl_ia_grid_bert`, `bl_ia_r1_gru`-`bl_ia_r3_gru`, `bl_ia_r1_bert`, `bl_ia_mp2`, `bl_ia_probe_gru`, `bl_ia_val10_gru`, `bl_ia_val10_bert`; width-capped GRU4Rec `bl_ia_cap_r1_gru`, `bl_ia_cap_r2_gru`, `bl_ia_cap_val10_gru` | `scripts/report_ic_grid.py --dir $I --models gru4rec,bert4rec --seeds 0-2` |
| REES46: SASRec, TxnCausal | `rz_0c`, `rz_p1_grid`, `rz_p1_r1`-`rz_p1_r4`, `rz_p1_probe_do0.8`, `rz_p1_val10` | `scripts/report_ic_grid.py --dir results/rees46 --protocol full_catalog` |

The first line of each queue file says what it is. `beat_0c.txt` is the first
confirmatory experiment (TxnCausal and SASRec at default hyperparameters, +0.0018).

## Pre-registered experiments

The models, seeds (100-119), primary comparison, gates, and decision rules of each
experiment below were committed to the research log of the development repository before
its test runs started. The hash is that commit.

| Experiment | Commit | Result (primary metric, overall) |
|---|---|---|
| Katcher's: TxnCausal vs SASRec at default hyperparameters | `b13a390` | +0.0018, not significant |
| Katcher's: TxnCausal-T+FW vs SASRec-T | `963f6b7` | +0.0095, all three gates |
| Katcher's: symmetric control, TxnCausal-T+FW vs SASRec-T+FW; training-set size | `2d60ed6` | +0.0050, confirmed (S1) |
| Katcher's: GRU4Rec-T+FW, BERT4Rec-T | `6110121` | +0.0137, +0.0140, confirmed (G1, B1) |
| Katcher's: decomposition P1, P2, P3 | `90203c3` | +0.0002 n.s., +0.0047, +0.0184 confirmed |
| Instacart: SASRec-T+FW | `8aaa358` | +0.0169, confirmed |
| Instacart: GRU4Rec-T+FW, BERT4Rec-T | `76d2489` | +0.0107 trend, +0.0141 confirmed |
| Instacart: width-capped GRU4Rec | `8b9f5af` | +0.0066, trend |
| REES46: SASRec-T+FW | `331efa1` | -0.0035 (SASRec higher) |

The development repository holds the research log with these registrations, in Korean,
together with exploratory code that the paper does not report. It is available to editors
and reviewers on request.

## Tests named in the paper

| Test | File | Property |
|---|---|---|
| `test_block_causal` | `tests/test_txn_causal.py` | a transaction state never sees later transactions |
| `test_intra_txn_permutation_invariant` | `tests/test_txn_causal.py` | scores do not depend on the order within a transaction |
| `test_h4_i2_loss_permutation_invariant` | `tests/test_txn_causal.py` | neither does the FW-weighted loss |
| `test_h4_weights_match_definition` | `tests/test_txn_causal.py` | FW |
| `test_causal_future_item_does_not_leak` | `tests/test_sasrec.py` | SASRec's causal mask |
| `test_final2_weights_newest_position` | `tests/test_sasrec.py` | FW on SASRec |
| `test_gru_packed_path_equals_stepping_each_user_alone` | `tests/test_seq_baselines.py` | batched GRU equals a step-by-step recurrence |
| `test_bert_eval_scores_do_not_depend_on_the_target` | `tests/test_seq_baselines.py` | the held-out target cannot leak into BERT4Rec |
| `test_variant_differs_from_its_parent_in_the_position_table_only` | `tests/test_ablation.py` | position variants |
| `test_sasrec_txnpos_still_orders_the_items_of_an_order`, `test_txn_causal_itempos_loses_the_invariance` | `tests/test_ablation.py` | the invariance needs both the positions and the mask |
| `test_bert_txnpos_is_invariant_within_an_order_and_blind_to_the_target` | `tests/test_ablation.py` | BERT4Rec variant |
| `test_aspublished_returns_bare_header_when_fits`, `test_canonical_cand_map_option_free_and_uniform` | `tests/test_baselines.py` | prompts and candidate texts of ConcurrentBERT-NSP |

## Notes

- This repository was assembled from the development repository by moving files, not by
  rewriting them. Twelve Katcher's runs and seven Instacart and REES46 runs, made with the
  same commands in both repositories, give byte-identical result files.
- Some comments name sections of an internal research plan ("Phase 0-B", "S3"). That
  document is not part of this repository.
- The item order inside a transaction is the order of the rows of the log. TxnCausal does
  not depend on it; SASRec, GRU4Rec, and BERT4Rec do.

## License

The code is released under the MIT License (`LICENSE`). `data/README.md` gives the terms
for the data. `src/mvcp/baselines/_concurrentbert_text.py` is taken from the repository of
Park and Oh (2024).

## Citation

```bibtex
@article{kim2026txncausal,
  author  = {Kim, Jiyeon and Oh, Jangmin},
  title   = {A Transaction-Causal Transformer for Next-Item Recommendation in
             E-Commerce with Co-Purchases},
  journal = {IEEE Access},
  year    = {2026},
  note    = {submitted}
}
```

Contact: Jangmin Oh (jangmin.oh@sungshin.ac.kr)
