# Data

## Katcher's

`train_data.csv`, `val_data.csv`, `test_data.csv`: the purchase log of Katcher's, a Korean
e-commerce platform, collected from April to July 2023 and first used by Park and Oh
(2024). One row is one purchased product option. The three files are the split of that
study: users are assigned 60/20/20 to training, validation, and test, and no user appears
in two files.

| | Users | Rows |
|---|---|---|
| `train_data.csv` | 10,071 | 64,676 |
| `val_data.csv` | 3,358 | 21,008 |
| `test_data.csv` | 3,358 | 21,317 |

The files are UTF-8 with a byte-order mark. Product and category names are in Korean.

### Columns

The ID-based models read four columns.

| Column | Meaning |
|---|---|
| `user_id` | user identifier |
| `order_id` | identifier of the order; the rows that share it form one transaction |
| `product_id` | product identifier; the item of the models (options of a product are not distinguished) |
| `initial_paid_at` | payment time in milliseconds; orders the transactions of a user, and the latest value marks the held-out purchase |

The text-based baseline (ConcurrentBERT-NSP) also reads `product_name`, `category_name`,
and `attribute_values`.

The other columns are those of the original log and are not read by the code: `code`,
`order_code`, `option_product_code`, `option_product_id`, `full_name`, `status_code`,
`status_changed_at`, `seller_product_id`, `product_sales_type`, `category_id`,
`root_category_id`, `root_category_name`, `original_price`, `unit_price`, `quantity`,
`total_product_price`, `initial_paid_method`, `initial_paid_amount`, `attribute_types`,
`attributes`, `day`.

### Negatives

`negatives/published.json` maps each test user to the 49 sampled negatives released with
Park and Oh (2024), as item indices. `uv run mvcp-prep` checks that the lists cover the
3,358 test users and contain neither a user's target nor an item of the user's history.

### Source

The raw log is public in the repository of Park and Oh (2024),
<https://github.com/msparkDev/ConcurrentBERT-Recommender>. The three files here are that
log after the preprocessing and the split of that study.

M. Park and J. Oh, "Enhancing E-Commerce Recommendation Systems with Multiple Item
Purchase Data: A Bidirectional Encoder Representations from Transformers-Based Approach,"
*Applied Sciences*, vol. 14, no. 16, 7255, 2024.

### Terms of use

The Katcher's data were provided by Katcher's for the study of Park and Oh (2024) and
were made public with that study. The files here are redistributed for research use. If
you use them, cite the paper of this repository and Park and Oh (2024).

## Instacart

`instacart/split.json` lists the users of an earlier two-department sample of the
Instacart data; the builder of this repository excludes them when it draws its sample.
The Instacart data themselves are not redistributed here. `README.md` in the root gives
the download command, the checksums of the raw files, and the builder.

"The Instacart Online Grocery Shopping Dataset 2017," Instacart. Released by Instacart for
non-commercial use.

## REES46

Not redistributed here; `README.md` in the root gives the download command.

M. Kechinov, "eCommerce purchase history from electronics store," Kaggle; data provided by
the REES46 Marketing Platform.
