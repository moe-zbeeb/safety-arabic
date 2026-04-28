---
license: apache-2.0
task_categories:
- text-generation
language:
- ar
- de
- es
- fr
- ru
- tr
- zh
tags:
- dpo
- rlhf
- multilingual
- beneficial
size_categories:
- 1K<n<10K
source_datasets:
- multilingual/orca_dpo_pairs
pretty_name: Beneficial DPO Dataset
---

# Beneficial DPO Dataset

A deterministic 10,000-row multilingual DPO subset sampled from `multilingual/orca_dpo_pairs`.

The dataset is balanced across seven language splits and keeps the DPO training fields `prompt`, `chosen`, and `rejected`, along with the original `system`, `question`, and `mllm_index` metadata.

## Source Split Counts

| Source split | Rows |
|---|---:|
| `ar_train` | 1429 |
| `de_train` | 1429 |
| `es_train` | 1429 |
| `fr_train` | 1429 |
| `ru_train` | 1428 |
| `tr_train` | 1428 |
| `zh_train` | 1428 |


## Columns

- `mllm_index`
- `system`
- `question`
- `chosen`
- `rejected`
- `id`
- `source_dataset`
- `source_split`
- `language`
- `prompt`

## Reproducibility

Sampling seed: `20260428`

Source dataset: `multilingual/orca_dpo_pairs`
