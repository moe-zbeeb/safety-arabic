# Ratio x Ordering Ablation Datasets

This folder contains the curated training datasets for the ratio x ordering ablation.

The goal is to separate two factors that were mixed together in the earlier curriculum experiments:

- refusal ratio
- example ordering

Each dataset here keeps the same total size, `10,000` examples, and changes only:

- the beneficial/refusal ratio
- the sequence order of those examples

## Folder Layout

There are 3 ratio folders:

- `80ben_20ref/`
- `90ben_10ref/`
- `95ben_5ref/`

Each ratio folder contains 4 dataset variants:

- `beneficial-first.jsonl`
- `refusal-first.jsonl`
- `random.jsonl`
- `interleaved.jsonl`

Each dataset also has a matching manifest file:

- `beneficial-first.manifest.json`
- `refusal-first.manifest.json`
- `random.manifest.json`
- `interleaved.manifest.json`

The top-level [`index.json`](/home/zbibm/Safety-Arabic/Training%20Corpus/ratio_ordering_ablation/index.json) summarizes all curated datasets and their checksums.

## Source Data

These curated datasets are derived from the fixed canonical ratio datasets in:

- [`curriculumLearningTrainingCorpus/hala_8000_ref_2000.jsonl`](/home/zbibm/Safety-Arabic/Training%20Corpus/curriculumLearningTrainingCorpus/hala_8000_ref_2000.jsonl)
- [`curriculumLearningTrainingCorpus/hala_9000_ref_1000.jsonl`](/home/zbibm/Safety-Arabic/Training%20Corpus/curriculumLearningTrainingCorpus/hala_9000_ref_1000.jsonl)
- [`curriculumLearningTrainingCorpus/hala_9500_ref_500.jsonl`](/home/zbibm/Safety-Arabic/Training%20Corpus/curriculumLearningTrainingCorpus/hala_9500_ref_500.jsonl)

The memberships are fixed by those source files. This curation step does not resample examples. It only changes the order.

Class labels are inferred from the existing `source` field:

- `source == "hammh0a/Hala-4.6M-SFT"` means `beneficial`
- any other row means `refusal`

## What Each Dataset Means

### `beneficial-first.jsonl`

This is a strict block curriculum:

- all beneficial examples come first
- all refusal examples come after

Example interpretation:

- `80ben_20ref/beneficial-first.jsonl` means 8,000 beneficial examples followed by 2,000 refusal examples
- `95ben_5ref/beneficial-first.jsonl` means 9,500 beneficial examples followed by 500 refusal examples

This matches the "beneficial first, then safety refusals" curriculum style.

### `refusal-first.jsonl`

This is the reverse block curriculum:

- all refusal examples come first
- all beneficial examples come after

This is useful for checking whether curriculum effects come from seeing refusal behavior early rather than late.

### `random.jsonl`

This uses the same fixed example membership for the ratio, but shuffles the full dataset into one mixed sequence.

Properties:

- exact class counts are preserved
- no block ordering is intended
- the shuffle is deterministic

This is the clean non-curriculum baseline for each ratio.

### `interleaved.jsonl`

This is a deterministic weighted interleave:

- beneficial and refusal examples are spread throughout the full sequence
- exact class counts are preserved
- the sequence is mixed by ratio rather than grouped into two large blocks

This is meant to test a structured mixed ordering that is not purely random and not block-based.

For example:

- `80ben_20ref/interleaved.jsonl` places refusal examples roughly every 5 positions
- `90ben_10ref/interleaved.jsonl` places refusal examples roughly every 10 positions
- `95ben_5ref/interleaved.jsonl` places refusal examples roughly every 20 positions

## Ratio Definitions

All datasets have exactly `10,000` rows.

- `80ben_20ref`: `8,000` beneficial + `2,000` refusal
- `90ben_10ref`: `9,000` beneficial + `1,000` refusal
- `95ben_5ref`: `9,500` beneficial + `500` refusal

## Manifest Files

Each `*.manifest.json` file records:

- ratio
- ordering
- source file
- dataset file
- project shuffle seed
- total row count
- beneficial count
- refusal count
- number of class runs in the final sequence
- first and last index of each class
- SHA-256 checksum of the dataset file

These manifests are intended to make later training runs auditable and reproducible.

## How The Datasets Were Generated

The datasets in this folder were created by:

- [`curate_ratio_ordering_ablation.py`](/home/zbibm/Safety-Arabic/Training%20Corpus/curate_ratio_ordering_ablation.py)

To regenerate them:

```bash
cd /home/zbibm/Safety-Arabic
python 'Training Corpus/curate_ratio_ordering_ablation.py' --force
```

The script uses a fixed project shuffle constant:

- `20260409`

That constant is only used for deterministic mixed orderings such as `random`.

## Validation Guarantees

The generation script validates:

- each dataset has exactly `10,000` rows
- beneficial/refusal counts match the target ratio exactly
- `beneficial-first` and `refusal-first` are truly block ordered
- `random` is not block ordered
- `interleaved` contains both classes across the sequence rather than collapsing into two blocks

## Recommended Usage

Use this folder as the dataset source for the next training grid.

Suggested convention:

- train one model against the 12 curated datasets here
- keep training hyperparameters fixed
- compare outcomes by ratio first, then by ordering within each ratio

That gives a cleaner answer to whether prior gains came from:

- fewer refusal examples
- the order in which refusal examples were presented
- or an interaction between both
