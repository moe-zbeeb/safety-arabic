# Curriculum Learning Datasets

This folder contains the original curriculum-learning training datasets used in the earlier ratio sweep experiments.

Each file has exactly `10,000` training examples and mixes:

- beneficial examples from `hammh0a/Hala-4.6M-SFT`
- refusal-style safety examples from the Arabic safety corpus

## What The File Names Mean

- `hala_7000_ref_3000.jsonl`: `7,000` beneficial + `3,000` refusal
- `hala_7500_ref_2500.jsonl`: `7,500` beneficial + `2,500` refusal
- `hala_8000_ref_2000.jsonl`: `8,000` beneficial + `2,000` refusal
- `hala_8500_ref_1500.jsonl`: `8,500` beneficial + `1,500` refusal
- `hala_9000_ref_1000.jsonl`: `9,000` beneficial + `1,000` refusal
- `hala_9500_ref_500.jsonl`: `9,500` beneficial + `500` refusal

## Important Ordering Detail

These datasets are not just different ratios. They are also ordered as a block curriculum:

- beneficial examples come first
- refusal examples come after

So these files change both:

- class ratio
- presentation order

That is why the newer ablation datasets in [`../ratio_ordering_ablation/README.md`](/home/zbibm/Safety-Arabic/Training%20Corpus/ratio_ordering_ablation/README.md) were created: to separate ratio effects from ordering effects.

## Data Format

Each row is a JSON object with the training fields already expected by the current fine-tuning scripts:

- `prompt_ar`
- `response_ar`
- optional metadata such as `source`

The `source` field can be used to infer class membership:

- `source == "hammh0a/Hala-4.6M-SFT"` means beneficial
- other rows correspond to refusal/safety examples

## Recommended Usage

Use this folder if you want to reproduce the original curriculum-style experiments.

Use [`../ratio_ordering_ablation/`](/home/zbibm/Safety-Arabic/Training%20Corpus/ratio_ordering_ablation) if you want to compare:

- beneficial-first
- refusal-first
- random
- interleaved

at fixed ratios.
