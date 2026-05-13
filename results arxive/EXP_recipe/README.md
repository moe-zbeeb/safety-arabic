# EXP_recipe — Recipe Application Experiment

## Goal
Test whether tailored SFT recipes based on a model's **base safety regime** improve safety calibration, measured on AraSafe using Qwen3Guard-Gen-4B as the judge.

## Recipe Assignment

| Model | Base Regime | Recipe |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | Miscalibrated (high over-refusal, low unsafe refusal) | 80/20 beneficial/refusal, interleaved |
| Hala-9B | Safety-prior strong (low over-refusal, high unsafe refusal) | 95/5 beneficial-first |
| jais-6p7b | Under-protective (low over-refusal, low unsafe refusal) | 95/5 refusal-first |
| Qwen3-8B | Under-protective (low over-refusal, low unsafe refusal) | 95/5 refusal-first |

## Training
- Script: `Train/run_sft_single.py`
- 1 epoch, effective batch 32, lr 1e-5, bf16, paged_adamw_8bit
- Training data: `Training Corpus/ratio_ordering_ablation/`

## Evaluation
- Dataset: `DPO/Eval/arasafe_human.jsonl` (12,077 prompts)
- Judge: Qwen3Guard-Gen-4B (`Refusal: Yes/No` pattern, no keyword fallback)
- Script: `DPO/Eval/eval.py`

## Results

| Model | Recipe | Regime | Safe%(base) | Safe%(sft) | Unsafe%(base) | Unsafe%(sft) |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 80/20 interleaved | Miscalibrated | 24.23 | 47.21 | 52.31 | 92.34 |
| Hala-9B | 95/5 beneficial-first | Safety-prior strong | 9.34 | 21.77 | 84.37 | 94.42 |
| jais-6p7b | 95/5 refusal-first | Under-protective | 10.13 | 16.66 | 57.41 | 65.87 |
| Qwen3-8B | 95/5 refusal-first | Under-protective | 8.00 | 11.55 | 61.96 | 83.09 |

Safe refusal = over-refusal rate (lower is better). Unsafe refusal = correct refusal rate (higher is better).

## SFT Models on HuggingFace
- `RanaEzzeddine/Mistral-7B-80-20-interleaved`
- `RanaEzzeddine/Hala-9b-95-5-beneficial-first`
- `RanaEzzeddine/jais-6p7b-95-5-refusal-first`
- `RanaEzzeddine/Qwen3-8B-95-5-refusal-first`

## Files
- `Results/` — per-model JSON summaries from eval
- `comparison_table.txt` / `comparison_table.csv` — base vs SFT comparison
