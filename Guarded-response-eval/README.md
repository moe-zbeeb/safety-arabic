# Inference-time safety

This folder contains the inference-time safety evaluation pipeline used for the
`EXP10` comparison:

- base model only
- base model + external guard
- SFT-aligned model only
- hybrid: SFT-aligned model + external guard

The main benchmark is AraSafe, and the final refusal judgment is done by
`Qwen/Qwen3Guard-Gen-4B`.

## What the current evaluator does

[`Eval/eval_guarded.py`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/eval_guarded.py)
implements the current runtime-guard setup:

- the base or SFT model generates the answer in chunks
- the external guard inspects the growing response during generation
- if the guard flags the response as unsafe, generation stops early
- no canned refusal text is injected
- the final visible output is then judged by `Qwen/Qwen3Guard-Gen-4B`

This is a runtime stopping experiment, not a post-hoc response replacement
experiment.

## Files

- [`Eval/eval_guarded.py`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/eval_guarded.py) — guarded evaluator
- [`Eval/run_guard_sweep.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/run_guard_sweep.sh) — multi-GPU launcher for guarded sweeps
- [`Eval/run_exp10_base_sft_guard.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/run_exp10_base_sft_guard.sh) — wrapper for the standard 5 base + 5 SFT guarded sweep
- [`Eval/base_sft_guard_manifest.txt`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/base_sft_guard_manifest.txt) — manifest for the guarded base/SFT runs
- [`build_four_setup_comparison.py`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/build_four_setup_comparison.py) — builds the base / base+guard / SFT / hybrid comparison table

## Manifest format

Each manifest line uses:

```text
short_name|variant_name|model_path_or_hf_id|modes|optional_tokenizer_override
```

Examples:

```text
allam_base|base|humain-ai/ALLaM-7B-Instruct-preview|response|
allam_sft|sft|MHK-22/ALLaM-7B-SFT-safe|response|humain-ai/ALLaM-7B-Instruct-preview
```

Notes:

- `modes` is usually `response` for the current experiment
- the tokenizer column is optional
- the tokenizer override is useful for SFT checkpoints that should reuse the
  base model tokenizer

## Main guarded sweep

The current guarded sweep is:

- 5 base models + external guard
- 5 SFT models + external guard

The generic launcher defaults to:

- guard: `meta-llama/Llama-Guard-3-1B`
- judge: `Qwen/Qwen3Guard-Gen-4B`
- mode: `response`

The two shell scripts have different roles:

- [`Eval/run_guard_sweep.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/run_guard_sweep.sh)
  is the generic guarded launcher.
- [`Eval/run_exp10_base_sft_guard.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/run_exp10_base_sft_guard.sh)
  is the fixed wrapper for the standard 5-base + 5-SFT experiment.

Run it with:

```bash
NUM_GPUS=2 \
OUTPUT_DIR=/workspace/Safety-Arabic/results\ arxive/EXP10 \
bash Guarded-response-eval/Eval/run_exp10_base_sft_guard.sh
```

You can override the defaults:

```bash
NUM_GPUS=2 \
MANIFEST=/tmp/custom_manifest.txt \
GUARD_MODEL=/workspace/Safety-Arabic/models/Llama-Guard-3-1B \
JUDGE_MODEL=Qwen/Qwen3Guard-Gen-4B \
OUTPUT_DIR=/workspace/Safety-Arabic/results\ arxive/EXP10 \
bash Guarded-response-eval/Eval/run_exp10_base_sft_guard.sh
```

## Custom runs

To run a smaller custom experiment, create your own manifest and call
[`Eval/run_guard_sweep.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Guarded-response-eval/Eval/run_guard_sweep.sh)
directly:

```text
short_name|variant_name|model_path_or_hf_id|modes|optional_tokenizer_override
```

Then run:

```bash
SETUPS_FILE=/tmp/custom_manifest.txt \
GUARD_MODES=response \
INCLUDE_UNGUARDED=all \
OUTPUT_DIR=/workspace/Safety-Arabic/output/response_only_guard_runs \
bash Guarded-response-eval/Eval/run_guard_sweep.sh
```

## Outputs

Each run writes a summary JSON like:

- `allam_base__base_guard_response_summary.json`
- `allam_sft__sft_guard_response_summary.json`

Typical summary fields include:

- `safe_refusal_rate`
- `unsafe_refusal_rate`
- `refusal_rate`
- `intervention_rate`
- `stopped_early_count`
- `parse_failure_rate`

## Four-setup comparison

Build the 4-column comparison table with:

```bash
python Guarded-response-eval/build_four_setup_comparison.py \
  --exp-dir "/workspace/Safety-Arabic/results arxive/EXP10"
```

This writes:

- `four_setup_comparison.csv`
- `four_setup_comparison.md`

## Practical interpretation

This folder is used to support the comparison:

1. base only
2. base + external guard
3. SFT only
4. SFT + external guard

The guarded runs here provide rows 2 and 4. The script combines them with the
project's base-only and SFT-only reference values to produce the final table.
