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

[`eval_guarded.py`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/eval_guarded.py)
implements the current runtime-guard setup:

- the base or SFT model generates the answer in chunks
- the external guard inspects the growing response during generation
- if the guard flags the response as unsafe, generation stops early
- no canned refusal text is injected
- the final visible output is then judged by `Qwen/Qwen3Guard-Gen-4B`

This is a runtime stopping experiment, not a post-hoc response replacement
experiment.

## Files

- [`eval_guarded.py`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/eval_guarded.py) — guarded evaluator
- [`run_grid.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/run_grid.sh) — multi-GPU launcher
- [`run_base_sft_guard_experiment.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/run_base_sft_guard_experiment.sh) — wrapper for the 5 base + 5 SFT guarded sweep
- [`exp7_base_sft_guard_manifest.txt`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/exp7_base_sft_guard_manifest.txt) — manifest for the guarded base/SFT runs
- [`example_setups_response_only.txt`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/example_setups_response_only.txt) — example custom manifest
- [`aggregate_exp10.py`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/aggregate_exp10.py) — summary aggregator

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

The wrapper defaults to:

- guard: `meta-llama/Llama-Guard-3-1B`
- judge: `Qwen/Qwen3Guard-Gen-4B`
- mode: `response`

Run it with:

```bash
NUM_GPUS=2 \
OUTPUT_DIR=/workspace/Safety-Arabic/results\ arxive/EXP10 \
bash Inference-time-safety/run_base_sft_guard_experiment.sh
```

You can override the defaults:

```bash
NUM_GPUS=2 \
MANIFEST=/tmp/custom_manifest.txt \
GUARD_MODEL=/workspace/Safety-Arabic/models/Llama-Guard-3-1B \
JUDGE_MODEL=Qwen/Qwen3Guard-Gen-4B \
OUTPUT_DIR=/workspace/Safety-Arabic/results\ arxive/EXP10 \
bash Inference-time-safety/run_base_sft_guard_experiment.sh
```

## Custom runs

To run a smaller custom experiment, create a manifest and call
[`run_grid.sh`](/Users/ranaezzeddine/Desktop/Safety-Arabic/Inference-time-safety/run_grid.sh)
directly:

```bash
cp Inference-time-safety/example_setups_response_only.txt /tmp/response_only_setups.txt
# edit /tmp/response_only_setups.txt

SETUPS_FILE=/tmp/response_only_setups.txt \
GUARD_MODES=response \
INCLUDE_UNGUARDED=all \
OUTPUT_DIR=/workspace/Safety-Arabic/output/response_only_guard_runs \
bash Inference-time-safety/run_grid.sh
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

## Aggregation

Aggregate an experiment directory with:

```bash
python Inference-time-safety/aggregate_exp10.py \
  --exp-dir "/workspace/Safety-Arabic/results arxive/EXP10"
```

This writes:

- `grid_summary.csv`
- `results_table.md`

## Practical interpretation

This folder is used to support the comparison:

1. base only
2. base + external guard
3. SFT only
4. SFT + external guard

The guarded runs here provide rows 2 and 4. The unguarded base and SFT results
come from the matching evaluation pipeline outside this folder.
