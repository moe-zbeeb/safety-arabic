# Inference-time safety

This folder evaluates AraSafe with an external Qwen3Guard interception layer.

The original version of this repo framed the sweep as a fixed `EXP10` grid:

- 5 base models
- 5 aligned checkpoints
- 3 guarded modes (`prompt`, `response`, `both`)

That is still supported, but the launcher is now configurable so you can also
run narrower experiments such as:

- response-only guarding
- base + SFT checkpoints
- arbitrary subsets of models
- unguarded + guarded paired runs for the same setup

## Files

- `eval_guarded.py` — evaluator. Reuses `DPO/Eval/eval.py` for generation + judging,
  then inserts guard interception before the final judge pass.
- `run_grid.sh` — configurable multi-GPU launcher.
- `aggregate_exp10.py` — recursive summary aggregator for both the original
  archive layout and custom flat output folders.
- `canned_refusal_ar.txt` — fixed Arabic refusal string substituted on guard intervention.
- `example_setups_response_only.txt` — template for response-only runs over custom checkpoints.

## Launcher behavior

If you run `run_grid.sh` without extra configuration, it preserves the legacy
matrix based on the five base models and five aligned checkpoints.

You can override that with:

- `SETUPS_FILE` — custom setup manifest.
- setup entries may use either local model directories or Hugging Face repo IDs such as `MHK-22/<model-name>`.
- `GUARD_MODES` — comma-separated guard modes, e.g. `response` or `prompt,response`.
- `INCLUDE_UNGUARDED` — `none`, `all`, or a comma-separated list of variants that
  should also run in unguarded mode.

Each setup-file line uses this format:

```text
short_name|variant_name|/absolute/path/to/model/or/HF-repo-id|comma,separated,modes
```

Notes:

- Lines starting with `#` are ignored.
- If the modes column is omitted or empty, the launcher falls back to
  `GUARD_MODES` plus `INCLUDE_UNGUARDED`.
- `none` means unguarded evaluation.
- Run names become:
  - `short__variant` for unguarded runs
  - `short__variant_guard_response` for guarded runs

## Example: response-only base + SFT sweep

If your experiment is basically “run the base model and a set of SFT checkpoints,
then compare unguarded vs response-guarded”, create a manifest from the example
file and point it to your actual checkpoint paths:

```bash
cp Inference-time-safety/example_setups_response_only.txt /tmp/exp7_setups.txt
# edit the paths or Hugging Face repo IDs in /tmp/exp7_setups.txt

SETUPS_FILE=/tmp/exp7_setups.txt GUARD_MODES=response INCLUDE_UNGUARDED=all OUTPUT_DIR=/home/zbibm/Safety-Arabic/output/exp7_response_only bash Inference-time-safety/run_grid.sh
```

That gives you paired `none` and `response` runs for every listed setup.
If you only want guarded runs, set:

```bash
INCLUDE_UNGUARDED=none
```

## Aggregation

Aggregate any output folder recursively:

```bash
python Inference-time-safety/aggregate_exp10.py   --exp-dir /home/zbibm/Safety-Arabic/output/exp7_response_only   --title "Experiment 7 — Response-only inference-time safety"
```

This writes:

- `grid_summary.csv`
- `results_table.md`

## Legacy EXP10 behavior

The original fixed matrix still works out of the box:

```bash
bash Inference-time-safety/run_grid.sh
```

with default setup names:

- `allam7b__base`
- `allam7b__aligned`
- `allam7b__base_guard_prompt`
- `allam7b__aligned_guard_response`
- etc.


## Exact setup for your base+guard and SFT+guard experiment

For the 5-model family experiment where each family has exactly two guarded runs:

- `base + guard`
- `SFT + guard`

use these files directly:

- `exp7_base_sft_guard_manifest.txt` — 10 guarded jobs (5 base + 5 SFT)
- `run_base_sft_guard_experiment.sh` — wrapper that forces `response` mode and disables unguarded reruns

Workflow:

```bash
# 1) Edit the SFT repo IDs in the manifest
nano Inference-time-safety/exp7_base_sft_guard_manifest.txt

# 2) Run the experiment on 2 GPUs
NUM_GPUS=2 OUTPUT_DIR=/workspace/Safety-Arabic/output/exp7_base_sft_guard bash Inference-time-safety/run_base_sft_guard_experiment.sh
```

Aggregate after completion:

```bash
python Inference-time-safety/aggregate_exp10.py   --exp-dir /workspace/Safety-Arabic/output/exp7_base_sft_guard   --title "Experiment 7 — Base+guard and SFT+guard"
```
