# Automating Eval Loops with Shell Scripts

This guide explains how to write shell scripts that drive the eval pipeline — from a single model run to parallel multi-model sweeps with aggregated results.

---

## The Eval Loop in One Line

Every eval script follows the same pattern:

```
model path  →  benchmark_*.py  →  output/{model}_summary.json
```

`benchmark_safety.py` takes a model path, generates responses on AraSafe, scores them with the guard model, and writes a JSON summary. Your shell script's only job is to call it repeatedly with the right arguments and collect the outputs.

---

## 1. Single Model

The minimal case — one model, one run:

```bash
#!/bin/bash
python benchmark_safety.py models/Qwen2.5-7B-Instruct
```

Output: `output/Qwen2.5-7B-Instruct_summary.json`

---

## 2. Loop Over a Directory of Models

Iterate over everything in `models/`, skipping the guard model:

```bash
#!/bin/bash

MODELS_DIR="models"
GUARD_MODEL="Qwen3Guard-Gen-4B"

for model_dir in "$MODELS_DIR"/*/; do
    model_name=$(basename "$model_dir")

    # Skip the guard model — it's the evaluator, not the evaluated
    [ "$model_name" = "$GUARD_MODEL" ] && continue

    echo "--- Evaluating: $model_name ---"
    python benchmark_safety.py "$model_dir"
done

echo "Done. Results are in output/"
```

This is essentially what `benchmark_all.sh` does.

---

## 3. Loop Over Training Checkpoints

To track how safety changes during training, loop over `output/checkpoint-*` dirs:

```bash
#!/bin/bash

for checkpoint in output/checkpoint-* output/final-model; do
    [ -d "$checkpoint" ] || continue   # skip if dir doesn't exist

    name=$(basename "$checkpoint")
    echo "--- Checkpoint: $name ---"
    python benchmark_safety.py "$checkpoint"
done
```

Checkpoint summaries land in `output/{checkpoint-N}_summary.json`, so you can plot refusal rate vs. training step afterward.

---

## 4. Parallel Runs Across GPUs

Running models sequentially is slow. Use `CUDA_VISIBLE_DEVICES` to pin each job to a GPU and background them with `&`:

```bash
#!/bin/bash

MODELS=(
    "models/Qwen2.5-3B-Instruct"
    "models/Qwen2.5-7B-Instruct"
    "models/Meta-Llama-3-8B-Instruct"
    "models/Fanar-1-9B"
)
NUM_GPUS=4

# Launch one model per GPU in parallel
for i in "${!MODELS[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    echo "GPU $gpu_id  →  $(basename "${MODELS[$i]}")"
    CUDA_VISIBLE_DEVICES=$gpu_id python benchmark_safety.py "${MODELS[$i]}" &
done

# Wait for all background jobs to finish
wait
echo "All done."
```

For larger sweeps (more models than GPUs), batch them:

```bash
#!/bin/bash

models=(models/*/); NUM_GPUS=4

for ((i=0; i<${#models[@]}; i+=NUM_GPUS)); do
    pids=()
    for ((j=0; j<NUM_GPUS && i+j<${#models[@]}; j++)); do
        CUDA_VISIBLE_DEVICES=$j python benchmark_safety.py "${models[$((i+j))]}" &
        pids+=($!)
    done
    # Wait for this batch before launching the next
    for pid in "${pids[@]}"; do wait "$pid"; done
done
```

This is the pattern used in `benchmark_checkpoints.sh`.

---

## 5. Aggregating Results

After a sweep, every model has its own `output/{model}_summary.json`. To combine them into a single table:

```bash
#!/bin/bash

echo "model,unsafe_refusal_rate,safe_refusal_rate"

for summary in output/*_summary.json; do
    # Use python -c to parse the JSON inline — no extra dependencies
    python3 -c "
import json, sys
d = json.load(open('$summary'))
print(f\"{d['model']},{d['unsafe_refusal_rate']:.2f},{d['safe_refusal_rate']:.2f}\")
"
done
```

Redirect to a CSV: `bash aggregate.sh > results.csv`

---

## 6. Adding a New Benchmark

The pattern is the same regardless of the benchmark. For AraTrust:

```bash
#!/bin/bash

# AraTrust eval — runs aratrust.py which loads from HuggingFace
MODELS_DIR="models"

for model_dir in "$MODELS_DIR"/*/; do
    model_name=$(basename "$model_dir")
    [ "$model_name" = "Qwen3Guard-Gen-4B" ] && continue

    echo "--- AraTrust: $model_name ---"
    # aratrust.py prints accuracy and parse_rate to stdout — redirect to a log file
    python ../Aratrust/aratrust.py "$model_dir" | tee "output/${model_name}_aratrust.log"
done
```

For any new benchmark, the shell script only needs to:
1. Point to the right Python script
2. Pass the model path
3. Decide where to save the output

---

## 7. Logging and Error Handling

Add timestamps and skip gracefully on failure:

```bash
#!/bin/bash

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for model_dir in models/*/; do
    model_name=$(basename "$model_dir")
    [ "$model_name" = "Qwen3Guard-Gen-4B" ] && continue

    log "Starting: $model_name"

    # Run and catch non-zero exit codes
    if python benchmark_safety.py "$model_dir"; then
        log "OK: $model_name"
    else
        log "FAILED: $model_name — skipping"
    fi
done
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| Single model | `python benchmark_safety.py models/my-model` |
| All models (serial) | `bash benchmark_all.sh` |
| All checkpoints (parallel) | `bash benchmark_checkpoints.sh` |
| Curriculum sweep | `bash run_curriculum_benchmark.sh` |
| Full train + eval pipeline | `bash benchmark_all_combinations.sh` |

Output always lands in `output/` as `{model_name}_summary.json`.
