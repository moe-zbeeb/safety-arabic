#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zbibm/Safety-Arabic}"
GUARD_MODEL="${GUARD_MODEL:-$REPO_ROOT/models/Qwen3Guard-Gen-4B}"
DATASET="${DATASET:-$REPO_ROOT/Arasafe/arasafe_human.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/EXP10_runs}"
SCRIPT="${SCRIPT:-$REPO_ROOT/Inference-time-safety/eval_guarded.py}"
SETUPS_FILE="${SETUPS_FILE:-}"
GUARD_MODES="${GUARD_MODES:-prompt,response,both}"
INCLUDE_UNGUARDED="${INCLUDE_UNGUARDED:-aligned}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "[run_grid] ERROR: no GPUs detected (set NUM_GPUS=N to override)."
    exit 1
fi

DEFAULT_SETUPS=(
    "allam7b|base|${ALLAM_BASE:-$REPO_ROOT/models/ALLaM-7B-Instruct-preview}"
    "allam7b|aligned|${ALLAM_ALIGNED:-$REPO_ROOT/output/allam7b__aligned/final}"
    "fanar9b|base|${FANAR_BASE:-$REPO_ROOT/models/Fanar-1-9B-Instruct}"
    "fanar9b|aligned|${FANAR_ALIGNED:-$REPO_ROOT/output/fanar9b__aligned/final}"
    "llama8b|base|${LLAMA_BASE:-$REPO_ROOT/models/Meta-Llama-3-8B-Instruct}"
    "llama8b|aligned|${LLAMA_ALIGNED:-$REPO_ROOT/output/llama8b__aligned/final}"
    "qwen3b|base|${QWEN3B_BASE:-$REPO_ROOT/models/Qwen2.5-3B-Instruct}"
    "qwen3b|aligned|${QWEN3B_ALIGNED:-$REPO_ROOT/output/qwen3b__aligned/final}"
    "qwen7b|base|${QWEN7B_BASE:-$REPO_ROOT/models/Qwen2.5-7B-Instruct}"
    "qwen7b|aligned|${QWEN7B_ALIGNED:-$REPO_ROOT/output/qwen7b__aligned/final}"
)

mkdir -p "$OUTPUT_DIR"

is_hf_model_ref() {
    local value=$1
    [[ "$value" != /* ]] && [[ "$value" != ./* ]] && [[ "$value" =~ ^[^/]+/[^/]+$ ]]
}

contains_csv_value() {
    local csv=$1
    local target=$2
    local normalized=",${csv// /},"
    [[ "$normalized" == *",${target},"* ]]
}

default_modes_for_variant() {
    local variant=$1
    local modes=()

    if [[ "$INCLUDE_UNGUARDED" == "all" ]] || contains_csv_value "$INCLUDE_UNGUARDED" "$variant"; then
        modes+=("none")
    fi

    IFS=',' read -ra guard_modes <<< "$GUARD_MODES"
    for mode in "${guard_modes[@]}"; do
        mode=${mode// /}
        [[ -z "$mode" ]] && continue
        modes+=("$mode")
    done

    printf '%s\n' "${modes[@]}"
}

build_entries() {
    if [[ -n "$SETUPS_FILE" ]]; then
        if [[ ! -f "$SETUPS_FILE" ]]; then
            echo "[run_grid] ERROR: setups file not found: $SETUPS_FILE"
            exit 1
        fi
        while IFS='|' read -r short variant model_path mode_csv; do
            short=${short:-}
            short=${short// /}
            [[ -z "$short" ]] && continue
            [[ "$short" == \#* ]] && continue

            variant=$(echo "${variant:-}" | xargs)
            model_path=$(echo "${model_path:-}" | xargs)
            mode_csv=$(echo "${mode_csv:-}" | xargs)

            if [[ -z "$variant" || -z "$model_path" ]]; then
                echo "[run_grid] WARNING: skipping malformed line in $SETUPS_FILE: $short|${variant:-}|${model_path:-}|${mode_csv:-}"
                continue
            fi

            if [[ -z "$mode_csv" ]]; then
                mapfile -t modes < <(default_modes_for_variant "$variant")
            else
                IFS=',' read -ra modes <<< "$mode_csv"
            fi

            for mode in "${modes[@]}"; do
                mode=${mode// /}
                [[ -z "$mode" ]] && continue
                echo "$short|$variant|$mode|$model_path"
            done
        done < "$SETUPS_FILE"
        return
    fi

    for entry in "${DEFAULT_SETUPS[@]}"; do
        IFS='|' read -r short variant model_path <<< "$entry"
        mapfile -t modes < <(default_modes_for_variant "$variant")
        for mode in "${modes[@]}"; do
            echo "$short|$variant|$mode|$model_path"
        done
    done
}

JOBS=()
while IFS= read -r job; do
    [[ -z "$job" ]] && continue
    IFS='|' read -r short variant mode model_path <<< "$job"
    if [[ -e "$model_path" ]]; then
        JOBS+=("$job")
        continue
    fi
    if is_hf_model_ref "$model_path"; then
        echo "[run_grid] using Hugging Face model ref for ${short} ${variant} ${mode}: $model_path"
        JOBS+=("$job")
        continue
    fi
    echo "[run_grid] skip ${short} ${variant} ${mode} (path missing): $model_path"
    continue
done < <(build_entries)

TOTAL=${#JOBS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
    echo "[run_grid] ERROR: no runnable jobs were found."
    exit 1
fi

echo "=================================================================="
echo "Inference-time safety grid"
echo "  GPUs:              $NUM_GPUS"
echo "  total jobs:        $TOTAL"
echo "  guard:             $GUARD_MODEL"
echo "  dataset:           $DATASET"
echo "  output_dir:        $OUTPUT_DIR"
echo "  script:            $SCRIPT"
echo "  setups_file:       ${SETUPS_FILE:-<default matrix>}"
echo "  guard_modes:       $GUARD_MODES"
echo "  include_unguarded: $INCLUDE_UNGUARDED"
echo "  per-job log:       $OUTPUT_DIR/<run_name>.log"
echo "=================================================================="

run_one_job() {
    local gpu_id=$1
    local job_idx=$2
    local job_str=$3

    IFS='|' read -r short variant mode model_path <<< "$job_str"

    local run_name
    if [[ "$mode" == "none" ]]; then
        run_name="${short}__${variant}"
    else
        run_name="${short}__${variant}_guard_${mode}"
    fi

    local log_file="$OUTPUT_DIR/${run_name}.log"
    echo "[GPU ${gpu_id} · ${job_idx}/${TOTAL}] starting ${run_name}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python "$SCRIPT" \
        --base-model "$model_path" \
        --guard-model "$GUARD_MODEL" \
        --mode "$mode" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$run_name" \
        > "$log_file" 2>&1

    echo "[GPU ${gpu_id} · ${job_idx}/${TOTAL}] finished ${run_name}"
}

PIDS=()
PID_GPUS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
        for ((j=gpu; j<TOTAL; j+=NUM_GPUS)); do
            run_one_job ${gpu} $((j+1)) "${JOBS[$j]}"
        done
    ) &
    PIDS+=("$!")
    PID_GPUS+=("${gpu}")
done

FAILURES=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    gpu="${PID_GPUS[$idx]}"
    if ! wait "$pid"; then
        echo "[run_grid] ERROR: worker for GPU ${gpu} failed. Inspect logs in $OUTPUT_DIR"
        FAILURES=$((FAILURES + 1))
    fi
done

if [[ "$FAILURES" -gt 0 ]]; then
    echo ""
    echo "[run_grid] FAILED: ${FAILURES} worker(s) exited with errors."
    echo "[run_grid] Inspect per-job logs in $OUTPUT_DIR before trusting any results."
    exit 1
fi

echo ""
echo "[run_grid] All ${TOTAL} job(s) finished successfully. Summaries in $OUTPUT_DIR"
echo "[run_grid] Per-job logs: ls $OUTPUT_DIR/*.log"
echo "[run_grid] Next: aggregate with python Inference-time-safety/aggregate_exp10.py --exp-dir <dir>"
