#!/bin/bash
# ---------------------------------------------------------------------------
# Ratio × Ordering Ablation — per-model runner
#
# Trains ONE model on all 12 ablation datasets (3 ratios × 4 orderings)
# using all available GPUs via accelerate, then evaluates each resulting
# model with the AraSafe benchmark.
#
# Usage:
#   ./run_ratio_ordering_ablation.sh <MODEL_NAME> [--test]
#
# Examples:
#   ./run_ratio_ordering_ablation.sh Fanar-1-9B          # full epoch
#   ./run_ratio_ordering_ablation.sh Fanar-1-9B --test   # 5 steps only
# ---------------------------------------------------------------------------
set -euo pipefail

TEST_MODE=false
MODEL=""
for arg in "$@"; do
    case "$arg" in
        --test) TEST_MODE=true ;;
        *)      MODEL="$arg" ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <MODEL_NAME> [--test]"
    exit 1
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${BASE_DIR}/models"
DATASETS_DIR="${BASE_DIR}/Training Corpus/ratio_ordering_ablation"
OUTPUT_BASE="${BASE_DIR}/output_ratio_ordering_ablation"
RESULTS_BASE="${BASE_DIR}/results_ratio_ordering_ablation"
TRAIN_SCRIPT="${BASE_DIR}/Train/train.py"
BENCHMARK_SCRIPT="${BASE_DIR}/benchmarking-scripts/benchmark_safety.py"
LOG_DIR="${BASE_DIR}/logs_ratio_ordering_ablation"
NUM_GPUS="${NUM_GPUS:-4}"

MODEL_PATH="${MODELS_DIR}/${MODEL}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model directory not found: ${MODEL_PATH}"
    exit 1
fi

RATIOS=("80ben_20ref" "90ben_10ref" "95ben_5ref")
ORDERINGS=("beneficial-first" "refusal-first" "random" "interleaved")
TOTAL=$(( ${#RATIOS[@]} * ${#ORDERINGS[@]} ))
CURRENT=0

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${MODEL}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${MODEL}] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log "=============================================="
log "RATIO × ORDERING ABLATION"
log "Model:       ${MODEL}"
log "Model path:  ${MODEL_PATH}"
log "Datasets:    ${DATASETS_DIR}"
log "Output:      ${OUTPUT_BASE}/${MODEL}"
log "Results:     ${RESULTS_BASE}/${MODEL}"
log "GPUs:        ${NUM_GPUS}"
log "Test mode:   ${TEST_MODE}"
log "Total runs:  ${TOTAL}"
log "=============================================="

for RATIO in "${RATIOS[@]}"; do
    for ORDERING in "${ORDERINGS[@]}"; do
        ((CURRENT++)) || true

        RUN_NAME="${RATIO}__${ORDERING}"
        DATASET_FILE="${DATASETS_DIR}/${RATIO}/${ORDERING}.jsonl"
        RUN_OUTPUT="${OUTPUT_BASE}/${MODEL}/${RUN_NAME}"
        RESULTS_DIR="${RESULTS_BASE}/${MODEL}"
        RESULT_FILE="${RESULTS_DIR}/${RUN_NAME}_summary.json"

        log ""
        log "[${CURRENT}/${TOTAL}] --- ${RUN_NAME} ---"

        if [ -f "$RESULT_FILE" ]; then
            log "  SKIP: result already exists"
            continue
        fi

        if [ ! -f "$DATASET_FILE" ]; then
            log "  ERROR: dataset not found: ${DATASET_FILE}"
            continue
        fi

        mkdir -p "$RUN_OUTPUT"
        mkdir -p "$RESULTS_DIR"

        # ---- Generate patched training script ----
        TEMP_TRAIN=$(mktemp "${BASE_DIR}/.train_ablation_XXXXXX.py")
        trap "rm -f '$TEMP_TRAIN'" EXIT

        python3 -c "
import sys
src, dst = sys.argv[1], sys.argv[2]
dataset_path, model_path, output_dir = sys.argv[3], sys.argv[4], sys.argv[5]
test_mode = sys.argv[6] == 'true'
with open(src) as f:
    code = f.read()
code = code.replace('/home/zbibm/Safety-Arabic/Training Corpus/mix70-30.jsonl', dataset_path)
code = code.replace('/home/zbibm/Safety-Arabic/models/jais-7b-chat', model_path)
code = code.replace('/home/zbibm/Safety-Arabic/models/Fanar-1-9B', model_path)
code = code.replace('./output', output_dir)
code = code.replace('./logs', output_dir + '/logs')
code = code.replace('save_steps=50', 'save_steps=999999999')
if test_mode:
    code = code.replace('num_train_epochs=1', 'max_steps=5')
with open(dst, 'w') as f:
    f.write(code)
" "$TRAIN_SCRIPT" "$TEMP_TRAIN" "$DATASET_FILE" "$MODEL_PATH" "$RUN_OUTPUT" "$TEST_MODE"

        # ---- Train on all GPUs ----
        log "  TRAIN START  (accelerate × ${NUM_GPUS} GPUs)"
        TRAIN_START=$(date +%s)

        set +eo pipefail
        accelerate launch --num_processes "${NUM_GPUS}" "$TEMP_TRAIN" 2>&1 | tee -a "$LOG_FILE"
        TRAIN_EXIT=${PIPESTATUS[0]}
        set -eo pipefail

        TRAIN_END=$(date +%s)
        TRAIN_SECS=$((TRAIN_END - TRAIN_START))
        log "  TRAIN END    ${TRAIN_SECS}s  exit=${TRAIN_EXIT}"

        rm -f "$TEMP_TRAIN"

        if [ "$TRAIN_EXIT" -ne 0 ]; then
            log "  ERROR: training failed — skipping eval"
            continue
        fi

        if [ ! -d "${RUN_OUTPUT}/final-model" ]; then
            log "  ERROR: final-model directory missing after training"
            continue
        fi

        # ---- Evaluate with AraSafe ----
        log "  EVAL START   (AraSafe benchmark)"
        EVAL_START=$(date +%s)

        cd "$BASE_DIR"
        set +eo pipefail
        CUDA_VISIBLE_DEVICES=0 python3 "$BENCHMARK_SCRIPT" "${RUN_OUTPUT}/final-model" 2>&1 | tee -a "$LOG_FILE"
        EVAL_EXIT=${PIPESTATUS[0]}
        set -eo pipefail

        EVAL_END=$(date +%s)
        EVAL_SECS=$((EVAL_END - EVAL_START))
        log "  EVAL END     ${EVAL_SECS}s  exit=${EVAL_EXIT}"

        # ---- Save results ----
        BENCH_OUTPUT="${BASE_DIR}/output/final-model_summary.json"
        if [ -f "$BENCH_OUTPUT" ]; then
            mv "$BENCH_OUTPUT" "$RESULT_FILE"
            log "  RESULT: ${RESULT_FILE}"
        else
            log "  WARNING: expected benchmark output not found at ${BENCH_OUTPUT}"
        fi

        # ---- Cleanup intermediate checkpoints ----
        find "$RUN_OUTPUT" -maxdepth 1 -name "checkpoint-*" -type d -exec rm -rf {} + 2>/dev/null || true
        log "  CLEANUP: removed intermediate checkpoints"

        log "  DONE: ${RUN_NAME}  train=${TRAIN_SECS}s  eval=${EVAL_SECS}s"
    done
done

log ""
log "=============================================="
log "COMPLETED ALL ${TOTAL} RUNS FOR: ${MODEL}"
log "=============================================="
