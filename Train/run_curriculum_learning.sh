#!/bin/bash
#
# Curriculum Learning Training Wrapper Script
# Trains all models on all HALA ladder datasets
#
# Usage: ./run_curriculum_learning.sh [--dry-run] [--model MODEL_NAME] [--dataset DATASET_NAME]
#

set -e

# Base paths
BASE_DIR="/home/zbibm/Safety-Arabic"
MODELS_DIR="${BASE_DIR}/models"
DATASETS_DIR="${BASE_DIR}/Training Corpus/hala_ladder_outputs"
OUTPUT_BASE="${BASE_DIR}/output_curriculum_learning"
TRAIN_SCRIPT="${BASE_DIR}/Train/train_flexible.py"
SAVE_STEPS=1000000000

# Models to train (excluding Qwen3Guard which is a safety classifier)
MODELS=(
    "allam"
    "Fanar-1-9B"
    "Meta-Llama-3-8B-Instruct"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
)

# Datasets (ordered by refusal ratio - curriculum learning order)
DATASETS=(
    "hala_7000_ref_3000.jsonl"  # 70% beneficial, 30% refusals
    "hala_7500_ref_2500.jsonl"  # 75% beneficial, 25% refusals
    "hala_8000_ref_2000.jsonl"  # 80% beneficial, 20% refusals
    "hala_8500_ref_1500.jsonl"  # 85% beneficial, 15% refusals
    "hala_9000_ref_1000.jsonl"  # 90% beneficial, 10% refusals
    "hala_9500_ref_500.jsonl"   # 95% beneficial, 5% refusals
)

# Dataset short names for folder naming
declare -A DATASET_SHORT_NAMES
DATASET_SHORT_NAMES["hala_7000_ref_3000.jsonl"]="70ben_30ref"
DATASET_SHORT_NAMES["hala_7500_ref_2500.jsonl"]="75ben_25ref"
DATASET_SHORT_NAMES["hala_8000_ref_2000.jsonl"]="80ben_20ref"
DATASET_SHORT_NAMES["hala_8500_ref_1500.jsonl"]="85ben_15ref"
DATASET_SHORT_NAMES["hala_9000_ref_1000.jsonl"]="90ben_10ref"
DATASET_SHORT_NAMES["hala_9500_ref_500.jsonl"]="95ben_5ref"

# Parse arguments
DRY_RUN=false
FILTER_MODEL=""
FILTER_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run           Print commands without executing"
            echo "  --model MODEL       Train only specified model"
            echo "  --dataset DATASET   Train only on specified dataset"
            echo "  --help              Show this help message"
            echo ""
            echo "Available models:"
            for model in "${MODELS[@]}"; do
                echo "  - $model"
            done
            echo ""
            echo "Available datasets:"
            for dataset in "${DATASETS[@]}"; do
                echo "  - $dataset"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output base directory
mkdir -p "${OUTPUT_BASE}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "CURRICULUM LEARNING TRAINING STARTED"
log "=============================================="
log "Base directory: ${BASE_DIR}"
log "Output directory: ${OUTPUT_BASE}"
log "=============================================="

# Count total combinations
TOTAL_COMBINATIONS=0
for model in "${MODELS[@]}"; do
    if [[ -n "$FILTER_MODEL" && "$model" != "$FILTER_MODEL" ]]; then
        continue
    fi
    for dataset in "${DATASETS[@]}"; do
        if [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]]; then
            continue
        fi
        ((TOTAL_COMBINATIONS++)) || true
    done
done

log "Total training combinations: ${TOTAL_COMBINATIONS}"
log "=============================================="

# Training counter
CURRENT=0

# Iterate through models
for model in "${MODELS[@]}"; do
    # Skip if filtering by model
    if [[ -n "$FILTER_MODEL" && "$model" != "$FILTER_MODEL" ]]; then
        continue
    fi

    MODEL_PATH="${MODELS_DIR}/${model}"

    # Check if model exists
    if [[ ! -d "$MODEL_PATH" ]]; then
        log "WARNING: Model not found: ${MODEL_PATH}, skipping..."
        continue
    fi

    # Create model-specific output directory
    MODEL_OUTPUT_DIR="${OUTPUT_BASE}/${model}_curriculum_learning"
    mkdir -p "${MODEL_OUTPUT_DIR}"

    log ""
    log "=============================================="
    log "MODEL: ${model}"
    log "=============================================="

    # Iterate through datasets
    for dataset in "${DATASETS[@]}"; do
        # Skip if filtering by dataset
        if [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]]; then
            continue
        fi

        ((CURRENT++)) || true

        DATASET_PATH="${DATASETS_DIR}/${dataset}"
        DATASET_SHORT="${DATASET_SHORT_NAMES[$dataset]}"
        RUN_OUTPUT_DIR="${MODEL_OUTPUT_DIR}/${DATASET_SHORT}"

        # Check if dataset exists
        if [[ ! -f "$DATASET_PATH" ]]; then
            log "WARNING: Dataset not found: ${DATASET_PATH}, skipping..."
            continue
        fi

        # Check if already trained (final-model exists)
        if [[ -d "${RUN_OUTPUT_DIR}/final-model" ]]; then
            log "[${CURRENT}/${TOTAL_COMBINATIONS}] SKIPPING (already trained): ${model} + ${dataset}"
            continue
        fi

        log ""
        log "[${CURRENT}/${TOTAL_COMBINATIONS}] Training: ${model} + ${dataset}"
        log "  Dataset: ${DATASET_PATH}"
        log "  Output: ${RUN_OUTPUT_DIR}"

        # Build command
        CMD="python ${TRAIN_SCRIPT} \
            --model_path \"${MODEL_PATH}\" \
            --dataset_path \"${DATASET_PATH}\" \
            --output_dir \"${RUN_OUTPUT_DIR}\" \
            --num_epochs 1 \
            --batch_size 16 \
            --gradient_accumulation_steps 2 \
            --learning_rate 1e-5 \
            --save_steps ${SAVE_STEPS}"

        if [[ "$DRY_RUN" == true ]]; then
            log "  [DRY-RUN] Would execute:"
            log "  $CMD"
        else
            log "  Starting training..."
            START_TIME=$(date +%s)

            # Execute training
            eval $CMD

            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            log "  Training completed in ${DURATION} seconds"
        fi
    done
done

log ""
log "=============================================="
log "CURRICULUM LEARNING TRAINING COMPLETED"
log "=============================================="
log "Results saved to: ${OUTPUT_BASE}"
log "=============================================="

# Print summary
echo ""
echo "Output directory structure:"
if command -v tree &> /dev/null; then
    tree -L 2 "${OUTPUT_BASE}" 2>/dev/null || ls -la "${OUTPUT_BASE}"
else
    ls -la "${OUTPUT_BASE}"
fi
