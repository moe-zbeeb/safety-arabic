#!/bin/bash

# Configuration
MODELS_DIR="/home/zbibm/Safety-Arabic/models"
DATASETS_DIR="/home/zbibm/Safety-Arabic/Training Corpus"
OUTPUT_BASE_DIR="./output"
RESULTS_BASE_DIR="./results_all_combinations"
LOG_FILE="./pipeline_execution.log"
NUM_GPUS=4
TRAINING_SCRIPT="Train/train_flexible.py"

# Create results directory
mkdir -p "$RESULTS_BASE_DIR"

# Initialize log file
echo "Pipeline started at $(date)" > "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"

# Get list of models (excluding guard model and Qwen 3B)
models=()
for model_path in "$MODELS_DIR"/*; do
    model_name=$(basename "$model_path")
    # Skip the guard model and Qwen2.5-3B-Instruct
    if [ "$model_name" != "Qwen3Guard-Gen-4B" ] && [ "$model_name" != "Qwen2.5-3B-Instruct" ]; then
        models+=("$model_path")
    fi
done

# Get list of datasets
datasets=()
for dataset_path in "$DATASETS_DIR"/*.jsonl; do
    if [ -f "$dataset_path" ]; then
        datasets+=("$dataset_path")
    fi
done

{
    echo "========================================="
    echo "TRAINING AND BENCHMARKING PIPELINE"
    echo "========================================="
    echo "Models found: ${#models[@]}"
    for model in "${models[@]}"; do
        echo "  - $(basename "$model")"
    done
    echo ""
    echo "Datasets found: ${#datasets[@]}"
    for dataset in "${datasets[@]}"; do
        echo "  - $(basename "$dataset")"
    done
    echo "========================================="
    echo ""
} | tee -a "$LOG_FILE"

# Function to run benchmark on a specific GPU
run_benchmark() {
    local checkpoint_dir=$1
    local gpu_id=$2
    local results_dir=$3
    local checkpoint_name=$(basename "$checkpoint_dir")

    echo "[GPU $gpu_id] Benchmarking: $checkpoint_name"
    CUDA_VISIBLE_DEVICES=$gpu_id python benchmark_safety.py "$checkpoint_dir" > /dev/null 2>&1

    # Move the summary JSON to results directory
    if [ -f "$OUTPUT_BASE_DIR/${checkpoint_name}_summary.json" ]; then
        mv "$OUTPUT_BASE_DIR/${checkpoint_name}_summary.json" "$results_dir/"
        echo "[GPU $gpu_id] ✓ Saved summary for $checkpoint_name"
    fi
}

# Main loop: iterate through all model-dataset combinations
for model_path in "${models[@]}"; do
    model_name=$(basename "$model_path")

    for dataset_path in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset_path" .jsonl)
        combination_name="${model_name}_${dataset_name}"
        results_dir="$RESULTS_BASE_DIR/$combination_name"

        {
            echo ""
            echo "========================================="
            echo "PROCESSING: $combination_name"
            echo "Model: $model_name"
            echo "Dataset: $dataset_name"
            echo "Started: $(date)"
            echo "========================================="

            # Create results directory for this combination
            mkdir -p "$results_dir"

            # Clean output directory before training
            if [ -d "$OUTPUT_BASE_DIR" ]; then
                echo "Cleaning output directory..."
                rm -rf "$OUTPUT_BASE_DIR"
            fi
            mkdir -p "$OUTPUT_BASE_DIR"

            # Train the model
            echo ""
            echo "--- STEP 1: Training ---"
            python "$TRAINING_SCRIPT" --model "$model_path" --dataset "$dataset_path" --output "$OUTPUT_BASE_DIR"

            if [ $? -ne 0 ]; then
                echo "ERROR: Training failed for $combination_name"
                continue
            fi

            echo "✓ Training completed"

            # Collect all checkpoint directories
            echo ""
            echo "--- STEP 2: Collecting checkpoints ---"
            checkpoints=()
            for checkpoint_dir in "$OUTPUT_BASE_DIR"/checkpoint-* "$OUTPUT_BASE_DIR"/final-model; do
                if [ -d "$checkpoint_dir" ]; then
                    checkpoints+=("$checkpoint_dir")
                    echo "  Found: $(basename "$checkpoint_dir")"
                fi
            done
            echo "Total checkpoints: ${#checkpoints[@]}"

            # Benchmark all checkpoints in parallel batches
            echo ""
            echo "--- STEP 3: Benchmarking checkpoints ---"
            for ((i=0; i<${#checkpoints[@]}; i+=NUM_GPUS)); do
                pids=()

                for ((j=0; j<NUM_GPUS && i+j<${#checkpoints[@]}; j++)); do
                    gpu_id=$j
                    checkpoint_dir="${checkpoints[$((i+j))]}"
                    run_benchmark "$checkpoint_dir" "$gpu_id" "$results_dir" &
                    pids+=($!)
                done

                # Wait for current batch to complete
                for pid in "${pids[@]}"; do
                    wait $pid
                done
            done

            echo "✓ All checkpoints benchmarked"

            # Create a combined summary for this combination
            echo ""
            echo "--- STEP 4: Creating combined summary ---"
            python - <<EOF
import json
import glob
from pathlib import Path

results_dir = Path("$results_dir")
summaries = []

for summary_file in sorted(results_dir.glob("*_summary.json")):
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        summaries.append(data)

combined = {
    "model_name": "$model_name",
    "dataset_name": "$dataset_name",
    "combination": "$combination_name",
    "num_checkpoints": len(summaries),
    "checkpoints": summaries
}

with open(results_dir / "combined_summary.json", 'w', encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"✓ Combined summary created with {len(summaries)} checkpoints")
EOF

            # Clean up checkpoints to save memory
            echo ""
            echo "--- STEP 5: Cleaning up checkpoints ---"
            rm -rf "$OUTPUT_BASE_DIR"
            echo "✓ Checkpoints removed to save memory"

            echo ""
            echo "========================================="
            echo "✓ COMPLETED: $combination_name"
            echo "Finished: $(date)"
            echo "Results saved in: $results_dir"
            echo "========================================="

        } 2>&1 | tee -a "$LOG_FILE"
    done
done

{
    echo ""
    echo "========================================="
    echo "ALL COMBINATIONS COMPLETED!"
    echo "========================================="
    echo "Results directory: $RESULTS_BASE_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Summary of combinations:"
    for combo_dir in "$RESULTS_BASE_DIR"/*; do
        if [ -d "$combo_dir" ]; then
            combo_name=$(basename "$combo_dir")
            num_summaries=$(ls -1 "$combo_dir"/*_summary.json 2>/dev/null | wc -l)
            echo "  $combo_name: $num_summaries checkpoints"
        fi
    done
    echo "========================================="
    echo "Pipeline finished at $(date)"
} | tee -a "$LOG_FILE"
