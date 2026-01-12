#!/bin/bash

CHECKPOINTS_DIR="output"
NUM_GPUS=4

# Collect all checkpoint directories
checkpoints=()
for checkpoint_dir in "$CHECKPOINTS_DIR"/checkpoint-* "$CHECKPOINTS_DIR"/final-model; do
    if [ -d "$checkpoint_dir" ]; then
        checkpoints+=("$checkpoint_dir")
    fi
done

# Function to run benchmark on a specific GPU
run_benchmark() {
    local checkpoint_dir=$1
    local gpu_id=$2
    local checkpoint_name=$(basename "$checkpoint_dir")

    echo "========================================="
    echo "Running benchmark for: $checkpoint_name on GPU $gpu_id"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=$gpu_id python benchmark_safety.py "$checkpoint_dir"
}

# Run benchmarks in parallel batches of 4
for ((i=0; i<${#checkpoints[@]}; i+=NUM_GPUS)); do
    pids=()

    for ((j=0; j<NUM_GPUS && i+j<${#checkpoints[@]}; j++)); do
        gpu_id=$j
        checkpoint_dir="${checkpoints[$((i+j))]}"
        run_benchmark "$checkpoint_dir" "$gpu_id" &
        pids+=($!)
    done

    # Wait for current batch to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo ""
done

echo "========================================="
echo "All checkpoint benchmarks completed!"
echo "========================================="
