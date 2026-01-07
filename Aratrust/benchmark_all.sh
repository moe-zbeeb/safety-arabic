#!/bin/bash

# Benchmark all checkpoints and log results
OUTPUT_DIR="output"
SCRIPT="aratrust.py"
LOG_FILE="benchmark_results.log"
SUMMARY_FILE="accuracy_summary.txt"

# Clear previous results
> "$LOG_FILE"
> "$SUMMARY_FILE"

echo "==================================================================" | tee -a "$LOG_FILE"
echo "Starting AraTrust Benchmarking - $(date)" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List of checkpoints to benchmark (in order)
CHECKPOINTS=(
    "checkpoint-50"
    "checkpoint-100"
    "checkpoint-150"
    "checkpoint-200"
    "checkpoint-250"
    "checkpoint-300"
    "checkpoint-313"
)

# Loop through each checkpoint
for checkpoint in "${CHECKPOINTS[@]}"; do
    MODEL_PATH="$OUTPUT_DIR/$checkpoint"

    echo "==================================================================" | tee -a "$LOG_FILE"
    echo "Benchmarking: $checkpoint" | tee -a "$LOG_FILE"
    echo "==================================================================" | tee -a "$LOG_FILE"

    # Run the benchmark and capture output
    python "$SCRIPT" --model-path "$MODEL_PATH" 2>&1 | tee -a "$LOG_FILE"

    # Extract accuracy from the last run and append to summary
    ACCURACY=$(grep "Accuracy:" "$LOG_FILE" | tail -1 | awk '{print $2}')
    echo "$checkpoint: $ACCURACY" >> "$SUMMARY_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "Completed: $checkpoint" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "==================================================================" | tee -a "$LOG_FILE"
echo "All benchmarks completed - $(date)" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "==================================================================" | tee -a "$LOG_FILE"
echo "ACCURACY SUMMARY" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
cat "$SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Results saved to:"
echo "  - Full log: $LOG_FILE"
echo "  - Summary: $SUMMARY_FILE"
