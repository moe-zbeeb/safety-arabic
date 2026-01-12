#!/bin/bash

MODELS_DIR="models"
GUARD_MODEL="Qwen3Guard-Gen-4B"

for model_dir in "$MODELS_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")

        if [ "$model_name" != "$GUARD_MODEL" ]; then
            echo "========================================="
            echo "Running benchmark for: $model_name"
            echo "========================================="
            python benchmark_safety.py "$model_dir"
            echo ""
        fi
    fi
done

echo "========================================="
echo "All benchmarks completed!"
echo "========================================="
