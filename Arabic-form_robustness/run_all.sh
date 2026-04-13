#!/bin/bash
#chmod +x run_all.sh
#./run_all.sh

MODELS=(
    "/workspace/models/ALLaM-7B-Instruct-preview"
    "/workspace/models/Fanar-1-9B-Instruct"
    "/workspace/models/Meta-Llama-3-8B-Instruct"
    "/workspace/models/Qwen2.5-3B-Instruct"
    "/workspace/models/Qwen2.5-7B-Instruct"
)

mkdir -p logs

for model in "${MODELS[@]}"; do
    name=$(basename "$model")
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    if [ ! -d "$model" ]; then
        echo "SKIP: $model not found"
        continue
    fi
    python /workspace/benchmark_boundary.py "$model" 2>&1 | tee "logs/${name}.log"
done

echo "All done."
