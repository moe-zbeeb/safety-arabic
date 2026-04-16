#!/bin/bash
set -e  # stop on any error

RUN=fanar-100refusals
CKPT_DIR=/workspace/output
EVAL=/workspace/Eval/eval.py

# evaluate base model first (step 0)
python "$EVAL" \
    --model /workspace/models/Fanar-1-9B \
    --run-name "${RUN}__base"

# evaluate each checkpoint (sorted numerically, not alphabetically)
for ckpt in $(ls -d $CKPT_DIR/checkpoint-* | sort -V); do
    echo "=== Evaluating $ckpt ==="
    python "$EVAL" \
        --model "$ckpt" \
        --run-name "$RUN"
done