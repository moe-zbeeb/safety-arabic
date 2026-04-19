RUN=fanar-100refusals
RUN_DIR=/workspace/output/$RUN

# evaluate base model first (step 0, optional)
python /workspace/Eval/eval.py \
    --model /workspace/models/Fanar-1-9B \
    --run-name ${RUN}__base

# evaluate each checkpoint
for ckpt in $RUN_DIR/checkpoint-*; do
    python /workspace/Eval/eval.py \
        --model "$ckpt" \
        --run-name "$RUN"
done

# evaluate final model
python /workspace/Eval/eval.py \
    --model $RUN_DIR/final-model \
    --run-name ${RUN}__final