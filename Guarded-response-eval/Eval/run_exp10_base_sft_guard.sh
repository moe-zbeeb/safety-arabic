#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Safety-Arabic}"
MANIFEST="${MANIFEST:-$REPO_ROOT/Guarded-response-eval/Eval/base_sft_guard_manifest.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/base_sft_guard_sweep}"
GUARD_MODEL="${GUARD_MODEL:-meta-llama/Llama-Guard-3-1B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3Guard-Gen-4B}"
NUM_GPUS="${NUM_GPUS:-2}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[run_exp10_base_sft_guard] ERROR: manifest not found: $MANIFEST"
    exit 1
fi

export REPO_ROOT
export SETUPS_FILE="$MANIFEST"
export GUARD_MODES="response"
export INCLUDE_UNGUARDED="none"
export OUTPUT_DIR
export GUARD_MODEL
export JUDGE_MODEL
export NUM_GPUS

bash "$REPO_ROOT/Guarded-response-eval/Eval/run_guard_sweep.sh"
