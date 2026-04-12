#!/bin/bash
# ---------------------------------------------------------------------------
# Ratio × Ordering Ablation — master launcher
#
# Runs all 5 models sequentially through the full 12-dataset ablation grid.
# Each model uses all 4 GPUs for training (via accelerate).
#
# Performs pre-flight checks before starting:
#   - generates JSONL datasets if missing
#   - verifies model directories exist
#   - verifies AraSafe benchmark data and guard model
#
# Usage:
#   ./launch_all_ablation.sh                   # full run
#   ./launch_all_ablation.sh --test            # smoke test (5 training steps)
# ---------------------------------------------------------------------------
set -euo pipefail

TEST_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --test) TEST_FLAG="--test" ;;
    esac
done

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${BASE_DIR}/run_ratio_ordering_ablation.sh"
DATASETS_DIR="${BASE_DIR}/Training Corpus/ratio_ordering_ablation"
CURATE_SCRIPT="${BASE_DIR}/Training Corpus/curate_ratio_ordering_ablation.py"

MODELS=(
    "allam"
    "Fanar-1-9B"
    "Meta-Llama-3-8B-Instruct"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
)

declare -A MODEL_HF_IDS=(
    ["allam"]="humain-ai/ALLaM-7B-Instruct-preview"
    ["Fanar-1-9B"]="QCRI/Fanar-1-9B"
    ["Meta-Llama-3-8B-Instruct"]="meta-llama/Meta-Llama-3-8B-Instruct"
    ["Qwen2.5-3B-Instruct"]="Qwen/Qwen2.5-3B-Instruct"
    ["Qwen2.5-7B-Instruct"]="Qwen/Qwen2.5-7B-Instruct"
)
GUARD_HF_ID="Qwen/Qwen3Guard-Gen-4B"

download_model() {
    local name="$1"
    local hf_id="$2"
    local dest="${BASE_DIR}/models/${name}"
    mkdir -p "${BASE_DIR}/models"
    echo "  Downloading ${name} from ${hf_id} ..."
    local token_arg=""
    if [ -n "${HF_TOKEN:-}" ]; then
        token_arg="--token ${HF_TOKEN}"
    fi
    huggingface-cli download "$hf_id" --local-dir "$dest" $token_arg
}

if [ -n "$TEST_FLAG" ]; then
    MODE_LABEL="SMOKE TEST (5 training steps per run)"
else
    MODE_LABEL="FULL RUN (1 epoch per run)"
fi

echo "=============================================="
echo "RATIO × ORDERING ABLATION — ${MODE_LABEL}"
echo "=============================================="
echo "Models: ${#MODELS[@]}"
for m in "${MODELS[@]}"; do echo "  - $m"; done
echo "Datasets: 12 (3 ratios × 4 orderings)"
echo "Total runs: $(( ${#MODELS[@]} * 12 ))"
echo "=============================================="
echo ""

# ===================== PRE-FLIGHT CHECKS =====================

echo "--- Pre-flight: datasets ---"
MISSING=0
for ratio in 80ben_20ref 90ben_10ref 95ben_5ref; do
    for ordering in beneficial-first refusal-first random interleaved; do
        if [ ! -f "${DATASETS_DIR}/${ratio}/${ordering}.jsonl" ]; then
            ((MISSING++)) || true
        fi
    done
done

if [ "$MISSING" -gt 0 ]; then
    echo "  ${MISSING}/12 JSONL files missing — generating now..."
    python3 "$CURATE_SCRIPT" --force
    echo "  Done."
else
    echo "  All 12 JSONL files present."
fi

echo ""
echo "--- Pre-flight: models ---"
ALL_OK=true
for m in "${MODELS[@]}"; do
    if [ -d "${BASE_DIR}/models/${m}" ]; then
        echo "  ✓ models/${m}"
    else
        echo "  ✗ models/${m}  NOT FOUND — downloading..."
        download_model "$m" "${MODEL_HF_IDS[$m]}"
        if [ -d "${BASE_DIR}/models/${m}" ]; then
            echo "  ✓ models/${m}  downloaded"
        else
            echo "  ✗ models/${m}  download FAILED"
            ALL_OK=false
        fi
    fi
done

echo ""
echo "--- Pre-flight: AraSafe data ---"
if [ -f "${BASE_DIR}/Arasafe/arasafe_human.jsonl" ]; then
    echo "  ✓ Arasafe/arasafe_human.jsonl"
else
    echo "  ✗ Arasafe/arasafe_human.jsonl  NOT FOUND"
    ALL_OK=false
fi

echo ""
echo "--- Pre-flight: guard model ---"
if [ -d "${BASE_DIR}/models/Qwen3Guard-Gen-4B" ]; then
    echo "  ✓ models/Qwen3Guard-Gen-4B"
else
    echo "  ✗ models/Qwen3Guard-Gen-4B  NOT FOUND — downloading..."
    download_model "Qwen3Guard-Gen-4B" "$GUARD_HF_ID"
    if [ ! -d "${BASE_DIR}/models/Qwen3Guard-Gen-4B" ]; then
        echo "  ✗ models/Qwen3Guard-Gen-4B  download FAILED"
        ALL_OK=false
    fi
fi

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: one or more pre-flight checks failed. Fix the issues above and re-run."
    exit 1
fi

echo ""
echo "All pre-flight checks passed."
echo ""

# ===================== RUN MODELS =====================

OVERALL_START=$(date +%s)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTING: ${MODEL}"
    echo "=============================================="

    MODEL_START=$(date +%s)
    bash "$RUNNER" "$MODEL" $TEST_FLAG
    MODEL_END=$(date +%s)
    MODEL_SECS=$((MODEL_END - MODEL_START))
    MODEL_MINS=$((MODEL_SECS / 60))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED: ${MODEL}  (${MODEL_MINS}m ${MODEL_SECS}s)"
done

OVERALL_END=$(date +%s)
TOTAL_SECS=$((OVERALL_END - OVERALL_START))
HOURS=$((TOTAL_SECS / 3600))
MINS=$(( (TOTAL_SECS % 3600) / 60 ))

echo ""
echo "=============================================="
echo "ALL MODELS COMPLETE"
echo "Total wall time: ${HOURS}h ${MINS}m"
echo "=============================================="
echo ""

# ===================== COLLECT RESULTS =====================

echo "Collecting results..."
python3 "${BASE_DIR}/collect_ablation_results.py"
