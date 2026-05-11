#!/usr/bin/env bash
# run_variants.sh
#
# Evaluate all 5 models on all 5 boundary-set variants (EXP6a).
# Run from the repo root on RunPod:
#
#   cd /workspace
#   bash boundary_set_robustness/run_variants.sh 2>&1 | tee logs/exp6a_master.log
#
# Results  → /workspace/results arxive/EXP6a/Results/
# Plots    → /workspace/results arxive/EXP6a/plots/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCHMARK="${SCRIPT_DIR}/benchmark_variants.py"
VIS_SCRIPT="${SCRIPT_DIR}/vis.py"

RESULTS_DIR="${REPO_ROOT}/results arxive/EXP6a/Results"
PLOTS_DIR="${REPO_ROOT}/results arxive/EXP6a/plots"
LOG_DIR="${REPO_ROOT}/results arxive/EXP6a/logs"

MODELS=(
    "/workspace/models/ALLaM-7B-Instruct-preview"
    "/workspace/models/Fanar-1-9B-Instruct"
    "/workspace/models/Meta-Llama-3-8B-Instruct"
    "/workspace/models/Qwen2.5-3B-Instruct"
    "/workspace/models/Qwen2.5-7B-Instruct"
)

mkdir -p "${RESULTS_DIR}" "${PLOTS_DIR}" "${LOG_DIR}"

echo "======================================================="
echo "  EXP6a — Boundary-set robustness (updated ARABIZI)"
echo "  $(date)"
echo "  Models : ${#MODELS[@]}"
echo "  Results: ${RESULTS_DIR}"
echo "======================================================="

for model in "${MODELS[@]}"; do
    name="$(basename "${model}")"
    log_file="${LOG_DIR}/${name}.log"

    echo ""
    echo "-------------------------------------------------------"
    echo "  Model: ${name}"
    echo "  Log  : ${log_file}"
    echo "-------------------------------------------------------"

    python "${BENCHMARK}" "${model}" 2>&1 | tee "${log_file}"

    echo "[run_variants] Finished ${name}"
done

echo ""
echo "======================================================="
echo "  All models done. Generating visualizations..."
echo "======================================================="

python "${VIS_SCRIPT}" \
    --input  "${RESULTS_DIR}" \
    --Results "${PLOTS_DIR}"

echo ""
echo "======================================================="
echo "  EXP6a complete."
echo "  Results : ${RESULTS_DIR}"
echo "  Plots   : ${PLOTS_DIR}"
echo "  Logs    : ${LOG_DIR}"
echo "  $(date)"
echo "======================================================="
