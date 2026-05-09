#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=selection_incremental
#SBATCH --output=logs/selection_incremental_%j.out
#SBATCH --error=logs/selection_incremental_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# Run all incremental selection strategies:
#   random, stratified, maxdist, lcmd

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

BASE_CONFIG="${BASE_CONFIG:-configs/selection/selection_exp6.yaml}"
EMBEDDING_PATH="${EMBEDDING_PATH:-$DAYCENT_ROOT/output/static_emb_32}"

STEP_SIZE="${STEP_SIZE:-5}"
MAX_POINTS="${MAX_POINTS:-50}"
SEED="${SEED:-42}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-42}"

for STRATEGY in random stratified maxdist lcmd; do
    EXTRA_ARGS=()
    EXPERIMENT_NAME="exp6_${STRATEGY}"

    if [ "$STRATEGY" = "stratified" ]; then
        EXTRA_ARGS+=(--feature-groups spatial elevation climate soil)
    fi

    if [ "$STRATEGY" = "maxdist" ] || [ "$STRATEGY" = "lcmd" ]; then
        EXTRA_ARGS+=(--embedding-path "$EMBEDDING_PATH")
    fi

    python run_ensemble_experiment.py \
        --base-config "$BASE_CONFIG" \
        --strategy "$STRATEGY" \
        --step-size "$STEP_SIZE" \
        --max-points "$MAX_POINTS" \
        --seed "$SEED" \
        --ensemble-seeds $ENSEMBLE_SEEDS \
        --experiment-name "$EXPERIMENT_NAME" \
        "${EXTRA_ARGS[@]}" \
        "$@"
done
