#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=random_stratified
#SBATCH --output=logs/random_stratified_%j.out
#SBATCH --error=logs/random_stratified_%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# Run random and stratified incremental baselines.

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

BASE_CONFIG="${BASE_CONFIG:-configs/selection/selection_exp6.yaml}"
STEP_SIZE="${STEP_SIZE:-5}"
MAX_POINTS="${MAX_POINTS:-50}"
SEED="${SEED:-42}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-42}"

python run_ensemble_experiment.py \
    --base-config "$BASE_CONFIG" \
    --strategy random \
    --step-size "$STEP_SIZE" \
    --max-points "$MAX_POINTS" \
    --seed "$SEED" \
    --ensemble-seeds $ENSEMBLE_SEEDS \
    --experiment-name "${RANDOM_EXPERIMENT_NAME:-exp6_random}" \
    "$@"

python run_ensemble_experiment.py \
    --base-config "$BASE_CONFIG" \
    --strategy stratified \
    --step-size "$STEP_SIZE" \
    --max-points "$MAX_POINTS" \
    --seed "$SEED" \
    --ensemble-seeds $ENSEMBLE_SEEDS \
    --experiment-name "${STRATIFIED_EXPERIMENT_NAME:-exp6_stratified}" \
    --feature-groups spatial elevation climate soil \
    "$@"
