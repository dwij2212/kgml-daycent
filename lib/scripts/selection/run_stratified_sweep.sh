#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=stratified_sweep
#SBATCH --output=logs/stratified_sweep_%j.out
#SBATCH --error=logs/stratified_sweep_%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# Run the stratified incremental baseline.

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

python run_ensemble_experiment.py \
    --base-config "${BASE_CONFIG:-configs/selection/selection_exp6.yaml}" \
    --strategy stratified \
    --step-size "${STEP_SIZE:-5}" \
    --max-points "${MAX_POINTS:-50}" \
    --seed "${SEED:-42}" \
    --ensemble-seeds ${ENSEMBLE_SEEDS:-42} \
    --experiment-name "${EXPERIMENT_NAME:-exp6_stratified}" \
    --feature-groups spatial elevation climate soil \
    "$@"
