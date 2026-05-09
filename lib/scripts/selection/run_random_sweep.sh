#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=random_sweep
#SBATCH --output=logs/random_sweep_%j.out
#SBATCH --error=logs/random_sweep_%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# Run fixed-budget random baselines across selection seeds.

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

for SEED in ${SELECTION_SEEDS:-1 2 3 4 5 6 7 8 9 10}; do
    python run_ensemble_experiment.py \
        --base-config "${BASE_CONFIG:-configs/selection/selection_exp6.yaml}" \
        --strategy random \
        --step-size "${N_POINTS:-30}" \
        --max-points "${N_POINTS:-30}" \
        --seed "$SEED" \
        --ensemble-seeds ${ENSEMBLE_SEEDS:-42} \
        --experiment-name "${EXPERIMENT_NAME:-exp6_random}" \
        "$@"
done
