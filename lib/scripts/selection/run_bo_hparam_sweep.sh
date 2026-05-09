#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=bo_hparam_n20
#SBATCH --output=logs/bo_hparam_n20_%j.out
#SBATCH --error=logs/bo_hparam_n20_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# Run a BO hyperparameter sweep.

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

EMBEDDING_PATH="${EMBEDDING_PATH:-$DAYCENT_ROOT/output/static_emb_32}"

python run_bo_hparam_sweep.py \
    --base-config "${BASE_CONFIG:-configs/selection/selection_exp6.yaml}" \
    --n-points "${N_POINTS:-20}" \
    --n-iterations "${N_ITERATIONS:-20}" \
    --seeds ${BO_SEEDS:-42} \
    --embedding-path "$EMBEDDING_PATH" \
    --experiment-name "${EXPERIMENT_NAME:-exp6_bo_hparam_n20}" \
    --sweep-label "${SWEEP_LABEL:-hparam_sweep_n20}" \
    --score-metric "${SCORE_METRIC:-yield_r2}" \
    --Q-values ${Q_VALUES:-200 400 600 800} \
    --max-radius-values ${MAX_RADIUS_VALUES:-3 5 10 15} \
    --epsilon-factor-values ${EPSILON_FACTOR_VALUES:-0.3} \
    --fail-tol-values ${FAIL_TOL_VALUES:-20} \
    --succ-tol-values ${SUCC_TOL_VALUES:-10} \
    --shrink-tol-values ${SHRINK_TOL_VALUES:-5} \
    --reuse-existing \
    "$@"
