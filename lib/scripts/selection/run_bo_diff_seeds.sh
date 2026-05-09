#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=bo_diff_seeds_n20
#SBATCH --output=logs/bo_diff_seeds_n20_%j.out
#SBATCH --error=logs/bo_diff_seeds_n20_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# Run GraphBO with fixed hparams across different BO seeds.

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

EMBEDDING_PATH="${EMBEDDING_PATH:-$DAYCENT_ROOT/output/static_emb_32}"

python run_bo_diff_seeds.py \
    --base-config "${BASE_CONFIG:-configs/selection/selection_exp6.yaml}" \
    --n-points "${N_POINTS:-20}" \
    --n-iterations "${N_ITERATIONS:-20}" \
    --seeds ${BO_SEEDS:-42 123 456 789} \
    --embedding-path "$EMBEDDING_PATH" \
    --experiment-name "${EXPERIMENT_NAME:-exp6_bo_hparam_n20}" \
    --sweep-label "${SWEEP_LABEL:-diff_seeds_q600_r3_n20}" \
    --score-metric "${SCORE_METRIC:-yield_r2}" \
    --Q "${Q:-600}" \
    --max-radius "${MAX_RADIUS:-3}" \
    --epsilon-factor "${EPSILON_FACTOR:-0.3}" \
    --fail-tol "${FAIL_TOL:-20}" \
    --succ-tol "${SUCC_TOL:-10}" \
    --shrink-tol "${SHRINK_TOL:-5}" \
    "$@"
