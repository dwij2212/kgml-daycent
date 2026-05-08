#!/bin/bash -l
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

# ============================================================
# Run a targeted BO framework-hyperparameter sweep at a fixed 20-point
# budget, then refresh the convergence/final-best curves.
#
# The full planned grid is Q={200,400,600,800} x radius={3,5,10,15}.
# Completed runs already cover Q=200/400 for all radii and Q=600,r=3.
# These are the most informative pending runs to add next:
#   - Q=600, radius=5: nearest-radius check around the current best Q=600,r=3.
#   - Q=800, radius=3/5: tests whether a larger candidate subgraph helps while
#     keeping radius low, since max_radius often stops mattering once Q is hit.
#
# Usage:
#   sbatch scripts/selection/run_bo_hparam_sweep.sh
#   sbatch scripts/selection/run_bo_hparam_sweep.sh --seeds 42 123
#   bash   scripts/selection/run_bo_hparam_sweep.sh --skip-train
# ============================================================

cd /projects/standard/kumarv/shared/dwij/daycent/lib

PYTHON_BIN=/users/6/mehta423/anaconda3/envs/wstatt/bin/python
export MPLCONFIGDIR=/tmp/matplotlib-${USER}

COMMON_ARGS=(
    --base-config configs/selection/selection_exp6.yaml
    --n-points 20
    --n-iterations 20
    --seeds 42
    --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32
    --experiment-name exp6_bo_hparam_n20
    --score-metric yield_r2
    --epsilon-factor-values 0.3
    --fail-tol-values 20
    --succ-tol-values 10
    --shrink-tol-values 5
    --reuse-existing
)

# Explicit Q/radius pairs. Keep each pair as a one-cell grid to avoid running
# the whole Cartesian product of pending values.
TARGET_COMBOS=(
    "600 5"
    "800 3"
    "800 5"
)

for combo in "${TARGET_COMBOS[@]}"; do
    read -r Q_VALUE MAX_RADIUS_VALUE <<< "${combo}"
    echo
    echo "Running targeted BO hparam combo: Q=${Q_VALUE}, max_radius=${MAX_RADIUS_VALUE}"

    "${PYTHON_BIN}" run_bo_hparam_sweep.py \
        "${COMMON_ARGS[@]}" \
        --sweep-label hparam_sweep_pending_n20 \
        --Q-values "${Q_VALUE}" \
        --max-radius-values "${MAX_RADIUS_VALUE}" \
        "$@"
done

# Refresh the main aggregate tables/plots across the full intended grid. This
# is plot-only, so still-missing combinations are audited without being run.
"${PYTHON_BIN}" run_bo_hparam_sweep.py \
    "${COMMON_ARGS[@]}" \
    --sweep-label hparam_sweep_n20 \
    --Q-values 200 400 600 800 \
    --max-radius-values 3 5 10 15 \
    --plot-only \
    "$@"
