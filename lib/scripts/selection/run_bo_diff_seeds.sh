#!/bin/bash -l
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

# ============================================================
# Run BO with fixed framework hparams across multiple BO seeds.
#
# This isolates the effect of the first random k-subset selected by
# BOGraphStrategy before the GP/EI loop takes over.
#
# Safety:
#   - Complete existing seed runs are reused.
#   - Incomplete existing seed artifacts cause the Python runner to stop
#     instead of overwriting them.
#   - Aggregate CSV/PNG reports go to a timestamped directory if the fixed
#     report directory already contains files.
#   - A preflight preview checks that these seeds produce distinct initial
#     subsets before any training starts.
#
# Default hparams use the current best completed setting:
#   Q=600, max_radius=3, n_points=20
#
# Usage:
#   sbatch scripts/selection/run_bo_diff_seeds.sh
#   sbatch scripts/selection/run_bo_diff_seeds.sh --seeds 42 123 456
#   bash   scripts/selection/run_bo_diff_seeds.sh --plot-only
# ============================================================

cd /projects/standard/kumarv/shared/dwij/daycent/lib

PYTHON_BIN=/users/6/mehta423/anaconda3/envs/wstatt/bin/python
export MPLCONFIGDIR=/tmp/matplotlib-${USER}

"${PYTHON_BIN}" run_bo_diff_seeds.py \
    --base-config configs/selection/selection_exp6.yaml \
    --n-points 20 \
    --n-iterations 20 \
    --seeds 42 123 456 789 \
    --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
    --experiment-name exp6_bo_hparam_n20 \
    --sweep-label diff_seeds_q600_r3_n20 \
    --score-metric yield_r2 \
    --Q 600 \
    --max-radius 3 \
    --epsilon-factor 0.3 \
    --fail-tol 20 \
    --succ-tol 10 \
    --shrink-tol 5 \
    --reuse-existing \
    "$@"
