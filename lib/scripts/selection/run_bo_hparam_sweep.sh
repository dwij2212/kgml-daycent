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
# Run a BO framework-hyperparameter sweep at a fixed 20-point
# budget, then plot the convergence/final-best curves.
#
# Usage:
#   sbatch scripts/selection/run_bo_hparam_sweep.sh
#   sbatch scripts/selection/run_bo_hparam_sweep.sh --seeds 42 123
#   bash   scripts/selection/run_bo_hparam_sweep.sh --skip-train
# ============================================================

cd /projects/standard/kumarv/shared/dwij/daycent/lib

PYTHON_BIN=/users/6/mehta423/anaconda3/envs/wstatt/bin/python
export MPLCONFIGDIR=/tmp/matplotlib-${USER}

"${PYTHON_BIN}" run_bo_hparam_sweep.py \
    --base-config configs/selection/selection_exp6.yaml \
    --n-points 20 \
    --n-iterations 20 \
    --seeds 42 \
    --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
    --experiment-name exp6_bo_hparam_n20 \
    --score-metric yield_r2 \
    --Q-values 200 400 600 800 \
    --max-radius-values 3 5 10 15 \
    --epsilon-factor-values 0.3 \
    --fail-tol-values 20 \
    --succ-tol-values 10 \
    --shrink-tol-values 5 \
    --reuse-existing \
    "$@"
