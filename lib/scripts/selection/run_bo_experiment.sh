#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=bo_graph_exp
#SBATCH --output=logs/bo_graph_%j.out
#SBATCH --error=logs/bo_graph_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# ============================================================
# Run a Bayesian Optimisation subset-selection experiment.
#
# The BO loop proposes a complete k-subset at each iteration,
# trains an LSTM surrogate, evaluates on the test set, and
# feeds the R² score back to guide the next proposal.
#
# Usage:
#   sbatch scripts/selection/run_bo_experiment.sh
#   sbatch scripts/selection/run_bo_experiment.sh --n-iterations 20
#   bash   scripts/selection/run_bo_experiment.sh --skip-train  # dry-run
# ============================================================

cd /projects/standard/kumarv/shared/dwij/daycent/lib

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wstatt

EXTRA_ARGS="$@"

# python run_bo_experiment.py \
#     --base-config configs/selection/selection_base.yaml \
#     --n-points 200 \
#     --n-iterations 50 \
#     --seed 42 \
#     --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
#     --experiment-name exp5_bo \
#     --score-metric yield_r2 \
#     --Q 400 \
#     --max-radius 5 \
#     --epsilon-factor 0.3 \
#     --fail-tol 20 \
#     $EXTRA_ARGS


for N_POINTS in $(seq 50 50 300); do
    python run_bo_experiment.py \
        --base-config configs/selection/selection_base.yaml \
        --n-points "$N_POINTS" \
        --n-iterations 10 \
        --seed 42 \
        --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
        --experiment-name exp5_bo \
        --score-metric yield_r2 \
        --Q 400 \
        --max-radius 5 \
        --epsilon-factor 0.3 \
        --fail-tol 20 \
        $EXTRA_ARGS
done