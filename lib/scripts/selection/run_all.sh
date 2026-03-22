#!/bin/bash

#!/bin/bash -l
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

# ============================================================
# Run a budget sweep with the random strategy.
#
# Usage:
#   bash scripts/run_random_sweep.sh
#   bash scripts/run_random_sweep.sh --skip-train  # eval only
# ============================================================

# python run_ensemble_experiment.py \
#       --base-config configs/selection/selection_base.yaml \
#       --strategy random --step-size 50 --max-points 300 \
#       --seed 42 \
#       --ensemble-seeds 42 123 456 789 1024 \
#       --experiment-name exp5_random

# python run_ensemble_experiment.py \
#       --base-config configs/selection/selection_base.yaml \
#       --strategy stratified --step-size 50 --max-points 300 \
#       --seed 42 \
#       --ensemble-seeds 42 123 456 789 1024 \
#       --experiment-name exp5_stratified

python run_ensemble_experiment.py \
      --base-config configs/selection/selection_base.yaml \
      --strategy lcmd --step-size 50 --max-points 300 \
      --seed 42 \
      --embedding-path /users/6/mehta423/projects/daycent/output/inverse_1/eval \
      --ensemble-seeds 42 123 456 789 1024 \
      --experiment-name exp5_lcmd

# python run_ensemble_experiment.py \
#       --base-config configs/selection/selection_base.yaml \
#       --strategy maxdist --step-size 50 --max-points 300 \
#       --seed 42 \
#       --embedding-path /users/6/mehta423/projects/daycent/output/inverse_1/eval \
#       --ensemble-seeds 42 123 456 789 1024 \
#       --experiment-name exp5_mcdist