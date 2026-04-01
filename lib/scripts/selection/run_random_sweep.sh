#!/bin/bash

#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=random_sweep
#SBATCH --output=logs/random_sweep_%j.out
#SBATCH --error=logs/random_sweep_%j.err
#SBATCH --time=2:00:00
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


cd /projects/standard/kumarv/shared/dwij/daycent/lib

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wstatt

EXTRA_ARGS="$@"

python run_ensemble_experiment.py \
    --base-config configs/selection/selection_base.yaml \
    --strategy random \
    --n-points 25 50 100 150 200 250 300 350 400 \
    --seed 42 \
    $EXTRA_ARGS

