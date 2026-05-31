#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=inverse_train
#SBATCH --output=logs/inverse_%j.out
#SBATCH --error=logs/inverse_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# ============================================================
# Train and evaluate the DayCent inverse model.
#
# Usage:
#   bash scripts/inverse/train_eval.sh [config_name]
#   bash scripts/inverse/train_eval.sh inverse_1
# ============================================================

cd /projects/standard/kumarv/shared/dwij/daycent/lib

source ~/anaconda3/etc/profile.d/conda.sh
conda activate wstatt

CONFIG_NAME="${1:-inverse_1}"

echo "=== Training inverse model: ${CONFIG_NAME} ==="
python train_inverse.py --config configs/inverse/${CONFIG_NAME}.yaml

echo "=== Evaluating inverse model: ${CONFIG_NAME} ==="
python eval_inverse.py --config configs/inverse/${CONFIG_NAME}.yaml
