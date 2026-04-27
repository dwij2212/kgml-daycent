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

for seed in $(seq 1 1 30); do
      python run_ensemble_experiment.py \
            --base-config configs/selection/selection_exp6.yaml \
            --strategy random --step-size 30 --max-points 30 \
            --seed "$seed" \
            --ensemble-seeds 42 \
            --experiment-name exp6_random
done

# python run_ensemble_experiment.py \
#       --base-config configs/selection/selection_exp6.yaml \
#       --strategy stratified --step-size 5 --max-points 50 \
#       --seed 42 \
#       --ensemble-seeds 42 123 456 789 1024 \
#       --experiment-name exp6_stratified

# python run_ensemble_experiment.py \
#       --base-config configs/selection/selection_exp6.yaml \
#       --strategy lcmd --step-size 5 --max-points 50 \
#       --seed 42 \
#       --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
#       --ensemble-seeds 42 123 456 789 1024 \
#       --experiment-name exp6_lcmd

# python run_ensemble_experiment.py \
#       --base-config configs/selection/selection_exp6.yaml \
#       --strategy maxdist --step-size 5 --max-points 50 \
#       --seed 42 \
#       --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
#       --ensemble-seeds 42 123 456 789 1024 \
#       --experiment-name exp6_maxdist

python run_bo_experiment.py \
      --base-config configs/selection/selection_exp6.yaml \
      --n-points 30 \
      --n-iterations 30 \
      --seed 42 \
      --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/static_emb_32 \
      --experiment-name exp6_bo \
      --score-metric yield_r2 \
      --Q 400 \
      --max-radius 10 \
      --epsilon-factor 0.3 \
      --fail-tol 20 \
      $EXTRA_ARGS
