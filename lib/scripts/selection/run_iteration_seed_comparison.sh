#!/usr/bin/env bash
#SBATCH --account=kumarv
#SBATCH --job-name=iter_seed_cmp_n20
#SBATCH --output=logs/iter_seed_cmp_n20_%j.out
#SBATCH --error=logs/iter_seed_cmp_n20_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehta423@umn.edu

# For each initial seed:
#   1. run N_ITERATIONS random fixed-budget trials
#   2. start GraphBO from random iteration 0's exact selection_result.json
#   3. compare random iterations vs GraphBO iterations

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

PYTHON="${PYTHON:-/users/6/mehta423/anaconda3/envs/wstatt/bin/python}"

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

BASE_CONFIG="${BASE_CONFIG:-configs/selection/selection_exp6.yaml}"
EMBEDDING_PATH="${EMBEDDING_PATH:-$DAYCENT_ROOT/output/static_emb_32}"

N_POINTS="${N_POINTS:-20}"
N_ITERATIONS="${N_ITERATIONS:-20}"
SEEDS="${SEEDS:-42 123 456 789}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-42}"

RANDOM_EXPERIMENT_NAME="${RANDOM_EXPERIMENT_NAME:-exp6_random_same_init_n${N_POINTS}}"
BO_EXPERIMENT_NAME="${BO_EXPERIMENT_NAME:-exp6_bo_same_init_n${N_POINTS}}"
RUN_LABEL="${RUN_LABEL:-same_initial_subset}"
COMPARE_DIR="${COMPARE_DIR:-$DAYCENT_OUTPUT_ROOT/selection/exp6_same_init_compare_n${N_POINTS}}"

SCORE_METRIC="${SCORE_METRIC:-yield_r2}"
Q="${Q:-600}"
MAX_RADIUS="${MAX_RADIUS:-3}"
EPSILON_FACTOR="${EPSILON_FACTOR:-0.3}"
FAIL_TOL="${FAIL_TOL:-20}"
SUCC_TOL="${SUCC_TOL:-10}"
SHRINK_TOL="${SHRINK_TOL:-5}"

read -r -a SEED_ARRAY <<< "$SEEDS"
read -r -a ENSEMBLE_SEED_ARRAY <<< "$ENSEMBLE_SEEDS"

EXTRA_ARGS=()
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-train)
fi
if [[ "${SKIP_PLOTS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-plots)
fi

RANDOM_DIR="$DAYCENT_OUTPUT_ROOT/selection/$RANDOM_EXPERIMENT_NAME/random"
RANDOM_ITER_DIR="$DAYCENT_OUTPUT_ROOT/selection/$RANDOM_EXPERIMENT_NAME/random_iterations"
BO_DIR="$DAYCENT_OUTPUT_ROOT/selection/$BO_EXPERIMENT_NAME/bo_graph/$RUN_LABEL"

echo "Python: $PYTHON"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "n_points=$N_POINTS n_iterations=$N_ITERATIONS"
echo "Random output: $RANDOM_DIR"
echo "Random iteration output: $RANDOM_ITER_DIR"
echo "GraphBO output: $BO_DIR"

mkdir -p "$RANDOM_ITER_DIR"

for seed in "${SEED_ARRAY[@]}"; do
    for iter in $(seq 0 $((N_ITERATIONS - 1))); do
        if [[ "$iter" == "0" ]]; then
            random_seed="$seed"
        else
            random_seed=$((seed * 100000 + iter))
        fi

        echo
        echo "=== seed $seed random iteration $iter: selection_seed=$random_seed ==="
        "$PYTHON" run_ensemble_experiment.py \
            --base-config "$BASE_CONFIG" \
            --strategy random \
            --step-size "$N_POINTS" \
            --max-points "$N_POINTS" \
            --seed "$random_seed" \
            --ensemble-seeds "${ENSEMBLE_SEED_ARRAY[@]}" \
            --experiment-name "$RANDOM_EXPERIMENT_NAME" \
            "${EXTRA_ARGS[@]}"

        actual_random_dir="$RANDOM_DIR/n${N_POINTS}_ss${random_seed}"
        iter_random_dir="$RANDOM_ITER_DIR/iter${iter}_n${N_POINTS}_s${seed}"
        mkdir -p "$iter_random_dir"
        ln -sfn "$actual_random_dir/ensemble_summary.json" "$iter_random_dir/ensemble_summary.json"
        ln -sfn "$actual_random_dir/selection_result.json" "$iter_random_dir/selection_result.json"
    done

    initial_subset_file="$RANDOM_DIR/n${N_POINTS}_ss${seed}/selection_result.json"
    if [[ ! -f "$initial_subset_file" ]]; then
        echo "Missing random selection file: $initial_subset_file" >&2
        exit 1
    fi

    echo
    echo "=== seed $seed: GraphBO from random initial subset ==="
    "$PYTHON" run_bo_experiment.py \
        --base-config "$BASE_CONFIG" \
        --n-points "$N_POINTS" \
        --n-iterations "$N_ITERATIONS" \
        --seed "$seed" \
        --embedding-path "$EMBEDDING_PATH" \
        --experiment-name "$BO_EXPERIMENT_NAME" \
        --run-label "$RUN_LABEL" \
        --score-metric "$SCORE_METRIC" \
        --initial-subset-file "$initial_subset_file" \
        --Q "$Q" \
        --max-radius "$MAX_RADIUS" \
        --epsilon-factor "$EPSILON_FACTOR" \
        --fail-tol "$FAIL_TOL" \
        --succ-tol "$SUCC_TOL" \
        --shrink-tol "$SHRINK_TOL" \
        "${EXTRA_ARGS[@]}"
done

mkdir -p "$COMPARE_DIR"
"$PYTHON" compare_selection_iterations.py \
    --n-points "$N_POINTS" \
    --random-dir "$RANDOM_ITER_DIR" \
    --bo-dir "$BO_DIR" \
    --save-dir "$COMPARE_DIR"

echo
echo "Done."
echo "Compare outputs: $COMPARE_DIR"
