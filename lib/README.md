# DayCent Selection Code Walkthrough

This folder contains the emulator, data loading, and selection code for the
Bayesian selection graph experiments in the report.

The central question is: given a pool of spatial DayCent locations, which
fixed-size subset should be used to train the emulator so that it generalizes
to held-out locations, scenarios, and years?

## Quick Start

Run commands from this directory:

```bash
cd /projects/standard/kumarv/shared/dwij/daycent/lib
export DAYCENT_ROOT=/projects/standard/kumarv/shared/dwij/daycent
```

Create and activate your own Python environment before running anything:

```bash
git clone <repo-url>
cd daycent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-selection.txt
export DAYCENT_ROOT="$(pwd)"
cd lib
```

For GPU runs, install the Torch build that matches your CUDA setup before
running the selection scripts.

Then edit or copy `configs/selection/selection_exp6.yaml` so
`data.base_dir` points to your local DayCent data directory. That directory
must contain the same `inputs/` and `outputs/` layout expected by
`data.preprocessing`.

For local clones, outputs default to:

```text
daycent/output/
daycent/data/
```

You can override those without editing code:

```bash
export DAYCENT_ROOT=/path/to/daycent
export DAYCENT_OUTPUT_ROOT=/path/to/outputs
export DAYCENT_PROCESSED_ROOT=/path/to/processed-cache
```

Use a dry run first to check imports, point loading, graph construction, and
output paths without training the emulator:

```bash
python run_bo_experiment.py \
  --base-config configs/selection/selection_exp6.yaml \
  --n-points 20 \
  --n-iterations 3 \
  --seed 42 \
  --embedding-path "$DAYCENT_ROOT/output/static_emb_32" \
  --experiment-name exp6_bo_smoke \
  --score-metric yield_r2 \
  --Q 200 \
  --max-radius 3 \
  --skip-train \
  --skip-plots
```

Run the main fixed-budget GraphBO experiment:

```bash
python run_bo_experiment.py \
  --base-config configs/selection/selection_exp6.yaml \
  --n-points 20 \
  --n-iterations 20 \
  --seed 42 \
  --embedding-path "$DAYCENT_ROOT/output/static_emb_32" \
  --experiment-name exp6_bo \
  --score-metric yield_r2 \
  --Q 600 \
  --max-radius 3
```

Or use the wrapper:

```bash
bash scripts/selection/run_bo_experiment.sh
```

For Slurm:

```bash
sbatch scripts/selection/run_bo_experiment.sh
```

You can override wrapper defaults without editing the script:

```bash
N_POINTS=30 Q=800 MAX_RADIUS=5 sbatch scripts/selection/run_bo_experiment.sh
```

## What to Read First

Read the code in this order:

1. `configs/selection/selection_exp6.yaml`
   Defines the final split used in the report. The candidate pool has 417
   training locations and the held-out test set has 83 disjoint locations.

2. `run_bo_experiment.py`
   Drives the ask-tell BO loop. Each iteration asks `BOGraphStrategy` for a
   complete k-location subset, trains the emulator on that subset, evaluates
   the held-out test split, and tells the score back to BO.

3. `selection/bo_graph.py`
   Implements the GraphComBO-style search. It builds an embedding-space base
   graph over locations, lifts it into a local graph of k-subsets, fits a GP
   with a graph diffusion kernel, and uses expected improvement.

4. `run_ensemble_experiment.py`
   Runs incremental baselines with optional training-seed ensembles. Use this
   for random, stratified, MaxDist, and LCMD budget curves.

5. `selection/embedding_strategy.py`, `selection/random_strategy.py`,
   `selection/stratified_strategy.py`
   Implement the non-BO selection strategies.

6. `compare_selection_runs.py`, `compare_selection_iterations.py`,
   `run_bo_hparam_sweep.py`, `run_bo_diff_seeds.py`
   Analysis and plotting helpers for the tables and curves in the report.

The notebooks are useful history, but they are not the clean entry point for
sharing or reproducing the selection experiments.

## Folder Map

```text
lib/
  configs/selection/
    selection_exp6.yaml          final split for the report
    selection_base.yaml          older exp5 split
    point_sets/                  CSV point lists used by the configs

  selection/
    base.py                      SelectionResult and BaseStrategy
    registry.py                  strategy lookup by string name
    random_strategy.py           uniform random baseline
    stratified_strategy.py       KMeans stratification over static/site features
    embedding_strategy.py        MaxDist and LCMD embedding baselines
    bo_graph.py                  graph Bayesian optimization over subsets
    visualize.py                 selection maps and feature coverage plots

  run_selection_experiment.py    single incremental selection train/eval loop
  run_ensemble_experiment.py     incremental loop with training-seed ensembles
  run_bo_experiment.py           fixed-budget GraphBO ask-tell loop
  run_bo_hparam_sweep.py         BO Q/radius/hparam sweep wrapper
  run_bo_diff_seeds.py           BO initialization sensitivity wrapper
  compare_selection_runs.py      aggregate budget-curve results
  compare_selection_iterations.py fixed-budget BO vs random iteration plots
  build_static_embeddings.py     build avg_codes.npy/avg_codes_pids.npy

  scripts/selection/             Slurm/local wrappers for common runs
```

## Experimental Split

Use `configs/selection/selection_exp6.yaml` for the final experiments.

The important split settings are:

- Pool: `configs/selection/point_sets/pool_exp6_417.csv`
- Test: `configs/selection/point_sets/test_exp6_83.csv`
- Train scenarios: 1 through 10
- Train years: 2002 through 2020
- Test scenarios: 11, 12, 13
- Test years: 2020 through 2024

The training points are not hard-coded in the YAML. Selection scripts inject
them at runtime by replacing `data.train.points`.

## Embeddings

GraphBO, MaxDist, and LCMD need an embedding directory containing:

```text
avg_codes.npy
avg_codes_pids.npy
```

The default path used by scripts is:

```text
daycent/output/static_emb_32
```

That format is compatible with embeddings produced by the inverse-model
pipeline or by `build_static_embeddings.py`.

To rebuild static-feature embeddings:

```bash
python build_static_embeddings.py \
  --config configs/inverse/inverse_1.yaml \
  --topk 32 \
  --output-dir "$DAYCENT_ROOT/output/static_emb_32"
```

## Main Run Types

### All Incremental Strategies

Use this to run the incremental baselines in sequence:

```bash
bash scripts/selection/run_all.sh
```

This runs `random`, `stratified`, `maxdist`, and `lcmd` through
`run_ensemble_experiment.py`. Override the budget with environment variables:

```bash
STEP_SIZE=5 MAX_POINTS=50 bash scripts/selection/run_all.sh
```

### Fixed-Budget GraphBO

Use this for the report's graph-based Bayesian optimization results:

```bash
bash scripts/selection/run_bo_experiment.sh
```

Outputs go to:

```text
daycent/output/selection/<experiment>/bo_graph/<optional-run-label>/sweep_s<seed>/
```

Important files:

- `bo_results.csv`: one row per BO iteration
- `best_subset.json`: best observed subset and score
- `bo_convergence.png`: per-iteration and best-so-far curves
- `iter*_n*_s*/summary.json`: train/eval metrics for each proposed subset
- `iter*_n*_s*/selection_result.json`: selected point IDs and BO metadata

### BO Hyperparameter Sweep

Use this to test candidate graph size `Q` and local radius:

```bash
bash scripts/selection/run_bo_hparam_sweep.sh
```

By default this runs the report grid over `Q={200,400,600,800}` and
`max_radius={3,5,10,15}`. Narrow it with environment variables:

```bash
Q_VALUES="600 800" MAX_RADIUS_VALUES="3 5" bash scripts/selection/run_bo_hparam_sweep.sh
```

### BO Seed Sensitivity

Use this to test whether different initial random subsets change the final BO
result:

```bash
bash scripts/selection/run_bo_diff_seeds.sh
```

For plotting existing runs only:

```bash
bash scripts/selection/run_bo_diff_seeds.sh --plot-only
```

### Random Fixed-Budget Baseline

Use this to evaluate random subsets at a fixed budget across selection seeds:

```bash
N_POINTS=30 SELECTION_SEEDS="1 2 3 4 5 6 7 8 9 10" \
  bash scripts/selection/run_random_sweep.sh
```

### Incremental Baselines

Use this for random and stratified budget curves:

```bash
STEP_SIZE=5 MAX_POINTS=50 bash scripts/selection/run_random_and_stratified.sh
```

Use MaxDist or LCMD directly through `run_ensemble_experiment.py`:

```bash
python run_ensemble_experiment.py \
  --base-config configs/selection/selection_exp6.yaml \
  --strategy lcmd \
  --step-size 5 \
  --max-points 50 \
  --seed 42 \
  --embedding-path "$DAYCENT_ROOT/output/static_emb_32" \
  --ensemble-seeds 42 \
  --experiment-name exp6_lcmd
```

## How the GraphBO Loop Works

1. `run_bo_experiment.py` reads the base config and resolves CSV point lists.
2. It loads raw DayCent data once using all pool points, then reuses those
   DataFrames across BO iterations.
3. `BOGraphStrategy` loads location embeddings and builds an epsilon-ball base
   graph over candidate locations.
4. The first BO iteration samples one random k-subset.
5. Around the current best subset, the strategy builds a local combo-subgraph.
   Nodes are k-subsets. Edges are one-location swaps allowed by the base graph.
6. The strategy fits a GP on observed subset scores inside the local subgraph.
7. Expected improvement chooses the next unevaluated subset.
8. The emulator is trained and evaluated on that subset.
9. `strategy.tell(subset, score)` records the score and updates the trust-region
   state.

This is why BO runs are expensive: every iteration is a full train/eval run.

## Strategy Summary

- `random`: samples k points uniformly from the remaining pool.
- `stratified`: clusters static/site features with KMeans and selects cluster
  representatives.
- `maxdist`: starts from a small random warm start, then greedily picks the
  point farthest from the already selected set in embedding space.
- `lcmd`: density-aware MaxDist variant. It finds the selected center with the
  largest residual cluster mass and adds the farthest point from that cluster.
- `bo_graph`: proposes complete fixed-size subsets and can replace earlier
  locations through graph-neighbor subset swaps.

## Notes and Gotchas

- The code assumes `daycent/lib` is the working directory. The shell wrappers
  `cd` there before running Python.
- The shell wrappers call plain `python`, so activate the environment you want
  before running them.
- Output paths default to the repository's `output/` and `data/` directories,
  or to `DAYCENT_OUTPUT_ROOT` / `DAYCENT_PROCESSED_ROOT` when set.
- `run_bo_experiment.py` maximizes the selected `--score-metric`. For the report
  runs, use `yield_r2`.
- `selection_base.yaml` is the older exp5 split. Use `selection_exp6.yaml` for
  the final spatially held-out protocol.
- `--skip-train` in BO is a dry run: it uses synthetic random scores and is only
  meant to check the mechanics.
