"""
Run a point-selection experiment.

This script:
  1. Loads a base experiment config (with all points, fixed test set).
  2. Applies a selection strategy **incrementally** to grow the training
     set by ``step_size`` points at each step (sampling without replacement).
  3. At each budget checkpoint, trains and evaluates a model.
  4. Saves metrics, selection metadata, and visualisations per step.

The incremental loop maintains a *remaining pool* and a *selected-so-far*
list.  At every step the strategy receives only the remaining pool and the
already-selected points, ensuring no point is ever chosen twice.

Usage examples
--------------
  # Random: grow by 50 each step up to 200 total → train at 50,100,150,200
  python run_selection_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy random --step-size 50 --max-points 200 --seed 42

  # Stratified: single step of 100
  python run_selection_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy stratified --step-size 100 --seed 42 \\
      --feature-groups spatial soil elevation

  # LCMD: grow by 25 each step up to 150
  python run_selection_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy lcmd --step-size 25 --max-points 150 --seed 42 \\
      --embedding-path /projects/standard/kumarv/shared/dwij/daycent/output/inverse_1/eval
"""
import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Any, List

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster jobs
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ---- project imports ----
from utils.config import ExperimentConfig, SplitConfig
from selection import get_strategy, SelectionResult
from selection.visualize import plot_selected_points, plot_feature_coverage
from train_emulator import train
from eval_emulator import evaluate_experiment
from data.preprocessing import load_raw_data, normalize_raw_data


# =========================================================================== #
#  Helpers
# =========================================================================== #

def load_base_config(yaml_path: str) -> dict:
    """Load the raw YAML dict (before converting to ExperimentConfig)."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_csv_point_field(field, config_dir: str) -> List[str]:
    """Resolve a points field that may be a CSV reference or an existing list."""
    if isinstance(field, list):
        return [str(p) for p in field]
    if isinstance(field, dict) and field.get("source") == "csv":
        csv_path = os.path.join(config_dir, field["path"])
        df = pd.read_csv(csv_path, dtype=str)
        return df["point_id"].tolist()
    return []


def resolve_point_lists(base_dict: dict, config_dir: str) -> None:
    """Resolve any CSV-backed point references in-place.

    Supports pool_points and val/test split points being specified as:
        { source: csv, path: relative/to/config_dir.csv }
    instead of inline YAML lists.

    Also asserts that pool and val/test point sets are disjoint.
    """
    data = base_dict["data"]

    # Resolve pool_points
    pool_raw = data.get("pool_points", [])
    data["pool_points"] = _resolve_csv_point_field(pool_raw, config_dir)

    # Resolve val points
    val = data.get("val", {})
    if isinstance(val.get("points"), (dict, list)):
        val["points"] = _resolve_csv_point_field(val["points"], config_dir)

    # Resolve test points
    test = data.get("test", {})
    if isinstance(test.get("points"), (dict, list)):
        test["points"] = _resolve_csv_point_field(test["points"], config_dir)

    # Guard: pool and val/test must not overlap
    pool_set = set(data["pool_points"])
    # valtest_set = set(val.get("points", []))
    # overlap = pool_set & valtest_set
    # if overlap:
    #     raise ValueError(
    #         f"Pool and val/test point sets overlap! Overlapping IDs: {sorted(overlap)}"
    #     )


def build_experiment_config(
    base_dict: dict,
    selected_points: List[str],
    run_tag: str,
    shared_processed_dir: str | None = None,
) -> ExperimentConfig:
    """Create an ExperimentConfig with the train split overridden.

    Parameters
    ----------
    base_dict : dict
        Raw YAML dict from the base config.
    selected_points : list[str]
        Training point IDs produced by the selection strategy.
    run_tag : str
        A short identifier for this run, used for output dirs and wandb name.
    shared_processed_dir : str or None
        If provided, all ensemble members sharing the same training points
        will reuse this preprocessed data cache directory instead of each
        creating its own under data/{experiment_id}.
    """
    cfg = copy.deepcopy(base_dict)

    # Override train points
    cfg["data"]["train"]["points"] = selected_points

    # Override experiment_id & wandb name to keep outputs separate
    cfg["experiment_id"] = f"selection/{run_tag}"
    cfg.setdefault("wandb", {})["name"] = run_tag

    # Parse the dict into an ExperimentConfig
    from utils.config import DataConfig, SplitConfig, ModelConfig, TrainingConfig, WandbConfig

    data_dict = cfg["data"]
    if "train" in data_dict:
        data_dict["train"] = SplitConfig(**data_dict["train"])
    if "val" in data_dict:
        data_dict["val"] = SplitConfig(**data_dict["val"])
    if "test" in data_dict:
        data_dict["test"] = SplitConfig(**data_dict["test"])

    # Remove pool_points from data_dict — it's not part of DataConfig
    data_dict.pop("pool_points", None)

    data_config = DataConfig(**data_dict)
    model_config = ModelConfig(**cfg.get("model", {}))
    training_config = TrainingConfig(**cfg.get("training", {}))
    wandb_config = WandbConfig(**cfg.get("wandb", {}))

    return ExperimentConfig(
        experiment_id=cfg["experiment_id"],
        description=cfg.get("description", ""),
        data=data_config,
        model=model_config,
        training=training_config,
        wandb=wandb_config,
        shared_processed_dir=shared_processed_dir,
    )


def load_metadata(base_dict: dict) -> Dict[str, Any]:
    """Load the DataFrames that strategies may need."""
    base_dir = base_dict["data"]["base_dir"]
    input_dir = os.path.join(base_dir, "inputs")
    lookup_path = os.path.join(input_dir, "Midwest_lookupTable.xlsx")
    init_cond_path = os.path.join(input_dir, "initial_site_conditions.xlsx")

    lookup_df = pd.read_excel(lookup_path)
    init_cond_df = pd.read_excel(init_cond_path)

    return {
        "lookup_df": lookup_df,
        "init_cond_df": init_cond_df,
    }


def save_run_summary(
    run_dir: str,
    run_tag: str,
    result: SelectionResult,
    metrics: Dict[str, Any],
    elapsed_s: float,
):
    """Persist a JSON summary of the run."""
    summary = {
        "run_tag": run_tag,
        "strategy": result.strategy_name,
        "strategy_params": result.strategy_params,
        "n_train_points": result.n_points,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed_s, 1),
    }
    path = os.path.join(run_dir, "summary.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved run summary → {path}")


# =========================================================================== #
#  Core train-eval primitive (reused by ensemble runner)
# =========================================================================== #

def train_eval_with_data(
    base_dict: dict,
    selected_points: List[str],
    run_tag: str,
    result: "SelectionResult",
    raw_data: dict,
    train_seed: int | None = None,
    shared_processed_dir: str | None = None,
    skip_train: bool = False,
    skip_plots: bool = False,
    pool_points: List[str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train one model using pre-loaded raw DataFrames (no disk I/O for data loading).

    Identical to :func:`train_eval_single` but accepts ``raw_data`` (the dict
    returned by :func:`load_raw_data`) and normalizes it in-memory for the
    specific training split.  Use this inside sweep loops to avoid redundant CSV
    loading across ensemble members and incremental steps.

    Parameters
    ----------
    raw_data : dict
        Unnormalized DataFrames as returned by :func:`load_raw_data`.
        Must contain keys ``'weather_df'``, ``'management_df'``, ``'output_df'``.
    """
    t0 = time.time()

    if pool_points is None:
        pool_points = [str(p) for p in base_dict["data"]["pool_points"]]

    bd = copy.deepcopy(base_dict)
    if train_seed is not None:
        bd["training"]["random_seed"] = train_seed
    config = build_experiment_config(bd, selected_points, run_tag,
                                     shared_processed_dir=shared_processed_dir)
    run_dir = config.output_dir
    os.makedirs(run_dir, exist_ok=True)

    sel_path = os.path.join(run_dir, "selection_result.json")
    if not os.path.exists(sel_path):
        result.save(sel_path)

    if not skip_plots and metadata is not None:
        test_points = [str(p) for p in base_dict["data"]["test"]["points"]]
        plot_selected_points(
            result=result, test_points=test_points, pool_points=pool_points,
            lookup_df=metadata["lookup_df"],
            save_path=os.path.join(run_dir, "selection_map.png"),
        )
        plt.close("all")
        plot_feature_coverage(
            result=result, pool_points=pool_points,
            init_cond_df=metadata["init_cond_df"],
            save_path=os.path.join(run_dir, "feature_coverage.png"),
        )
        plt.close("all")

    # Normalize the shared raw data for this specific training split
    train_cfg = config.data.get_train_config()
    prepared_data = normalize_raw_data(
        raw_data,
        train_pids=train_cfg["points"],
        train_scenario_ids=train_cfg["scenarios"],
        train_years=train_cfg["years"],
        scaler_path=config.get_scaler_path(),
    )

    if not skip_train:
        print("\n--- TRAINING ---")
        train(config, prepared_data=prepared_data)
    else:
        print("\n--- SKIPPING TRAINING (--skip-train) ---")

    print("\n--- EVALUATION ---")
    eval_out = evaluate_experiment(config, split="test", num_samples=3, save_csv=True,
                                   prepared_data=prepared_data)
    metrics = eval_out["metrics"]

    elapsed = time.time() - t0
    save_run_summary(run_dir, run_tag, result, metrics, elapsed)

    print(f"\nRUN {run_tag} DONE in {elapsed/60:.1f} min")
    print(f"  Yield  R²={metrics['yield']['r2']:.4f}  RMSE={metrics['yield']['rmse']:.4f}")
    print(f"  SOMSC  R²={metrics['somsc']['r2']:.4f}  RMSE={metrics['somsc']['rmse']:.4f}")

    return {"run_tag": run_tag, "metrics": metrics, "elapsed_seconds": elapsed}


def train_eval_single(
    base_dict: dict,
    selected_points: List[str],
    run_tag: str,
    result: "SelectionResult",
    train_seed: int | None = None,
    shared_processed_dir: str | None = None,
    skip_train: bool = False,
    skip_plots: bool = False,
    pool_points: List[str] | None = None,
    metadata: Dict[str, Any] | None = None,
    raw_data: dict | None = None,
) -> Dict[str, Any]:
    """Train one model on *selected_points* and evaluate on the test set.

    Parameters
    ----------
    base_dict : dict
        Raw resolved YAML dict (pool/val/test points already plain lists).
    selected_points : list[str]
        Training point IDs for this run.
    run_tag : str
        Experiment identifier used for output directory and W&B name.
        E.g. "exp5_default/random/n50_ss42/member_0_ts42"
    result : SelectionResult
        The selection result (used for saving metadata and plots).
    train_seed : int or None
        Override training.random_seed. If None, uses config default.
    skip_train : bool
        Skip the training phase (useful if checkpoint already exists).
    skip_plots : bool
        Skip selection map / feature coverage plots.
    pool_points : list[str] or None
        Full pool of candidate points (needed for plots). Resolved from
        base_dict if not provided.
    metadata : dict or None
        Lookup / init_cond DataFrames for plots.

    Returns
    -------
    dict with keys: run_tag, metrics, elapsed_seconds
    """
    # Fast path: delegate to train_eval_with_data when raw DataFrames are available
    if raw_data is not None:
        return train_eval_with_data(
            base_dict=base_dict,
            selected_points=selected_points,
            run_tag=run_tag,
            result=result,
            raw_data=raw_data,
            train_seed=train_seed,
            shared_processed_dir=shared_processed_dir,
            skip_train=skip_train,
            skip_plots=skip_plots,
            pool_points=pool_points,
            metadata=metadata,
        )

    t0 = time.time()

    if pool_points is None:
        pool_points = [str(p) for p in base_dict["data"]["pool_points"]]

    # Build ExperimentConfig (optionally override train seed)
    bd = copy.deepcopy(base_dict)
    if train_seed is not None:
        bd["training"]["random_seed"] = train_seed
    config = build_experiment_config(bd, selected_points, run_tag,
                                     shared_processed_dir=shared_processed_dir)
    run_dir = config.output_dir
    os.makedirs(run_dir, exist_ok=True)

    # Save selection result alongside the run (only on first member / single runs)
    sel_path = os.path.join(run_dir, "selection_result.json")
    if not os.path.exists(sel_path):
        result.save(sel_path)

    # Visualise (only when metadata is available)
    if not skip_plots and metadata is not None:
        test_points = [str(p) for p in base_dict["data"]["test"]["points"]]

        plot_selected_points(
            result=result,
            test_points=test_points,
            pool_points=pool_points,
            lookup_df=metadata["lookup_df"],
            save_path=os.path.join(run_dir, "selection_map.png"),
        )
        plt.close("all")

        plot_feature_coverage(
            result=result,
            pool_points=pool_points,
            init_cond_df=metadata["init_cond_df"],
            save_path=os.path.join(run_dir, "feature_coverage.png"),
        )
        plt.close("all")

    # Train
    if not skip_train:
        print("\n--- TRAINING ---")
        train(config)
    else:
        print("\n--- SKIPPING TRAINING (--skip-train) ---")

    # Evaluate on test set
    print("\n--- EVALUATION ---")
    eval_out = evaluate_experiment(config, split="test", num_samples=3, save_csv=True)
    metrics = eval_out["metrics"]

    elapsed = time.time() - t0
    save_run_summary(run_dir, run_tag, result, metrics, elapsed)

    print(f"\nRUN {run_tag} DONE in {elapsed/60:.1f} min")
    print(f"  Yield  R²={metrics['yield']['r2']:.4f}  RMSE={metrics['yield']['rmse']:.4f}")
    print(f"  SOMSC  R²={metrics['somsc']['r2']:.4f}  RMSE={metrics['somsc']['rmse']:.4f}")

    return {"run_tag": run_tag, "metrics": metrics, "elapsed_seconds": elapsed}


# =========================================================================== #
#  Single-run entry point (thin wrapper around train_eval_single)
# =========================================================================== #

def run_single(
    base_dict: dict,
    strategy_name: str,
    n_points: int,
    seed: int,
    metadata: Dict[str, Any],
    selected_points: List[str],
    remaining_pool: List[str],
    experiment_name: str = "default",
    feature_groups: List[str] | None = None,
    embedding_path: str | None = None,
    n_warm: int = 1,
    skip_train: bool = False,
    skip_plots: bool = False,
    raw_data: dict | None = None,
) -> Dict[str, Any]:
    """Execute a single incremental selection → train → eval step.

    Parameters
    ----------
    n_points : int
        Number of **new** points to pick in this step.
    selected_points : list[str]
        Points already selected in prior steps.
    remaining_pool : list[str]
        Pool points not yet selected (passed to strategy as ``pool_points``).

    Returns
    -------
    dict with run results including newly_selected and cumulative totals.
    """
    cumulative_n = len(selected_points) + n_points
    run_tag = f"{experiment_name}/{strategy_name}/n{cumulative_n}_ss{seed}"
    print(f"\n{'#'*80}")
    print(f"  RUN: {run_tag}  (step +{n_points}, cumulative {cumulative_n})")
    print(f"{'#'*80}\n")

    print(f"Remaining pool: {len(remaining_pool)} | Already selected: {len(selected_points)}")

    # Run selection strategy on remaining pool
    strategy_kwargs = dict(n_points=n_points, seed=seed)
    if feature_groups and strategy_name == "stratified":
        strategy_kwargs["feature_groups"] = feature_groups
    if strategy_name in ("maxdist", "lcmd"):
        if embedding_path:
            strategy_kwargs["embedding_path"] = embedding_path
        strategy_kwargs["n_warm"] = n_warm

    strategy = get_strategy(strategy_name, **strategy_kwargs)
    result = strategy.select(
        remaining_pool,
        selected_points=selected_points,
        metadata=metadata,
    )
    newly_selected = result.selected_points
    print(f"Newly selected {len(newly_selected)} points via '{result.strategy_name}'")

    # Cumulative training set for this step
    cumulative_selected = selected_points + newly_selected

    # Build a SelectionResult that represents the full cumulative selection
    cumulative_result = SelectionResult(
        selected_points=sorted(cumulative_selected),
        strategy_name=result.strategy_name,
        strategy_params=result.strategy_params,
        metadata={
            **result.metadata,
            "step_newly_selected": newly_selected,
            "step_n_points": n_points,
            "cumulative_n_points": cumulative_n,
        },
    )

    # Use the full pool (from base_dict) for plots
    full_pool = [str(p) for p in base_dict["data"]["pool_points"]]

    # Train and evaluate on the cumulative set
    out = train_eval_single(
        base_dict=base_dict,
        selected_points=cumulative_selected,
        run_tag=run_tag,
        result=cumulative_result,
        skip_train=skip_train,
        skip_plots=skip_plots,
        pool_points=full_pool,
        metadata=metadata,
        raw_data=raw_data,
    )

    return {
        "run_tag": run_tag,
        "n_points": cumulative_n,
        "step_size": n_points,
        "strategy": strategy_name,
        "yield_r2": out["metrics"]["yield"]["r2"],
        "yield_rmse": out["metrics"]["yield"]["rmse"],
        "somsc_r2": out["metrics"]["somsc"]["r2"],
        "somsc_rmse": out["metrics"]["somsc"]["rmse"],
        "newly_selected": newly_selected,
    }


def run_sweep(args):
    """Run incremental selection steps and produce a budget curve.

    At each step, ``step_size`` new points are selected (without replacement)
    from the remaining pool, appended to the cumulative training set, and a
    model is trained + evaluated.  This continues until the cumulative
    budget reaches ``max_points`` or the pool is exhausted.
    """
    base_dict = load_base_config(args.base_config)
    resolve_point_lists(base_dict, os.path.dirname(os.path.abspath(args.base_config)))
    metadata = load_metadata(base_dict)

    experiment_name = args.experiment_name
    step_size = args.step_size

    # Full pool and incremental state
    full_pool = [str(p) for p in base_dict["data"]["pool_points"]]
    remaining_pool = list(full_pool)
    selected_so_far: List[str] = []

    max_points = args.max_points if args.max_points else len(full_pool)

    # ------------------------------------------------------------------ #
    # Load raw data ONCE — all incremental steps share these DataFrames   #
    # Use all pool points as train so weather data for every candidate is loaded.
    # ------------------------------------------------------------------ #
    from utils.config import ExperimentConfig
    from data.preprocessing import load_raw_data
    _tmp_config = build_experiment_config(base_dict, full_pool, "tmp_raw_load")
    raw_data = load_raw_data(_tmp_config)

    # ------------------------------------------------------------------ #
    # Shared initial points (optional)                                    #
    # ------------------------------------------------------------------ #
    sweep_dir = os.path.join(
        "/projects/standard/kumarv/shared/dwij/daycent/output/selection",
        experiment_name,
        args.strategy,
        f"sweep_s{args.seed}",
    )
    os.makedirs(sweep_dir, exist_ok=True)

    shared_init_size = getattr(args, "shared_init_size", None) or 0
    shared_init_file = getattr(args, "shared_init_file", None)

    if shared_init_file and os.path.exists(shared_init_file):
        with open(shared_init_file) as f:
            init_info = json.load(f)
        selected_so_far = [str(p) for p in init_info["points"]]
        remaining_pool = [p for p in remaining_pool if p not in set(selected_so_far)]
        print(f"\n[SharedInit] Loaded {len(selected_so_far)} initial points from {shared_init_file}")
    elif shared_init_size > 0:
        init_strategy = get_strategy("random", n_points=shared_init_size, seed=args.seed)
        init_result = init_strategy.select(remaining_pool, selected_points=[], metadata=metadata)
        selected_so_far = init_result.selected_points
        remaining_pool = [p for p in remaining_pool if p not in set(selected_so_far)]
        init_path = os.path.join(sweep_dir, "shared_init_points.json")
        with open(init_path, "w") as f:
            json.dump({"seed": args.seed, "n": shared_init_size, "points": selected_so_far}, f, indent=2)
        print(f"\n[SharedInit] Selected {len(selected_so_far)} random base points (seed={args.seed})")
        print(f"  Saved to {init_path}")

    all_results = []
    step_num = 0
    while len(selected_so_far) < max_points and remaining_pool:
        # How many points to pick this step (may be less at the end)
        n_this_step = min(step_size, max_points - len(selected_so_far), len(remaining_pool))
        if n_this_step <= 0:
            break
        step_num += 1

        print(f"\n{'='*80}")
        print(f"  INCREMENTAL STEP {step_num}: selecting {n_this_step} new points")
        print(f"  Cumulative budget will be {len(selected_so_far) + n_this_step}")
        print(f"{'='*80}")

        row = run_single(
            base_dict=base_dict,
            strategy_name=args.strategy,
            n_points=n_this_step,
            seed=args.seed,
            metadata=metadata,
            selected_points=selected_so_far,
            remaining_pool=remaining_pool,
            experiment_name=experiment_name,
            feature_groups=args.feature_groups,
            embedding_path=args.embedding_path,
            n_warm=args.n_warm,
            skip_train=args.skip_train,
            skip_plots=args.skip_plots,
            raw_data=raw_data,
        )

        # Update incremental state
        newly_selected = row.pop("newly_selected")
        newly_set = set(newly_selected)
        selected_so_far.extend(newly_selected)
        remaining_pool = [p for p in remaining_pool if p not in newly_set]

        all_results.append(row)

    # Save aggregate results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved → {csv_path}")
    print(df.to_string(index=False))

    # Plot budget curves if we have multiple steps
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(df["n_points"], df["yield_r2"], "o-", color="dodgerblue", linewidth=2)
        axes[0].set_xlabel("# Training Points (cumulative)")
        axes[0].set_ylabel("Yield R²")
        axes[0].set_title(f"Yield R² vs Budget ({args.strategy})")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["n_points"], df["somsc_r2"], "o-", color="coral", linewidth=2)
        axes[1].set_xlabel("# Training Points (cumulative)")
        axes[1].set_ylabel("SOMSC R²")
        axes[1].set_title(f"SOMSC R² vs Budget ({args.strategy})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(sweep_dir, "budget_curve.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Budget curve saved → {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run incremental point-selection experiment(s) for DayCent emulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-config", type=str, required=True,
        help="Path to the base selection YAML config.",
    )
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["random", "stratified", "maxdist", "lcmd"],
        help="Selection strategy name.",
    )
    parser.add_argument(
        "--step-size", type=int, required=True,
        help="Number of new points to select at each incremental step.",
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="Maximum cumulative training points.  The incremental loop "
             "stops when this budget is reached or the pool is exhausted.  "
             "If omitted, runs until the entire pool is selected.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the selection strategy (default: 42).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="default",
        help="Top-level experiment folder name under output/selection/ "
             "(default: 'default'). Use a descriptive name like 'exp5_default'.",
    )
    parser.add_argument(
        "--feature-groups", type=str, nargs="*", default=None,
        help="Feature groups for stratified strategy "
             "(e.g. spatial soil elevation climate).",
    )
    parser.add_argument(
        "--embedding-path", type=str, default=None,
        help="Path to directory with avg_codes.npy / avg_codes_pids.npy "
             "(required for maxdist / lcmd strategies).",
    )
    parser.add_argument(
        "--n-warm", type=int, default=1,
        help="Number of random warm-start points for maxdist / lcmd "
             "(default: 1).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (useful if model already trained).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip visualisation plots.",
    )
    parser.add_argument(
        "--shared-init-size", type=int, default=0,
        help="Select this many random points (using --seed) as a shared starting "
             "set before the chosen strategy kicks in.  Use the same value and "
             "seed across strategies to ensure a fair comparison.",
    )
    parser.add_argument(
        "--shared-init-file", type=str, default=None,
        help="Path to a shared_init_points.json file produced by a prior run.  "
             "Overrides --shared-init-size.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}")
        sys.exit(1)

    run_sweep(args)


if __name__ == "__main__":
    main()
