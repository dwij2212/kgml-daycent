"""
Run a point-selection experiment.

This script:
  1. Loads a base experiment config (with all points, fixed test set).
  2. Applies a selection strategy to choose *n* training points.
  3. Overrides the config's train split with the selected points.
  4. Trains the model (reusing the existing training pipeline).
  5. Evaluates on the fixed test set.
  6. Saves metrics, selection metadata, and visualisations.

Usage examples
--------------
  # Random selection of 50 training points
  python run_selection_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy random --n-points 50 --seed 42

  # Stratified selection using spatial + soil features
  python run_selection_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy stratified --n-points 100 --seed 42 \\
      --feature-groups spatial soil elevation

  # Sweep over multiple budget sizes
  python run_selection_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy random --n-points 25 50 100 200 400 --seed 42
"""
import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---- project imports ----
from utils.config import ExperimentConfig, SplitConfig
from selection import get_strategy, SelectionResult
from selection.visualize import plot_selected_points, plot_feature_coverage
from train_experiment import train
from eval_experiment import evaluate_experiment


# =========================================================================== #
#  Helpers
# =========================================================================== #

def load_base_config(yaml_path: str) -> dict:
    """Load the raw YAML dict (before converting to ExperimentConfig)."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def build_experiment_config(
    base_dict: dict,
    selected_points: List[str],
    run_tag: str,
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
    pool_points = data_dict.pop("pool_points", None)

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
#  Main
# =========================================================================== #

def run_single(
    base_dict: dict,
    strategy_name: str,
    n_points: int,
    seed: int,
    metadata: Dict[str, Any],
    feature_groups: List[str] | None = None,
    skip_train: bool = False,
    skip_plots: bool = False,
):
    """Execute a single selection → train → eval cycle."""
    run_tag = f"{strategy_name}_n{n_points}_s{seed}"
    print(f"\n{'#'*80}")
    print(f"  RUN: {run_tag}")
    print(f"{'#'*80}\n")
    t0 = time.time()

    # 1. Get pool points from the base config
    pool_points = [str(p) for p in base_dict["data"]["pool_points"]]
    print(f"Pool size: {len(pool_points)} points")

    # 2. Run selection strategy
    strategy_kwargs = dict(n_points=n_points, seed=seed)
    if feature_groups and strategy_name == "stratified":
        strategy_kwargs["feature_groups"] = feature_groups

    strategy = get_strategy(strategy_name, **strategy_kwargs)
    result = strategy.select(pool_points, metadata=metadata)
    print(f"Selected {result.n_points} training points via '{result.strategy_name}'")

    # 3. Build config with overridden train points
    config = build_experiment_config(base_dict, result.selected_points, run_tag)
    run_dir = config.output_dir
    os.makedirs(run_dir, exist_ok=True)

    # 4. Save selection result
    result.save(os.path.join(run_dir, "selection_result.json"))

    # 5. Visualise
    if not skip_plots:
        test_points = [str(p) for p in base_dict["data"]["test"]["points"]]
        lookup_df = metadata["lookup_df"]
        init_cond_df = metadata["init_cond_df"]

        plot_selected_points(
            result=result,
            test_points=test_points,
            pool_points=pool_points,
            lookup_df=lookup_df,
            save_path=os.path.join(run_dir, "selection_map.png"),
        )
        plt.close("all")

        plot_feature_coverage(
            result=result,
            pool_points=pool_points,
            init_cond_df=init_cond_df,
            save_path=os.path.join(run_dir, "feature_coverage.png"),
        )
        plt.close("all")

    # 6. Train
    if not skip_train:
        print("\n--- TRAINING ---")
        train(config)
    else:
        print("\n--- SKIPPING TRAINING (--skip-train) ---")

    # 7. Evaluate on test set
    print("\n--- EVALUATION ---")
    eval_out = evaluate_experiment(config, split="test", num_samples=3, save_csv=True)
    metrics = eval_out["metrics"]

    elapsed = time.time() - t0
    save_run_summary(run_dir, run_tag, result, metrics, elapsed)

    print(f"\nRUN {run_tag} DONE in {elapsed/60:.1f} min")
    print(f"  Yield  R²={metrics['yield']['r2']:.4f}  RMSE={metrics['yield']['rmse']:.4f}")
    print(f"  SOMSC  R²={metrics['somsc']['r2']:.4f}  RMSE={metrics['somsc']['rmse']:.4f}")

    return {
        "run_tag": run_tag,
        "n_points": n_points,
        "strategy": strategy_name,
        "yield_r2": metrics["yield"]["r2"],
        "yield_rmse": metrics["yield"]["rmse"],
        "somsc_r2": metrics["somsc"]["r2"],
        "somsc_rmse": metrics["somsc"]["rmse"],
    }


def run_sweep(args):
    """Run all requested budget sizes and produce a comparison plot."""
    base_dict = load_base_config(args.base_config)
    metadata = load_metadata(base_dict)

    all_results = []
    for n in args.n_points:
        row = run_single(
            base_dict=base_dict,
            strategy_name=args.strategy,
            n_points=n,
            seed=args.seed,
            metadata=metadata,
            feature_groups=args.feature_groups,
            skip_train=args.skip_train,
            skip_plots=args.skip_plots,
        )
        all_results.append(row)

    # Save aggregate results
    sweep_dir = f"/users/6/mehta423/daycent/output/selection/sweep_{args.strategy}_s{args.seed}"
    os.makedirs(sweep_dir, exist_ok=True)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved → {csv_path}")
    print(df.to_string(index=False))

    # Plot budget curves if we have multiple n_points
    if len(args.n_points) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(df["n_points"], df["yield_r2"], "o-", color="dodgerblue", linewidth=2)
        axes[0].set_xlabel("# Training Points")
        axes[0].set_ylabel("Yield R²")
        axes[0].set_title(f"Yield R² vs Budget ({args.strategy})")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["n_points"], df["somsc_r2"], "o-", color="coral", linewidth=2)
        axes[1].set_xlabel("# Training Points")
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
        description="Run point-selection experiment(s) for DayCent emulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-config", type=str, required=True,
        help="Path to the base selection YAML config.",
    )
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["random", "stratified"],
        help="Selection strategy name.",
    )
    parser.add_argument(
        "--n-points", type=int, nargs="+", required=True,
        help="Number(s) of training points to select. "
             "Pass multiple values for a budget sweep.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--feature-groups", type=str, nargs="*", default=None,
        help="Feature groups for stratified strategy "
             "(e.g. spatial soil elevation climate).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (useful if model already trained).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip visualisation plots.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}")
        sys.exit(1)

    run_sweep(args)


if __name__ == "__main__":
    main()
