"""
Run an ensemble of models for a point-selection experiment.

For each (strategy, n_points, selection_seed) configuration:
  1. Select training points ONCE using the chosen strategy.
  2. Train N models with different training seeds (sequential, single GPU).
  3. Average metrics across ensemble members and report mean ± std.

The ensemble summary is backward-compatible with compare_selection_runs.py
(top-level ``metrics`` field uses mean values).

Usage examples
--------------
  # Ensemble of 5 random-selection runs at multiple budgets
  python run_ensemble_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy random --n-points 50 100 200 \\
      --seed 42 \\
      --ensemble-seeds 42 123 456 789 1024 \\
      --experiment-name exp5_default

  # Stratified with spatial+soil features
  python run_ensemble_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --strategy stratified --n-points 100 \\
      --seed 42 \\
      --ensemble-seeds 42 123 456 789 1024 \\
      --feature-groups spatial soil elevation \\
      --experiment-name exp5_default
"""
import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import ExperimentConfig
from selection import get_strategy, SelectionResult
from selection.visualize import plot_selected_points, plot_feature_coverage
from run_selection_experiment import (
    load_base_config,
    resolve_point_lists,
    load_metadata,
    build_experiment_config,
    train_eval_single,
)


# =========================================================================== #
#  Helpers
# =========================================================================== #

DEFAULT_ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]


def aggregate_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std across ensemble member metrics dicts."""
    result: Dict[str, Any] = {}
    tasks = list(all_metrics[0].keys())  # e.g. ["yield", "somsc"]
    for task in tasks:
        task_metrics = [m[task] for m in all_metrics]
        keys = list(task_metrics[0].keys())  # e.g. ["mse", "rmse", "mae", "r2"]
        result[task] = {}
        for k in keys:
            vals = [m[k] for m in task_metrics if m[k] is not None]
            if vals:
                result[task][k] = round(float(np.mean(vals)), 6)
                result[task][f"{k}_std"] = round(float(np.std(vals)), 6)
            else:
                result[task][k] = None
                result[task][f"{k}_std"] = None
    return result


def save_ensemble_summary(
    budget_dir: str,
    experiment_name: str,
    strategy_name: str,
    n_points: int,
    selection_seed: int,
    result: SelectionResult,
    all_metrics: List[Dict[str, Any]],
    member_run_tags: List[str],
    elapsed_s: float,
) -> None:
    """Save ensemble_summary.json (backward-compatible with compare_selection_runs.py)."""
    agg = aggregate_metrics(all_metrics)

    summary = {
        # --- top-level fields expected by compare_selection_runs.py ---
        "run_tag": f"{experiment_name}/{strategy_name}/n{n_points}_ss{selection_seed}",
        "strategy": result.strategy_name,
        "strategy_params": result.strategy_params,
        "n_train_points": result.n_points,
        "metrics": agg,          # mean values used for compat with single-run reader
        # --- ensemble-specific fields ---
        "ensemble": {
            "n_members": len(all_metrics),
            "member_run_tags": member_run_tags,
            "per_member_metrics": all_metrics,
            "aggregated": agg,
        },
        "elapsed_seconds": round(elapsed_s, 1),
    }

    path = os.path.join(budget_dir, "ensemble_summary.json")
    os.makedirs(budget_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved ensemble summary → {path}")


# =========================================================================== #
#  Core ensemble runner
# =========================================================================== #

def run_ensemble(
    base_dict: dict,
    strategy_name: str,
    n_points: int,
    selection_seed: int,
    ensemble_seeds: List[int],
    metadata: Dict[str, Any],
    experiment_name: str = "default",
    feature_groups: List[str] | None = None,
    skip_train: bool = False,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run ensemble for one (strategy, n_points, selection_seed) configuration.

    Returns a summary dict with aggregated metrics.
    """
    t0 = time.time()
    pool_points = [str(p) for p in base_dict["data"]["pool_points"]]

    # 1. Select training points ONCE
    strategy_kwargs = dict(n_points=n_points, seed=selection_seed)
    if feature_groups and strategy_name == "stratified":
        strategy_kwargs["feature_groups"] = feature_groups

    strategy = get_strategy(strategy_name, **strategy_kwargs)
    result = strategy.select(pool_points, metadata=metadata)
    print(f"\n[Ensemble] Selected {result.n_points} pts via '{strategy_name}' "
          f"(n={n_points}, ss={selection_seed})")

    # Budget-level directory (shared across all members)
    budget_tag = f"{experiment_name}/{strategy_name}/n{n_points}_ss{selection_seed}"
    budget_dir = os.path.join(
        "/users/6/mehta423/daycent/output/selection", budget_tag
    )
    os.makedirs(budget_dir, exist_ok=True)

    # Save selection result at budget level (shared)
    result.save(os.path.join(budget_dir, "selection_result.json"))

    # Plots at budget level (only once, not per member)
    if not skip_plots:
        test_points = [str(p) for p in base_dict["data"]["test"]["points"]]
        plot_selected_points(
            result=result,
            test_points=test_points,
            pool_points=pool_points,
            lookup_df=metadata["lookup_df"],
            save_path=os.path.join(budget_dir, "selection_map.png"),
        )
        plt.close("all")
        plot_feature_coverage(
            result=result,
            pool_points=pool_points,
            init_cond_df=metadata["init_cond_df"],
            save_path=os.path.join(budget_dir, "feature_coverage.png"),
        )
        plt.close("all")

    # Shared preprocessed data cache: all members reuse the same .npy files
    shared_processed_dir = os.path.join(
        "/users/6/mehta423/daycent/data/selection", budget_tag
    )

    # 2. Train each ensemble member sequentially
    all_metrics: List[Dict[str, Any]] = []
    member_run_tags: List[str] = []

    for i, train_seed in enumerate(ensemble_seeds):
        member_tag = f"{budget_tag}/member_{i}_ts{train_seed}"
        print(f"\n{'='*70}")
        print(f"  Ensemble member {i+1}/{len(ensemble_seeds)} | train_seed={train_seed}")
        print(f"  run_tag: {member_tag}")
        print(f"{'='*70}")

        out = train_eval_single(
            base_dict=base_dict,
            selected_points=result.selected_points,
            run_tag=member_tag,
            result=result,
            train_seed=train_seed,
            shared_processed_dir=shared_processed_dir,
            skip_train=skip_train,
            skip_plots=True,   # plots already saved at budget level
            pool_points=pool_points,
            metadata=None,     # skip per-member plots
        )
        all_metrics.append(out["metrics"])
        member_run_tags.append(member_tag)

    # 3. Aggregate and save ensemble summary
    elapsed = time.time() - t0
    save_ensemble_summary(
        budget_dir=budget_dir,
        experiment_name=experiment_name,
        strategy_name=strategy_name,
        n_points=n_points,
        selection_seed=selection_seed,
        result=result,
        all_metrics=all_metrics,
        member_run_tags=member_run_tags,
        elapsed_s=elapsed,
    )

    agg = aggregate_metrics(all_metrics)
    print(f"\n[Ensemble] n={n_points} DONE in {elapsed/60:.1f} min")
    print(f"  Yield  R²={agg['yield']['r2']:.4f} ± {agg['yield']['r2_std']:.4f}")
    print(f"  SOMSC  R²={agg['somsc']['r2']:.4f} ± {agg['somsc']['r2_std']:.4f}")

    return {
        "run_tag": budget_tag,
        "n_points": n_points,
        "strategy": strategy_name,
        "yield_r2_mean": agg["yield"]["r2"],
        "yield_r2_std": agg["yield"]["r2_std"],
        "yield_rmse_mean": agg["yield"]["rmse"],
        "somsc_r2_mean": agg["somsc"]["r2"],
        "somsc_r2_std": agg["somsc"]["r2_std"],
        "somsc_rmse_mean": agg["somsc"]["rmse"],
    }


# =========================================================================== #
#  Sweep and CLI
# =========================================================================== #

def run_ensemble_sweep(args) -> None:
    """Run ensembles for all requested budget sizes."""
    base_dict = load_base_config(args.base_config)
    resolve_point_lists(base_dict, os.path.dirname(os.path.abspath(args.base_config)))
    metadata = load_metadata(base_dict)

    all_rows = []
    for n in args.n_points:
        row = run_ensemble(
            base_dict=base_dict,
            strategy_name=args.strategy,
            n_points=n,
            selection_seed=args.seed,
            ensemble_seeds=args.ensemble_seeds,
            metadata=metadata,
            experiment_name=args.experiment_name,
            feature_groups=args.feature_groups,
            skip_train=args.skip_train,
            skip_plots=args.skip_plots,
        )
        all_rows.append(row)

    # Save aggregated CSV for the sweep
    sweep_dir = os.path.join(
        "/users/6/mehta423/daycent/output/selection",
        args.experiment_name,
        args.strategy,
        f"ensemble_sweep_ss{args.seed}",
    )
    os.makedirs(sweep_dir, exist_ok=True)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(sweep_dir, "ensemble_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEnsemble sweep results → {csv_path}")
    print(df.to_string(index=False))

    # Budget curve plot (mean ± std)
    if len(args.n_points) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ns = df["n_points"].values

        axes[0].plot(ns, df["yield_r2_mean"], "o-", color="dodgerblue", linewidth=2)
        axes[0].fill_between(
            ns,
            df["yield_r2_mean"] - df["yield_r2_std"],
            df["yield_r2_mean"] + df["yield_r2_std"],
            alpha=0.25, color="dodgerblue",
        )
        axes[0].set_xlabel("# Training Points")
        axes[0].set_ylabel("Yield R²")
        axes[0].set_title(f"Yield R² vs Budget ({args.strategy}, {len(args.ensemble_seeds)} members)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ns, df["somsc_r2_mean"], "o-", color="coral", linewidth=2)
        axes[1].fill_between(
            ns,
            df["somsc_r2_mean"] - df["somsc_r2_std"],
            df["somsc_r2_mean"] + df["somsc_r2_std"],
            alpha=0.25, color="coral",
        )
        axes[1].set_xlabel("# Training Points")
        axes[1].set_ylabel("SOMSC R²")
        axes[1].set_title(f"SOMSC R² vs Budget ({args.strategy}, {len(args.ensemble_seeds)} members)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(sweep_dir, "ensemble_budget_curve.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Budget curve → {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble point-selection experiments for DayCent emulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["random", "stratified"],
    )
    parser.add_argument("--n-points", type=int, nargs="+", required=True)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Selection strategy seed (controls WHICH points are chosen).",
    )
    parser.add_argument(
        "--ensemble-seeds", type=int, nargs="+",
        default=DEFAULT_ENSEMBLE_SEEDS,
        help="Training seeds for ensemble members (default: 42 123 456 789 1024).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="default",
        help="Top-level experiment folder name under output/selection/.",
    )
    parser.add_argument(
        "--feature-groups", type=str, nargs="*", default=None,
        help="Feature groups for stratified strategy.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}")
        sys.exit(1)

    run_ensemble_sweep(args)


if __name__ == "__main__":
    main()
