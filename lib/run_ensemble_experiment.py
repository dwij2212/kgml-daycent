"""
Run an ensemble of models for an incremental point-selection experiment.

For each incremental step:
  1. Select ``step_size`` new training points (without replacement) using
     the chosen strategy, appending to the cumulative training set.
  2. Train N models with different training seeds (sequential, single GPU).
  3. Average metrics across ensemble members and report mean ± std.

The selection is done ONCE (shared across ensemble members) and grows
incrementally.  The ensemble summary at each cumulative budget is
backward-compatible with compare_selection_runs.py.

Usage examples
--------------
  # Ensemble of 5 with random strategy, growing by 50 each step up to 200
  python run_ensemble_experiment.py \\
      --base-config configs/selection/selection_exp6.yaml \\
      --strategy random --step-size 50 --max-points 300 \\
      --seed 42 \\
      --ensemble-seeds 42 123 456 789 1024 \\
      --experiment-name exp5_default

  # LCMD ensemble growing by 25 each step up to 150
  python run_ensemble_experiment.py \\
      --base-config configs/selection/selection_exp6.yaml \\
      --strategy lcmd --step-size 25 --max-points 150 \\
      --seed 42 \\
      --embedding-path ../output/static_emb_32 \\
      --ensemble-seeds 42 123 456 789 1024 \\
      --experiment-name exp5_lcmd
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
from data.preprocessing import load_raw_data
from run_selection_experiment import (
    load_base_config,
    resolve_point_lists,
    load_metadata,
    build_experiment_config,
    train_eval_single,
)
from utils.paths import processed_root, selection_output_root


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
    cumulative_selected: List[str],
    step_result: SelectionResult,
    selection_seed: int,
    ensemble_seeds: List[int],
    metadata: Dict[str, Any],
    experiment_name: str = "default",
    skip_train: bool = False,
    skip_plots: bool = False,
    raw_data: dict | None = None,
) -> Dict[str, Any]:
    """Run ensemble for one cumulative budget checkpoint.

    Parameters
    ----------
    cumulative_selected : list[str]
        The full cumulative training set (all points selected so far).
    step_result : SelectionResult
        The SelectionResult for the *cumulative* selection (for saving).

    Returns a summary dict with aggregated metrics.
    """
    t0 = time.time()
    n_points = len(cumulative_selected)
    full_pool = [str(p) for p in base_dict["data"]["pool_points"]]

    print(f"\n[Ensemble] Training on {n_points} cumulative pts via '{strategy_name}' "
          f"(ss={selection_seed})")

    # Budget-level directory (shared across all members)
    budget_tag = f"{experiment_name}/{strategy_name}/n{n_points}_ss{selection_seed}"
    budget_dir = os.path.join(
        selection_output_root(), budget_tag
    )
    os.makedirs(budget_dir, exist_ok=True)

    # Save selection result at budget level (shared)
    step_result.save(os.path.join(budget_dir, "selection_result.json"))

    # Plots at budget level (only once, not per member)
    if not skip_plots:
        test_points = [str(p) for p in base_dict["data"]["test"]["points"]]
        plot_selected_points(
            result=step_result,
            test_points=test_points,
            pool_points=full_pool,
            lookup_df=metadata["lookup_df"],
            save_path=os.path.join(budget_dir, "selection_map.png"),
        )
        plt.close("all")
        plot_feature_coverage(
            result=step_result,
            pool_points=full_pool,
            init_cond_df=metadata["init_cond_df"],
            save_path=os.path.join(budget_dir, "feature_coverage.png"),
        )
        plt.close("all")

    # Shared preprocessed data cache: all members reuse the same .npy files
    shared_processed_dir = os.path.join(
        processed_root(), "selection", budget_tag
    )

    # Train each ensemble member sequentially
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
            selected_points=cumulative_selected,
            run_tag=member_tag,
            result=step_result,
            train_seed=train_seed,
            shared_processed_dir=shared_processed_dir,
            skip_train=skip_train,
            skip_plots=True,   # plots already saved at budget level
            pool_points=full_pool,
            metadata=None,     # skip per-member plots
            raw_data=raw_data,
        )
        all_metrics.append(out["metrics"])
        member_run_tags.append(member_tag)

    # Aggregate and save ensemble summary
    elapsed = time.time() - t0
    save_ensemble_summary(
        budget_dir=budget_dir,
        experiment_name=experiment_name,
        strategy_name=strategy_name,
        n_points=n_points,
        selection_seed=selection_seed,
        result=step_result,
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
    """Run incremental ensemble selection + training across a budget curve.

    At each step ``step_size`` new points are selected (without replacement)
    from the remaining pool, appended to the cumulative training set, and an
    ensemble of models is trained + evaluated.  Continues until the cumulative
    budget reaches ``max_points`` or the pool is exhausted.
    """
    base_dict = load_base_config(args.base_config)
    resolve_point_lists(base_dict, os.path.dirname(os.path.abspath(args.base_config)))
    metadata = load_metadata(base_dict)

    step_size = args.step_size

    # Full pool and incremental state
    full_pool = [str(p) for p in base_dict["data"]["pool_points"]]
    remaining_pool = list(full_pool)
    selected_so_far: List[str] = []

    max_points = args.max_points if args.max_points else len(full_pool)

    # ------------------------------------------------------------------ #
    # Load raw data ONCE — all members and steps share these DataFrames   #
    # Use all pool points as train so weather data for every candidate is loaded.
    # ------------------------------------------------------------------ #
    _tmp_config = build_experiment_config(base_dict, full_pool, "tmp_raw_load")
    raw_data = load_raw_data(_tmp_config)

    # ------------------------------------------------------------------ #
    # Shared initial points (optional)                                    #
    # ------------------------------------------------------------------ #
    sweep_dir = os.path.join(
        selection_output_root(),
        args.experiment_name,
        args.strategy,
        f"ensemble_sweep_ss{args.seed}",
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

    all_rows = []
    step_num = 0
    while len(selected_so_far) < max_points and remaining_pool:
        n_this_step = min(step_size, max_points - len(selected_so_far), len(remaining_pool))
        if n_this_step <= 0:
            break
        step_num += 1
        cumulative_n = len(selected_so_far) + n_this_step

        print(f"\n{'='*80}")
        print(f"  INCREMENTAL STEP {step_num}: selecting {n_this_step} new points")
        print(f"  Cumulative budget will be {cumulative_n}")
        print(f"  Remaining pool: {len(remaining_pool)} | Already selected: {len(selected_so_far)}")
        print(f"{'='*80}")

        # --- Selection (shared across all ensemble members) ---
        strategy_kwargs = dict(n_points=n_this_step, seed=args.seed)
        if args.feature_groups and args.strategy == "stratified":
            strategy_kwargs["feature_groups"] = args.feature_groups
        if args.strategy in ("maxdist", "lcmd"):
            if args.embedding_path:
                strategy_kwargs["embedding_path"] = args.embedding_path
            strategy_kwargs["n_warm"] = args.n_warm

        strategy = get_strategy(args.strategy, **strategy_kwargs)
        step_result = strategy.select(
            remaining_pool,
            selected_points=selected_so_far,
            metadata=metadata,
        )
        newly_selected = step_result.selected_points
        print(f"Newly selected {len(newly_selected)} points via '{step_result.strategy_name}'")

        # Cumulative training set
        cumulative_selected = selected_so_far + newly_selected

        # Build cumulative SelectionResult for saving / plotting
        cumulative_result = SelectionResult(
            selected_points=sorted(cumulative_selected),
            strategy_name=step_result.strategy_name,
            strategy_params=step_result.strategy_params,
            metadata={
                **step_result.metadata,
                "step_newly_selected": newly_selected,
                "step_n_points": n_this_step,
                "cumulative_n_points": cumulative_n,
            },
        )

        # --- Ensemble training at this budget checkpoint ---
        row = run_ensemble(
            base_dict=base_dict,
            strategy_name=args.strategy,
            cumulative_selected=cumulative_selected,
            step_result=cumulative_result,
            selection_seed=args.seed,
            ensemble_seeds=args.ensemble_seeds,
            metadata=metadata,
            experiment_name=args.experiment_name,
            skip_train=args.skip_train,
            skip_plots=args.skip_plots,
            raw_data=raw_data,
        )
        all_rows.append(row)

        # Update incremental state
        newly_set = set(newly_selected)
        selected_so_far.extend(newly_selected)
        remaining_pool = [p for p in remaining_pool if p not in newly_set]

    # Save aggregated CSV for the sweep
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(sweep_dir, "ensemble_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEnsemble sweep results → {csv_path}")
    print(df.to_string(index=False))

    # Budget curve plot (mean ± std)
    if len(all_rows) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ns = df["n_points"].values

        axes[0].plot(ns, df["yield_r2_mean"], "o-", color="dodgerblue", linewidth=2)
        axes[0].fill_between(
            ns,
            df["yield_r2_mean"] - df["yield_r2_std"],
            df["yield_r2_mean"] + df["yield_r2_std"],
            alpha=0.25, color="dodgerblue",
        )
        axes[0].set_xlabel("# Training Points (cumulative)")
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
        axes[1].set_xlabel("# Training Points (cumulative)")
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
        choices=["random", "stratified", "maxdist", "lcmd"],
    )
    parser.add_argument(
        "--step-size", type=int, required=True,
        help="Number of NEW points to select per incremental step.",
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="Stop when cumulative budget reaches this size "
             "(default: full pool).",
    )
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
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
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

    run_ensemble_sweep(args)


if __name__ == "__main__":
    main()
