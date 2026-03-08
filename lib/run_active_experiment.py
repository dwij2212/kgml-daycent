"""
Active point selection experiment.

Iteratively builds a training set by selecting the most informative points
using a model-dependent acquisition function. Each round:
  1. Train model on the current labeled set.
  2. Score remaining pool points with the acquisition function.
  3. Add the top-B points to the labeled set.
  4. Evaluate on the fixed test set.

This produces a learning curve (metrics vs. budget) where each step's
selection is *informed by the model*, unlike one-shot random / stratified
selection. Directly comparable to one-shot experiments at the same budgets.

Usage examples
--------------
  # Active loop with random acquisition (placeholder, for testing the loop)
  python run_active_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --acquisition random \\
      --seed-size 25 --batch-size 25 --n-rounds 8 \\
      --seed 42 \\
      --experiment-name exp5_default

  # Active loop with uncertainty acquisition (once implemented)
  python run_active_experiment.py \\
      --base-config configs/selection_base.yaml \\
      --acquisition uncertainty \\
      --seed-size 25 --batch-size 25 --n-rounds 8 \\
      --seed 42 \\
      --experiment-name exp5_default

The above produces budgets: 25, 50, 75, ..., 200 — comparable to one-shot
sweeps at the same budget sizes.
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

from selection.acquisition import get_acquisition, ACQUISITION_REGISTRY
from selection.base import SelectionResult
from selection.random_strategy import RandomStrategy
from run_selection_experiment import (
    load_base_config,
    resolve_point_lists,
    load_metadata,
    build_experiment_config,
)
from train_experiment import train
from eval_experiment import evaluate_experiment


# =========================================================================== #
#  Helpers
# =========================================================================== #

def _make_dummy_result(selected_points: List[str], strategy_name: str,
                       strategy_params: Dict[str, Any]) -> SelectionResult:
    """Wrap a list of selected points in a SelectionResult for logging."""
    return SelectionResult(
        selected_points=selected_points,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        metadata={},
    )


def save_round_summary(
    round_dir: str,
    round_i: int,
    n_total: int,
    new_points: List[str],
    current_train: List[str],
    metrics: Dict[str, Any],
    acquisition_scores: Dict[str, float] | None,
    elapsed_s: float,
) -> None:
    """Save per-round results to round_dir/summary.json."""
    summary = {
        "round": round_i,
        "n_train_points": n_total,
        "new_points_added": new_points,
        "current_train_points": current_train,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed_s, 1),
    }
    os.makedirs(round_dir, exist_ok=True)
    with open(os.path.join(round_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if acquisition_scores is not None:
        with open(os.path.join(round_dir, "acquisition_scores.json"), "w") as f:
            json.dump(acquisition_scores, f, indent=2)

    print(f"  Saved round {round_i} summary → {round_dir}/summary.json")


def save_active_summary(
    active_dir: str,
    experiment_name: str,
    acquisition_name: str,
    seed_size: int,
    batch_size: int,
    seed: int,
    round_rows: List[Dict[str, Any]],
) -> None:
    """Save active_summary.json: full metrics-vs-budget curve."""
    summary = {
        "experiment_name": experiment_name,
        "acquisition": acquisition_name,
        "seed_size": seed_size,
        "batch_size": batch_size,
        "seed": seed,
        "n_rounds": len(round_rows),
        "rounds": round_rows,
    }
    path = os.path.join(active_dir, "active_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nActive summary → {path}")


# =========================================================================== #
#  Core active loop
# =========================================================================== #

def run_active(
    base_dict: dict,
    acquisition_name: str,
    seed_size: int,
    batch_size: int,
    n_rounds: int,
    seed: int,
    metadata: Dict[str, Any],
    experiment_name: str = "default",
    skip_train: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the active selection loop.

    Returns list of per-round result dicts (n_points, metrics).
    """
    pool_points = [str(p) for p in base_dict["data"]["pool_points"]]
    acquisition = get_acquisition(acquisition_name, seed=seed)

    # Active run output dir
    active_tag = (
        f"{experiment_name}/active_{acquisition_name}/"
        f"seed{seed_size}_batch{batch_size}_ss{seed}"
    )
    active_dir = os.path.join(
        "/users/6/mehta423/daycent/output/selection", active_tag
    )
    os.makedirs(active_dir, exist_ok=True)

    # Round 0: random seed set for a warm start
    rng_strategy = RandomStrategy(n_points=seed_size, seed=seed)
    seed_result = rng_strategy.select(pool_points, metadata=metadata)
    current_train: List[str] = list(seed_result.selected_points)
    remaining_pool: List[str] = [p for p in pool_points if p not in set(current_train)]

    print(f"\n[Active] Starting with {len(current_train)} seed points "
          f"({acquisition_name} acquisition, {n_rounds} rounds, batch={batch_size})")

    round_rows: List[Dict[str, Any]] = []

    for round_i in range(n_rounds):
        t0 = time.time()
        n_total = len(current_train)
        round_dir = os.path.join(active_dir, f"round_{round_i}")

        print(f"\n{'='*70}")
        print(f"  [Active] Round {round_i} | n_train={n_total} | "
              f"remaining_pool={len(remaining_pool)}")
        print(f"{'='*70}")

        # 1. Build run tag and config
        run_tag = f"{active_tag}/round_{round_i}"
        bd = copy.deepcopy(base_dict)
        config = build_experiment_config(bd, current_train, run_tag)

        # 2. Train
        if not skip_train:
            print("--- TRAINING ---")
            train(config)
        else:
            print("--- SKIPPING TRAINING ---")

        # 3. Evaluate on test set
        print("--- EVALUATION ---")
        eval_out = evaluate_experiment(config, split="test", num_samples=3, save_csv=True)
        metrics = eval_out["metrics"]

        # 4. Score remaining pool with acquisition function
        #    (uses trained model if acquisition needs it)
        model_path = config.get_model_path()
        print(f"--- SCORING {len(remaining_pool)} remaining pool points ---")
        scores = acquisition.score(
            model_path=model_path,
            candidate_points=remaining_pool,
            current_train=current_train,
            metadata=metadata,
        )

        # 5. Select top-B points
        new_points = acquisition.select_top(scores, n=min(batch_size, len(remaining_pool)))
        new_points_set = set(new_points)
        current_train = current_train + new_points
        remaining_pool = [p for p in remaining_pool if p not in new_points_set]

        elapsed = time.time() - t0

        print(f"  Added {len(new_points)} new points. "
              f"Total train: {len(current_train)}, remaining: {len(remaining_pool)}")
        print(f"  Yield R²={metrics['yield']['r2']:.4f}  "
              f"SOMSC R²={metrics['somsc']['r2']:.4f}")

        row = {
            "round": round_i,
            "n_train_points": n_total,
            "yield_r2": metrics["yield"]["r2"],
            "yield_rmse": metrics["yield"]["rmse"],
            "somsc_r2": metrics["somsc"]["r2"],
            "somsc_rmse": metrics["somsc"]["rmse"],
            "elapsed_seconds": round(elapsed, 1),
        }
        round_rows.append(row)

        # Save per-round files
        save_round_summary(
            round_dir=round_dir,
            round_i=round_i,
            n_total=n_total,
            new_points=new_points,
            current_train=current_train,
            metrics=metrics,
            acquisition_scores=scores,
            elapsed_s=elapsed,
        )

        if not remaining_pool:
            print("Pool exhausted — stopping early.")
            break

    # Save aggregate learning-curve summary
    save_active_summary(
        active_dir=active_dir,
        experiment_name=experiment_name,
        acquisition_name=acquisition_name,
        seed_size=seed_size,
        batch_size=batch_size,
        seed=seed,
        round_rows=round_rows,
    )

    # Budget-curve plot
    df = pd.DataFrame(round_rows)
    if len(df) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(df["n_train_points"], df["yield_r2"], "o-", color="dodgerblue", linewidth=2)
        axes[0].set_xlabel("# Training Points")
        axes[0].set_ylabel("Yield R²")
        axes[0].set_title(f"Active ({acquisition_name}): Yield R²")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["n_train_points"], df["somsc_r2"], "o-", color="coral", linewidth=2)
        axes[1].set_xlabel("# Training Points")
        axes[1].set_ylabel("SOMSC R²")
        axes[1].set_title(f"Active ({acquisition_name}): SOMSC R²")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(active_dir, "active_budget_curve.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Budget curve → {fig_path}")

    return round_rows


# =========================================================================== #
#  CLI
# =========================================================================== #

def main():
    available_acquisitions = sorted(ACQUISITION_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Active point selection experiment for DayCent emulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument(
        "--acquisition", type=str, required=True,
        choices=available_acquisitions,
        help=f"Acquisition function. Available: {available_acquisitions}",
    )
    parser.add_argument(
        "--seed-size", type=int, default=25,
        help="Number of random seed points for round 0 (default: 25).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=25,
        help="Number of points to add per round (default: 25).",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=8,
        help="Number of active selection rounds (default: 8).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the initial random seed set and acquisition (default: 42).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="default",
        help="Top-level experiment folder name under output/selection/.",
    )
    parser.add_argument("--skip-train", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}")
        sys.exit(1)

    base_dict = load_base_config(args.base_config)
    resolve_point_lists(base_dict, os.path.dirname(os.path.abspath(args.base_config)))
    metadata = load_metadata(base_dict)

    run_active(
        base_dict=base_dict,
        acquisition_name=args.acquisition,
        seed_size=args.seed_size,
        batch_size=args.batch_size,
        n_rounds=args.n_rounds,
        seed=args.seed,
        metadata=metadata,
        experiment_name=args.experiment_name,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
