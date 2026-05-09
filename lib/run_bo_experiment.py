"""
Run a Bayesian Optimisation subset-selection experiment.

This script drives the ask-tell BO loop:

  1. ``BOGraphStrategy.select()`` proposes a **complete** k-subset.
  2. A full LSTM surrogate is trained on that subset and evaluated on the
     held-out test set (expensive).
  3. The resulting R² score is fed back via ``strategy.tell()``.
  4. Repeat for ``n_iterations`` rounds.

Unlike ``run_selection_experiment.py`` (which grows the training set
incrementally), every BO iteration proposes a fresh k-subset from the
full pool.

Usage
-----
  python run_bo_experiment.py \\
      --base-config configs/selection/selection_base.yaml \\
      --n-points 50 --n-iterations 50 --seed 42 \\
      --embedding-path ../output/inverse_1/eval \\
      --experiment-name exp5_bo --score-metric yield_r2

    python run_bo_experiment.py \
      --base-config configs/selection/selection_base.yaml \
      --n-points 100 --n-iterations 50 --seed 42 \
      --embedding-path ../output/static_emb_32 \
      --experiment-name exp5_bo --score-metric yield_r2
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from typing import Dict, Any, List

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'codex')}"),
)

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- project imports ----
from run_selection_experiment import (
    load_base_config,
    resolve_point_lists,
    load_metadata,
    build_experiment_config,
    train_eval_single,
)
from selection import get_strategy, SelectionResult
from selection.visualize import plot_selected_points, plot_feature_coverage
from data.preprocessing import load_raw_data
from utils.paths import selection_output_root


def subset_signature(points: List[str]) -> str:
    """Stable short hash for a selected subset."""
    payload = "\n".join(sorted(str(p) for p in points))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _run_label(args) -> str:
    return (getattr(args, "run_label", None) or "").strip().strip("/")


def _bo_run_prefix(args) -> str:
    """Path-like run prefix under daycent/output/selection."""
    parts = [args.experiment_name, "bo_graph"]
    label = _run_label(args)
    if label:
        parts.append(label)
    return "/".join(parts)


def bo_output_dir(args) -> str:
    """Directory containing aggregate files for one BO loop."""
    parts = [
        selection_output_root(),
        args.experiment_name,
        "bo_graph",
    ]
    label = _run_label(args)
    if label:
        parts.append(label)
    parts.append(f"sweep_s{args.seed}")
    return os.path.join(*parts)


# =========================================================================== #
#  BO loop
# =========================================================================== #

def run_bo_loop(args):
    """Run the full BO experiment."""
    base_dict = load_base_config(args.base_config)
    config_dir = os.path.dirname(os.path.abspath(args.base_config))
    resolve_point_lists(base_dict, config_dir)
    metadata = load_metadata(base_dict)

    pool_points = [str(p) for p in base_dict["data"]["pool_points"]]
    test_points = [str(p) for p in base_dict["data"]["test"]["points"]]
    print(f"Pool size: {len(pool_points)}")

    # ---- Load raw data ONCE — reused across all BO iterations ----
    # Use all pool points as train so weather data for every candidate is loaded.
    _tmp_config = build_experiment_config(base_dict, pool_points, "tmp_raw_load")
    raw_data = load_raw_data(_tmp_config)

    # ---- Instantiate BO strategy ----
    strategy = get_strategy(
        "bo_graph",
        n_points=args.n_points,
        seed=args.seed,
        embedding_path=args.embedding_path,
        epsilon_factor=args.epsilon_factor,
        Q=args.Q,
        max_radius=args.max_radius,
        fail_tol=args.fail_tol,
        succ_tol=args.succ_tol,
        shrink_tol=args.shrink_tol,
    )

    # ---- Output directory ----
    run_label = _run_label(args)
    run_prefix = _bo_run_prefix(args)
    out_dir = bo_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    run_args_path = os.path.join(out_dir, "bo_run_args.json")
    with open(run_args_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ---- BO loop ----
    all_results: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_subset = None
    first_subset = None

    for i in range(args.n_iterations):
        print(f"\n{'='*80}")
        print(f"  BO ITERATION {i + 1} / {args.n_iterations}")
        print(f"{'='*80}\n")

        # Ask
        result = strategy.select(pool_points, metadata=metadata)
        subset = result.selected_points
        if first_subset is None:
            first_subset = list(subset)

        run_tag = f"{run_prefix}/iter{i}_n{args.n_points}_s{args.seed}"

        if args.skip_train:
            # Save selection output and plots even in dry-run mode.
            run_cfg = build_experiment_config(base_dict, subset, run_tag)
            run_dir = run_cfg.output_dir
            sel_path = os.path.join(run_dir, "selection_result.json")
            if not os.path.exists(sel_path):
                result.save(sel_path)

            if not args.skip_plots:
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

            # Dry-run: use random score for testing the BO loop
            import numpy as np
            score = float(np.random.RandomState(args.seed + i).uniform(0.5, 0.95))
            metrics = {
                "yield": {"r2": score, "rmse": 0.0},
                "somsc": {"r2": score * 0.9, "rmse": 0.0},
            }
            elapsed = 0.0
            print(f"  [DRY-RUN] Random score = {score:.4f}")
        else:
            # Expensive: train + evaluate (data already loaded, no CSV I/O)
            out = train_eval_single(
                base_dict=base_dict,
                selected_points=subset,
                run_tag=run_tag,
                result=result,
                skip_plots=args.skip_plots,
                pool_points=pool_points,
                metadata=metadata,
                raw_data=raw_data,
            )
            metrics = out["metrics"]
            elapsed = out.get("elapsed_seconds", 0.0)

        # Parse score from metrics
        metric_group, metric_name = args.score_metric.split("_", 1)
        score = metrics[metric_group][metric_name]

        # Tell
        strategy.tell(subset, score)

        if score > best_score:
            best_score = score
            best_subset = subset

        row = {
            "experiment_name": args.experiment_name,
            "run_label": run_label,
            "seed": args.seed,
            "n_points": args.n_points,
            "Q": args.Q,
            "max_radius": args.max_radius,
            "epsilon_factor": args.epsilon_factor,
            "fail_tol": args.fail_tol,
            "succ_tol": args.succ_tol,
            "shrink_tol": args.shrink_tol,
            "iteration": i,
            "score": score,
            "best_score": best_score,
            "subset_signature": subset_signature(subset),
            "selected_points_json": json.dumps(sorted(str(p) for p in subset)),
            "ei_value": result.metadata.get("ei_value"),
            "gp_lengthscale": result.metadata.get("gp_lengthscale"),
            "gp_noise_var": result.metadata.get("gp_noise_var"),
            "n_local_obs": result.metadata.get("n_local_observations"),
            "subgraph_size": result.metadata.get("subgraph_size"),
            "n_restarts": result.metadata.get("n_restarts"),
            "n_total_obs": result.metadata.get("n_total_observations"),
            "elapsed_s": elapsed,
        }
        all_results.append(row)

        print(
            f"\n  BO iter {i}: {args.score_metric}={score:.4f}, "
            f"best={best_score:.4f}"
        )

    # ---- Save results ----
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, "bo_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBO results saved → {csv_path}")
    print(df.to_string(index=False))

    # Save best subset
    best_info = {
        "best_score": best_score,
        "score_metric": args.score_metric,
        "best_subset": best_subset,
        "first_subset": first_subset,
        "first_subset_signature": (
            subset_signature(first_subset) if first_subset is not None else None
        ),
        "n_iterations": args.n_iterations,
        "n_points": args.n_points,
        "seed": args.seed,
        "run_label": run_label,
        "Q": args.Q,
        "max_radius": args.max_radius,
        "epsilon_factor": args.epsilon_factor,
        "fail_tol": args.fail_tol,
        "succ_tol": args.succ_tol,
        "shrink_tol": args.shrink_tol,
    }
    best_path = os.path.join(out_dir, "best_subset.json")
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2, default=str)
    print(f"Best subset saved → {best_path}")

    # ---- Plot convergence curve ----
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iters = df["iteration"]

        axes[0].plot(iters, df["score"], "o-", color="dodgerblue",
                     alpha=0.5, label="Per-iteration")
        axes[0].plot(iters, df["best_score"], "s-", color="crimson",
                     linewidth=2, label="Best so far")
        axes[0].set_xlabel("BO Iteration")
        axes[0].set_ylabel(args.score_metric)
        axes[0].set_title(f"BO Convergence (k={args.n_points})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        ei_vals = df["ei_value"].dropna()
        if len(ei_vals) > 0:
            axes[1].plot(ei_vals.index, ei_vals.values, "o-",
                         color="forestgreen")
        axes[1].set_xlabel("BO Iteration")
        axes[1].set_ylabel("Expected Improvement")
        axes[1].set_title("Acquisition Value")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "bo_convergence.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Convergence plot saved → {fig_path}")


# =========================================================================== #
#  CLI
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Run a BO subset-selection experiment for DayCent emulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-config", type=str, required=True,
        help="Path to the base selection YAML config.",
    )
    parser.add_argument(
        "--n-points", type=int, required=True,
        help="Subset size k (number of training locations per BO iteration).",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=50,
        help="Number of BO iterations (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--embedding-path", type=str, required=True,
        help="Directory with avg_codes.npy / avg_codes_pids.npy.",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="exp5_bo",
        help="Top-level experiment folder name (default: exp5_bo).",
    )
    parser.add_argument(
        "--run-label", type=str, default="",
        help=(
            "Optional subfolder under <experiment-name>/bo_graph. "
            "Useful for hparam sweeps so runs do not overwrite one another."
        ),
    )
    parser.add_argument(
        "--score-metric", type=str, default="yield_r2",
        choices=["yield_r2", "yield_rmse", "somsc_r2", "somsc_rmse"],
        help="Metric to optimise (default: yield_r2).",
    )
    # --- BO hyperparameters ---
    parser.add_argument(
        "--Q", type=int, default=200,
        help="Combo-subgraph size (default: 200).",
    )
    parser.add_argument(
        "--max-radius", type=int, default=3,
        help="Max BFS hops for subgraph construction (default: 3).",
    )
    parser.add_argument(
        "--epsilon-factor", type=float, default=0.3,
        help="Base-graph threshold = mean_dist * factor (default: 0.3).",
    )
    parser.add_argument(
        "--fail-tol", type=int, default=20,
        help="Failures before restart (default: 20).",
    )
    parser.add_argument(
        "--succ-tol", type=int, default=10,
        help="Successes before expanding Q (default: 10).",
    )
    parser.add_argument(
        "--shrink-tol", type=int, default=5,
        help="Failures between shrink events (default: 5).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Dry-run: skip training, use random scores (for testing BO loop).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip selection visualisation plots.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}")
        sys.exit(1)

    run_bo_loop(args)


if __name__ == "__main__":
    main()
