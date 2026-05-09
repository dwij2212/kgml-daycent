"""
Run and plot a BO hyperparameter sweep.

This wraps run_bo_experiment.py for a fixed training budget, usually n=20,
over a grid of BO framework hyperparameters such as Q and max_radius.  Each
hyperparameter setting gets its own run label so per-iteration model outputs do
not overwrite one another.

Example:
  python run_bo_hparam_sweep.py \
      --base-config configs/selection/selection_base.yaml \
      --n-points 20 --n-iterations 20 --seeds 42 \
      --embedding-path ../output/static_emb_32 \
      --experiment-name exp6_bo_hparam_n20 \
      --Q-values 100 200 400 \
      --max-radius-values 3 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from argparse import Namespace
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'codex')}"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import selection_output_root

OUTPUT_ROOT = selection_output_root()


def _float_token(value: float) -> str:
    """Make a compact path-safe float token."""
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def make_run_label(hparams: dict[str, Any], prefix: str = "") -> str:
    """Stable label used both in paths and plot legends."""
    pieces = [
        f"q{hparams['Q']}",
        f"r{hparams['max_radius']}",
        f"eps{_float_token(hparams['epsilon_factor'])}",
        f"fail{hparams['fail_tol']}",
        f"succ{hparams['succ_tol']}",
        f"shrink{hparams['shrink_tol']}",
    ]
    label = "_".join(pieces)
    if prefix:
        return f"{prefix.strip('_')}_{label}"
    return label


def iter_hparam_grid(args: argparse.Namespace):
    keys = [
        "Q",
        "max_radius",
        "epsilon_factor",
        "fail_tol",
        "succ_tol",
        "shrink_tol",
    ]
    value_lists = [
        args.Q_values,
        args.max_radius_values,
        args.epsilon_factor_values,
        args.fail_tol_values,
        args.succ_tol_values,
        args.shrink_tol_values,
    ]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def make_bo_args(
    args: argparse.Namespace,
    hparams: dict[str, Any],
    seed: int,
    run_label: str,
) -> Namespace:
    """Build the Namespace expected by run_bo_experiment.run_bo_loop."""
    return Namespace(
        base_config=args.base_config,
        n_points=args.n_points,
        n_iterations=args.n_iterations,
        seed=seed,
        embedding_path=args.embedding_path,
        experiment_name=args.experiment_name,
        run_label=run_label,
        score_metric=args.score_metric,
        Q=hparams["Q"],
        max_radius=hparams["max_radius"],
        epsilon_factor=hparams["epsilon_factor"],
        fail_tol=hparams["fail_tol"],
        succ_tol=hparams["succ_tol"],
        shrink_tol=hparams["shrink_tol"],
        skip_train=args.skip_train,
        skip_plots=args.skip_plots,
    )


def expected_bo_csv(bo_args: Namespace) -> str:
    return os.path.join(bo_output_dir(bo_args), "bo_results.csv")


def bo_output_dir(bo_args: Namespace) -> str:
    """Mirror run_bo_experiment.bo_output_dir without importing the BO stack."""
    parts = [
        OUTPUT_ROOT,
        bo_args.experiment_name,
        "bo_graph",
    ]
    run_label = (bo_args.run_label or "").strip().strip("/")
    if run_label:
        parts.append(run_label)
    parts.append(f"sweep_s{bo_args.seed}")
    return os.path.join(*parts)


def result_is_complete(path: str, n_iterations: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        return len(pd.read_csv(path)) >= n_iterations
    except Exception:
        return False


def sweep_output_dir(args: argparse.Namespace) -> str:
    label = args.sweep_label or f"hparam_sweep_n{args.n_points}"
    return os.path.join(OUTPUT_ROOT, args.experiment_name, "bo_graph", label)


def collect_results(manifest: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for row in manifest:
        csv_path = row["bo_results_csv"]
        if not os.path.exists(csv_path):
            print(f"[warn] Missing BO results: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        if "n_iterations" in row and "iteration" in df.columns:
            df = df[df["iteration"].astype(int) < int(row["n_iterations"])]
        for key, value in row.items():
            if key not in df.columns:
                df[key] = value
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["run_label", "seed", "iteration"], kind="mergesort"
    ).reset_index(drop=True)
    return combined


def write_initial_subset_audit(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Check whether hparam settings share the same iteration-0 subset."""
    if df.empty or "subset_signature" not in df.columns:
        return pd.DataFrame()

    init = df[df["iteration"].astype(int) == 0].copy()
    cols = [
        "seed",
        "run_label",
        "Q",
        "max_radius",
        "epsilon_factor",
        "fail_tol",
        "succ_tol",
        "shrink_tol",
        "subset_signature",
        "selected_points_json",
    ]
    init = init[[c for c in cols if c in init.columns]].drop_duplicates()

    rows = []
    for seed, sub in init.groupby("seed", sort=True):
        signatures = sorted(sub["subset_signature"].dropna().unique())
        rows.append({
            "seed": seed,
            "n_hparam_runs": int(sub["run_label"].nunique()),
            "n_initial_subsets": int(len(signatures)),
            "same_initial_subset_across_hparams": len(signatures) == 1,
            "subset_signatures": ",".join(signatures),
        })

    audit = pd.DataFrame(rows)
    audit_path = os.path.join(out_dir, "bo_initial_subset_audit.csv")
    audit.to_csv(audit_path, index=False)

    init_path = os.path.join(out_dir, "bo_initial_subsets_by_run.csv")
    init.to_csv(init_path, index=False)

    if not audit.empty and audit["same_initial_subset_across_hparams"].all():
        print("\nInitial subset audit: PASS")
        print("  For each seed, every hparam setting started from the same subset.")
    else:
        print("\nInitial subset audit: WARNING")
        print("  At least one seed used different initial subsets across hparams.")

    print(f"  Audit CSV -> {audit_path}")
    print(f"  Initial subsets CSV -> {init_path}")
    return audit


def plot_sweep_results(df: pd.DataFrame, out_dir: str, score_metric: str) -> None:
    if df.empty:
        print("[warn] No results to plot.", file=sys.stderr)
        return

    labels = list(dict.fromkeys(df["run_label"].astype(str).tolist()))
    cmap = matplotlib.colormaps["tab20"].resampled(max(1, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax_best, ax_score = axes

    for idx, label in enumerate(labels):
        sub = df[df["run_label"].astype(str) == label].copy()
        grouped = (
            sub.groupby("iteration", sort=True)
            .agg(
                best_mean=("best_score", "mean"),
                best_std=("best_score", "std"),
                score_mean=("score", "mean"),
                score_std=("score", "std"),
                n_runs=("seed", "nunique"),
            )
            .reset_index()
        )
        x = grouped["iteration"].astype(int).to_numpy()
        color = cmap(idx)

        best_mean = grouped["best_mean"].astype(float).to_numpy()
        best_std = grouped["best_std"].fillna(0.0).astype(float).to_numpy()
        score_mean = grouped["score_mean"].astype(float).to_numpy()
        score_std = grouped["score_std"].fillna(0.0).astype(float).to_numpy()

        ax_best.plot(x, best_mean, "o-", color=color, linewidth=2, label=label)
        if grouped["n_runs"].max() > 1:
            ax_best.fill_between(
                x, best_mean - best_std, best_mean + best_std,
                color=color, alpha=0.15,
            )

        ax_score.plot(x, score_mean, "o-", color=color, linewidth=1.7, label=label)
        if grouped["n_runs"].max() > 1:
            ax_score.fill_between(
                x, score_mean - score_std, score_mean + score_std,
                color=color, alpha=0.12,
            )

    ax_best.set_title("Best Score So Far")
    ax_best.set_xlabel("BO iteration")
    ax_best.set_ylabel(score_metric)
    ax_best.grid(True, alpha=0.3)
    ax_best.legend(fontsize=7)

    ax_score.set_title("Per-Iteration Score")
    ax_score.set_xlabel("BO iteration")
    ax_score.set_ylabel(score_metric)
    ax_score.grid(True, alpha=0.3)
    ax_score.legend(fontsize=7)

    fig.suptitle("BO Hyperparameter Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "bo_hparam_convergence.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot -> {fig_path}")

    final_rows = (
        df.sort_values("iteration")
        .groupby(["run_label", "seed"], as_index=False)
        .tail(1)
    )
    final_summary = (
        final_rows.groupby("run_label", sort=False)
        .agg(
            final_best_mean=("best_score", "mean"),
            final_best_std=("best_score", "std"),
            n_seeds=("seed", "nunique"),
            Q=("Q", "first"),
            max_radius=("max_radius", "first"),
            epsilon_factor=("epsilon_factor", "first"),
            fail_tol=("fail_tol", "first"),
            succ_tol=("succ_tol", "first"),
            shrink_tol=("shrink_tol", "first"),
        )
        .reset_index()
    )
    final_summary["final_best_std"] = final_summary["final_best_std"].fillna(0.0)
    final_csv = os.path.join(out_dir, "bo_hparam_final_summary.csv")
    final_summary.to_csv(final_csv, index=False)
    print(f"Final summary CSV -> {final_csv}")

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(final_summary)), 5))
    x = np.arange(len(final_summary))
    ax.bar(
        x,
        final_summary["final_best_mean"].astype(float).to_numpy(),
        yerr=final_summary["final_best_std"].astype(float).to_numpy(),
        color=[cmap(i) for i in range(len(final_summary))],
        alpha=0.9,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(final_summary["run_label"], rotation=35, ha="right")
    ax.set_ylabel(score_metric)
    ax.set_title("Final Best Score by BO Hparams")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "bo_hparam_final_best.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Final-best plot -> {bar_path}")


def run_sweep(args: argparse.Namespace) -> None:
    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}", file=sys.stderr)
        sys.exit(1)

    out_dir = sweep_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    for hparams in iter_hparam_grid(args):
        run_label = make_run_label(hparams, prefix=args.run_label_prefix)
        for seed in args.seeds:
            bo_args = make_bo_args(args, hparams, seed=seed, run_label=run_label)
            csv_path = expected_bo_csv(bo_args)
            status = "pending"

            if args.plot_only:
                status = "plot_only"
            elif args.reuse_existing and result_is_complete(csv_path, args.n_iterations):
                status = "reused"
                print(f"\n[reuse] {run_label} seed={seed}: {csv_path}")
            else:
                print("\n" + "=" * 80)
                print(f"BO hparam run: {run_label} | seed={seed}")
                print("=" * 80)
                from run_bo_experiment import run_bo_loop
                run_bo_loop(bo_args)
                status = "ran"

            manifest.append({
                "experiment_name": args.experiment_name,
                "run_label": run_label,
                "seed": seed,
                "n_points": args.n_points,
                "n_iterations": args.n_iterations,
                "score_metric": args.score_metric,
                "bo_results_csv": csv_path,
                "bo_output_dir": bo_output_dir(bo_args),
                "status": status,
                **hparams,
            })

    manifest_path = os.path.join(out_dir, "bo_hparam_manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"\nManifest CSV -> {manifest_path}")

    combined = collect_results(manifest)
    if combined.empty:
        print("No BO results found to combine/plot.")
        return

    combined_path = os.path.join(out_dir, "bo_hparam_results.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Combined results CSV -> {combined_path}")

    write_initial_subset_audit(combined, out_dir)
    plot_sweep_results(combined, out_dir, args.score_metric)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a BO hparam sweep and plot the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default="exp6_bo_hparam_n20")
    parser.add_argument(
        "--sweep-label",
        type=str,
        default=None,
        help="Aggregate output folder under <experiment>/bo_graph/.",
    )
    parser.add_argument(
        "--run-label-prefix",
        type=str,
        default="",
        help="Optional prefix added to every individual hparam run label.",
    )
    parser.add_argument(
        "--score-metric",
        type=str,
        default="yield_r2",
        choices=["yield_r2", "yield_rmse", "somsc_r2", "somsc_rmse"],
    )
    parser.add_argument("--Q-values", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--max-radius-values", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--epsilon-factor-values", type=float, nargs="+", default=[0.3])
    parser.add_argument("--fail-tol-values", type=int, nargs="+", default=[20])
    parser.add_argument("--succ-tol-values", type=int, nargs="+", default=[10])
    parser.add_argument("--shrink-tol-values", type=int, nargs="+", default=[5])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip any hparam/seed run whose bo_results.csv already has enough rows.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only collect and plot expected existing results; do not run BO.",
    )

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
