"""
Compare fixed-budget selection strategy performance across iterations.

This script is intentionally strategy-aware for now:
- random: parses runs like n20_ss1/ensemble_summary.json
- bo_graph: parses runs like iter10_n20_s42/summary.json

Once run layouts are unified, these parser functions can be replaced by one
shared parser.

Usage:
    python compare_selection_iterations.py \
        --n-points 30 \
        --random-dir ../output/selection/exp6_random/random \
        --bo-dir ../output/selection/exp6_bo/bo_graph
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'codex')}"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.paths import selection_output_root


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric(metrics: dict, target: str, metric_name: str) -> float:
    return _safe_float(metrics.get(target, {}).get(metric_name))


def collect_random_iterations(random_dir: str, n_points: int | None = None) -> pd.DataFrame:
    """Parse random strategy runs.

    Supported layouts:
      - n20_ss42/ensemble_summary.json
      - iter0_n20_s42/ensemble_summary.json
    """
    rows = []
    pat = re.compile(r"^n(?P<n>\d+)_ss(?P<seed>\d+)$")
    iter_pat = re.compile(r"^iter(?P<iter>\d+)_n(?P<n>\d+)_s(?P<seed>\d+)$")

    iter_pattern = os.path.join(random_dir, "**", "iter*_n*_s*", "ensemble_summary.json")
    for path in sorted(glob.glob(iter_pattern, recursive=True)):
        run_dir = os.path.basename(os.path.dirname(path))
        m = iter_pat.match(run_dir)
        if not m:
            print(f"[warn] random: skipping unrecognized folder name: {run_dir}", file=sys.stderr)
            continue

        with open(path) as f:
            d = json.load(f)

        n_from_dir = int(m.group("n"))
        n_train = int(d.get("n_train_points", n_from_dir))
        if n_points is not None and n_train != n_points:
            continue

        metrics = d.get("metrics") or d.get("ensemble", {}).get("aggregated", {})
        strategy_params = d.get("strategy_params", {})

        rows.append({
            "strategy": "random",
            "run_tag": d.get("run_tag", run_dir),
            "iteration": int(m.group("iter")),
            "seed": int(m.group("seed")),
            "selection_seed": strategy_params.get("seed"),
            "n_points": n_train,
            "yield_r2": _metric(metrics, "yield", "r2"),
            "yield_rmse": _metric(metrics, "yield", "rmse"),
            "somsc_r2": _metric(metrics, "somsc", "r2"),
            "somsc_rmse": _metric(metrics, "somsc", "rmse"),
            "elapsed_s": _safe_float(d.get("elapsed_seconds")),
            "source_path": path,
        })

    pattern = os.path.join(random_dir, "n*_ss*", "ensemble_summary.json")
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.basename(os.path.dirname(path))
        m = pat.match(run_dir)
        if not m:
            print(f"[warn] random: skipping unrecognized folder name: {run_dir}", file=sys.stderr)
            continue

        with open(path) as f:
            d = json.load(f)

        n_from_dir = int(m.group("n"))
        n_train = int(d.get("n_train_points", n_from_dir))
        if n_points is not None and n_train != n_points:
            continue

        metrics = d.get("metrics") or d.get("ensemble", {}).get("aggregated", {})
        seed = int(m.group("seed"))

        rows.append({
            "strategy": "random",
            "run_tag": d.get("run_tag", run_dir),
            "iteration": seed,
            "seed": seed,
            "selection_seed": seed,
            "n_points": n_train,
            "yield_r2": _metric(metrics, "yield", "r2"),
            "yield_rmse": _metric(metrics, "yield", "rmse"),
            "somsc_r2": _metric(metrics, "somsc", "r2"),
            "somsc_rmse": _metric(metrics, "somsc", "rmse"),
            "elapsed_s": _safe_float(d.get("elapsed_seconds")),
            "source_path": path,
        })

    return pd.DataFrame(rows)


def collect_bo_graph_iterations(bo_dir: str, n_points: int | None = None) -> pd.DataFrame:
    """Parse bo_graph runs from iter*_n*_s*/summary.json.

    Supports both the original layout:
        <bo_dir>/iter0_n20_s42/summary.json
    and hparam-sweep labels:
        <bo_dir>/q200_r3_eps0p3_fail20_succ10_shrink5/iter0_n20_s42/summary.json
    """
    rows = []
    pat = re.compile(r"^iter(?P<iter>\d+)_n(?P<n>\d+)_s(?P<seed>\d+)$")

    pattern = os.path.join(bo_dir, "**", "iter*_n*_s*", "summary.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.basename(os.path.dirname(path))
        m = pat.match(run_dir)
        if not m:
            print(f"[warn] bo_graph: skipping unrecognized folder name: {run_dir}", file=sys.stderr)
            continue

        label_dir = os.path.dirname(os.path.dirname(path))
        hparam_label = os.path.relpath(label_dir, bo_dir)
        if hparam_label == ".":
            hparam_label = ""
        strategy_label = "bo_graph" if not hparam_label else f"bo_graph:{hparam_label}"

        with open(path) as f:
            d = json.load(f)

        n_train = int(d.get("n_train_points", int(m.group("n"))))
        if n_points is not None and n_train != n_points:
            continue

        metrics = d.get("metrics", {})

        rows.append({
            "strategy": strategy_label,
            "hparam_label": hparam_label,
            "run_tag": d.get("run_tag", run_dir),
            "iteration": int(m.group("iter")),
            "seed": int(m.group("seed")),
            "n_points": n_train,
            "yield_r2": _metric(metrics, "yield", "r2"),
            "yield_rmse": _metric(metrics, "yield", "rmse"),
            "somsc_r2": _metric(metrics, "somsc", "r2"),
            "somsc_rmse": _metric(metrics, "somsc", "rmse"),
            "elapsed_s": _safe_float(d.get("elapsed_seconds")),
            "source_path": path,
        })

    return pd.DataFrame(rows)


def summarize_iterations(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-strategy summary stats for quick comparison."""
    rows = []
    for strategy, sub in df.groupby("strategy", sort=False):
        sub = sub.sort_values("iteration")

        best_idx = sub["yield_r2"].astype(float).idxmax()
        best_row = sub.loc[best_idx]

        rows.append({
            "strategy": strategy,
            "n_points": int(sub["n_points"].iloc[0]),
            "n_iterations": int(sub["iteration"].nunique()),
            "n_seeds": int(sub["seed"].nunique()),
            "n_runs": int(len(sub)),
            "yield_r2_mean": float(sub["yield_r2"].mean()),
            "yield_r2_std": float(sub["yield_r2"].std(ddof=0)),
            "yield_r2_best": float(best_row["yield_r2"]),
            "yield_r2_best_iteration": int(best_row["iteration"]),
            "yield_rmse_mean": float(sub["yield_rmse"].mean()),
            "yield_rmse_std": float(sub["yield_rmse"].std(ddof=0)),
            "somsc_r2_mean": float(sub["somsc_r2"].mean()),
            "somsc_r2_std": float(sub["somsc_r2"].std(ddof=0)),
            "somsc_rmse_mean": float(sub["somsc_rmse"].mean()),
            "somsc_rmse_std": float(sub["somsc_rmse"].std(ddof=0)),
            "elapsed_s_total": float(sub["elapsed_s"].sum(skipna=True)),
        })

    return pd.DataFrame(rows)


def summarize_seed_iterations(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-strategy, per-seed best-metric summary stats."""
    rows = []
    for (strategy, seed), sub in df.groupby(["strategy", "seed"], sort=True):
        sub = sub.sort_values("iteration")

        best_r2_idx = sub["yield_r2"].astype(float).idxmax()
        best_rmse_idx = sub["yield_rmse"].astype(float).idxmin()
        best_r2_row = sub.loc[best_r2_idx]
        best_rmse_row = sub.loc[best_rmse_idx]

        rows.append({
            "strategy": strategy,
            "seed": int(seed),
            "n_points": int(sub["n_points"].iloc[0]),
            "n_iterations": int(sub["iteration"].nunique()),
            "n_runs": int(len(sub)),
            "yield_r2_mean": float(sub["yield_r2"].mean()),
            "yield_r2_best": float(best_r2_row["yield_r2"]),
            "yield_r2_best_iteration": int(best_r2_row["iteration"]),
            "yield_rmse_mean": float(sub["yield_rmse"].mean()),
            "yield_rmse_best": float(best_rmse_row["yield_rmse"]),
            "yield_rmse_best_iteration": int(best_rmse_row["iteration"]),
            "elapsed_s_total": float(sub["elapsed_s"].sum(skipna=True)),
        })

    return pd.DataFrame(rows)


def plot_iteration_comparison(df: pd.DataFrame, save_dir: str, n_points: int):
    """Plot yield metrics vs iteration for each strategy in a 2x2 grid.

    Top row: Yield R2 and Yield RMSE curves.
    Bottom row: Yield R2 running max and Yield RMSE running min (best-so-far).
    """
    strategies = sorted(df["strategy"].unique())
    curve_cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(strategies)))
    extrema_cmap = matplotlib.colormaps["Dark2"].resampled(max(1, len(strategies)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_yield_r2 = axes[0, 0]
    ax_yield_rmse = axes[0, 1]
    ax_yield_r2_max = axes[1, 0]
    ax_yield_rmse_min = axes[1, 1]

    for i, strategy in enumerate(strategies):
        sub = (
            df[df["strategy"] == strategy]
            .groupby("iteration", as_index=False)
            .agg({
                "yield_r2": "mean",
                "yield_rmse": "mean",
            })
            .sort_values("iteration")
        )
        x = sub["iteration"].astype(int).values
        curve_color = curve_cmap(i)
        extrema_color = extrema_cmap(i)
        linestyle = "--" if strategy == "random" else "-"

        y_r2 = sub["yield_r2"].astype(float).values
        y_rmse = sub["yield_rmse"].astype(float).values
        y_r2_max = sub["yield_r2"].astype(float).cummax().values
        y_rmse_min = sub["yield_rmse"].astype(float).cummin().values

        ax_yield_r2.plot(
            x,
            y_r2,
            "o" + linestyle,
            color=curve_color,
            linewidth=2,
            label=strategy,
        )
        ax_yield_rmse.plot(
            x,
            y_rmse,
            "o" + linestyle,
            color=curve_color,
            linewidth=2,
            label=strategy,
        )
        ax_yield_r2_max.plot(
            x,
            y_r2_max,
            "o" + linestyle,
            color=extrema_color,
            linewidth=2,
            label=f"{strategy} max",
        )
        ax_yield_rmse_min.plot(
            x,
            y_rmse_min,
            "o" + linestyle,
            color=extrema_color,
            linewidth=2,
            label=f"{strategy} min",
        )

    panel_config = [
        (ax_yield_r2, "Yield R2", "Yield R2 vs Iteration"),
        (ax_yield_rmse, "Yield RMSE", "Yield RMSE vs Iteration"),
        (ax_yield_r2_max, "Yield R2", "Yield R2 Running Max vs Iteration"),
        (ax_yield_rmse_min, "Yield RMSE", "Yield RMSE Running Min vs Iteration"),
    ]
    for ax, ylabel, title in panel_config:
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Strategy Comparison Across Iterations (n_points={n_points})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"strategy_iteration_comparison_n{n_points}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Iteration comparison plot saved -> {out_path}")


def _strategy_linestyle(strategy: str) -> str:
    return "--" if strategy == "random" else "-"


def plot_seed_iteration_comparison(df: pd.DataFrame, save_dir: str, n_points: int):
    """Plot seed-level yield metrics and best-so-far curves in one figure."""
    seeds = sorted(df["seed"].dropna().astype(int).unique())
    strategies = sorted(df["strategy"].unique())
    seed_cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(seeds)))
    seed_colors = {seed: seed_cmap(i) for i, seed in enumerate(seeds)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_yield_r2 = axes[0, 0]
    ax_yield_rmse = axes[0, 1]
    ax_yield_r2_max = axes[1, 0]
    ax_yield_rmse_min = axes[1, 1]

    for seed in seeds:
        for strategy in strategies:
            sub = (
                df[(df["seed"] == seed) & (df["strategy"] == strategy)]
                .sort_values("iteration")
            )
            if sub.empty:
                continue

            x = sub["iteration"].astype(int).values
            y_r2 = sub["yield_r2"].astype(float).values
            y_rmse = sub["yield_rmse"].astype(float).values
            label = f"s{seed} {strategy}"
            color = seed_colors[seed]
            linestyle = _strategy_linestyle(strategy)

            ax_yield_r2.plot(
                x,
                y_r2,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=1.8,
                alpha=0.85,
                label=label,
            )
            ax_yield_rmse.plot(
                x,
                y_rmse,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=1.8,
                alpha=0.85,
                label=label,
            )
            ax_yield_r2_max.plot(
                x,
                pd.Series(y_r2).cummax().values,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=label,
            )
            ax_yield_rmse_min.plot(
                x,
                pd.Series(y_rmse).cummin().values,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=label,
            )

    panel_config = [
        (ax_yield_r2, "Yield R2", "Yield R2 vs Iteration"),
        (ax_yield_rmse, "Yield RMSE", "Yield RMSE vs Iteration"),
        (ax_yield_r2_max, "Yield R2", "Yield R2 Running Max vs Iteration"),
        (ax_yield_rmse_min, "Yield RMSE", "Yield RMSE Running Min vs Iteration"),
    ]
    for ax, ylabel, title in panel_config:
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"Strategy Comparison Across Iterations by Seed (n_points={n_points})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"strategy_iteration_seed_comparison_n{n_points}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Seed-level iteration comparison plot saved -> {out_path}")


def plot_seed_running_best_comparison(df: pd.DataFrame, save_dir: str, n_points: int):
    """Plot best-so-far yield curves faceted by seed."""
    seeds = sorted(df["seed"].dropna().astype(int).unique())
    strategies = sorted(df["strategy"].unique())
    n_seeds = len(seeds)
    if n_seeds == 0:
        return

    curve_cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(strategies)))
    strategy_colors = {strategy: curve_cmap(i) for i, strategy in enumerate(strategies)}

    fig, axes = plt.subplots(
        n_seeds,
        2,
        figsize=(14, max(3.0 * n_seeds, 4.5)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, seed in enumerate(seeds):
        ax_r2 = axes[row_idx, 0]
        ax_rmse = axes[row_idx, 1]

        for strategy in strategies:
            sub = (
                df[(df["seed"] == seed) & (df["strategy"] == strategy)]
                .sort_values("iteration")
            )
            if sub.empty:
                continue

            x = sub["iteration"].astype(int).values
            y_r2_max = sub["yield_r2"].astype(float).cummax().values
            y_rmse_min = sub["yield_rmse"].astype(float).cummin().values
            linestyle = _strategy_linestyle(strategy)
            color = strategy_colors[strategy]

            ax_r2.plot(
                x,
                y_r2_max,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2,
                label=f"{strategy} max",
            )
            ax_rmse.plot(
                x,
                y_rmse_min,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2,
                label=f"{strategy} min",
            )

        ax_r2.set_title(f"Seed {seed}: Yield R2 Running Max")
        ax_rmse.set_title(f"Seed {seed}: Yield RMSE Running Min")
        ax_r2.set_ylabel("Yield R2")
        ax_rmse.set_ylabel("Yield RMSE")
        for ax in (ax_r2, ax_rmse):
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlabel("Iteration")

    fig.suptitle(f"Best-so-far Strategy Comparison by Seed (n_points={n_points})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"strategy_iteration_seed_running_best_n{n_points}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Seed-level running-best plot saved -> {out_path}")


def collect_all_strategies(
    random_dir: str,
    bo_dir: str,
    n_points: int,
) -> pd.DataFrame:
    """Collect iteration-level rows from strategy-specific parsers."""
    frames = []

    if os.path.isdir(random_dir):
        frames.append(collect_random_iterations(random_dir, n_points=n_points))
    else:
        print(f"[warn] random_dir not found: {random_dir}", file=sys.stderr)

    if os.path.isdir(bo_dir):
        frames.append(collect_bo_graph_iterations(bo_dir, n_points=n_points))
    else:
        print(f"[warn] bo_dir not found: {bo_dir}", file=sys.stderr)

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["strategy", "iteration", "seed"], kind="mergesort").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare strategy performance across iterations for fixed n_points.")
    parser.add_argument(
        "--random-dir",
        type=str,
        default=os.path.join(selection_output_root(), "exp6_random", "random"),
        help="Directory containing random runs (n*_ss*/ensemble_summary.json).",
    )
    parser.add_argument(
        "--bo-dir",
        type=str,
        default=os.path.join(selection_output_root(), "exp6_bo", "bo_graph"),
        help="Directory containing bo_graph runs (iter*_n*_s*/summary.json).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=20,
        help="Fixed training subset size to compare across iterations.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for CSV and plot. Defaults to common parent of input dirs.",
    )
    args = parser.parse_args()

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.commonpath([
            os.path.abspath(args.random_dir),
            os.path.abspath(args.bo_dir),
        ])
    os.makedirs(save_dir, exist_ok=True)

    df = collect_all_strategies(
        random_dir=args.random_dir,
        bo_dir=args.bo_dir,
        n_points=args.n_points,
    )
    if df.empty:
        print("No matching iteration results found.")
        sys.exit(0)

    print("\n=== ITERATION RESULTS ===")
    print(df.to_string(index=False))

    out_csv = os.path.join(save_dir, f"strategy_iteration_results_n{args.n_points}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nIteration CSV -> {out_csv}")

    summary_df = summarize_iterations(df)
    print("\n=== STRATEGY SUMMARY ===")
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(save_dir, f"strategy_iteration_summary_n{args.n_points}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV -> {summary_csv}")

    seed_summary_df = summarize_seed_iterations(df)
    print("\n=== STRATEGY SUMMARY BY SEED ===")
    print(seed_summary_df.to_string(index=False))

    seed_summary_csv = os.path.join(save_dir, f"strategy_iteration_seed_summary_n{args.n_points}.csv")
    seed_summary_df.to_csv(seed_summary_csv, index=False)
    print(f"Seed summary CSV -> {seed_summary_csv}")

    plot_iteration_comparison(df, save_dir=save_dir, n_points=args.n_points)
    plot_seed_iteration_comparison(df, save_dir=save_dir, n_points=args.n_points)
    plot_seed_running_best_comparison(df, save_dir=save_dir, n_points=args.n_points)


if __name__ == "__main__":
    main()
