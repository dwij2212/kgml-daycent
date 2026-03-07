"""
Compare selection experiment results across strategies and budget sizes.

Usage:
    python compare_selection_runs.py --runs-dir /users/6/mehta423/daycent/output/selection
    python compare_selection_runs.py --csv sweep_random.csv sweep_stratified.csv
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def collect_summaries(runs_dir: str) -> pd.DataFrame:
    """Walk the runs directory and collect all summary.json files."""
    rows = []
    pattern = os.path.join(runs_dir, "**", "summary.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            d = json.load(f)
        row = {
            "run_tag": d["run_tag"],
            "strategy": d["strategy"],
            "n_points": d["n_train_points"],
            "yield_mse": d["metrics"]["yield"]["mse"],
            "yield_rmse": d["metrics"]["yield"]["rmse"],
            "yield_r2": d["metrics"]["yield"]["r2"],
            "somsc_mse": d["metrics"]["somsc"]["mse"],
            "somsc_rmse": d["metrics"]["somsc"]["rmse"],
            "somsc_r2": d["metrics"]["somsc"]["r2"],
            "elapsed_s": d.get("elapsed_seconds", None),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, save_dir: str):
    """Generate comparison plots for all strategies found in *df*."""
    strategies = df["strategy"].unique()
    cmap = plt.cm.get_cmap("tab10", len(strategies))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, strat in enumerate(sorted(strategies)):
        sub = df[df["strategy"] == strat].sort_values("n_points")
        color = cmap(i)
        label = strat

        axes[0, 0].plot(sub["n_points"], sub["yield_r2"], "o-",
                        color=color, label=label, linewidth=2)
        axes[0, 1].plot(sub["n_points"], sub["yield_rmse"], "o-",
                        color=color, label=label, linewidth=2)
        axes[1, 0].plot(sub["n_points"], sub["somsc_r2"], "o-",
                        color=color, label=label, linewidth=2)
        axes[1, 1].plot(sub["n_points"], sub["somsc_rmse"], "o-",
                        color=color, label=label, linewidth=2)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("# Training Points")
        ax.legend()

    axes[0, 0].set_ylabel("Yield R²")
    axes[0, 0].set_title("Yield R² vs Budget")
    axes[0, 1].set_ylabel("Yield RMSE")
    axes[0, 1].set_title("Yield RMSE vs Budget")
    axes[1, 0].set_ylabel("SOMSC R²")
    axes[1, 0].set_title("SOMSC R² vs Budget")
    axes[1, 1].set_ylabel("SOMSC RMSE")
    axes[1, 1].set_title("SOMSC RMSE vs Budget")

    fig.suptitle("Selection Strategy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "strategy_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare selection runs.")
    parser.add_argument("--runs-dir", type=str,
                        default="/users/6/mehta423/daycent/output/selection",
                        help="Root directory containing selection run outputs.")
    parser.add_argument("--csv", type=str, nargs="*",
                        help="Explicit CSV paths to merge (from sweep results).")
    args = parser.parse_args()

    if args.csv:
        dfs = [pd.read_csv(p) for p in args.csv]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = collect_summaries(args.runs_dir)

    if df.empty:
        print("No results found.")
        sys.exit(0)

    print("\n=== ALL RESULTS ===")
    print(df.to_string(index=False))

    csv_out = os.path.join(args.runs_dir, "all_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nAggregated CSV → {csv_out}")

    plot_comparison(df, args.runs_dir)


if __name__ == "__main__":
    main()
