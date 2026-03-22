"""
Compare selection experiment results across strategies and budget sizes.

Usage:
    python compare_selection_runs.py --runs-dir /users/6/mehta423/projects/daycent/output/selection
    python compare_selection_runs.py --csv sweep_random.csv sweep_stratified.csv
    python compare_selection_runs.py --ensembles --runs-dir /users/6/mehta423/projects/daycent/output/selection
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def _row_from_ensemble_json(path: str) -> dict:
    """Build a stats row from a pre-aggregated ensemble_summary.json."""
    with open(path) as f:
        d = json.load(f)
    agg = d["ensemble"]["aggregated"]
    return {
        "run_tag": d["run_tag"],
        "strategy": d["strategy"],
        "n_points": d["n_train_points"],
        "n_members": d["ensemble"]["n_members"],
        "yield_r2":       agg["yield"]["r2"],
        "yield_r2_std":   agg["yield"]["r2_std"],
        "yield_rmse":     agg["yield"]["rmse"],
        "yield_rmse_std": agg["yield"]["rmse_std"],
        "somsc_r2":       agg["somsc"]["r2"],
        "somsc_r2_std":   agg["somsc"]["r2_std"],
        "somsc_rmse":     agg["somsc"]["rmse"],
        "somsc_rmse_std": agg["somsc"]["rmse_std"],
        "elapsed_s":      d.get("elapsed_seconds"),
    }


def _row_from_member_jsons(member_paths: list[str], run_tag: str) -> dict | None:
    """Compute mean/std from individual member summary.json files."""
    records = []
    for p in member_paths:
        with open(p) as f:
            d = json.load(f)
        records.append(d)

    if not records:
        return None

    strategy  = records[0]["strategy"]
    n_points  = records[0]["n_train_points"]

    def _stat(key_outer, key_inner):
        vals = [r["metrics"][key_outer][key_inner] for r in records
                if key_inner in r["metrics"].get(key_outer, {})]
        arr = np.array(vals, dtype=float)
        return float(arr.mean()) if arr.size else float("nan"), \
               float(arr.std())  if arr.size > 1 else 0.0

    yr2,   yr2s   = _stat("yield", "r2")
    yrmse, yrmses = _stat("yield", "rmse")
    sr2,   sr2s   = _stat("somsc", "r2")
    srmse, srmses = _stat("somsc", "rmse")

    return {
        "run_tag":        run_tag,
        "strategy":       strategy,
        "n_points":       n_points,
        "n_members":      len(records),
        "yield_r2":       yr2,   "yield_r2_std":   yr2s,
        "yield_rmse":     yrmse, "yield_rmse_std": yrmses,
        "somsc_r2":       sr2,   "somsc_r2_std":   sr2s,
        "somsc_rmse":     srmse, "somsc_rmse_std": srmses,
        "elapsed_s":      None,
    }


def collect_ensemble_summaries(runs_dir: str) -> pd.DataFrame:
    """
    Walk *runs_dir* recursively and collect ensemble statistics.

    Priority per budget directory:
      1. Use ensemble_summary.json if present (pre-aggregated).
      2. Otherwise aggregate from all member_*/summary.json files found.

    Multiple selection-seed runs (e.g. n100_ss42, n100_ss99) for the same
    strategy and n_points are grouped, and their mean ± std is returned so the
    final plot shows one band per (strategy, n_points) regardless of how many
    seed runs exist.
    """
    rows = []

    # Each budget dir sits two levels below a strategy dir:
    #   <runs_dir>/<experiment>/<strategy>/<budget_dir>/
    # We search for ensemble_summary.json OR member_*/summary.json anywhere.
    ensemble_jsons = sorted(glob.glob(
        os.path.join(runs_dir, "**", "ensemble_summary.json"), recursive=True))

    # Track dirs already handled via ensemble_summary to avoid double-counting.
    handled_dirs: set[str] = set()

    for epath in ensemble_jsons:
        try:
            row = _row_from_ensemble_json(epath)
            rows.append(row)
            handled_dirs.add(os.path.dirname(epath))
        except Exception as exc:
            print(f"[warn] skipping {epath}: {exc}", file=sys.stderr)

    # Find budget dirs that have member summaries but no ensemble_summary.json.
    member_jsons = sorted(glob.glob(
        os.path.join(runs_dir, "**", "summary.json"), recursive=True))

    # Group member summary paths by their parent budget directory.
    budget_dir_members: dict[str, list[str]] = {}
    for mpath in member_jsons:
        budget_dir = os.path.dirname(os.path.dirname(mpath))  # strip member_* level
        if budget_dir in handled_dirs:
            continue
        budget_dir_members.setdefault(budget_dir, []).append(mpath)

    for budget_dir, mpaths in budget_dir_members.items():
        run_tag = os.path.relpath(budget_dir, runs_dir)
        try:
            row = _row_from_member_jsons(mpaths, run_tag)
            if row:
                rows.append(row)
        except Exception as exc:
            print(f"[warn] skipping {budget_dir}: {exc}", file=sys.stderr)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # If multiple selection-seed runs share the same (strategy, n_points),
    # collapse them into a single row: mean of means, pooled std.
    metric_cols = ["yield_r2", "yield_rmse", "somsc_r2", "somsc_rmse"]
    std_cols    = [c + "_std" for c in metric_cols]

    def _pool(group: pd.DataFrame) -> pd.Series:
        # When include_groups=False the groupby keys are excluded from *group*,
        # but group.name holds the key tuple (strategy, n_points).
        strat_key, npts_key = group.name
        out = {
            "run_tag":   group["run_tag"].iloc[0],
            "strategy":  strat_key,
            "n_points":  npts_key,
            "n_members": int(group["n_members"].sum()),
        }
        for col, scol in zip(metric_cols, std_cols):
            means = group[col].values.astype(float)
            stds  = group[scol].values.astype(float)
            grand_mean = float(np.mean(means))
            # pooled std: sqrt(mean of variances + variance of means)
            pooled_std = float(np.sqrt(np.mean(stds**2 + (means - grand_mean)**2)))
            out[col]  = grand_mean
            out[scol] = pooled_std
        out["elapsed_s"] = group["elapsed_s"].sum(skipna=True)
        return pd.Series(out)

    df = (df.groupby(["strategy", "n_points"], sort=False)
            .apply(_pool, include_groups=False)
            .reset_index(drop=True))

    return df


def plot_comparison(df: pd.DataFrame, save_dir: str):
    """Generate comparison plots for all strategies found in *df*."""
    # Group by strategy and n_points to compute mean and std
    df_agg = df.groupby(["strategy", "n_points"]).agg({
        "yield_r2": ["mean", "std"],
        "yield_rmse": ["mean", "std"],
        "somsc_r2": ["mean", "std"],
        "somsc_rmse": ["mean", "std"],
    }).reset_index()
    
    # Flatten the multi-level columns
    df_agg.columns = [
        "strategy", "n_points",
        "yield_r2_mean", "yield_r2_std",
        "yield_rmse_mean", "yield_rmse_std",
        "somsc_r2_mean", "somsc_r2_std",
        "somsc_rmse_mean", "somsc_rmse_std"
    ]

    strategies = df_agg["strategy"].unique()
    cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(strategies)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plots = [
        ("yield_r2_mean", "yield_r2_std", "Yield R²", "Yield R² vs Budget", (0, 0)),
        ("yield_rmse_mean", "yield_rmse_std", "Yield RMSE", "Yield RMSE vs Budget", (0, 1)),
        ("somsc_r2_mean", "somsc_r2_std", "SOMSC R²", "SOMSC R² vs Budget", (1, 0)),
        ("somsc_rmse_mean", "somsc_rmse_std", "SOMSC RMSE", "SOMSC RMSE vs Budget", (1, 1)),
    ]

    for i, strat in enumerate(sorted(strategies)):
        sub = df_agg[df_agg["strategy"] == strat].sort_values("n_points")
        color = cmap(i)
        x = sub["n_points"]
        
        for mean_col, std_col, ylabel, title, (r, c) in plots:
            ax = axes[r, c]
            y = sub[mean_col]
            yerr = sub[std_col].fillna(0)
            
            ax.plot(x, y, "o-", color=color, label=strat, linewidth=2)

    for (_, _, ylabel, title, (r, c)) in plots:
        axes[r, c].set_ylabel(ylabel)
        axes[r, c].set_title(title)
        axes[r, c].set_xlabel("# Training Points")
        axes[r, c].grid(True, alpha=0.3)
        axes[r, c].legend(fontsize=8)

    fig.suptitle("Selection Strategy Comparison (mean)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "strategy_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved → {out_path}")


def plot_ensemble_comparison(df: pd.DataFrame, save_dir: str):
    """
    Like plot_comparison but draws mean lines with ±1 std shaded bands.
    Expects columns: yield_r2, yield_r2_std, yield_rmse, yield_rmse_std,
                     somsc_r2, somsc_r2_std, somsc_rmse, somsc_rmse_std.
    """
    strategies = df["strategy"].unique()
    cmap = matplotlib.colormaps["tab10"].resampled(len(strategies))

    metrics = [
        ("yield_r2",   "yield_r2_std",   "Yield R²",   "Yield R² vs Budget",   (0, 0)),
        ("yield_rmse", "yield_rmse_std", "Yield RMSE", "Yield RMSE vs Budget", (0, 1)),
        ("somsc_r2",   "somsc_r2_std",   "SOMSC R²",   "SOMSC R² vs Budget",   (1, 0)),
        ("somsc_rmse", "somsc_rmse_std", "SOMSC RMSE", "SOMSC RMSE vs Budget", (1, 1)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, strat in enumerate(sorted(strategies)):
        sub   = df[df["strategy"] == strat].sort_values("n_points")
        color = cmap(i)
        x     = sub["n_points"].values

        for mean_col, std_col, ylabel, title, (r, c) in metrics:
            ax   = axes[r, c]
            mean = sub[mean_col].values.astype(float)
            std  = sub[std_col].values.astype(float)
            n    = sub["n_members"].values.astype(float)
            # Use ±1 std; label shows member count if it varies
            n_label = f"{int(n.mean())} members" if n.min() == n.max() \
                      else f"{int(n.min())}–{int(n.max())} members"
            ax.plot(x, mean, "o-", color=color,
                    label=f"{strat} ({n_label})", linewidth=2)

    for (_, _, ylabel, title, (r, c)) in metrics:
        axes[r, c].set_ylabel(ylabel)
        axes[r, c].set_title(title)
        axes[r, c].set_xlabel("# Training Points")
        axes[r, c].grid(True, alpha=0.3)
        axes[r, c].legend(fontsize=8)

    fig.suptitle("Selection Strategy Comparison (mean)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "strategy_comparison_ensemble.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ensemble comparison plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare selection runs.")
    parser.add_argument("--runs-dir", type=str,
                        default="/users/6/mehta423/projects/daycent/output/selection",
                        help="Root directory containing selection run outputs.")
    parser.add_argument("--csv", type=str, nargs="*",
                        help="Explicit CSV paths to merge (from sweep results).")
    parser.add_argument("--ensembles", action="store_true",
                        help=(
                            "Search within --runs-dir for ensemble_summary.json files "
                            "(or fall back to member summary.json files). "
                            "Multiple selection-seed runs sharing the same strategy and "
                            "n_points are collapsed into a single mean ± std band per "
                            "strategy, so the plot stays readable."))
    args = parser.parse_args()

    if args.ensembles:
        df = collect_ensemble_summaries(args.runs_dir)

        if df.empty:
            print("No ensemble results found under:", args.runs_dir)
            sys.exit(0)

        print("\n=== ENSEMBLE RESULTS ===")
        print(df.to_string(index=False))

        csv_out = os.path.join(args.runs_dir, "ensemble_results.csv")
        df.to_csv(csv_out, index=False)
        print(f"\nEnsemble CSV → {csv_out}")

        plot_ensemble_comparison(df, args.runs_dir)
        return

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
