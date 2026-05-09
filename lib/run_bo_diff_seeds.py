"""
Run and visualize BO sensitivity to random initial subset / seed.

This is a focused companion to run_bo_hparam_sweep.py.  It keeps BO framework
hyperparameters fixed, varies the BO seed, and writes seed-level convergence,
final-best, and subset-overlap diagnostics.  The goal is to see how much the
first random k-subset changes the final BO result.

Example:
  python run_bo_diff_seeds.py \
      --base-config configs/selection/selection_exp6.yaml \
      --n-points 20 --n-iterations 20 \
      --seeds 42 123 456 789 \
      --embedding-path ../output/static_emb_32 \
      --experiment-name exp6_bo_hparam_n20 \
      --Q 600 --max-radius 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
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

from run_bo_hparam_sweep import (
    OUTPUT_ROOT,
    bo_output_dir,
    collect_results,
    expected_bo_csv,
    make_bo_args,
    make_run_label,
    result_is_complete,
)


def _float_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def hparams_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "Q": args.Q,
        "max_radius": args.max_radius,
        "epsilon_factor": args.epsilon_factor,
        "fail_tol": args.fail_tol,
        "succ_tol": args.succ_tol,
        "shrink_tol": args.shrink_tol,
    }


def diff_seed_output_dir(args: argparse.Namespace, run_label: str) -> str:
    label = args.sweep_label
    if not label:
        label = (
            f"diff_seeds_{run_label}_n{args.n_points}"
            f"_iters{args.n_iterations}"
        )
    return os.path.join(OUTPUT_ROOT, args.experiment_name, "bo_graph", label)


def prepare_report_output_dir(args: argparse.Namespace, run_label: str) -> str:
    """Create a report dir without clobbering existing report artifacts."""
    out_dir = diff_seed_output_dir(args, run_label)
    if (
        os.path.isdir(out_dir)
        and os.listdir(out_dir)
        and not args.allow_overwrite_reports
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{out_dir}_{stamp}"
        suffix = 1
        while os.path.exists(out_dir):
            out_dir = f"{diff_seed_output_dir(args, run_label)}_{stamp}_{suffix}"
            suffix += 1
        print(
            "[no-clobber] Report directory already has files; "
            f"writing this report to {out_dir}"
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def bo_run_artifacts(bo_args: argparse.Namespace) -> list[str]:
    """Return existing BO outputs that would be reused or overwritten."""
    paths: list[str] = []
    csv_path = expected_bo_csv(bo_args)
    if os.path.exists(csv_path):
        paths.append(csv_path)

    sweep_dir = bo_output_dir(bo_args)
    if os.path.isdir(sweep_dir) and os.listdir(sweep_dir):
        paths.append(sweep_dir)

    run_label = str(getattr(bo_args, "run_label", "") or "").strip().strip("/")
    iter_parent = os.path.join(
        OUTPUT_ROOT,
        bo_args.experiment_name,
        "bo_graph",
        run_label,
    )
    iter_pattern = os.path.join(
        iter_parent,
        f"iter*_n{bo_args.n_points}_s{bo_args.seed}",
    )
    paths.extend(sorted(glob.glob(iter_pattern)))
    return sorted(set(paths))


def parse_points(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {str(p) for p in json.loads(value)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def preview_initial_subsets(
    args: argparse.Namespace,
    hparams: dict[str, Any],
    run_label: str,
) -> pd.DataFrame:
    """Run only the first BO ask per seed to verify initial subset diversity."""
    from run_bo_experiment import subset_signature
    from run_selection_experiment import (
        load_base_config,
        load_metadata,
        resolve_point_lists,
    )
    from selection import get_strategy

    base_dict = load_base_config(args.base_config)
    config_dir = os.path.dirname(os.path.abspath(args.base_config))
    resolve_point_lists(base_dict, config_dir)
    metadata = load_metadata(base_dict)
    pool_points = [str(p) for p in base_dict["data"]["pool_points"]]

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        strategy = get_strategy(
            "bo_graph",
            n_points=args.n_points,
            seed=seed,
            embedding_path=args.embedding_path,
            epsilon_factor=args.epsilon_factor,
            Q=args.Q,
            max_radius=args.max_radius,
            fail_tol=args.fail_tol,
            succ_tol=args.succ_tol,
            shrink_tol=args.shrink_tol,
        )
        result = strategy.select(pool_points, metadata=metadata)
        subset = sorted(str(p) for p in result.selected_points)
        rows.append({
            "seed": int(seed),
            "run_label": run_label,
            "Q": hparams["Q"],
            "max_radius": hparams["max_radius"],
            "initial_subset_signature": subset_signature(subset),
            "selected_points_json": json.dumps(subset),
            "subgraph_size": result.metadata.get("subgraph_size"),
        })

    return pd.DataFrame(rows)


def normalized_auc(sub: pd.DataFrame) -> float:
    sub = sub.sort_values("iteration")
    x = sub["iteration"].astype(float).to_numpy()
    y = sub["best_score"].astype(float).to_numpy()
    if len(x) < 2 or x.max() == x.min():
        return float(y[-1])
    return float(np.trapezoid(y, x) / (x.max() - x.min()))


def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, sub in df.groupby("seed", sort=True):
        sub = sub.sort_values("iteration")
        best_idx = sub["score"].astype(float).idxmax()
        best_row = sub.loc[best_idx]
        first_row = sub.iloc[0]
        last_row = sub.iloc[-1]

        rows.append({
            "seed": int(seed),
            "run_label": str(last_row["run_label"]),
            "Q": int(last_row["Q"]),
            "max_radius": int(last_row["max_radius"]),
            "n_points": int(last_row["n_points"]),
            "n_iterations": int(sub["iteration"].nunique()),
            "initial_subset_signature": str(first_row["subset_signature"]),
            "best_subset_signature": str(best_row["subset_signature"]),
            "final_subset_signature": str(last_row["subset_signature"]),
            "initial_points_json": first_row["selected_points_json"],
            "best_points_json": best_row["selected_points_json"],
            "final_points_json": last_row["selected_points_json"],
            "initial_score": float(first_row["score"]),
            "best_iteration": int(best_row["iteration"]),
            "best_score": float(best_row["score"]),
            "final_best_score": float(last_row["best_score"]),
            "mean_score": float(sub["score"].mean()),
            "normalized_auc": normalized_auc(sub),
            "n_unique_subsets": int(sub["subset_signature"].nunique()),
            "max_local_obs": int(sub["n_local_obs"].max()),
            "max_subgraph_size": int(sub["subgraph_size"].max()),
            "max_restarts": int(sub["n_restarts"].max()),
            "elapsed_h": float(sub["elapsed_s"].sum(skipna=True) / 3600.0),
            "bo_results_csv": str(last_row["bo_results_csv"]),
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["final_best_score", "normalized_auc", "mean_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def build_subset_movement(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, sub in df.groupby("seed", sort=True):
        sub = sub.sort_values("iteration")
        first = parse_points(sub.iloc[0]["selected_points_json"])
        prev: set[str] | None = None

        for _, row in sub.iterrows():
            current = parse_points(row["selected_points_json"])
            rows.append({
                "seed": int(seed),
                "run_label": str(row["run_label"]),
                "Q": int(row["Q"]),
                "max_radius": int(row["max_radius"]),
                "iteration": int(row["iteration"]),
                "score": float(row["score"]),
                "best_score": float(row["best_score"]),
                "overlap_initial_jaccard": jaccard(current, first),
                "overlap_previous_jaccard": (
                    np.nan if prev is None else jaccard(current, prev)
                ),
                "swap_count_from_previous": (
                    np.nan if prev is None else len(current - prev)
                ),
            })
            prev = current
    return pd.DataFrame(rows)


def pairwise_jaccard(summary: pd.DataFrame, points_col: str) -> pd.DataFrame:
    seeds = summary["seed"].astype(int).tolist()
    sets = {
        int(row["seed"]): parse_points(row[points_col])
        for _, row in summary.iterrows()
    }
    mat = pd.DataFrame(index=seeds, columns=seeds, dtype=float)
    for a in seeds:
        for b in seeds:
            mat.loc[a, b] = jaccard(sets[a], sets[b])
    return mat


def plot_seed_convergence(
    df: pd.DataFrame,
    out_dir: str,
    score_metric: str,
) -> None:
    seeds = sorted(df["seed"].astype(int).unique())
    cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(seeds)))

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, seed in enumerate(seeds):
        sub = df[df["seed"].astype(int) == seed].sort_values("iteration")
        color = cmap(idx)
        ax.plot(
            sub["iteration"],
            sub["score"],
            "o--",
            color=color,
            alpha=0.35,
            linewidth=1.4,
        )
        ax.plot(
            sub["iteration"],
            sub["best_score"],
            "o-",
            color=color,
            linewidth=2.2,
            label=f"seed {seed}",
        )

    ax.set_title("BO convergence across initial-subset seeds")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel(score_metric)
    ax.grid(True, alpha=0.3)
    ax.legend(title="BO seed", fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, "bo_diff_seed_convergence.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Seed convergence plot -> {path}")


def plot_seed_final_summary(
    summary: pd.DataFrame,
    out_dir: str,
    score_metric: str,
) -> None:
    if summary.empty:
        return

    plot_df = summary.sort_values("seed")
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(plot_df)), 5.5))
    x = np.arange(len(plot_df))
    bars = ax.bar(
        x,
        plot_df["final_best_score"].astype(float).to_numpy(),
        color=matplotlib.colormaps["viridis"].resampled(len(plot_df))(range(len(plot_df))),
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"s{int(row.seed)}\ninit {row.initial_subset_signature}"
            for row in plot_df.itertuples()
        ],
        rotation=0,
    )
    ax.set_ylabel(score_metric)
    ax.set_title("Final best score by random initial subset")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, row in zip(bars, plot_df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row.final_best_score:.3f}\niter {int(row.best_iteration)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "bo_diff_seed_final_best.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Final-best-by-seed plot -> {path}")


def plot_subset_jaccard(
    initial_jaccard: pd.DataFrame,
    best_jaccard: pd.DataFrame,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        (initial_jaccard, "Initial subset Jaccard"),
        (best_jaccard, "Best subset Jaccard"),
    ]
    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat.astype(float).to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_xticklabels([str(c) for c in mat.columns])
        ax.set_yticklabels([str(i) for i in mat.index])
        ax.set_xlabel("seed")
        ax.set_ylabel("seed")
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                ax.text(j, i, f"{float(mat.iloc[i, j]):.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Subset overlap across BO seeds", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "bo_diff_seed_subset_jaccard.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Subset Jaccard plot -> {path}")


def plot_seed_mechanics(
    df: pd.DataFrame,
    out_dir: str,
) -> None:
    diagnostics = [
        ("ei_value", "Expected improvement", False),
        ("gp_lengthscale", "GP lengthscale", True),
        ("n_local_obs", "Local observations", False),
        ("subgraph_size", "Combo-subgraph size", False),
        ("elapsed_s", "Elapsed time per iteration (min)", False),
        ("overlap_initial_jaccard", "Subset overlap vs initial", False),
    ]

    movement = build_subset_movement(df)
    plot_df = df.merge(
        movement[["seed", "iteration", "overlap_initial_jaccard"]],
        on=["seed", "iteration"],
        how="left",
    )

    seeds = sorted(plot_df["seed"].astype(int).unique())
    cmap = matplotlib.colormaps["tab10"].resampled(max(1, len(seeds)))
    colors = {seed: cmap(i) for i, seed in enumerate(seeds)}

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    axes = axes.ravel()
    for ax, (col, title, log_y) in zip(axes, diagnostics):
        for seed in seeds:
            sub = plot_df[plot_df["seed"].astype(int) == seed].sort_values("iteration")
            y = sub[col].copy()
            if col == "elapsed_s":
                y = y / 60.0
            valid = y.notna()
            if not valid.any():
                continue
            ax.plot(
                sub.loc[valid, "iteration"],
                y.loc[valid],
                "o-",
                color=colors[seed],
                linewidth=1.4,
                markersize=3,
                label=f"seed {seed}",
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if log_y:
            ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=min(4, len(seeds)), frameon=False)
    fig.supxlabel("BO iteration")
    fig.suptitle("BO mechanics across seeds", fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    path = os.path.join(out_dir, "bo_diff_seed_mechanics.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Seed mechanics plot -> {path}")


def run_diff_seed_sweep(args: argparse.Namespace) -> None:
    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}", file=sys.stderr)
        sys.exit(1)

    hparams = hparams_from_args(args)
    run_label = make_run_label(hparams, prefix=args.run_label_prefix)
    out_dir = prepare_report_output_dir(args, run_label)

    preview = preview_initial_subsets(args, hparams, run_label)
    preview_path = os.path.join(out_dir, "bo_initial_subset_seed_preview.csv")
    preview.to_csv(preview_path, index=False)
    n_preview_unique = int(preview["initial_subset_signature"].nunique())
    print(f"Initial subset preview CSV -> {preview_path}")
    print(
        f"Preview initial subset signatures: {n_preview_unique} unique across "
        f"{len(preview)} seeds"
    )
    if n_preview_unique != len(preview) and not args.allow_duplicate_initial_subsets:
        duplicates = preview[
            preview.duplicated("initial_subset_signature", keep=False)
        ].sort_values("initial_subset_signature")
        raise RuntimeError(
            "Different seeds did not produce unique initial subsets. "
            "Refusing to run because this would not isolate initial-subset "
            f"variation.\n{duplicates.to_string(index=False)}"
        )

    manifest: list[dict[str, Any]] = []
    for seed in args.seeds:
        bo_args = make_bo_args(args, hparams, seed=seed, run_label=run_label)
        csv_path = expected_bo_csv(bo_args)
        status = "pending"
        existing_artifacts = bo_run_artifacts(bo_args)

        if args.plot_only:
            status = "plot_only"
        elif result_is_complete(csv_path, args.n_iterations):
            status = "reused"
            print(f"\n[reuse] seed={seed}: {csv_path}")
        elif existing_artifacts and not args.allow_overwrite_bo_runs:
            artifact_text = "\n  ".join(existing_artifacts[:20])
            if len(existing_artifacts) > 20:
                artifact_text += f"\n  ... {len(existing_artifacts) - 20} more"
            raise RuntimeError(
                f"Refusing to run seed={seed} because existing incomplete BO "
                "artifacts were found and would risk overwriting.\n"
                f"  {artifact_text}\n"
                "Use a new seed/run-label or pass --allow-overwrite-bo-runs "
                "only if you intentionally want to replace them."
            )
        else:
            print("\n" + "=" * 80)
            print(
                f"BO diff-seed run: seed={seed} | run_label={run_label} "
                f"| Q={args.Q} max_radius={args.max_radius}"
            )
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

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(out_dir, "bo_diff_seed_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nManifest CSV -> {manifest_path}")

    combined = collect_results(manifest)
    if combined.empty:
        print("No BO results found to combine/plot.")
        return

    combined = combined.sort_values(["seed", "iteration"], kind="mergesort").reset_index(drop=True)
    combined_path = os.path.join(out_dir, "bo_diff_seed_results.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Combined seed results CSV -> {combined_path}")

    summary = build_seed_summary(combined)
    summary_path = os.path.join(out_dir, "bo_diff_seed_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Seed summary CSV -> {summary_path}")

    initial_subset_audit = summary[[
        "seed",
        "initial_subset_signature",
        "initial_score",
        "best_iteration",
        "final_best_score",
        "elapsed_h",
    ]].copy()
    initial_subset_audit_path = os.path.join(out_dir, "bo_initial_subset_by_seed.csv")
    initial_subset_audit.to_csv(initial_subset_audit_path, index=False)
    print(f"Initial subset audit CSV -> {initial_subset_audit_path}")
    print(
        "Initial subset signatures: "
        f"{summary['initial_subset_signature'].nunique()} unique across "
        f"{summary['seed'].nunique()} seeds"
    )
    if not summary.empty:
        checked = summary.merge(
            preview[["seed", "initial_subset_signature"]],
            on="seed",
            how="left",
            suffixes=("_actual", "_preview"),
        )
        checked["matches_preview"] = (
            checked["initial_subset_signature_actual"]
            == checked["initial_subset_signature_preview"]
        )
        checked_path = os.path.join(out_dir, "bo_initial_subset_preview_check.csv")
        checked[[
            "seed",
            "initial_subset_signature_actual",
            "initial_subset_signature_preview",
            "matches_preview",
        ]].to_csv(checked_path, index=False)
        print(f"Initial subset preview check CSV -> {checked_path}")
        if not checked["matches_preview"].all():
            raise RuntimeError(
                "At least one actual initial subset did not match the "
                "preflight preview. This should not happen for deterministic "
                "BO seeds."
            )

    movement = build_subset_movement(combined)
    movement_path = os.path.join(out_dir, "bo_diff_seed_subset_movement.csv")
    movement.to_csv(movement_path, index=False)
    print(f"Subset movement CSV -> {movement_path}")

    initial_jaccard = pairwise_jaccard(summary, "initial_points_json")
    best_jaccard = pairwise_jaccard(summary, "best_points_json")
    initial_jaccard_path = os.path.join(out_dir, "bo_initial_subset_jaccard.csv")
    best_jaccard_path = os.path.join(out_dir, "bo_best_subset_jaccard.csv")
    initial_jaccard.to_csv(initial_jaccard_path)
    best_jaccard.to_csv(best_jaccard_path)
    print(f"Initial subset Jaccard CSV -> {initial_jaccard_path}")
    print(f"Best subset Jaccard CSV -> {best_jaccard_path}")

    plot_seed_convergence(combined, out_dir, args.score_metric)
    plot_seed_final_summary(summary, out_dir, args.score_metric)
    plot_subset_jaccard(initial_jaccard, best_jaccard, out_dir)
    plot_seed_mechanics(combined, out_dir)

    best = summary.iloc[0]
    print("\n=== BEST SEED ===")
    print(
        f"seed={int(best['seed'])} final_best={best['final_best_score']:.4f} "
        f"best_iteration={int(best['best_iteration'])} "
        f"initial_signature={best['initial_subset_signature']}"
    )
    print("\n=== SEED SUMMARY ===")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run/plot BO sensitivity to different random initial subsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
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
        help="Optional prefix added to the fixed-hparam run label.",
    )
    parser.add_argument(
        "--score-metric",
        type=str,
        default="yield_r2",
        choices=["yield_r2", "yield_rmse", "somsc_r2", "somsc_rmse"],
    )
    parser.add_argument("--Q", type=int, default=600)
    parser.add_argument("--max-radius", type=int, default=3)
    parser.add_argument("--epsilon-factor", type=float, default=0.3)
    parser.add_argument("--fail-tol", type=int, default=20)
    parser.add_argument("--succ-tol", type=int, default=10)
    parser.add_argument("--shrink-tol", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Complete existing BO runs are "
            "always reused unless --allow-overwrite-bo-runs is set."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only collect and plot expected existing results; do not run BO.",
    )
    parser.add_argument(
        "--allow-overwrite-bo-runs",
        action="store_true",
        help=(
            "Dangerous: allow running a seed even if existing incomplete BO "
            "artifacts for that seed/run-label are present."
        ),
    )
    parser.add_argument(
        "--allow-overwrite-reports",
        action="store_true",
        help=(
            "Allow overwriting aggregate diff-seed CSV/PNG report files. By "
            "default, a timestamped report directory is created instead."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-initial-subsets",
        action="store_true",
        help=(
            "Allow running even if the preflight seed preview finds duplicate "
            "initial subsets."
        ),
    )

    args = parser.parse_args()
    run_diff_seed_sweep(args)


if __name__ == "__main__":
    main()
