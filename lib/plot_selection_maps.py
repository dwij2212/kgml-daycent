"""
Plot selection maps from existing selection_result.json files.

Usage:
    python plot_selection_maps.py \
        --base-config configs/selection/selection_base.yaml

    python plot_selection_maps.py \
        --base-config configs/selection/selection_exp6.yaml \
        --runs-dir /projects/standard/kumarv/shared/dwij/daycent/output/selection/exp6_bo/bo_graph

    python plot_selection_maps.py \
        --base-config configs/selection/selection_exp6.yaml \
        --selection-json \
        /projects/standard/kumarv/shared/dwij/daycent/output/selection/exp6_bo/bo_graph/iter0_n25_s42/selection_result.json
"""
import argparse
import glob
import json
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_selection_experiment import (
    load_base_config,
    resolve_point_lists,
    load_metadata,
)
from selection import SelectionResult
from selection.visualize import plot_selected_points


def _load_selection_result(path: str) -> SelectionResult:
    """Load SelectionResult with a small compatibility fallback."""
    try:
        return SelectionResult.load(path)
    except Exception:
        with open(path, "r") as f:
            d = json.load(f)

        selected_points = [str(p) for p in d.get("selected_points", [])]
        strategy_name = d.get("strategy_name") or d.get("strategy") or "unknown"

        return SelectionResult(
            selected_points=selected_points,
            strategy_name=strategy_name,
            strategy_params=d.get("strategy_params", {}),
            metadata=d.get("metadata", {}),
        )


def collect_selection_jsons(runs_dir: str, explicit_paths: List[str] | None = None) -> List[str]:
    """Return sorted selection_result.json paths."""
    if explicit_paths:
        paths = [p for p in explicit_paths if os.path.basename(p) == "selection_result.json"]
        return sorted(paths)

    pattern = os.path.join(runs_dir, "**", "selection_result.json")
    return sorted(glob.glob(pattern, recursive=True))


def main():
    parser = argparse.ArgumentParser(description="Plot maps from saved selection_result.json files.")
    parser.add_argument(
        "--base-config", type=str,
        default="configs/selection/selection_base.yaml",
        help="Path to base selection YAML used to recover pool/test points.",
    )
    parser.add_argument(
        "--runs-dir", type=str,
        default="/projects/standard/kumarv/shared/dwij/daycent/output/selection",
        help="Root directory to search for selection_result.json files.",
    )
    parser.add_argument(
        "--selection-json", type=str, nargs="*", default=None,
        help="Optional explicit selection_result.json paths. If given, --runs-dir search is skipped.",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Optional strategy filter (e.g. random, bo_graph).",
    )
    parser.add_argument(
        "--output-name", type=str, default="selection_map.png",
        help="Output image filename in each run directory (default: selection_map.png).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output maps.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: config not found: {args.base_config}")
        sys.exit(1)

    base_dict = load_base_config(args.base_config)
    resolve_point_lists(base_dict, os.path.dirname(os.path.abspath(args.base_config)))
    metadata = load_metadata(base_dict)

    pool_points = [str(p) for p in base_dict["data"]["pool_points"]]
    test_points = [str(p) for p in base_dict["data"]["test"]["points"]]

    sel_jsons = collect_selection_jsons(args.runs_dir, args.selection_json)
    if not sel_jsons:
        print("No selection_result.json files found.")
        return

    n_plotted = 0
    n_skipped_existing = 0
    n_skipped_strategy = 0
    n_failed = 0

    for sel_path in sel_jsons:
        try:
            result = _load_selection_result(sel_path)

            if args.strategy and result.strategy_name != args.strategy:
                n_skipped_strategy += 1
                continue

            run_dir = os.path.dirname(sel_path)
            out_path = os.path.join(run_dir, args.output_name)
            if os.path.exists(out_path) and not args.overwrite:
                n_skipped_existing += 1
                continue

            plot_selected_points(
                result=result,
                test_points=test_points,
                pool_points=pool_points,
                lookup_df=metadata["lookup_df"],
                save_path=out_path,
            )
            plt.close("all")
            n_plotted += 1
        except Exception as exc:
            n_failed += 1
            print(f"[warn] failed for {sel_path}: {exc}", file=sys.stderr)

    print("\n=== Selection Map Plotting Summary ===")
    print(f"Found selection_result.json files : {len(sel_jsons)}")
    print(f"Plotted                        : {n_plotted}")
    print(f"Skipped (already exists)       : {n_skipped_existing}")
    print(f"Skipped (strategy mismatch)    : {n_skipped_strategy}")
    print(f"Failed                         : {n_failed}")


if __name__ == "__main__":
    main()
