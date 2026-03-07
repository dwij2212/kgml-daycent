"""
Visualisation helpers for point selection experiments.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import SelectionResult


def plot_selected_points(
    result: SelectionResult,
    test_points: List[str],
    pool_points: List[str],
    lookup_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot the selected training points, test points, and background pool
    on an elevation-coloured map of the Midwest.

    Parameters
    ----------
    result : SelectionResult
        Output of a selection strategy.
    test_points : list[str]
        Fixed test point IDs.
    pool_points : list[str]
        All candidate training point IDs (superset of ``result.selected_points``).
    lookup_df : pd.DataFrame
        Must contain columns ``id``, ``POINT_X``, ``POINT_Y``, ``elevation``.
    save_path : str, optional
        If given, save figure to this path.
    title : str, optional
        Override the default title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lookup_df = lookup_df.copy()
    lookup_df["id"] = lookup_df["id"].astype(str)

    pool_df = lookup_df[lookup_df["id"].isin(pool_points)]
    train_df = lookup_df[lookup_df["id"].isin(result.selected_points)]
    test_df = lookup_df[lookup_df["id"].isin(test_points)]
    unselected_df = pool_df[~pool_df["id"].isin(result.selected_points)]

    fig, ax = plt.subplots(figsize=(14, 10))

    # 1. Background: full pool with elevation
    sc = ax.scatter(
        pool_df["POINT_X"], pool_df["POINT_Y"],
        c=pool_df["elevation"], cmap="terrain", alpha=0.20, s=15,
        label="Pool (elevation)",
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Elevation (m)")

    # 2. Unselected pool points (grey)
    ax.scatter(
        unselected_df["POINT_X"], unselected_df["POINT_Y"],
        c="lightgrey", s=20, alpha=0.4, edgecolors="none",
        label=f"Unselected pool ({len(unselected_df)})",
    )

    # 3. Selected training points (blue)
    ax.scatter(
        train_df["POINT_X"], train_df["POINT_Y"],
        c="dodgerblue", s=80, marker="o", edgecolors="black", linewidths=0.8,
        label=f"Selected train ({len(train_df)})",
        zorder=5,
    )

    # 4. Test points (red)
    ax.scatter(
        test_df["POINT_X"], test_df["POINT_Y"],
        c="red", s=80, marker="^", edgecolors="black", linewidths=0.8,
        label=f"Test ({len(test_df)})",
        zorder=5,
    )

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    default_title = (
        f"Selection: {result.strategy_name} | "
        f"n_train={result.n_points} | "
        f"n_test={len(test_points)}"
    )
    ax.set_title(title or default_title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved selection plot → {save_path}")

    return fig


def plot_feature_coverage(
    result: SelectionResult,
    pool_points: List[str],
    init_cond_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare feature distributions of selected vs. full pool.

    Draws violin plots of a handful of key features showing the pool
    distribution overlaid with the selected-subset distribution.
    """
    init_cond_df = init_cond_df.copy()
    init_cond_df["id"] = init_cond_df["id"].astype(str)

    if feature_cols is None:
        feature_cols = [
            "SITLAT", "SITLNG", "ELEV",
            "SOM1CI(1,1)", "SOM2CI(1,1)", "SOM3CI(1)",
            "SLSAND(1)", "SLCLAY(1)", "SLPH(1)",
        ]
    feature_cols = [c for c in feature_cols if c in init_cond_df.columns]

    pool_df = init_cond_df[init_cond_df["id"].isin(pool_points)]
    sel_df = init_cond_df[init_cond_df["id"].isin(result.selected_points)]

    n_feats = len(feature_cols)
    ncols = min(3, n_feats)
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        pool_vals = pool_df[col].dropna().values
        sel_vals = sel_df[col].dropna().values

        ax.hist(pool_vals, bins=30, alpha=0.4, color="grey",
                density=True, label="Pool")
        ax.hist(sel_vals, bins=15, alpha=0.6, color="dodgerblue",
                density=True, label="Selected")
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Feature coverage – {result.strategy_name} "
        f"(n={result.n_points})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved coverage plot → {save_path}")

    return fig
