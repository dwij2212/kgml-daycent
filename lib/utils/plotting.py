"""
Shared plotting utilities for DayCent experiments.

Functions here are deliberately data-agnostic: they accept plain arrays /
dicts and return matplotlib Figure objects.  Domain-specific wrappers in
eval_emulator.py, eval_inverse.py, and selection/visualize.py build on
these primitives.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Optional[str], dpi: int = 150) -> None:
    """Optionally save *fig* to *path* and close it."""
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Time-series overlay  (emulator & inverse reconstruction)
# ---------------------------------------------------------------------------

def plot_timeseries(
    true: np.ndarray,
    pred: np.ndarray,
    title: str = "",
    xlabel: str = "Time step",
    ylabel: str = "Value",
    metrics: Optional[Dict[str, float]] = None,
    true_label: str = "True",
    pred_label: str = "Predicted",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Single-panel true vs predicted time-series.

    Parameters
    ----------
    true, pred : 1-D np.ndarray
    metrics : optional dict of scalar metrics to append to title (e.g. r2, rmse)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(true))
    ax.plot(t, true, "o-", label=true_label, linewidth=2, markersize=5,
            color="#2E86AB", alpha=0.85)
    ax.plot(t, pred, "s--", label=pred_label, linewidth=2, markersize=5,
            color="#A23B72", alpha=0.75)

    full_title = title
    if metrics:
        parts = [f"{k.upper()}={v:.4f}" for k, v in metrics.items()]
        full_title += "  |  " + "  ".join(parts)
    ax.set_title(full_title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dual_timeseries(
    series: List[Tuple[np.ndarray, np.ndarray, str]],
    titles: List[str],
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    suptitle: str = "",
    metrics_list: Optional[List[Optional[Dict[str, float]]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Vertically-stacked subplots, one per (true, pred, label) tuple.

    Used by the emulator evaluation to show Yield and SOMSC on the same figure.
    """
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n),
                             gridspec_kw={"hspace": 0.4})
    if n == 1:
        axes = [axes]

    for i, (true, pred, var_label) in enumerate(series):
        ax = axes[i]
        t = np.arange(len(true))
        ax.plot(t, true, "o-", label="True", linewidth=1.8, markersize=5,
                color="#2E86AB", alpha=0.85)
        ax.plot(t, pred, "s--", label="Predicted", linewidth=1.8, markersize=5,
                color="#A23B72", alpha=0.75)

        title = titles[i] if titles else var_label
        if metrics_list and metrics_list[i]:
            m = metrics_list[i]
            title += f"  |  MSE={m.get('mse', 0):.2f}  R²={m.get('r2', 0):.3f}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabels[i] if xlabels else "Index", fontsize=10)
        ax.set_ylabel(ylabels[i] if ylabels else var_label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Scatter (true vs predicted)
# ---------------------------------------------------------------------------

def plot_scatter(
    true: np.ndarray,
    pred: np.ndarray,
    title: str = "",
    xlabel: str = "True",
    ylabel: str = "Predicted",
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scatter plot with identity line.  Can receive an existing *ax*."""
    owned = ax is None
    if owned:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    d_min = min(true.min(), pred.min())
    d_max = max(true.max(), pred.max())
    span = d_max - d_min
    lo, hi = d_min - 0.05 * span, d_max + 0.05 * span

    ax.scatter(true, pred, alpha=0.5, s=10, c="steelblue", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect fit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    full_title = title
    if metrics:
        parts = [f"{k.upper()}={v:.3f}" for k, v in metrics.items()
                 if isinstance(v, (int, float))]
        full_title += "\n" + "  ".join(parts)
    ax.set_title(full_title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)

    if owned:
        plt.tight_layout()
        _save(fig, save_path)
    return fig


def plot_scatter_grid(
    channel_results: List[Dict],
    true: np.ndarray,
    pred: np.ndarray,
    ncols: int = 3,
    max_plots: int = 9,
    title: str = "Scatter plots",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grid of scatter plots for multiple channels/features.

    Parameters
    ----------
    channel_results : list of dicts from ``compute_per_channel_metrics``
        Each dict must have at least: ``channel_idx``, ``channel_name``,
        ``corr``, ``r2``.
    true, pred : np.ndarray, shape (N, C) or (N, T, C)
    """
    results = sorted(channel_results, key=lambda r: r["corr"], reverse=True)
    results = results[:max_plots]

    n = len(results)
    cols = min(ncols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, res in enumerate(results):
        c_idx = res["channel_idx"]
        name = res["channel_name"]
        t = true[..., c_idx].flatten()
        p = pred[..., c_idx].flatten()
        plot_scatter(
            t, p,
            title=name,
            metrics={"corr": res["corr"], "r2": res["r2"]},
            ax=axes[i],
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Horizontal bar chart  (feature R² / correlation)
# ---------------------------------------------------------------------------

def plot_bar_h(
    names: Sequence[str],
    values: Sequence[float],
    title: str = "",
    xlabel: str = "Value",
    sort: bool = True,
    colormap: str = "RdYlBu",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart coloured by value (good for R² / correlation).

    Parameters
    ----------
    names : sequence of feature/channel names (y-axis)
    values : corresponding metric values
    sort : if True, sort by value ascending (so highest is at the top)
    """
    names_arr  = np.array(list(names))
    values_arr = np.array(list(values), dtype=float)

    if sort:
        idx = np.argsort(values_arr)
        names_arr  = names_arr[idx]
        values_arr = values_arr[idx]

    fig_h = max(5, 0.4 * len(names_arr))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    norm = plt.Normalize(vmin=values_arr.min(), vmax=values_arr.max())
    cmap = matplotlib.colormaps[colormap]
    colors = cmap(norm(values_arr))

    y_pos = np.arange(len(names_arr))
    ax.barh(y_pos, values_arr, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_arr, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(values_arr):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Budget-curve plot  (selection experiments)
# ---------------------------------------------------------------------------

def plot_budget_curve(
    n_points: Sequence[int],
    metrics_by_task: Dict[str, Dict[str, Sequence[float]]],
    title_prefix: str = "",
    std_by_task: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plots of metric vs training budget, one panel per task.

    Parameters
    ----------
    n_points : sequence of budget sizes (x-axis)
    metrics_by_task : ``{task_name: {metric_name: [values]}}``
        e.g. ``{"yield": {"r2": [...], "rmse": [...]}, "somsc": {...}}``
    std_by_task : optional matching structure with std values for shading
    title_prefix : prepended to each panel title
    """
    tasks = list(metrics_by_task.keys())
    n_tasks = len(tasks)
    colors = ["dodgerblue", "coral", "mediumseagreen", "orchid"]

    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        task_metrics = metrics_by_task[task]
        task_std     = (std_by_task or {}).get(task, {})
        ns = np.array(n_points)

        for j, (metric_name, vals) in enumerate(task_metrics.items()):
            color = colors[j % len(colors)]
            vals_arr = np.array(vals)
            ax.plot(ns, vals_arr, "o-", color=color, linewidth=2,
                    label=metric_name)
            if metric_name in task_std:
                std_arr = np.array(task_std[metric_name])
                ax.fill_between(ns, vals_arr - std_arr, vals_arr + std_arr,
                                alpha=0.2, color=color)

        ax.set_xlabel("# Training Points", fontsize=11)
        ax.set_title(f"{title_prefix}{task}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Histogram overlay  (feature coverage for selection)
# ---------------------------------------------------------------------------

def plot_distribution_comparison(
    pool_values: Dict[str, np.ndarray],
    selected_values: Dict[str, np.ndarray],
    title: str = "Feature coverage",
    ncols: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay histograms comparing pool vs selected distributions per feature.

    Parameters
    ----------
    pool_values, selected_values : ``{feature_name: 1-D np.ndarray}``
    """
    feature_names = list(pool_values.keys())
    n_feats = len(feature_names)
    cols = min(ncols, n_feats)
    rows = math.ceil(n_feats / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten()

    for i, feat in enumerate(feature_names):
        ax = axes_flat[i]
        pool_v = pool_values[feat]
        sel_v  = selected_values[feat]

        ax.hist(pool_v, bins=30, alpha=0.4, color="grey",
                density=True, label="Pool")
        ax.hist(sel_v,  bins=15, alpha=0.6, color="dodgerblue",
                density=True, label="Selected")
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Spatial scatter map  (selection point map)
# ---------------------------------------------------------------------------

def plot_spatial_points(
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    pool_mask: Optional[np.ndarray] = None,
    title: str = "Point selection map",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Elevation-coloured scatter map of pool / train / test points.

    All arrays are indexed identically (one entry per spatial point).

    Parameters
    ----------
    lon, lat, elevation : 1-D arrays for all known points
    train_mask : boolean mask — True for selected training points
    test_mask  : boolean mask — True for test points
    pool_mask  : boolean mask — True for candidate pool (defaults to ~test_mask)
    """
    if pool_mask is None:
        pool_mask = ~test_mask

    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: full pool coloured by elevation
    sc = ax.scatter(
        lon[pool_mask], lat[pool_mask],
        c=elevation[pool_mask], cmap="terrain", alpha=0.20, s=15,
        label="Pool (elevation)",
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Elevation (m)")

    # Unselected pool
    unsel = pool_mask & ~train_mask
    ax.scatter(lon[unsel], lat[unsel],
               c="lightgrey", s=20, alpha=0.4, edgecolors="none",
               label=f"Unselected pool ({unsel.sum()})")

    # Selected training
    ax.scatter(lon[train_mask], lat[train_mask],
               c="dodgerblue", s=80, marker="o",
               edgecolors="black", linewidths=0.8,
               label=f"Selected train ({train_mask.sum()})", zorder=5)

    # Test
    ax.scatter(lon[test_mask], lat[test_mask],
               c="red", s=80, marker="^",
               edgecolors="black", linewidths=0.8,
               label=f"Test ({test_mask.sum()})", zorder=5)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig
