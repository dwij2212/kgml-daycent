"""
Create slide-ready figures for the yearly SOMSC state model presentation.

The script reads the saved yearly evaluation CSVs and writes conceptual
diagrams plus result figures into the experiment output directory.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


POOL_COLS = ["som1c_soil", "som2c_soil", "som3c", "som2c_surface"]
SOIL_POOL_COLS = ["som1c_soil", "som2c_soil", "som3c"]
POOL_LABELS = {
    "som1c_soil": "SOM1C soil\nactive",
    "som2c_soil": "SOM2C soil\nslow",
    "som3c": "SOM3C\npassive",
    "som2c_surface": "SOM2C surface\nauxiliary",
}
POOL_COLORS = {
    "som1c_soil": "#2A9D8F",
    "som2c_soil": "#E9C46A",
    "som3c": "#4E79A7",
    "som2c_surface": "#C85A54",
}
MODE_LABELS = {
    "teacher_forced": "Teacher forced",
    "rollout": "Rollout",
}
MODE_COLORS = {
    "teacher_forced": "#2A9D8F",
    "rollout": "#4E79A7",
    "persistence": "#8D99AE",
}


def set_slide_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def add_slide_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    title_size: int = 23,
    subtitle_size: int = 13,
) -> None:
    fig.text(
        0.05,
        0.95,
        title,
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
        va="top",
        ha="left",
    )
    fig.text(
        0.05,
        0.885,
        subtitle,
        fontsize=subtitle_size,
        color="#4B5563",
        va="top",
        ha="left",
    )


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fc: str = "#F6F7F9",
    ec: str = "#233142",
    color: str = "#111827",
    fontsize: int = 16,
    weight: str = "normal",
    radius: float = 0.025,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
    )
    return box


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#233142",
    lw: float = 2.2,
    mutation_scale: float = 18,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def load_predictions(base_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for mode in ["teacher_forced", "rollout"]:
        path = base_dir / f"test_{mode}_yearly_state_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction CSV: {path}")
        df = pd.read_csv(path)
        df = df[df["somsc_mask"].astype(float) > 0].copy()
        frames[mode] = df
    return frames


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(residual**2)) / denom
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}


def compute_metrics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode, df in frames.items():
        for target_name, true_col, pred_col in [
            ("SOMSC level", "somsc_true", "somsc_pred"),
            ("Annual SOMSC change", "somsc_delta_true", "somsc_delta_pred"),
            ("No-change reference", "somsc_true", "somsc_persistence_pred"),
        ]:
            if target_name == "No-change reference" and mode != "teacher_forced":
                continue
            metrics = regression_metrics(df[true_col], df[pred_col])
            rows.append({"mode": mode, "target": target_name, **metrics})

        for pool in POOL_COLS:
            metrics = regression_metrics(
                df[f"{pool}_delta_true"],
                df[f"{pool}_delta_pred"],
            )
            rows.append(
                {
                    "mode": mode,
                    "target": f"{pool} annual change",
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def figure_yearly_problem(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "Reframing SOMSC as a yearly state transition",
        fontsize=27,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    ax.text(
        0.05,
        0.865,
        "Instead of supervising every month, learn the annual December-to-December update.",
        fontsize=17,
        color="#4B5563",
        va="top",
    )

    ax.text(0.08, 0.71, "Before", fontsize=18, fontweight="bold", color="#6B7280")
    ax.text(0.54, 0.71, "Now", fontsize=18, fontweight="bold", color="#2A6F62")

    draw_box(
        ax,
        (0.06, 0.47),
        0.33,
        0.16,
        "Monthly framing\n12 SOMSC labels per year",
        fc="#F4F5F7",
        ec="#9CA3AF",
        fontsize=17,
        weight="bold",
    )
    months_x = np.linspace(0.09, 0.36, 12)
    for idx, x in enumerate(months_x, start=1):
        ax.scatter(x, 0.38, s=90, color="#C85A54", zorder=3)
        if idx in [1, 6, 12]:
            ax.text(x, 0.32, str(idx), ha="center", fontsize=11, color="#6B7280")
    ax.text(0.225, 0.27, "Targets at each month", ha="center", fontsize=14, color="#6B7280")

    draw_box(
        ax,
        (0.52, 0.55),
        0.14,
        0.13,
        "Dec state\nY - 1",
        fc="#E8F3EF",
        ec="#2A9D8F",
        fontsize=15,
        weight="bold",
    )
    draw_box(
        ax,
        (0.52, 0.29),
        0.14,
        0.16,
        "Monthly drivers\nweather +\nmanagement",
        fc="#FDF5DF",
        ec="#E9C46A",
        fontsize=14,
        weight="bold",
    )
    draw_box(
        ax,
        (0.72, 0.42),
        0.12,
        0.15,
        "Learn\nannual\nDelta pools",
        fc="#EEF2FF",
        ec="#4E79A7",
        fontsize=14,
        weight="bold",
    )
    draw_box(
        ax,
        (0.89, 0.47),
        0.09,
        0.13,
        "Dec state\nY",
        fc="#E8F3EF",
        ec="#2A9D8F",
        fontsize=14,
        weight="bold",
    )
    draw_arrow(ax, (0.66, 0.615), (0.72, 0.535), color="#2A9D8F")
    draw_arrow(ax, (0.66, 0.37), (0.72, 0.465), color="#B08A24")
    draw_arrow(ax, (0.84, 0.495), (0.89, 0.535), color="#4E79A7")

    ax.text(
        0.67,
        0.18,
        "Training sample: one calendar year -> one December target",
        ha="center",
        fontsize=18,
        color="#111827",
        fontweight="bold",
    )
    ax.text(
        0.67,
        0.125,
        "This matches the evaluation question: can we step the SOC state forward one year?",
        ha="center",
        fontsize=15,
        color="#4B5563",
    )

    save_figure(fig, output_dir, "01_yearly_problem_framing")


def figure_pool_decomposition(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    df = frames["teacher_forced"]
    means = {pool: float(df[f"{pool}_true"].mean()) for pool in POOL_COLS}
    soil_total = sum(means[pool] for pool in SOIL_POOL_COLS)

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.94,
        "Target becomes a mass-balanced SOC state",
        fontsize=22,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    ax.text(
        0.05,
        0.885,
        "Predict pool changes, then recombine soil pools into December SOMSC.",
        fontsize=13.5,
        color="#4B5563",
        va="top",
    )

    draw_box(
        ax,
        (0.06, 0.54),
        0.18,
        0.13,
        "Original label\nSOMSC",
        fc="#F4F5F7",
        ec="#6B7280",
        fontsize=17,
        weight="bold",
    )
    draw_arrow(ax, (0.24, 0.605), (0.32, 0.605), color="#6B7280")

    x0 = 0.34
    width = 0.38
    y0 = 0.53
    height = 0.17
    current = x0
    for pool in SOIL_POOL_COLS:
        share = means[pool] / soil_total
        seg_w = width * share
        rect = Rectangle(
            (current, y0),
            seg_w,
            height,
            facecolor=POOL_COLORS[pool],
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(rect)
        label = POOL_LABELS[pool].split("\n")[0]
        if seg_w > 0.06:
            ax.text(
                current + seg_w / 2,
                y0 + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=11.5,
                color="white" if pool in ["som1c_soil", "som3c"] else "#111827",
                fontweight="bold",
            )
        current += seg_w
    ax.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="#111827", linewidth=1.5))
    ax.text(
        x0 + width / 2,
        y0 + height + 0.045,
        "SOMSC soil total",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        x0 + width / 2,
        0.39,
        "SOMSC = SOM1C_soil + SOM2C_soil + SOM3C",
        ha="center",
        fontsize=13.5,
        color="#111827",
    )
    ax.text(
        x0 + width / 2,
        0.335,
        "SOM1C_soil = SOMSC - SOM2C_soil - SOM3C",
        ha="center",
        fontsize=11.5,
        color="#4B5563",
    )

    draw_box(
        ax,
        (0.80, 0.50),
        0.14,
        0.17,
        "SOM2C\nsurface\nstate",
        fc="#FCEBE8",
        ec=POOL_COLORS["som2c_surface"],
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.87,
        0.405,
        "Auxiliary state:\npredicted, not summed\ninto SOMSC",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4B5563",
    )

    legend_x = 0.12
    legend_y = 0.18
    for idx, pool in enumerate(SOIL_POOL_COLS + ["som2c_surface"]):
        x = legend_x + idx * 0.21
        ax.add_patch(Rectangle((x, legend_y), 0.028, 0.028, color=POOL_COLORS[pool]))
        ax.text(
            x + 0.038,
            legend_y + 0.014,
            POOL_LABELS[pool],
            va="center",
            fontsize=10.8,
            color="#111827",
            linespacing=1.1,
        )

    save_figure(fig, output_dir, "02_soc_pool_decomposition")


def figure_results_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    tf_delta = metrics_df[(metrics_df["mode"] == "teacher_forced") & (metrics_df["target"] == "Annual SOMSC change")].iloc[0]
    ro_delta = metrics_df[(metrics_df["mode"] == "rollout") & (metrics_df["target"] == "Annual SOMSC change")].iloc[0]
    baseline = metrics_df[(metrics_df["target"] == "No-change reference")].iloc[0]
    tf_level = metrics_df[(metrics_df["mode"] == "teacher_forced") & (metrics_df["target"] == "SOMSC level")].iloc[0]
    ro_level = metrics_df[(metrics_df["mode"] == "rollout") & (metrics_df["target"] == "SOMSC level")].iloc[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.33, 7.5),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.77, wspace=0.30)
    add_slide_header(
        fig,
        "Yearly state model: test-set results",
        "Teacher forced = one-step accuracy. Rollout = accumulated trajectory stability.",
    )

    ax = axes[0]
    labels = ["No-change\nreference", "Teacher\nforced", "Rollout"]
    values = [baseline["rmse"], tf_delta["rmse"], ro_delta["rmse"]]
    colors = [MODE_COLORS["persistence"], MODE_COLORS["teacher_forced"], MODE_COLORS["rollout"]]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel("Annual change RMSE (g C/m2)")
    ax.set_title("Annual SOMSC change", pad=10, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.035,
            f"{value:.1f}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    ax.text(
        1,
        max(values) * 0.82,
        f"R2 = {tf_delta['r2']:.3f}",
        ha="center",
        fontsize=12,
        color=MODE_COLORS["teacher_forced"],
        fontweight="bold",
    )
    ax.text(
        2,
        max(values) * 0.70,
        f"R2 = {ro_delta['r2']:.3f}",
        ha="center",
        fontsize=12,
        color=MODE_COLORS["rollout"],
        fontweight="bold",
    )

    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.00, 0.98, "Held-out test metrics", fontsize=15, fontweight="bold", va="top")

    card_specs = [
        (
            0.00,
            0.59,
            "Teacher forced",
            MODE_COLORS["teacher_forced"],
            [
                ("Level RMSE", f"{tf_level['rmse']:.1f}"),
                ("Level R2", f"{tf_level['r2']:.4f}"),
                ("Change RMSE", f"{tf_delta['rmse']:.1f}"),
                ("Change R2", f"{tf_delta['r2']:.3f}"),
            ],
        ),
        (
            0.52,
            0.59,
            "Rollout",
            MODE_COLORS["rollout"],
            [
                ("Level RMSE", f"{ro_level['rmse']:.1f}"),
                ("Level R2", f"{ro_level['r2']:.4f}"),
                ("Change RMSE", f"{ro_delta['rmse']:.1f}"),
                ("Change R2", f"{ro_delta['r2']:.3f}"),
            ],
        ),
    ]
    for x0, y0, title, color, rows in card_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                0.44,
                0.33,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                linewidth=1.2,
                facecolor="#F8FAFC",
                edgecolor="#D1D5DB",
            )
        )
        ax.text(x0 + 0.03, y0 + 0.28, title, fontsize=12.5, fontweight="bold", color=color)
        for idx, (label, value) in enumerate(rows):
            y = y0 + 0.215 - idx * 0.062
            ax.text(x0 + 0.03, y, label, fontsize=10.8, color="#4B5563", va="center")
            ax.text(x0 + 0.40, y, value, fontsize=11.2, color="#111827", ha="right", va="center", fontweight="bold")

    ax.text(
        0.02,
        0.42,
        fill("Readout: the annual state model more than halves annual-change RMSE relative to carrying the previous year's state forward.", 50),
        fontsize=11.5,
        color="#4B5563",
        va="top",
    )

    save_figure(fig, output_dir, "03_results_metrics_summary")


def figure_true_vs_pred(frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.76, wspace=0.16)
    add_slide_header(
        fig,
        "Predicted vs true December SOMSC",
        "Each point is one scenario, location, and year in the held-out test set.",
    )

    all_true = pd.concat([frames[m]["somsc_true"] for m in frames])
    all_pred = pd.concat([frames[m]["somsc_pred"] for m in frames])
    lo = float(min(all_true.min(), all_pred.min()))
    hi = float(max(all_true.max(), all_pred.max()))
    pad = (hi - lo) * 0.04
    lo -= pad
    hi += pad

    for ax, mode in zip(axes, ["teacher_forced", "rollout"]):
        df = frames[mode]
        metrics = metrics_df[(metrics_df["mode"] == mode) & (metrics_df["target"] == "SOMSC level")].iloc[0]
        ax.scatter(
            df["somsc_true"],
            df["somsc_pred"],
            s=13,
            alpha=0.38,
            color=MODE_COLORS[mode],
            edgecolors="none",
        )
        ax.plot([lo, hi], [lo, hi], color="#111827", lw=1.5, ls="--", alpha=0.75)
        ax.set_title(MODE_LABELS[mode], fontweight="bold", pad=8)
        ax.set_xlabel("True SOMSC (g C/m2)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(
            0.05,
            0.93,
            f"RMSE {metrics['rmse']:.1f}\nR2 {metrics['r2']:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=11.8,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1D5DB"),
        )
    axes[0].set_ylabel("Predicted SOMSC (g C/m2)")

    save_figure(fig, output_dir, "04_true_vs_predicted_somsc")


def select_trajectory(frames: dict[str, pd.DataFrame]) -> tuple[int, int]:
    tf = frames["teacher_forced"]
    ro = frames["rollout"]
    merged = tf.merge(
        ro,
        on=["scenario_id", "point_id", "year"],
        suffixes=("_tf", "_ro"),
    )
    candidates = []
    for (scenario_id, point_id), group in merged.groupby(["scenario_id", "point_id"]):
        if group["year"].nunique() < 4:
            continue
        true_range = group["somsc_true_tf"].max() - group["somsc_true_tf"].min()
        rollout_rmse = np.sqrt(np.mean((group["somsc_pred_ro"] - group["somsc_true_tf"]) ** 2))
        score = true_range / (rollout_rmse + 1.0)
        candidates.append((score, int(scenario_id), int(point_id)))
    if not candidates:
        first = tf.iloc[0]
        return int(first["scenario_id"]), int(first["point_id"])
    _, scenario_id, point_id = max(candidates)
    return scenario_id, point_id


def figure_rollout_trajectory(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    scenario_id, point_id = select_trajectory(frames)
    merged = frames["teacher_forced"].merge(
        frames["rollout"],
        on=["scenario_id", "point_id", "year"],
        suffixes=("_tf", "_ro"),
    )
    group = merged[(merged["scenario_id"] == scenario_id) & (merged["point_id"] == point_id)].sort_values("year")

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.77)
    add_slide_header(
        fig,
        "Rollout trajectory check",
        "Rollout feeds the model's predicted pool state into the next year.",
    )

    years = group["year"].astype(int)
    ax.plot(years, group["somsc_true_tf"], color="#111827", marker="o", lw=3, label="True")
    ax.plot(years, group["somsc_pred_tf"], color=MODE_COLORS["teacher_forced"], marker="s", lw=2.5, label="Teacher forced")
    ax.plot(years, group["somsc_pred_ro"], color=MODE_COLORS["rollout"], marker="^", lw=2.5, label="Rollout")
    ax.plot(years, group["somsc_persistence_pred_tf"], color=MODE_COLORS["persistence"], marker=".", lw=2.0, ls=":", label="Previous true state")
    ax.set_xlabel("Year")
    ax.set_ylabel("December SOMSC (g C/m2)")
    ax.text(
        0.01,
        0.97,
        f"Scenario {scenario_id}, point {point_id}",
        transform=ax.transAxes,
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB"),
    )
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.set_xticks(list(years))
    ax.margins(x=0.04, y=0.14)

    ro_rmse = regression_metrics(group["somsc_true_tf"], group["somsc_pred_ro"])["rmse"]
    tf_rmse = regression_metrics(group["somsc_true_tf"], group["somsc_pred_tf"])["rmse"]
    ax.text(
        0.02,
        0.05,
        f"Example trajectory RMSE: teacher forced {tf_rmse:.1f}, rollout {ro_rmse:.1f}",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1D5DB"),
    )

    save_figure(fig, output_dir, "05_rollout_trajectory_example")


def figure_pool_delta_rmse(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    pool_targets = [f"{pool} annual change" for pool in POOL_COLS]
    x = np.arange(len(pool_targets))
    width = 0.36
    tf_vals = [
        metrics_df[(metrics_df["mode"] == "teacher_forced") & (metrics_df["target"] == target)]["rmse"].iloc[0]
        for target in pool_targets
    ]
    ro_vals = [
        metrics_df[(metrics_df["mode"] == "rollout") & (metrics_df["target"] == target)]["rmse"].iloc[0]
        for target in pool_targets
    ]

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.suptitle("Pool-level annual change errors", fontsize=27, fontweight="bold", x=0.05, ha="left", y=0.96)
    fig.text(
        0.05,
        0.89,
        "The model learns annual deltas for each SOC state variable, then sums the soil pools for SOMSC.",
        fontsize=16,
        color="#4B5563",
    )

    bars1 = ax.bar(x - width / 2, tf_vals, width, label="Teacher forced", color=MODE_COLORS["teacher_forced"])
    bars2 = ax.bar(x + width / 2, ro_vals, width, label="Rollout", color=MODE_COLORS["rollout"])
    ax.set_ylabel("Pool annual change RMSE\n(g C/m2)")
    ax.set_xticks(x)
    ax.set_xticklabels([POOL_LABELS[pool] for pool in POOL_COLS])
    ax.legend(frameon=True)
    ax.set_ylim(0, max(tf_vals + ro_vals) * 1.25)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(tf_vals + ro_vals) * 0.025, f"{value:.1f}", ha="center", fontsize=12.5)

    save_figure(fig, output_dir, "06_pool_delta_rmse")


def write_metrics_table(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    table = metrics_df.copy()
    table["mode"] = table["mode"].map(lambda x: MODE_LABELS.get(x, x))
    table = table[["mode", "target", "n", "rmse", "mae", "r2"]]
    table.to_csv(output_dir / "metrics_for_slides.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make yearly SOMSC slide figures.")
    parser.add_argument(
        "--base-dir",
        default="daycent/output/yearly_december_somsc_state_v1",
        help="Experiment directory containing saved yearly prediction CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for slide figures. Defaults to <base-dir>/slide_figures.",
    )
    args = parser.parse_args()

    set_slide_style()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "slide_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_predictions(base_dir)
    metrics_df = compute_metrics(frames)
    write_metrics_table(metrics_df, output_dir)

    figure_yearly_problem(output_dir)
    figure_pool_decomposition(frames, output_dir)
    figure_results_summary(metrics_df, output_dir)
    figure_true_vs_pred(frames, metrics_df, output_dir)
    figure_rollout_trajectory(frames, output_dir)
    figure_pool_delta_rmse(metrics_df, output_dir)

    print(f"Wrote slide figures to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
