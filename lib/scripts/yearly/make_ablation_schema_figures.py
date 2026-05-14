"""
Create simple method-schema figures for yearly SOMSC ablation experiments.

The goal is to make the ablation logic readable: what previous state is shown
to the model, what target is supervised, and which scientific question each
case answers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


CASE_ORDER = [
    "scalar_prev_somsc",
    "scalar_full_pools",
    "three_pool_somsc_context",
    "three_pool_no_aux",
    "four_state_current",
]
CASE_INFO = {
    "scalar_prev_somsc": {
        "label": "A. Scalar target\nprev SOMSC",
        "context": "previous\nSOMSC only",
        "target": "predict annual\nSOMSC delta",
        "test": "Is yearly delta framing enough?",
        "color": "#8D99AE",
    },
    "scalar_full_pools": {
        "label": "B. Scalar target\nfull pools",
        "context": "previous\n4-pool state",
        "target": "predict annual\nSOMSC delta",
        "test": "Does hidden pool composition solve most of it?",
        "color": "#2A9D8F",
    },
    "three_pool_somsc_context": {
        "label": "C. 3-pool target\nprev SOMSC",
        "context": "previous\nSOMSC only",
        "target": "predict 3 soil\npool deltas",
        "test": "Is pool supervision enough without pool state?",
        "color": "#C08497",
    },
    "three_pool_no_aux": {
        "label": "D. 3-pool target\nno aux",
        "context": "previous\n3 soil pools",
        "target": "predict 3 soil\npool deltas",
        "test": "How much do soil pools alone explain?",
        "color": "#E9C46A",
    },
    "four_state_current": {
        "label": "E. 4-state\ncurrent",
        "context": "previous\n3 soil + surface",
        "target": "predict 3 soil +\nsurface deltas",
        "test": "Does surface auxiliary improve the transition?",
        "color": "#4E79A7",
    },
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    fc="#F7F8FA",
    ec="#374151",
    color="#111827",
    fontsize=10.5,
    weight="normal",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.12,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, *, color="#374151", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def metric_text(summary: pd.DataFrame, case: str) -> str:
    rows = summary[summary["case"] == case]
    tf = rows[rows["mode"] == "teacher_forced"]
    ro = rows[rows["mode"] == "rollout"]
    parts = []
    if not tf.empty:
        parts.append(f"TF {float(tf['delta_rmse'].iloc[0]):.1f}")
    if not ro.empty:
        parts.append(f"RO {float(ro['delta_rmse'].iloc[0]):.1f}")
    else:
        parts.append("RO n/a")
    return "\n".join(parts)


def figure_case_schemas(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.96,
        "Ablation experiment design",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111827",
    )
    ax.text(
        0.03,
        0.905,
        "All cases use the same 12-month weather and management sequence. "
        "Only the previous-state representation and supervised target are changed.",
        fontsize=12.5,
        ha="left",
        va="top",
        color="#4B5563",
    )

    headers = [
        ("Case", 0.035),
        ("Previous state input", 0.205),
        ("Shared encoder", 0.405),
        ("Supervised target", 0.595),
        ("Question isolated", 0.765),
        ("Delta RMSE", 0.92),
    ]
    for text, x in headers:
        ax.text(x, 0.815, text, fontsize=10.5, fontweight="bold", color="#374151")

    y0 = 0.70
    dy = 0.13
    for idx, case in enumerate(CASE_ORDER):
        info = CASE_INFO[case]
        y = y0 - idx * dy
        color = info["color"]

        box(ax, 0.03, y, 0.13, 0.075, info["label"], fc=color, ec=color, color="white", weight="bold")
        box(ax, 0.19, y, 0.13, 0.075, info["context"], fc="#F9FAFB", ec=color)
        box(
            ax,
            0.37,
            y,
            0.13,
            0.075,
            "LSTM over\nannual drivers",
            fc="#EFF6FF",
            ec="#4E79A7",
        )
        box(ax, 0.57, y, 0.13, 0.075, info["target"], fc="#F9FAFB", ec=color)
        box(
            ax,
            0.74,
            y,
            0.15,
            0.075,
            fill(info["test"], width=21),
            fc="#FFFFFF",
            ec="#D1D5DB",
            fontsize=9.5,
        )
        box(
            ax,
            0.91,
            y,
            0.07,
            0.075,
            metric_text(summary, case),
            fc="#FFFFFF",
            ec="#D1D5DB",
            fontsize=9.5,
            weight="bold",
        )
        arrow(ax, 0.32, y + 0.037, 0.37, y + 0.037, color="#6B7280")
        arrow(ax, 0.50, y + 0.037, 0.57, y + 0.037, color="#6B7280")

    ax.text(
        0.91,
        0.13,
        "TF = teacher forced\nRO = rollout\ng C m-2 yr-1",
        fontsize=9.5,
        color="#6B7280",
        ha="left",
        va="top",
    )

    save(fig, out_dir, "05_ablation_case_schemas")


def contrast_panel(
    ax,
    title: str,
    left_label: str,
    left_value: float,
    right_label: str,
    right_value: float,
    note: str,
    *,
    left_color="#8D99AE",
    right_color="#4E79A7",
):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=15, fontweight="bold", va="top", color="#111827")

    box(ax, 0.05, 0.52, 0.32, 0.22, left_label, fc=left_color, ec=left_color, color="white", weight="bold")
    box(ax, 0.63, 0.52, 0.32, 0.22, right_label, fc=right_color, ec=right_color, color="white", weight="bold")
    arrow(ax, 0.39, 0.63, 0.61, 0.63, color="#374151", lw=2.2)

    ax.text(0.21, 0.42, f"{left_value:.1f}", ha="center", va="center", fontsize=23, fontweight="bold")
    ax.text(0.79, 0.42, f"{right_value:.1f}", ha="center", va="center", fontsize=23, fontweight="bold")
    ax.text(0.50, 0.42, "RMSE", ha="center", va="center", fontsize=10, color="#6B7280")

    ax.text(
        0.05,
        0.18,
        fill(note, width=52),
        fontsize=10.5,
        color="#4B5563",
        va="top",
        ha="left",
    )


def figure_question_map(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6))
    fig.text(
        0.035,
        0.97,
        "How to read the ablations",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111827",
    )
    fig.text(
        0.035,
        0.90,
        "Each contrast changes one modeling assumption and asks what information the yearly transition needs.",
        fontsize=12.5,
        ha="left",
        va="top",
        color="#4B5563",
    )

    tf = summary[summary["mode"] == "teacher_forced"].set_index("case")
    ro = summary[summary["mode"] == "rollout"].set_index("case")

    contrast_panel(
        axes[0],
        "1. Yearly delta alone?",
        "prev SOMSC\nscalar target",
        float(tf.loc["scalar_prev_somsc", "delta_rmse"]),
        "zero-change\nbaseline",
        float(tf.loc["scalar_prev_somsc", "zero_delta_rmse"]),
        "Only using aggregate previous SOMSC barely improves over carrying the previous state forward.",
        left_color="#8D99AE",
        right_color="#D1D5DB",
    )
    contrast_panel(
        axes[1],
        "2. Hidden pool state?",
        "prev SOMSC\nscalar target",
        float(tf.loc["scalar_prev_somsc", "delta_rmse"]),
        "full pools\nscalar target",
        float(tf.loc["scalar_full_pools", "delta_rmse"]),
        "Showing the previous DayCent pool composition makes a scalar SOMSC-delta target much easier.",
        left_color="#8D99AE",
        right_color="#2A9D8F",
    )
    contrast_panel(
        axes[2],
        "3. Surface auxiliary?",
        "3 soil pools\nno surface",
        float(ro.loc["three_pool_no_aux", "delta_rmse"]),
        "3 soil pools\n+ surface",
        float(ro.loc["four_state_current", "delta_rmse"]),
        "Adding the surface auxiliary state sharply improves rollout, suggesting it carries transition information.",
        left_color="#E9C46A",
        right_color="#4E79A7",
    )

    fig.subplots_adjust(top=0.78, left=0.03, right=0.99, bottom=0.05, wspace=0.22)
    save(fig, out_dir, "06_ablation_question_map")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make simple ablation schema figures.")
    parser.add_argument(
        "--summary",
        default="daycent/output/yearly_december_somsc_state_v1_ablation_summary_fair.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="daycent/output/yearly_december_somsc_state_v1/ablation_figures",
    )
    args = parser.parse_args()

    set_style()
    summary = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir)
    figure_case_schemas(summary, out_dir)
    figure_question_map(summary, out_dir)
    print(f"Wrote ablation schema figures to: {out_dir}")


if __name__ == "__main__":
    main()
