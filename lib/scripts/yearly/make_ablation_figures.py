"""
Create compact ablation figures for the yearly SOMSC state-model discovery.

Reads:
    daycent/output/yearly_december_somsc_state_v1_ablation_summary_fair.csv
    daycent/output/yearly_december_somsc_state_v1_ablation_pool_summary.csv

Writes PNG and SVG figures to:
    daycent/output/yearly_december_somsc_state_v1/ablation_figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_LABELS = {
    "four_state_current": "4-state\ncurrent",
    "scalar_full_pools": "Scalar target\nfull pools",
    "three_pool_no_aux": "3-pool\nno aux",
    "scalar_prev_somsc": "Scalar target\nprev SOMSC",
    "three_pool_somsc_context": "3-pool target\nprev SOMSC",
}
CASE_ORDER = [
    "four_state_current",
    "scalar_full_pools",
    "three_pool_no_aux",
    "scalar_prev_somsc",
    "three_pool_somsc_context",
]
MODE_LABELS = {
    "teacher_forced": "Teacher forced",
    "rollout": "Rollout",
}
MODE_COLORS = {
    "teacher_forced": "#2A9D8F",
    "rollout": "#4E79A7",
}
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


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linestyle": "-",
        }
    )


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.05,
        0.965,
        title,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111827",
    )
    fig.text(
        0.05,
        0.905,
        fill(subtitle, width=118),
        fontsize=12.5,
        ha="left",
        va="top",
        color="#4B5563",
    )


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["case_order"] = frame["case"].map({case: idx for idx, case in enumerate(CASE_ORDER)})
    return frame.sort_values(["case_order", "mode"])


def figure_delta_rmse(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    add_header(
        fig,
        "Ablation: annual SOMSC change is only easy with pool-state information",
        "Lower is better. The dashed line is the zero-change baseline: carry previous December SOMSC forward.",
    )

    plot_df = ordered_summary(summary)
    x_cases = [case for case in CASE_ORDER if case in set(plot_df["case"])]
    x = np.arange(len(x_cases))
    width = 0.34
    offsets = {"teacher_forced": -width / 2, "rollout": width / 2}

    for mode in ["teacher_forced", "rollout"]:
        values = []
        for case in x_cases:
            match = plot_df[(plot_df["case"] == case) & (plot_df["mode"] == mode)]
            values.append(float(match["delta_rmse"].iloc[0]) if not match.empty else np.nan)
        ax.bar(
            x + offsets[mode],
            values,
            width=width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
        for xpos, value in zip(x + offsets[mode], values):
            if np.isfinite(value):
                ax.text(
                    xpos,
                    value + 1.5,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#111827",
                )

    baseline = float(summary["zero_delta_rmse"].iloc[0])
    ax.axhline(
        baseline,
        color="#6B7280",
        lw=1.8,
        linestyle="--",
        label=f"Zero-change baseline ({baseline:.1f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in x_cases])
    ax.set_ylabel("Annual SOMSC change RMSE (g C m-2 yr-1)")
    ax.set_ylim(0, max(70, baseline + 10))
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.82), frameon=False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.78, left=0.09, right=0.98, bottom=0.18)
    save(fig, out_dir, "01_ablation_delta_rmse")


def figure_improvement(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    add_header(
        fig,
        "Improvement over no-change is concentrated in the pool-aware models",
        "Percent reduction in annual-change RMSE relative to predicting zero annual SOMSC change.",
    )

    plot_df = ordered_summary(summary)
    x_cases = [case for case in CASE_ORDER if case in set(plot_df["case"])]
    x = np.arange(len(x_cases))
    width = 0.34
    offsets = {"teacher_forced": -width / 2, "rollout": width / 2}

    for mode in ["teacher_forced", "rollout"]:
        values = []
        for case in x_cases:
            match = plot_df[(plot_df["case"] == case) & (plot_df["mode"] == mode)]
            values.append(
                float(match["improvement_vs_zero_delta_pct"].iloc[0])
                if not match.empty
                else np.nan
            )
        ax.bar(
            x + offsets[mode],
            values,
            width=width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
        for xpos, value in zip(x + offsets[mode], values):
            if np.isfinite(value):
                ax.text(
                    xpos,
                    value + 1.2,
                    f"{value:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#111827",
                )

    ax.axhline(0, color="#374151", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in x_cases])
    ax.set_ylabel("RMSE reduction vs zero-change baseline")
    ax.set_ylim(-5, 65)
    ax.legend(loc="upper right", frameon=False)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.78, left=0.09, right=0.98, bottom=0.18)
    save(fig, out_dir, "02_improvement_over_zero_change")


def figure_key_contrasts(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=True)
    add_header(
        fig,
        "Two mechanisms explain the gain",
        "Full previous pool composition makes even a scalar target work; adding the surface auxiliary improves rollout sharply.",
    )

    teacher = summary[summary["mode"] == "teacher_forced"].set_index("case")
    rollout = summary[summary["mode"] == "rollout"].set_index("case")

    panels = [
        (
            axes[0],
            teacher,
            ["scalar_prev_somsc", "scalar_full_pools", "four_state_current"],
            "State observability\n(teacher forced)",
            ["prev SOMSC\nscalar target", "full pools\nscalar target", "full pools\npool target"],
        ),
        (
            axes[1],
            rollout,
            ["three_pool_no_aux", "four_state_current"],
            "Surface auxiliary effect\n(rollout)",
            ["3 soil pools", "3 soil pools\n+ surface aux"],
        ),
    ]

    for ax, frame, cases, title, labels in panels:
        values = [float(frame.loc[case, "delta_rmse"]) for case in cases]
        colors = ["#8D99AE", "#2A9D8F", "#4E79A7"][: len(cases)]
        bars = ax.bar(np.arange(len(cases)), values, color=colors, width=0.62)
        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xticks(np.arange(len(cases)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 70)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.3,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    axes[0].set_ylabel("Annual SOMSC change RMSE (g C m-2 yr-1)")
    fig.subplots_adjust(top=0.74, left=0.08, right=0.98, bottom=0.20, wspace=0.15)
    save(fig, out_dir, "03_key_contrasts")


def figure_pool_behavior(pool_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12.5, 7.0))
    add_header(
        fig,
        "Errors concentrate in biologically active pools, not just the largest stock",
        "Final 4-state model, teacher-forced evaluation. SOM3C is large but stable; SOM1C is small but dynamic.",
    )

    data = pool_summary[
        (pool_summary["case"] == "four_state_current")
        & (pool_summary["mode"] == "teacher_forced")
    ].copy()
    order = ["som1c_soil", "som2c_soil", "som3c", "som2c_surface"]
    data["pool_order"] = data["pool"].map({pool: idx for idx, pool in enumerate(order)})
    data = data.sort_values("pool_order")

    x = np.arange(len(data))
    width = 0.34
    stock_bars = ax1.bar(
        x - width / 2,
        data["stock_mean"],
        width=width,
        color="#D1D5DB",
        label="Mean stock",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean stock (g C m-2, log scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([POOL_LABELS[p] for p in data["pool"]])
    ax1.set_ylim(1, max(data["stock_mean"]) * 2.8)

    ax2 = ax1.twinx()
    delta_points = ax2.scatter(
        x + width / 2,
        data["delta_rmse"],
        s=115,
        color=[POOL_COLORS[p] for p in data["pool"]],
        edgecolor="white",
        linewidth=1.5,
        label="Delta RMSE",
        zorder=4,
    )
    ax2.plot(
        x + width / 2,
        data["delta_rmse"],
        color="#111827",
        linewidth=1.2,
        alpha=0.45,
        zorder=3,
    )
    ax2.set_ylabel("Annual delta RMSE (g C m-2 yr-1)")
    ax2.set_ylim(0, max(data["delta_rmse"]) * 1.35)

    for xpos, row in zip(x + width / 2, data.itertuples()):
        ax2.text(
            xpos,
            row.delta_rmse + 0.9,
            f"{row.delta_rmse:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )

    ax1.legend(handles=[stock_bars, delta_points], loc="upper left", frameon=False)
    fig.subplots_adjust(top=0.78, left=0.10, right=0.90, bottom=0.18)
    save(fig, out_dir, "04_pool_stock_vs_delta_error")


def write_email(summary: pd.DataFrame, out_dir: Path) -> None:
    tf = summary[summary["mode"] == "teacher_forced"].set_index("case")
    rollout = summary[summary["mode"] == "rollout"].set_index("case")
    four_tf = tf.loc["four_state_current"]
    four_rollout = rollout.loc["four_state_current"]
    scalar_full = tf.loc["scalar_full_pools"]
    scalar_prev = tf.loc["scalar_prev_somsc"]
    three_rollout = rollout.loc["three_pool_no_aux"]

    email = f"""Subject: Interesting SOMSC ablation result: pool state seems to be the key

Hi all,

I finished a small ablation study on the yearly SOMSC model, and the result is pretty interesting. The main gain does not seem to come simply from using an LSTM or predicting annual deltas. It comes from representing the previous soil carbon state as DayCent pools instead of only using aggregate SOMSC.

The strongest comparison is:

- Scalar SOMSC-delta model with only previous aggregate SOMSC: RMSE {scalar_prev.delta_rmse:.1f} g C/m2/yr
- Scalar SOMSC-delta model with previous full pool state: RMSE {scalar_full.delta_rmse:.1f} g C/m2/yr
- Current 4-state pool-transition model: teacher-forced RMSE {four_tf.delta_rmse:.1f}, rollout RMSE {four_rollout.delta_rmse:.1f} g C/m2/yr

The zero-change baseline is {four_tf.zero_delta_rmse:.1f} g C/m2/yr, so the current model reduces annual-change RMSE by about {four_tf.improvement_vs_zero_delta_pct:.0f}% teacher-forced and {four_rollout.improvement_vs_zero_delta_pct:.0f}% in rollout.

This suggests aggregate SOMSC is a lossy state variable: two sites can have similar total SOMSC but different pool composition, and therefore different next-year dynamics. Once the model sees the pool composition, the transition becomes much easier to learn.

The auxiliary surface pool also seems important. Removing `som2c_surface` worsens rollout annual-change RMSE from {four_rollout.delta_rmse:.1f} to {three_rollout.delta_rmse:.1f} g C/m2/yr, which suggests that the surface/residue-like state carries useful information for subsequent soil-pool updates.

I also looked at pool-level errors. The errors are not simply largest where the stock is largest: SOM3C is a very large passive stock but has tiny annual movement, while SOM1C_soil is a small active pool with much larger relative turnover. This supports the idea that the model is learning a structured transition over biologically different carbon pools, rather than just fitting aggregate level persistence.

I put the ablation figures here:
{out_dir}

Best,
"""
    path = out_dir / "email_draft.md"
    path.write_text(email)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make yearly SOMSC ablation figures.")
    parser.add_argument(
        "--summary",
        default="daycent/output/yearly_december_somsc_state_v1_ablation_summary_fair.csv",
        help="Ablation summary CSV.",
    )
    parser.add_argument(
        "--pool-summary",
        default="daycent/output/yearly_december_somsc_state_v1_ablation_pool_summary.csv",
        help="Pool-level ablation summary CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="daycent/output/yearly_december_somsc_state_v1/ablation_figures",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    pool_summary = pd.read_csv(args.pool_summary)
    out_dir = Path(args.out_dir)
    set_style()

    figure_delta_rmse(summary, out_dir)
    figure_improvement(summary, out_dir)
    figure_key_contrasts(summary, out_dir)
    figure_pool_behavior(pool_summary, out_dir)
    write_email(summary, out_dir)

    print(f"Wrote ablation figures and email draft to: {out_dir}")


if __name__ == "__main__":
    main()
