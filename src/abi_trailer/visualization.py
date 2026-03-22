"""Visualization helpers for the public demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACCENT = "#0ea5a4"
ACCENT_DARK = "#0f766e"
BASELINE = "#cbd5e1"
BASELINE_DARK = "#64748b"
GRID = "#e2e8f0"
TEXT = "#0f172a"
SUBTEXT = "#475569"
CARRIER_COLORS = {
    "BBIG": "#94a3b8",
    "BVHG": "#64748b",
    "OTHER": "#cbd5e1",
    "UNKNOWN": "#cbd5e1",
}


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": SUBTEXT,
            "xtick.color": SUBTEXT,
            "ytick.color": SUBTEXT,
            "text.color": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "font.size": 10,
        }
    )


def _style_axes(ax: plt.Axes, *, show_x_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    if show_x_grid:
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def plot_distance_vs_weight(df: pd.DataFrame, output_path: str | Path) -> Path:
    _apply_theme()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    if "carrier_group" in df.columns:
        for carrier, group in df.groupby("carrier_group", sort=True):
            ax.scatter(
                group["distance_miles"],
                group["tractor_weight_lbs"],
                s=48,
                alpha=0.72,
                color=CARRIER_COLORS.get(str(carrier), BASELINE),
                edgecolors="white",
                linewidths=0.7,
                label=str(carrier),
            )
    else:
        ax.scatter(
            df["distance_miles"],
            df["tractor_weight_lbs"],
            s=48,
            alpha=0.72,
            color=BASELINE_DARK,
            edgecolors="white",
            linewidths=0.7,
        )

    slope, intercept = np.polyfit(df["distance_miles"], df["tractor_weight_lbs"], deg=1)
    x = np.linspace(df["distance_miles"].min(), df["distance_miles"].max(), 200)
    y = intercept + slope * x
    ax.plot(x, y, color=ACCENT, linewidth=2.8)
    ax.fill_between(x, y - 140, y + 140, color=ACCENT, alpha=0.08)

    corr = np.corrcoef(df["distance_miles"], df["tractor_weight_lbs"])[0, 1]
    ax.text(
        0.98,
        0.04,
        f"Positive relationship  |  r = {corr:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=ACCENT_DARK,
        bbox={"facecolor": "#f0fdfa", "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )

    ax.set_title("Distance and Tractor Weight")
    ax.set_xlabel("Delivery distance (miles)")
    ax.set_ylabel("Tractor weight (lbs)")
    ax.legend(frameon=False, ncol=min(df["carrier_group"].nunique(), 3), loc="upper left")
    _style_axes(ax)
    return _save_figure(fig, output_path)


def plot_method_cost_comparison(summary_df: pd.DataFrame, output_path: str | Path) -> Path:
    _apply_theme()
    chart_df = summary_df.sort_values("average_cost", ascending=False).copy()
    chart_df["label"] = chart_df["method"].str.upper()
    colors = [ACCENT if method == "dta" else BASELINE for method in chart_df["method"]]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.barh(chart_df["label"], chart_df["average_cost"], color=colors, edgecolor="none", height=0.64)

    ax.set_title("Average Mismatch Cost by Method")
    ax.set_xlabel("Average cost (lower is better)")
    ax.set_ylabel("")
    _style_axes(ax, show_x_grid=True)

    x_max = float(chart_df["average_cost"].max())
    ax.set_xlim(0, x_max * 1.2)

    for bar, method, value in zip(bars, chart_df["method"], chart_df["average_cost"], strict=True):
        ax.text(
            bar.get_width() + x_max * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color=ACCENT_DARK if method == "dta" else SUBTEXT,
            fontweight="bold" if method == "dta" else "normal",
        )

    best_row = chart_df.sort_values("average_cost", ascending=True).iloc[0]
    if best_row["method"] == "dta":
        best_y = chart_df.index[chart_df["method"] == "dta"][0]
        ax.text(
            x_max * 1.18,
            best_y,
            "Best",
            va="center",
            ha="right",
            fontsize=10,
            color=ACCENT_DARK,
            fontweight="bold",
        )

    return _save_figure(fig, output_path)


def plot_one_day_inventory_flow(dta_details: pd.DataFrame, output_path: str | Path) -> Path:
    _apply_theme()
    one_day = dta_details[dta_details["ship_date"] == dta_details["ship_date"].min()].reset_index(drop=True)
    x = np.arange(1, len(one_day) + 1)

    unique_levels = np.sort(one_day["selected_trailer_weight_lbs"].unique())
    level_map = {level: idx + 1 for idx, level in enumerate(unique_levels)}
    inverse_labels = [f"{int(level):,} lbs" for level in unique_levels]
    mapped_levels = one_day["selected_trailer_weight_lbs"].map(level_map)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    ax_top.plot(
        x,
        one_day["tractor_weight_lbs"],
        color=ACCENT_DARK,
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.6,
    )
    ax_top.fill_between(x, one_day["tractor_weight_lbs"], one_day["tractor_weight_lbs"].min() - 200, color=ACCENT, alpha=0.10)
    ax_top.set_title("One-Day Dynamic Assignment Flow")
    ax_top.set_ylabel("Tractor weight (lbs)")
    ax_top.text(
        0.01,
        0.92,
        "Observed tractor sequence",
        transform=ax_top.transAxes,
        fontsize=10,
        color=SUBTEXT,
    )
    _style_axes(ax_top)

    ax_bottom.step(x, mapped_levels, where="mid", color=ACCENT, linewidth=2.6)
    ax_bottom.scatter(x, mapped_levels, s=70, color=ACCENT, edgecolors="white", linewidths=1.0, zorder=3)
    ax_bottom.set_ylabel("Assigned trailer")
    ax_bottom.set_xlabel("Shipment sequence")
    ax_bottom.set_yticks(list(level_map.values()))
    ax_bottom.set_yticklabels(inverse_labels)
    ax_bottom.text(
        0.01,
        0.88,
        "DTA selects among trailer levels over time",
        transform=ax_bottom.transAxes,
        fontsize=10,
        color=SUBTEXT,
    )
    _style_axes(ax_bottom)

    if len(x) <= 12:
        ax_bottom.set_xticks(x)

    return _save_figure(fig, output_path)


def plot_inventory_sensitivity(sensitivity_df: pd.DataFrame, output_path: str | Path) -> Path:
    _apply_theme()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        sensitivity_df["extra_inventory_per_level"],
        sensitivity_df["average_cost"],
        marker="o",
        markersize=6,
        linewidth=2.2,
        color=ACCENT,
    )
    ax.set_title("Inventory Sensitivity")
    ax.set_xlabel("Extra inventory per level")
    ax.set_ylabel("Average cost")
    _style_axes(ax)
    return _save_figure(fig, output_path)
