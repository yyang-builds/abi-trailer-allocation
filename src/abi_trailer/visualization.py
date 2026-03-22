"""Visualization helpers for the public demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_distance_vs_weight(df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["distance_miles"], df["tractor_weight_lbs"], alpha=0.8)
    slope, intercept = np.polyfit(df["distance_miles"], df["tractor_weight_lbs"], deg=1)
    x = np.linspace(df["distance_miles"].min(), df["distance_miles"].max(), 100)
    ax.plot(x, intercept + slope * x, color="tab:red")
    ax.set_title("Distance vs. Tractor Weight")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Tractor weight (lbs)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return Path(output_path)


def plot_method_cost_comparison(summary_df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary_df["method"], summary_df["average_cost"], color="tab:blue")
    ax.set_title("Average Mismatch Cost by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Average cost")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return Path(output_path)


def plot_one_day_inventory_flow(dta_details: pd.DataFrame, output_path: str | Path) -> Path:
    one_day = dta_details[dta_details["ship_date"] == dta_details["ship_date"].min()].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(one_day.index + 1, one_day["tractor_weight_lbs"], label="Tractor weight", marker="o")
    ax.plot(one_day.index + 1, one_day["selected_trailer_weight_lbs"], label="Selected trailer", marker="s")
    ax.set_title("One-Day Assignment Flow")
    ax.set_xlabel("Shipment sequence")
    ax.set_ylabel("Weight (lbs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return Path(output_path)


def plot_inventory_sensitivity(sensitivity_df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        sensitivity_df["extra_inventory_per_level"],
        sensitivity_df["average_cost"],
        marker="o",
        color="tab:green",
    )
    ax.set_title("Inventory Sensitivity")
    ax.set_xlabel("Extra inventory per level")
    ax.set_ylabel("Average cost")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return Path(output_path)
