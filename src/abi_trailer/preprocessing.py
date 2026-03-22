"""Preprocessing utilities for experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def split_train_test_by_date(df: pd.DataFrame, train_fraction: float = 0.67) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data along unique dates while preserving time order."""

    unique_dates = np.array(sorted(df["ship_date"].dt.normalize().unique()))
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct dates are required for a train/test split.")

    train_count = max(1, int(len(unique_dates) * train_fraction))
    train_count = min(train_count, len(unique_dates) - 1)
    train_dates = set(unique_dates[:train_count])

    train_df = df[df["ship_date"].dt.normalize().isin(train_dates)].copy()
    test_df = df[~df["ship_date"].dt.normalize().isin(train_dates)].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def even_inventory_counts(total_shipments: int, n_levels: int, extra_per_level: int = 0) -> list[int]:
    """Allocate near-even inventory counts across trailer levels."""

    base = total_shipments // n_levels
    remainder = total_shipments % n_levels
    counts = [base + extra_per_level for _ in range(n_levels)]
    for idx in range(remainder):
        counts[idx] += 1
    return counts


def make_tractor_bin_edges(weights: Iterable[float], width_lbs: float = 500.0) -> np.ndarray:
    """Construct evenly spaced tractor-weight bins."""

    values = np.asarray(list(weights), dtype=float)
    start = np.floor(values.min() / width_lbs) * width_lbs
    stop = np.ceil(values.max() / width_lbs) * width_lbs + width_lbs
    return np.arange(start, stop + width_lbs, width_lbs)


def bin_tractor_weight(weight: float, bin_edges: np.ndarray) -> int:
    """Map a tractor weight to an integer bin identifier."""

    return int(np.digitize([weight], bin_edges[1:-1], right=False)[0])
