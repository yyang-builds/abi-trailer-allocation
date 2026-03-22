"""Simulation helpers for static and dynamic policies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from abi_trailer.cost import choose_best_trailer, compute_mismatch_cost
from abi_trailer.preprocessing import even_inventory_counts


def simulate_direct_targets(
    df: pd.DataFrame,
    target_weights: np.ndarray,
    *,
    max_gross_weight_lbs: float = 80_000.0,
    underage_cost_per_lb_mile: float = 0.00005,
    overage_cost_per_lb: float = 0.023,
    method_name: str = "direct",
) -> pd.DataFrame:
    """Evaluate one target weight per shipment."""

    result = df.copy()
    result["selected_trailer_weight_lbs"] = target_weights
    result["cost"] = compute_mismatch_cost(
        trailer_weight=result["selected_trailer_weight_lbs"].to_numpy(dtype=float),
        tractor_weight=result["tractor_weight_lbs"].to_numpy(dtype=float),
        distance_miles=result["distance_miles"].to_numpy(dtype=float),
        max_gross_weight_lbs=max_gross_weight_lbs,
        underage_cost_per_lb_mile=underage_cost_per_lb_mile,
        overage_cost_per_lb=overage_cost_per_lb,
    )
    result["method"] = method_name
    return result


def simulate_greedy_inventory_policy(
    df: pd.DataFrame,
    trailer_levels: np.ndarray,
    *,
    extra_inventory_per_level: int = 0,
    max_gross_weight_lbs: float = 80_000.0,
    underage_cost_per_lb_mile: float = 0.00005,
    overage_cost_per_lb: float = 0.023,
    method_name: str = "inventory_policy",
) -> pd.DataFrame:
    """Evaluate a static set of trailer levels with day-level greedy matching."""

    records: list[dict[str, float | int | str]] = []
    trailer_levels = np.asarray(trailer_levels, dtype=float)

    for ship_date, day_df in df.groupby(df["ship_date"].dt.normalize(), sort=True):
        inventory = even_inventory_counts(
            total_shipments=len(day_df),
            n_levels=len(trailer_levels),
            extra_per_level=extra_inventory_per_level,
        )

        for _, row in day_df.iterrows():
            available_idx = [idx for idx, count in enumerate(inventory) if count > 0]
            available_levels = trailer_levels[available_idx]
            best_local_idx, trailer_weight, cost = choose_best_trailer(
                trailer_levels=available_levels,
                tractor_weight=float(row["tractor_weight_lbs"]),
                distance_miles=float(row["distance_miles"]),
                max_gross_weight_lbs=max_gross_weight_lbs,
                underage_cost_per_lb_mile=underage_cost_per_lb_mile,
                overage_cost_per_lb=overage_cost_per_lb,
            )
            selected_idx = available_idx[best_local_idx]
            inventory[selected_idx] -= 1
            records.append(
                {
                    "ship_date": str(ship_date.date()),
                    "shipment_id": row["shipment_id"],
                    "distance_miles": float(row["distance_miles"]),
                    "tractor_weight_lbs": float(row["tractor_weight_lbs"]),
                    "selected_level_idx": int(selected_idx),
                    "selected_trailer_weight_lbs": float(trailer_weight),
                    "cost": float(cost),
                    "method": method_name,
                }
            )

    return pd.DataFrame(records)
