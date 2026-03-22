"""Cost utilities for trailer loading decisions."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_mismatch_cost(
    trailer_weight: float | np.ndarray,
    tractor_weight: float | np.ndarray,
    distance_miles: float | np.ndarray,
    *,
    max_gross_weight_lbs: float = 80_000.0,
    underage_cost_per_lb_mile: float = 0.00005,
    overage_cost_per_lb: float = 0.023,
) -> float | np.ndarray:
    """Compute the mismatch cost for one or many shipments."""

    trailer = np.asarray(trailer_weight, dtype=float)
    tractor = np.asarray(tractor_weight, dtype=float)
    distance = np.asarray(distance_miles, dtype=float)

    total_weight = trailer + tractor
    underweight = np.maximum(max_gross_weight_lbs - total_weight, 0.0)
    overweight = np.maximum(total_weight - max_gross_weight_lbs, 0.0)

    return underweight * distance * underage_cost_per_lb_mile + overweight * overage_cost_per_lb


def compute_reward(**kwargs: float | np.ndarray) -> float | np.ndarray:
    """Return reward as the negative mismatch cost."""

    return -compute_mismatch_cost(**kwargs)


def choose_best_trailer(
    trailer_levels: Iterable[float],
    tractor_weight: float,
    distance_miles: float,
    **cost_kwargs: float,
) -> tuple[int, float, float]:
    """Choose the trailer level with the lowest immediate mismatch cost."""

    levels = np.asarray(list(trailer_levels), dtype=float)
    costs = compute_mismatch_cost(
        trailer_weight=levels,
        tractor_weight=tractor_weight,
        distance_miles=distance_miles,
        **cost_kwargs,
    )
    best_idx = int(np.argmin(costs))
    return best_idx, float(levels[best_idx]), float(costs[best_idx])
