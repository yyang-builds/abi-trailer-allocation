from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import numpy as np

from abi_trailer.cost import choose_best_trailer, compute_mismatch_cost


def test_compute_mismatch_cost_underweight_case() -> None:
    cost = compute_mismatch_cost(
        trailer_weight=60_000,
        tractor_weight=18_000,
        distance_miles=100,
        max_gross_weight_lbs=80_000,
        underage_cost_per_lb_mile=0.00005,
        overage_cost_per_lb=0.023,
    )
    assert cost == 10.0


def test_choose_best_trailer_returns_lowest_cost_level() -> None:
    idx, trailer_weight, cost = choose_best_trailer(
        trailer_levels=np.array([60_000, 61_500, 63_000]),
        tractor_weight=18_500,
        distance_miles=120,
    )
    assert idx in {0, 1, 2}
    assert trailer_weight in {60_000.0, 61_500.0, 63_000.0}
    assert cost >= 0.0
