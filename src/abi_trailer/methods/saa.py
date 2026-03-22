"""Sample average approximation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from abi_trailer.cost import compute_mismatch_cost


class SAAModel:
    """SAA model for static target weights and multi-level trailer design."""

    def __init__(
        self,
        *,
        max_gross_weight_lbs: float = 80_000.0,
        underage_cost_per_lb_mile: float = 0.00005,
        overage_cost_per_lb: float = 0.023,
        n_levels: int = 3,
    ) -> None:
        self.max_gross_weight_lbs = max_gross_weight_lbs
        self.underage_cost_per_lb_mile = underage_cost_per_lb_mile
        self.overage_cost_per_lb = overage_cost_per_lb
        self.n_levels = n_levels
        self.single_target_: float | None = None
        self.levels_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "SAAModel":
        weights = df["tractor_weight_lbs"].to_numpy(dtype=float)
        distances = df["distance_miles"].to_numpy(dtype=float)
        self.single_target_ = self._optimize_target(weights, distances)

        sorted_df = df.sort_values("tractor_weight_lbs").reset_index(drop=True)
        index_partitions = np.array_split(np.arange(len(sorted_df)), self.n_levels)
        levels = []
        for idxs in index_partitions:
            part = sorted_df.iloc[idxs]
            levels.append(
                self._optimize_target(
                    part["tractor_weight_lbs"].to_numpy(dtype=float),
                    part["distance_miles"].to_numpy(dtype=float),
                )
            )
        self.levels_ = np.asarray(levels, dtype=float)
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        if self.single_target_ is None:
            raise RuntimeError("Fit the model before calling predict().")
        return np.repeat(self.single_target_, n_rows)

    def recommend_levels(self) -> np.ndarray:
        if self.levels_ is None:
            raise RuntimeError("Fit the model before calling recommend_levels().")
        return self.levels_.copy()

    def _optimize_target(self, weights: np.ndarray, distances: np.ndarray) -> float:
        candidate_targets = np.unique(self.max_gross_weight_lbs - weights)
        costs = []
        for q in candidate_targets:
            cost = compute_mismatch_cost(
                trailer_weight=np.repeat(q, len(weights)),
                tractor_weight=weights,
                distance_miles=distances,
                max_gross_weight_lbs=self.max_gross_weight_lbs,
                underage_cost_per_lb_mile=self.underage_cost_per_lb_mile,
                overage_cost_per_lb=self.overage_cost_per_lb,
            ).mean()
            costs.append(cost)
        return float(candidate_targets[int(np.argmin(costs))])
