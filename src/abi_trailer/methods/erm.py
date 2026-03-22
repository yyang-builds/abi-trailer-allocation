"""Empirical risk minimization with a linear trailer-weight rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ERMModel:
    """Learn a linear trailer-weight rule q(d) = q0 + q1 d."""

    def __init__(
        self,
        *,
        max_gross_weight_lbs: float = 80_000.0,
        underage_cost_per_lb_mile: float = 0.00005,
        overage_cost_per_lb: float = 0.023,
    ) -> None:
        self.max_gross_weight_lbs = max_gross_weight_lbs
        self.underage_cost_per_lb_mile = underage_cost_per_lb_mile
        self.overage_cost_per_lb = overage_cost_per_lb
        self.intercept_: float | None = None
        self.slope_: float | None = None

    def fit(self, df: pd.DataFrame) -> "ERMModel":
        distances = df["distance_miles"].to_numpy(dtype=float)
        weights = df["tractor_weight_lbs"].to_numpy(dtype=float)
        n = len(df)

        n_vars = 4 + 2 * n
        c = np.zeros(n_vars)
        c[4 : 4 + n] = distances * self.underage_cost_per_lb_mile / n
        c[4 + n :] = self.overage_cost_per_lb / n

        a_ub = []
        b_ub = []

        for idx, (distance, weight) in enumerate(zip(distances, weights, strict=True)):
            row_u = np.zeros(n_vars)
            row_u[0] = -1.0
            row_u[1] = 1.0
            row_u[2] = -distance
            row_u[3] = distance
            row_u[4 + idx] = -1.0
            a_ub.append(row_u)
            b_ub.append(-(self.max_gross_weight_lbs - weight))

            row_v = np.zeros(n_vars)
            row_v[0] = 1.0
            row_v[1] = -1.0
            row_v[2] = distance
            row_v[3] = -distance
            row_v[4 + n + idx] = -1.0
            a_ub.append(row_v)
            b_ub.append(self.max_gross_weight_lbs - weight)

        bounds = [(0.0, None)] * n_vars
        result = linprog(c=c, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(f"ERM optimization failed: {result.message}")

        solution = result.x
        self.intercept_ = float(solution[0] - solution[1])
        self.slope_ = float(solution[2] - solution[3])
        return self

    def predict(self, distance_miles: pd.Series | np.ndarray | float) -> np.ndarray:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("Fit the model before calling predict().")
        distance = np.asarray(distance_miles, dtype=float)
        return np.maximum(self.intercept_ + self.slope_ * distance, 0.0)
