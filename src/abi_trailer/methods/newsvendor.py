"""Newsvendor benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class NewsvendorModel:
    """Gaussian benchmark for trailer target weights."""

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
        self.mean_weight_: float | None = None
        self.std_weight_: float | None = None

    def fit(self, df: pd.DataFrame) -> "NewsvendorModel":
        weights = df["tractor_weight_lbs"].to_numpy(dtype=float)
        self.mean_weight_ = float(weights.mean())
        self.std_weight_ = float(weights.std(ddof=1) if len(weights) > 1 else 0.0)
        return self

    def predict(self, distance_miles: pd.Series | np.ndarray | float) -> np.ndarray:
        if self.mean_weight_ is None or self.std_weight_ is None:
            raise RuntimeError("Fit the model before calling predict().")

        distance = np.asarray(distance_miles, dtype=float)
        underage = distance * self.underage_cost_per_lb_mile
        critical_ratio = underage / (underage + self.overage_cost_per_lb)
        critical_ratio = np.clip(critical_ratio, 1e-6, 1 - 1e-6)
        tractor_quantile = self.mean_weight_ + self.std_weight_ * norm.ppf(critical_ratio)
        return np.maximum(self.max_gross_weight_lbs - tractor_quantile, 0.0)
