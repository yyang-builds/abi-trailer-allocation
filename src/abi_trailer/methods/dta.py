"""Dynamic trailer assignment with tabular Q-learning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from abi_trailer.cost import compute_mismatch_cost
from abi_trailer.preprocessing import bin_tractor_weight, even_inventory_counts, make_tractor_bin_edges


@dataclass
class DTAResult:
    average_cost: float
    details: pd.DataFrame


class DTAModel:
    """Train a tabular Q-learning policy for dynamic trailer assignment."""

    def __init__(
        self,
        trailer_levels: list[float] | np.ndarray,
        *,
        max_gross_weight_lbs: float = 80_000.0,
        underage_cost_per_lb_mile: float = 0.00005,
        overage_cost_per_lb: float = 0.023,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        tractor_bin_width_lbs: float = 500.0,
        bootstrap_episodes: int = 500,
        extra_inventory_per_level: int = 0,
        random_seed: int = 42,
    ) -> None:
        self.trailer_levels = np.asarray(trailer_levels, dtype=float)
        self.max_gross_weight_lbs = max_gross_weight_lbs
        self.underage_cost_per_lb_mile = underage_cost_per_lb_mile
        self.overage_cost_per_lb = overage_cost_per_lb
        self.epsilon = epsilon
        self.gamma = gamma
        self.tractor_bin_width_lbs = tractor_bin_width_lbs
        self.bootstrap_episodes = bootstrap_episodes
        self.extra_inventory_per_level = extra_inventory_per_level
        self.random_seed = random_seed
        self.q_table_: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(self._zero_values)
        self.visit_counts_: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(self._zero_values)
        self.bin_edges_: np.ndarray | None = None

    def _zero_values(self) -> np.ndarray:
        return np.zeros(len(self.trailer_levels), dtype=float)

    def fit(self, df: pd.DataFrame) -> "DTAModel":
        rng = np.random.default_rng(self.random_seed)
        self.bin_edges_ = make_tractor_bin_edges(df["tractor_weight_lbs"], self.tractor_bin_width_lbs)

        daily_sizes = (
            df.groupby(df["ship_date"].dt.normalize(), sort=True)
            .size()
            .to_numpy(dtype=int)
        )
        rows = df[["tractor_weight_lbs", "distance_miles"]].to_numpy(dtype=float)

        for _ in range(self.bootstrap_episodes):
            n_shipments = int(rng.choice(daily_sizes))
            sample_idx = rng.integers(0, len(rows), size=n_shipments)
            sampled = rows[sample_idx]
            inventory = tuple(
                even_inventory_counts(
                    total_shipments=n_shipments,
                    n_levels=len(self.trailer_levels),
                    extra_per_level=self.extra_inventory_per_level,
                )
            )

            for step in range(n_shipments):
                tractor_weight, distance = sampled[step]
                tractor_bin = bin_tractor_weight(float(tractor_weight), self.bin_edges_)
                state = inventory + (tractor_bin,)
                action = self._choose_action(state, explore=True, rng=rng)
                reward = -float(
                    compute_mismatch_cost(
                        trailer_weight=self.trailer_levels[action],
                        tractor_weight=tractor_weight,
                        distance_miles=distance,
                        max_gross_weight_lbs=self.max_gross_weight_lbs,
                        underage_cost_per_lb_mile=self.underage_cost_per_lb_mile,
                        overage_cost_per_lb=self.overage_cost_per_lb,
                    )
                )
                next_inventory = list(inventory)
                next_inventory[action] -= 1
                inventory = tuple(next_inventory)

                next_value = 0.0
                if step < n_shipments - 1 and sum(inventory) > 0:
                    next_bin = bin_tractor_weight(float(sampled[step + 1][0]), self.bin_edges_)
                    next_state = inventory + (next_bin,)
                    available = self._available_actions(next_state)
                    next_value = float(np.max(self.q_table_[next_state][available])) if available else 0.0

                self.visit_counts_[state][action] += 1.0
                alpha = 1.0 / self.visit_counts_[state][action]
                td_target = reward + self.gamma * next_value
                self.q_table_[state][action] += alpha * (td_target - self.q_table_[state][action])

        return self

    def evaluate(self, df: pd.DataFrame) -> DTAResult:
        if self.bin_edges_ is None:
            raise RuntimeError("Fit the model before calling evaluate().")

        records: list[dict[str, float | int | str]] = []
        total_costs: list[float] = []

        for ship_date, day_df in df.groupby(df["ship_date"].dt.normalize(), sort=True):
            inventory = tuple(
                even_inventory_counts(
                    total_shipments=len(day_df),
                    n_levels=len(self.trailer_levels),
                    extra_per_level=self.extra_inventory_per_level,
                )
            )
            for _, row in day_df.iterrows():
                tractor_bin = bin_tractor_weight(float(row["tractor_weight_lbs"]), self.bin_edges_)
                state = inventory + (tractor_bin,)
                action = self._choose_action(state, explore=False, rng=None)
                trailer_weight = float(self.trailer_levels[action])
                cost = float(
                    compute_mismatch_cost(
                        trailer_weight=trailer_weight,
                        tractor_weight=float(row["tractor_weight_lbs"]),
                        distance_miles=float(row["distance_miles"]),
                        max_gross_weight_lbs=self.max_gross_weight_lbs,
                        underage_cost_per_lb_mile=self.underage_cost_per_lb_mile,
                        overage_cost_per_lb=self.overage_cost_per_lb,
                    )
                )
                total_costs.append(cost)
                records.append(
                    {
                        "ship_date": str(ship_date.date()),
                        "shipment_id": row["shipment_id"],
                        "distance_miles": float(row["distance_miles"]),
                        "tractor_weight_lbs": float(row["tractor_weight_lbs"]),
                        "selected_level_idx": int(action),
                        "selected_trailer_weight_lbs": trailer_weight,
                        "cost": cost,
                        "inventory_before": str(inventory),
                    }
                )
                next_inventory = list(inventory)
                next_inventory[action] -= 1
                inventory = tuple(next_inventory)

        return DTAResult(average_cost=float(np.mean(total_costs)), details=pd.DataFrame(records))

    def _available_actions(self, state: tuple[int, ...]) -> list[int]:
        inventory = state[:-1]
        return [idx for idx, count in enumerate(inventory) if count > 0]

    def _choose_action(
        self,
        state: tuple[int, ...],
        *,
        explore: bool,
        rng: np.random.Generator | None,
    ) -> int:
        available = self._available_actions(state)
        if not available:
            raise RuntimeError("No available actions for the current inventory state.")

        if explore and rng is not None and rng.random() < self.epsilon:
            return int(rng.choice(available))

        q_values = self.q_table_[state]
        best_local_idx = int(np.argmax(q_values[available]))
        return int(available[best_local_idx])
