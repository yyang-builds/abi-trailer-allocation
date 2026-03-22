"""Generate public demo figures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from abi_trailer.evaluation import inventory_sensitivity, run_public_experiment
from abi_trailer.visualization import (
    ensure_output_dir,
    plot_distance_vs_weight,
    plot_inventory_sensitivity,
    plot_method_cost_comparison,
    plot_one_day_inventory_flow,
)


def main() -> None:
    artifacts = run_public_experiment(PROJECT_ROOT, PROJECT_ROOT / "configs" / "experiment_public.yaml")
    out_dir = ensure_output_dir(PROJECT_ROOT / artifacts["config"]["paths"]["outputs_dir"])

    plot_distance_vs_weight(artifacts["train_df"], out_dir / "distance_vs_weight.png")
    plot_method_cost_comparison(artifacts["summary"], out_dir / "method_cost_comparison.png")

    dta_details = artifacts["results"][artifacts["results"]["method"] == "dta"].copy()
    plot_one_day_inventory_flow(dta_details, out_dir / "one_day_assignment_flow.png")

    sensitivity = inventory_sensitivity(
        artifacts["train_df"],
        artifacts["test_df"],
        artifacts["models"]["saa"].recommend_levels(),
        artifacts["config"],
        extra_inventory_values=[0, 1, 2, 3],
    )
    sensitivity.to_csv(out_dir / "inventory_sensitivity.csv", index=False)
    plot_inventory_sensitivity(sensitivity, out_dir / "inventory_sensitivity.png")

    artifacts["summary"].to_csv(out_dir / "summary.csv", index=False)
    print("Figures written to:", out_dir)


if __name__ == "__main__":
    main()
