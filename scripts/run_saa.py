"""Run the public SAA demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from abi_trailer.data_loader import load_shipments_csv, load_yaml_config
from abi_trailer.methods import SAAModel
from abi_trailer.preprocessing import split_train_test_by_date
from abi_trailer.simulation import simulate_greedy_inventory_policy
from abi_trailer.visualization import ensure_output_dir


def main() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs" / "experiment_public.yaml")
    df = load_shipments_csv(PROJECT_ROOT / config["data"]["path"])
    train_df, test_df = split_train_test_by_date(df, config["split"]["train_fraction"])

    common_kwargs = {
        "max_gross_weight_lbs": config["cost"]["max_gross_weight_lbs"],
        "underage_cost_per_lb_mile": config["cost"]["underage_cost_per_lb_mile"],
        "overage_cost_per_lb": config["cost"]["overage_cost_per_lb"],
    }
    model = SAAModel(n_levels=config["methods"]["n_levels"], **common_kwargs).fit(train_df)
    result = simulate_greedy_inventory_policy(test_df, model.recommend_levels(), method_name="saa", **common_kwargs)

    out_dir = ensure_output_dir(PROJECT_ROOT / config["paths"]["outputs_dir"])
    result.to_csv(out_dir / "saa_results.csv", index=False)
    print("SAA levels:", model.recommend_levels().tolist())
    print("Average cost:", round(float(result["cost"].mean()), 4))


if __name__ == "__main__":
    main()
