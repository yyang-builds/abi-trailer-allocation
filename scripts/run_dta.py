"""Run the public DTA demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from abi_trailer.data_loader import load_shipments_csv, load_yaml_config
from abi_trailer.methods import DTAModel, SAAModel
from abi_trailer.preprocessing import split_train_test_by_date
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
    saa_model = SAAModel(n_levels=config["methods"]["n_levels"], **common_kwargs).fit(train_df)
    dta_model = DTAModel(
        trailer_levels=saa_model.recommend_levels(),
        epsilon=config["dta"]["epsilon"],
        gamma=config["dta"]["gamma"],
        tractor_bin_width_lbs=config["dta"]["tractor_bin_width_lbs"],
        bootstrap_episodes=config["dta"]["bootstrap_episodes"],
        extra_inventory_per_level=config["dta"]["extra_inventory_per_level"],
        random_seed=config["dta"]["random_seed"],
        **common_kwargs,
    ).fit(train_df)
    result = dta_model.evaluate(test_df)

    out_dir = ensure_output_dir(PROJECT_ROOT / config["paths"]["outputs_dir"])
    result.details.to_csv(out_dir / "dta_results.csv", index=False)
    print("DTA levels:", saa_model.recommend_levels().tolist())
    print("Average cost:", round(result.average_cost, 4))


if __name__ == "__main__":
    main()
