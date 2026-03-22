"""Experiment orchestration and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from abi_trailer.data_loader import load_shipments_csv, load_yaml_config
from abi_trailer.methods import DTAModel, ERMModel, NewsvendorModel, SAAModel
from abi_trailer.preprocessing import split_train_test_by_date
from abi_trailer.simulation import simulate_direct_targets, simulate_greedy_inventory_policy


def run_public_experiment(project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Run the bundled public demo experiment."""

    project_root = Path(project_root)
    config = load_yaml_config(config_path)
    data_path = project_root / config["data"]["path"]
    df = load_shipments_csv(data_path)
    train_df, test_df = split_train_test_by_date(df, train_fraction=config["split"]["train_fraction"])

    cost_cfg = config["cost"]
    method_cfg = config["methods"]
    dta_cfg = config["dta"]
    common_kwargs = {
        "max_gross_weight_lbs": cost_cfg["max_gross_weight_lbs"],
        "underage_cost_per_lb_mile": cost_cfg["underage_cost_per_lb_mile"],
        "overage_cost_per_lb": cost_cfg["overage_cost_per_lb"],
    }

    newsvendor = NewsvendorModel(**common_kwargs).fit(train_df)
    newsvendor_results = simulate_direct_targets(
        test_df,
        newsvendor.predict(test_df["distance_miles"]),
        method_name="newsvendor",
        **common_kwargs,
    )

    saa = SAAModel(n_levels=method_cfg["n_levels"], **common_kwargs).fit(train_df)
    saa_results = simulate_greedy_inventory_policy(
        test_df,
        saa.recommend_levels(),
        method_name="saa",
        **common_kwargs,
    )

    erm = ERMModel(**common_kwargs).fit(train_df)
    erm_results = simulate_direct_targets(
        test_df,
        erm.predict(test_df["distance_miles"]),
        method_name="erm",
        **common_kwargs,
    )

    dta = DTAModel(
        trailer_levels=saa.recommend_levels(),
        epsilon=dta_cfg["epsilon"],
        gamma=dta_cfg["gamma"],
        tractor_bin_width_lbs=dta_cfg["tractor_bin_width_lbs"],
        bootstrap_episodes=dta_cfg["bootstrap_episodes"],
        extra_inventory_per_level=dta_cfg["extra_inventory_per_level"],
        random_seed=dta_cfg["random_seed"],
        **common_kwargs,
    ).fit(train_df)
    dta_result = dta.evaluate(test_df)
    dta_results = dta_result.details.copy()
    dta_results["method"] = "dta"

    all_results = pd.concat(
        [newsvendor_results, saa_results, erm_results, dta_results],
        ignore_index=True,
        sort=False,
    )
    summary = (
        all_results.groupby("method", as_index=False)["cost"]
        .mean()
        .sort_values("cost", ascending=True)
        .reset_index(drop=True)
        .rename(columns={"cost": "average_cost"})
    )

    return {
        "config": config,
        "train_df": train_df,
        "test_df": test_df,
        "results": all_results,
        "summary": summary,
        "models": {
            "newsvendor": newsvendor,
            "saa": saa,
            "erm": erm,
            "dta": dta,
        },
    }


def inventory_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trailer_levels: list[float] | pd.Series,
    config: dict[str, Any],
    extra_inventory_values: list[int],
) -> pd.DataFrame:
    """Evaluate DTA sensitivity to additional inventory."""

    rows = []
    common_kwargs = {
        "max_gross_weight_lbs": config["cost"]["max_gross_weight_lbs"],
        "underage_cost_per_lb_mile": config["cost"]["underage_cost_per_lb_mile"],
        "overage_cost_per_lb": config["cost"]["overage_cost_per_lb"],
    }

    for extra in extra_inventory_values:
        model = DTAModel(
            trailer_levels=list(trailer_levels),
            epsilon=config["dta"]["epsilon"],
            gamma=config["dta"]["gamma"],
            tractor_bin_width_lbs=config["dta"]["tractor_bin_width_lbs"],
            bootstrap_episodes=config["dta"]["bootstrap_episodes"],
            extra_inventory_per_level=extra,
            random_seed=config["dta"]["random_seed"],
            **common_kwargs,
        ).fit(train_df)
        result = model.evaluate(test_df)
        rows.append({"extra_inventory_per_level": extra, "average_cost": result.average_cost})

    return pd.DataFrame(rows)
