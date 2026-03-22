"""Data and config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "shipment_id",
    "ship_date",
    "distance_miles",
    "tractor_weight_lbs",
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "inherits" in config:
        inherited_path = Path(path).resolve().parents[1] / Path(config["inherits"])
        base = load_yaml_config(inherited_path)
        config = deep_merge(base, {k: v for k, v in config.items() if k != "inherits"})

    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_shipments_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate shipment data."""

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ship_date"] = pd.to_datetime(df["ship_date"])
    if "carrier_group" not in df.columns:
        df["carrier_group"] = "UNKNOWN"
    return df.sort_values(["ship_date", "shipment_id"]).reset_index(drop=True)


def load_demo_dataset(project_root: str | Path) -> pd.DataFrame:
    """Load the synthetic demo dataset bundled with the repository."""

    return load_shipments_csv(Path(project_root) / "data" / "demo_synthetic_shipments.csv")
