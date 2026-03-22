from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from abi_trailer.data_loader import load_yaml_config


def test_config_inheritance_loads_defaults() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs" / "experiment_public.yaml")
    assert config["data"]["path"] == "data/demo_synthetic_shipments.csv"
    assert config["dta"]["bootstrap_episodes"] == 750
    assert config["cost"]["max_gross_weight_lbs"] == 80_000
