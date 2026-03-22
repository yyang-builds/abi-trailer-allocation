from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from abi_trailer.data_loader import load_demo_dataset
from abi_trailer.methods import DTAModel, ERMModel, NewsvendorModel, SAAModel
from abi_trailer.preprocessing import split_train_test_by_date


def test_core_models_fit_and_predict() -> None:
    df = load_demo_dataset(PROJECT_ROOT)
    train_df, test_df = split_train_test_by_date(df, train_fraction=0.67)

    nv = NewsvendorModel().fit(train_df)
    saa = SAAModel().fit(train_df)
    erm = ERMModel().fit(train_df)

    assert len(nv.predict(test_df["distance_miles"])) == len(test_df)
    assert len(saa.recommend_levels()) == 3
    assert len(erm.predict(test_df["distance_miles"])) == len(test_df)


def test_dta_smoke_evaluation() -> None:
    df = load_demo_dataset(PROJECT_ROOT)
    train_df, test_df = split_train_test_by_date(df, train_fraction=0.67)
    levels = SAAModel().fit(train_df).recommend_levels()

    dta = DTAModel(trailer_levels=levels, bootstrap_episodes=20, random_seed=7).fit(train_df)
    result = dta.evaluate(test_df)

    assert result.average_cost >= 0.0
    assert not result.details.empty
