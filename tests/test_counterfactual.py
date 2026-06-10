from __future__ import annotations

import pandas as pd

from run_counterfactual import compare_scenarios


def test_compare_scenarios_single_model_filters_can_use_different_model_names() -> None:
    actual = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "model_name": ["actual_model", "actual_model"],
            "forecast": [110.0, 120.0],
        }
    )
    counterfactual = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "model_name": ["cf_model", "cf_model"],
            "forecast": [100.0, 90.0],
        }
    )

    out = compare_scenarios(actual, counterfactual)

    assert out["incremental_sales"].tolist() == [10.0, 30.0]
    assert out["actual_model"].unique().tolist() == ["actual_model"]
    assert out["counterfactual_model"].unique().tolist() == ["cf_model"]


def test_compare_scenarios_preserves_intervals() -> None:
    actual = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "model_name": ["m"],
            "forecast": [120.0],
            "p10": [100.0],
            "p90": [140.0],
            "p025": [90.0],
            "p975": [150.0],
        }
    )
    counterfactual = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "model_name": ["m"],
            "forecast": [100.0],
            "p10": [80.0],
            "p90": [120.0],
            "p025": [70.0],
            "p975": [130.0],
        }
    )

    out = compare_scenarios(actual, counterfactual)

    assert float(out.loc[0, "incremental_lower_80"]) == -20.0
    assert float(out.loc[0, "incremental_upper_80"]) == 60.0
    assert float(out.loc[0, "incremental_lower_95"]) == -40.0
    assert float(out.loc[0, "incremental_upper_95"]) == 80.0

