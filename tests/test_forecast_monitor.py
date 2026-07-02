from __future__ import annotations

import pandas as pd

from run_forecast_monitor import append_monitor_rows, cusum_series


def test_cusum_series_accumulates_one_sided_bias() -> None:
    bias = pd.Series([0.03, 0.03, -0.02], index=pd.date_range("2026-01-01", periods=3, freq="D"))

    out = cusum_series(bias, slack=0.01)

    assert out["positive_cusum"].tolist() == [0.019999999999999997, 0.039999999999999994, 0.009999999999999995]
    assert out["negative_cusum"].tolist() == [0.0, 0.0, 0.01]


def test_append_monitor_rows_is_idempotent(tmp_path) -> None:
    log_csv = tmp_path / "monitoring_log.csv"
    row = {
        "source_run_id": "run-1",
        "source_forecast_csv": "forecast.csv",
        "row_type": "model",
        "model_name": "final_28d_seasonal_naive_production",
        "comparison_model": "",
        "evaluation_window": "<= 28d",
        "rows": 3,
        "wape": 0.1,
        "bias_pct": 0.01,
        "coverage_80": 1.0,
        "coverage_95": 1.0,
        "wape_diff_vs_final": "",
        "bias_diff_vs_final": "",
        "verdict": "",
    }

    assert append_monitor_rows(log_csv, [row]) == 1
    assert append_monitor_rows(log_csv, [row]) == 0
    assert len(pd.read_csv(log_csv)) == 1
