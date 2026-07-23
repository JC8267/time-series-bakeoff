from __future__ import annotations

import pandas as pd

from run_forecast_monitor import append_monitor_rows, cusum_series, grid_path_for_final, infer_grain, model_summary_rows


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
        "horizon": 28,
        "grain": "daily",
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


def test_append_monitor_rows_recognizes_legacy_key(tmp_path) -> None:
    log_csv = tmp_path / "monitor.csv"
    legacy = {
        "logged_at": "2026-07-01T00:00:00+00:00",
        "source_run_id": "run-1",
        "source_forecast_csv": "forecast.csv",
        "row_type": "model",
        "model_name": "final_28d_seasonal_naive_production",
        "comparison_model": "",
        "evaluation_window": "<= 28d",
        "rows": 3,
    }
    pd.DataFrame([legacy]).to_csv(log_csv, index=False)
    current = {
        **legacy,
        "horizon": 28,
        "grain": "daily",
    }

    assert append_monitor_rows(log_csv, [current]) == 0


def test_monitor_uses_explicit_custom_horizon_metadata(tmp_path) -> None:
    realized = pd.DataFrame(
        {
            "run_id": ["run-45", "run-45"],
            "model_name": ["final_custom_model", "final_custom_model"],
            "ds": pd.to_datetime(["2026-07-01", "2026-07-30"]),
            "training_cutoff": pd.to_datetime(["2026-06-30", "2026-06-30"]),
            "grain": ["daily", "daily"],
            "horizon": [45, 45],
            "horizon_step": [1, 30],
            "actual": [100.0, 110.0],
            "forecast": [98.0, 108.0],
            "p10": [90.0, 100.0],
            "p90": [110.0, 120.0],
            "p025": [85.0, 95.0],
            "p975": [115.0, 125.0],
        }
    )

    rows = model_summary_rows(realized, "run-45", tmp_path / "forecast.csv", "daily")

    assert infer_grain(realized) == "daily"
    assert len(rows) == 1
    assert rows[0]["horizon"] == 45
    assert rows[0]["grain"] == "daily"
    assert rows[0]["evaluation_window"] == "<= 45d"
    manifest = {"interval_sources": {"45": {"grid": "cached/45/grid.csv"}}}
    assert grid_path_for_final("final_custom_model", 45, "daily", manifest).as_posix() == "cached/45/grid.csv"
