from __future__ import annotations

import numpy as np
import pandas as pd

from run_production_forecast import aggregate_interval_row, cumulative_terminal_errors


def write_grid(tmp_path, path_errors: list[float], daily_forecast: list[float] | None = None):
    """Write complete synthetic 365-day paths with one constant daily error."""
    rows = []
    forecast_values = daily_forecast or [100.0] * len(path_errors)
    for path_no, (error, forecast_value) in enumerate(zip(path_errors, forecast_values)):
        cutoff = pd.Timestamp("2022-01-01") + pd.Timedelta(days=31 * path_no)
        for step in range(1, 366):
            forecast = float(forecast_value)
            rows.append(
                {
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(days=step),
                    "horizon_step": step,
                    "actual": forecast + error,
                    "forecast": forecast,
                    "model_name": "seasonal_naive_lag_364",
                }
            )
    grid_path = tmp_path / "grid.csv"
    pd.DataFrame(rows).to_csv(grid_path, index=False)
    return grid_path


def test_cumulative_terminal_errors_sums_each_path_and_keeps_horizons_separate(tmp_path) -> None:
    grid_path = write_grid(tmp_path, [1.0, 2.0, 3.0])

    paths_180 = cumulative_terminal_errors(grid_path, 180, interval_windows=3)
    paths_365 = cumulative_terminal_errors(grid_path, 365, interval_windows=3)

    np.testing.assert_allclose(paths_180["cumulative_error"], [180.0, 360.0, 540.0])
    np.testing.assert_allclose(paths_365["cumulative_error"], [365.0, 730.0, 1095.0])
    assert paths_180["forecast_sum"].eq(18_000.0).all()
    assert paths_365["forecast_sum"].eq(36_500.0).all()


def test_cumulative_paths_use_only_completed_prior_cutoffs(tmp_path) -> None:
    grid_path = write_grid(tmp_path, [10.0, 10.0, 10.0, 100.0])

    # The fourth path runs past this as-of date and must not calibrate itself.
    paths = cumulative_terminal_errors(
        grid_path,
        horizon=180,
        interval_windows=8,
        as_of=pd.Timestamp("2022-09-01"),
    )

    assert len(paths) == 3
    assert paths["cutoff"].max() == pd.Timestamp("2022-03-04")
    assert paths["cumulative_error"].max() == 1_800.0


def test_aggregate_interval_row_applies_additive_and_relative_calibration(tmp_path) -> None:
    # Historical paths have equal percentage errors but different absolute errors.
    grid_path = write_grid(tmp_path, [10.0, 20.0, 30.0], [100.0, 200.0, 300.0])
    forecast = np.full(180, 50.0)
    cutoff = pd.Timestamp("2023-01-01")

    additive = aggregate_interval_row(
        grid_path,
        source_horizon=180,
        horizon=180,
        forecast=forecast,
        training_cutoff=cutoff,
        interval_windows=3,
        interval_mode="additive",
        run_id="test-additive",
    )
    relative = aggregate_interval_row(
        grid_path,
        source_horizon=180,
        horizon=180,
        forecast=forecast,
        training_cutoff=cutoff,
        interval_windows=3,
        interval_mode="relative",
        run_id="test-relative",
    )

    expected_additive_median = 9_000.0 + np.quantile([1_800.0, 3_600.0, 5_400.0], 0.50)
    assert additive["calibrated_median"] == expected_additive_median
    assert relative["calibrated_median"] == 9_000.0 * 1.10
    assert additive["calibrated_median"] != relative["calibrated_median"]
    assert additive["calibration_mode"] == "additive"
    assert relative["calibration_mode"] == "relative"


def test_aggregate_interval_rows_are_ordered_and_distinct_for_180_and_365(tmp_path) -> None:
    # The last 185 days carry a larger error, so 365d calibration must differ
    # from the independent 180d terminal calibration.
    rows = []
    for path_no in range(3):
        cutoff = pd.Timestamp("2022-01-01") + pd.Timedelta(days=31 * path_no)
        for step in range(1, 366):
            error = 1.0 if step <= 180 else 4.0
            rows.append(
                {
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(days=step),
                    "horizon_step": step,
                    "actual": 50.0 + error,
                    "forecast": 50.0,
                    "model_name": "seasonal_naive_lag_364",
                }
            )
    grid_path = tmp_path / "grid.csv"
    pd.DataFrame(rows).to_csv(grid_path, index=False)

    row_180 = aggregate_interval_row(
        grid_path,
        source_horizon=180,
        horizon=180,
        forecast=np.full(180, 50.0),
        training_cutoff=pd.Timestamp("2023-01-01"),
        interval_windows=3,
        interval_mode="additive",
        run_id="test-180",
    )
    row_365 = aggregate_interval_row(
        grid_path,
        source_horizon=365,
        horizon=365,
        forecast=np.full(365, 50.0),
        training_cutoff=pd.Timestamp("2023-01-01"),
        interval_windows=3,
        interval_mode="additive",
        run_id="test-365",
    )

    for row in (row_180, row_365):
        assert row["p025"] <= row["p10"] <= row["calibrated_median"] <= row["p90"] <= row["p975"]
    assert row_180["horizon"] == 180
    assert row_365["horizon"] == 365
    assert row_180["calibrated_median"] != row_365["calibrated_median"]
