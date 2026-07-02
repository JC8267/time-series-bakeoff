from __future__ import annotations

import numpy as np
import pandas as pd

from run_bakeoff import (
    gregorian_easter,
    daily_reagg_level_scaled_seasonal_naive,
    daily_reagg_seasonal_naive,
    holiday_anchor_aware_source_dates,
    level_scaled_seasonal_naive,
    model_specs,
    seasonal_average,
    seasonal_naive,
)
from run_multifold_final_validation import WEEKLY_DEFAULT_POOL, default_grid_step_for_grain
from run_weekly_production_forecast import parameter_free_member_forecast


def test_seasonal_naive_uses_last_value_when_history_shorter_than_lag() -> None:
    train = pd.DataFrame({"y": [10.0, 20.0, 30.0]})

    out = seasonal_naive(train, horizon=2, lag=7)

    np.testing.assert_array_equal(out, np.asarray([30.0, 30.0]))


def test_seasonal_naive_uses_lagged_positions() -> None:
    train = pd.DataFrame({"y": np.arange(1.0, 11.0)})

    out = seasonal_naive(train, horizon=3, lag=7)

    np.testing.assert_array_equal(out, np.asarray([4.0, 5.0, 6.0]))


def test_seasonal_naive_tiles_last_season_when_horizon_exceeds_lag() -> None:
    train = pd.DataFrame({"y": np.arange(1.0, 15.0)})

    out = seasonal_naive(train, horizon=10, lag=7)

    expected = np.asarray([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 8.0, 9.0, 10.0])
    np.testing.assert_array_equal(out, expected)


def test_gregorian_easter_known_dates() -> None:
    assert gregorian_easter(2024) == pd.Timestamp("2024-03-31")
    assert gregorian_easter(2025) == pd.Timestamp("2025-04-20")
    assert gregorian_easter(2026) == pd.Timestamp("2026-04-05")


def test_holiday_anchor_aware_source_dates_use_anchor_rules() -> None:
    dates = pd.Series(pd.to_datetime(["2025-12-25", "2025-11-27", "2025-03-15", "2025-12-17", "2025-12-18"]))

    out = holiday_anchor_aware_source_dates(dates, default_lag=364)

    expected = pd.Series(pd.to_datetime(["2024-12-25", "2024-11-28", "2024-03-16", "2024-12-18", "2024-12-18"]))
    pd.testing.assert_series_equal(out.reset_index(drop=True), expected)


def test_holiday_anchor_aware_source_dates_respect_weekly_default_lag() -> None:
    out = holiday_anchor_aware_source_dates(pd.Series(pd.to_datetime(["2025-03-15"])), default_lag=365)

    assert out.iloc[0] == pd.Timestamp("2024-03-15")


def test_weekly_model_specs_include_annual_seasonal_naive() -> None:
    names = [spec.name for spec in model_specs(horizon=4, grain="weekly")]

    assert "seasonal_naive_lag_52" in names
    assert "seasonal_naive_scaled_lag_52_w4" in names
    assert "seasonal_naive_scaled_lag_52_w8_damped" in names
    assert "seasonal_average_lag_52_y2" in names
    assert "window_average_4" in names
    assert "seasonal_naive_lag_364" not in names
    assert "seasonal_naive_holiday_aligned_lag_52" not in names


def test_level_scaled_seasonal_naive_uses_recent_yoy_ratio() -> None:
    train = pd.DataFrame({"y": [10.0] * 7 + [5.0] * 7})

    out = level_scaled_seasonal_naive(train, horizon=3, lag=7, window=7)
    damped = level_scaled_seasonal_naive(train, horizon=3, lag=7, window=7, damping=0.5)

    np.testing.assert_array_equal(out, np.asarray([2.5, 2.5, 2.5]))
    np.testing.assert_array_equal(damped, np.asarray([3.75, 3.75, 3.75]))


def test_seasonal_average_uses_recency_weighted_available_years() -> None:
    train = pd.DataFrame({"y": [10.0] * 7 + [20.0] * 7 + [30.0] * 7})

    out = seasonal_average(train, horizon=2, base_lag=7, years=3)

    np.testing.assert_allclose(out, np.asarray([70.0 / 3.0, 70.0 / 3.0]))


def test_daily_reagg_seasonal_naive_sums_daily_forecast_to_week() -> None:
    train = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", "2024-01-10", freq="D"),
            "y": np.arange(1.0, 11.0),
        }
    )

    out = daily_reagg_seasonal_naive(
        train,
        pd.Series(pd.to_datetime(["2024-01-17"])),
        lag=7,
        weekly_freq="W-WED",
    )

    np.testing.assert_array_equal(out, np.asarray([49.0]))


def test_daily_reagg_level_scaled_seasonal_naive_sums_scaled_daily_forecast_to_week() -> None:
    train = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", "2024-01-14", freq="D"),
            "y": [10.0] * 7 + [5.0] * 7,
        }
    )

    out = daily_reagg_level_scaled_seasonal_naive(
        train,
        pd.Series(pd.to_datetime(["2024-01-21"])),
        lag=7,
        window=7,
        weekly_freq="W-SUN",
    )

    np.testing.assert_array_equal(out, np.asarray([17.5]))


def test_multifold_grid_step_default_is_grain_aware() -> None:
    assert default_grid_step_for_grain("daily") == 28
    assert default_grid_step_for_grain("weekly") == 4


def test_weekly_default_pool_includes_daily_reaggregation_winner() -> None:
    assert "seasonal_naive_lag_52" in WEEKLY_DEFAULT_POOL
    assert "daily_reagg_seasonal_naive_lag_365" in WEEKLY_DEFAULT_POOL


def test_weekly_challenger_member_forecast_dispatches_native_seasonal_naive() -> None:
    weekly = pd.DataFrame({"ds": pd.date_range("2024-01-03", periods=52, freq="W-WED"), "y": np.arange(1.0, 53.0)})
    daily = pd.DataFrame({"ds": [], "y": []})
    dates = pd.Series(pd.to_datetime(["2025-01-01"]))

    out = parameter_free_member_forecast("seasonal_naive_lag_52", daily, weekly, dates, horizon=1, weekly_freq="W-WED")

    np.testing.assert_array_equal(out, np.asarray([1.0]))
