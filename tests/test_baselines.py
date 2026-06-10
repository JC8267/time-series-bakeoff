from __future__ import annotations

import numpy as np
import pandas as pd

from run_bakeoff import seasonal_naive


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

