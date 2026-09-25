import numpy as np

from run_amzn_item_forecast import allocate_to_items, store_forecast


def test_store_forecast_uses_damped_annual_pattern_and_short_history_fallback():
    history = np.r_[np.ones(56), np.full(308, 2.0), np.full(56, 4.0)]
    forecast, model = store_forecast(history, 365)
    assert model == "seasonal_naive_scaled_lag_364_w56_damped"
    assert np.all(forecast[:308] == 5.0)
    assert np.all(forecast[308:364] == 10.0)
    assert forecast[364] == forecast[0]

    fallback, model = store_forecast(np.arange(28, dtype=float), 2)
    assert model == "window_average_28_fallback"
    assert np.all(fallback == 13.5)

    _, point, p10, p90, _ = allocate_to_items(
        {"a": np.ones(364), "b": np.full(364, 3.0)},
        np.array([40.0]),
        np.array([20.0]),
        np.array([60.0]),
    )
    assert np.allclose(point.sum(axis=0), 40.0)
    assert np.allclose(p10.sum(axis=0), 20.0)
    assert np.allclose(p90.sum(axis=0), 60.0)
