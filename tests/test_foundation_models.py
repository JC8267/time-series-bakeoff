from __future__ import annotations

import pandas as pd

from run_prophet_timesfm_benchmark import forecast_chronos2


class FakeChronos2Pipeline:
    def predict_df(self, df, **kwargs):
        horizon = kwargs["prediction_length"]
        timestamps = pd.date_range(pd.Timestamp(df["timestamp"].max()) + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.DataFrame(
            {
                "item_id": ["sales"] * horizon,
                "timestamp": timestamps,
                "target_name": ["target"] * horizon,
                "predictions": [10.0] * horizon,
                "0.025": [-1.0, 2.0],
                "0.1": [3.0, 4.0],
                "0.5": [5.0, 6.0],
                "0.9": [7.0, 8.0],
                "0.975": [9.0, 10.0],
            }
        )


def test_forecast_chronos2_uses_median_and_clips_negative_quantiles() -> None:
    train = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=3, freq="D"), "y": [1.0, 2.0, 3.0]})

    out = forecast_chronos2(FakeChronos2Pipeline(), train, horizon=2, max_context=8192)

    assert out["forecast"].tolist() == [5.0, 6.0]
    assert out["p50"].tolist() == [5.0, 6.0]
    assert out["p025"].tolist() == [0.0, 2.0]
