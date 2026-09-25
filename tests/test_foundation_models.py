from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from run_prophet_timesfm_benchmark import (
    forecast_chronos2,
    forecast_timesfm3,
    load_timesfm,
    load_timesfm3,
)


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


def test_load_timesfm_disables_request_time_torch_compile(monkeypatch) -> None:
    calls = {}

    class FakeModel:
        def compile(self, config) -> None:
            calls["forecast_config"] = config

    class FakeTimesFM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return FakeModel()

    fake_torch = types.ModuleType("torch")
    fake_torch.set_float32_matmul_precision = lambda value: calls.update(matmul_precision=value)
    fake_timesfm = types.ModuleType("timesfm")
    fake_timesfm.ForecastConfig = lambda **kwargs: kwargs
    fake_timesfm.TimesFM_2p5_200M_torch = FakeTimesFM
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "timesfm", fake_timesfm)

    load_timesfm(1024, 45)

    assert calls["load_kwargs"] == {"torch_compile": False}
    assert calls["forecast_config"]["max_horizon"] == 45


def test_load_timesfm3_uses_official_checkpoint(monkeypatch) -> None:
    calls = {}

    class FakeEvaluator:
        def __init__(self, config) -> None:
            calls["config"] = config

    fake_torch = types.ModuleType("torch")
    fake_torch.set_float32_matmul_precision = lambda value: calls.update(matmul_precision=value)
    fake_timesfm3 = types.ModuleType("timesfm3")
    fake_timesfm3.ModelConfig = lambda **kwargs: kwargs
    fake_timesfm3.TimesFM3Evaluator = FakeEvaluator
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "timesfm3", fake_timesfm3)

    load_timesfm3()

    assert calls["matmul_precision"] == "high"
    assert calls["config"] == {
        "checkpoint_path": "google/timesfm-3.0-pytorch",
        "per_core_batch_size": 1,
    }


def test_forecast_timesfm3_maps_official_quantiles() -> None:
    calls = {}

    class FakeResult:
        forecast = np.array([5.0, 6.0])
        quantiles = np.arange(18, dtype=float).reshape(2, 9)

    class FakeModel:
        def predict_batch(self, **kwargs):
            calls.update(kwargs)
            return iter([FakeResult()])

    train = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    out = forecast_timesfm3(FakeModel(), train, horizon=2, max_context=2)

    assert calls["contexts"][0].tolist() == [2.0, 3.0]
    assert calls["use_symmetric_averaging"] is False
    assert out["forecast"].tolist() == [5.0, 6.0]
    assert out["p10"].tolist() == [0.0, 9.0]
    assert out["p50"].tolist() == [4.0, 13.0]
    assert out["p90"].tolist() == [8.0, 17.0]
    assert out[["p025", "p975"]].isna().all().all()
