from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from run_new_model_screen import score_predictions, tabfm_examples, tabpfn_predict, tirex_predict, validate_forecast


def test_validate_forecast_rejects_degraded_output() -> None:
    with pytest.raises(ValueError, match="finite predictions"):
        validate_forecast([1.0, np.nan], 2)


def test_tirex_predict_uses_native_median_and_context_limit(monkeypatch) -> None:
    calls = {}

    class FakeTimeseriesType:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    fake_tirex = types.ModuleType("tirex2")
    fake_tirex.TimeseriesType = FakeTimeseriesType
    monkeypatch.setitem(sys.modules, "tirex2", fake_tirex)

    class FakeModel:
        quantiles = torch.tensor([0.1, 0.5, 0.9])

        def forecast(self, *args, **kwargs):
            return [np.array([[[1.0, 2.0], [3.0, -4.0], [5.0, 6.0]]])]

    out = tirex_predict(FakeModel(), np.arange(5.0), horizon=2, context_limit=3)

    assert calls["target"].shape == (1, 3)
    assert out.tolist() == [3.0, 0.0]


def test_tabpfn_predict_uses_median_target_and_context_limit() -> None:
    calls = {}

    class FakePipeline:
        def predict_df(self, context, **kwargs):
            calls["context"] = context
            calls.update(kwargs)
            return pd.DataFrame({"target": [4.0, -1.0]})

    train = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=5), "y": np.arange(5.0)})
    out = tabpfn_predict(FakePipeline(), train, horizon=2, context_limit=3)

    assert calls["context"]["target"].tolist() == [2.0, 3.0, 4.0]
    assert calls["quantiles"] == [0.5]
    assert out.tolist() == [4.0, 0.0]


def test_score_predictions_keeps_only_fixed_candidate_blend() -> None:
    frame = pd.DataFrame(
        {
            "horizon": [28, 28, 28, 28],
            "cutoff": [pd.Timestamp("2026-01-01")] * 2 + [pd.Timestamp("2026-02-01")] * 2,
            "ds": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-02-02", "2026-02-03"]),
            "actual": [100.0] * 4,
            "seasonal_naive_lag_364": [90.0] * 4,
            "timesfm_2_5": [80.0] * 4,
            "candidate": [110.0] * 4,
            "runtime_seconds": [1.0] * 4,
        }
    )
    spec = {"standalone_name": "candidate", "blend_name": "candidate_blend"}

    folds, summary = score_predictions(frame, spec)

    assert set(folds["model_name"]) == {
        "seasonal_naive_lag_364",
        "blend_snl75_tfm25",
        "candidate",
        "candidate_blend",
    }
    verdict = summary.loc[summary["model_name"].eq("candidate_blend"), "verdict"].item()
    assert verdict == "keep_for_fresh_confirmation"


def test_tabfm_examples_use_only_origin_available_targets() -> None:
    train = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=500), "y": np.arange(500.0)})

    X_train, y_train, X_future = tabfm_examples(train, horizon=2)

    assert len(X_future) == 2
    assert X_future["lag_364"].tolist() == [136.0, 137.0]
    assert X_future["origin_last_28_mean"].tolist() == [485.5, 485.5]
    assert len(X_train) == len(y_train)
