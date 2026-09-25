from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from run_neuralprophet_screen import neuralprophet_predict, screen, segment_score
from score_neuralprophet_ensembles import score


def test_neuralprophet_uses_cutoff_only_and_forecast_diagonal(monkeypatch) -> None:
    calls = []

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit(self, frame, **kwargs):
            assert frame["y"].tolist() == [1.0, 2.0, 3.0]

        def make_future_dataframe(self, frame, **kwargs):
            assert kwargs["periods"] == 2
            return pd.DataFrame({"ds": pd.date_range("2026-01-04", periods=2), "y": [np.nan, np.nan]})

        def predict(self, frame):
            return frame.assign(yhat1=[4.0, 999.0], yhat2=[np.nan, -5.0])

    monkeypatch.setitem(sys.modules, "neuralprophet", types.SimpleNamespace(NeuralProphet=FakeModel, set_log_level=lambda _: None))
    train = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3), "y": [1.0, 2.0, 3.0]})
    spec = {"n_lags": 91, "epochs": 40, "batch_size": 64, "learning_rate": 0.01}
    assert neuralprophet_predict(train, 2, spec).tolist() == [4.0, 0.0]
    assert calls[0]["n_forecasts"] == 2
    spec["n_lags"] = 0
    assert neuralprophet_predict(train, 2, spec).tolist() == [4.0, 999.0]
    assert calls[1]["n_forecasts"] == 1


def test_segment_score_splits_nov_dec_from_rest() -> None:
    frame = pd.DataFrame({"horizon": [28, 28], "ds": pd.to_datetime(["2025-12-01", "2026-01-01"]),
                          "actual": [100.0, 100.0], "seasonal_naive_lag_364": [90.0, 90.0],
                          "timesfm_2_5": [80.0, 80.0], "candidate": [100.0, 100.0]})
    scored = segment_score(frame, {"standalone_name": "candidate", "blend_name": "candidate_blend"})
    assert set(scored["segment"]) == {"nov_dec", "rest"}
    assert scored.loc[scored["model_name"].eq("candidate_blend"), "wape"].tolist() == [0.075, 0.075]


def test_ensemble_score_requires_both_models_and_uses_fixed_weights() -> None:
    base = {"horizon": 28, "cutoff": pd.Timestamp("2026-01-01"), "ds": pd.Timestamp("2026-01-02"),
            "actual": 100.0, "seasonal_naive_lag_364": 90.0, "timesfm_2_5": 80.0}
    rows = [{**base, "neuralprophet_trend": 110.0, "neuralprophet_ar": np.nan},
            {**base, "neuralprophet_trend": np.nan, "neuralprophet_ar": 100.0}]
    formulas = {"formulas": {"combined": {"seasonal_naive_lag_364": 0.75,
                                            "neuralprophet_trend": 0.125, "neuralprophet_ar": 0.125}}}
    folds, _ = score(pd.DataFrame(rows), formulas)
    assert folds.loc[folds["model_name"].eq("combined"), "wape"].item() == 0.0625


def test_annual_screen_uses_only_registered_evaluation_cutoffs(monkeypatch) -> None:
    import run_neuralprophet_screen as module

    baseline = pd.DataFrame({"cutoff": pd.to_datetime(["2025-01-01", "2025-02-01"]),
                             "ds": pd.to_datetime(["2025-01-02", "2025-02-02"]),
                             "actual": [100.0, 100.0], "seasonal_naive_lag_364": [90.0, 90.0],
                             "timesfm_2_5": [80.0, 80.0]})
    monkeypatch.setattr(module, "registered_baseline", lambda *args: baseline)
    monkeypatch.setattr(module, "neuralprophet_predict", lambda *args: np.array([95.0]))
    registration = {"evaluation_cutoffs": {"1": ["2025-02-01"]},
                    "candidates": {"neuralprophet_trend": {"standalone_name": "neuralprophet_trend"}}}
    sales = pd.DataFrame({"ds": pd.date_range("2024-12-01", "2025-02-01"), "y": 100.0})
    frame, status = screen(registration, sales, "neuralprophet_trend", 1, False)
    assert status["status"] == "completed"
    assert frame["cutoff"].tolist() == [pd.Timestamp("2025-02-01")]
