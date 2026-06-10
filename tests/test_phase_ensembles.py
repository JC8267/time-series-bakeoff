from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_phase5 import make_ensembles


def build_base(models: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in models:
        for step, ds in enumerate(pd.date_range("2026-01-01", periods=3, freq="D"), start=1):
            rows.append(
                {
                    "unique_id": "total_sales",
                    "horizon": 91,
                    "cutoff": pd.Timestamp("2025-12-31"),
                    "ds": ds,
                    "horizon_step": step,
                    "actual": 100.0,
                    "model_name": model_name,
                    "forecast": 100.0 + step,
                }
            )
    return pd.DataFrame(rows)


def make_metadata(ridge_models: list[str]) -> dict:
    return {
        "91": {
            "inverse_wape_models": ["model_a", "model_b"],
            "convex_stack_weights": {"model_a": 0.6, "model_b": 0.4},
            "ridge_coefficients": {m: 0.5 for m in ridge_models},
            "ridge_intercept": 1.0,
        }
    }


def test_make_ensembles_builds_ridge_stack_when_all_models_present() -> None:
    base = build_base(["model_a", "model_b"])
    metadata = make_metadata(["model_a", "model_b"])

    out = make_ensembles(base, metadata, 91)

    ridge = out[out["model_name"] == "phase5_ridge_stack"].sort_values("horizon_step")
    np.testing.assert_allclose(ridge["forecast"].to_numpy(), np.asarray([102.0, 103.0, 104.0]))


def test_make_ensembles_fails_loudly_when_ridge_model_missing() -> None:
    base = build_base(["model_a", "model_b"])
    metadata = make_metadata(["model_a", "model_b", "model_c"])

    with pytest.raises(ValueError, match="model_c"):
        make_ensembles(base, metadata, 91)
