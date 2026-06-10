from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import score_group
from run_bakeoff import add_conformal_intervals


def build_preds(n_cutoffs: int, horizon: int = 3) -> pd.DataFrame:
    rows = []
    for c in range(n_cutoffs):
        cutoff = pd.Timestamp("2025-01-01") + pd.Timedelta(days=28 * c)
        for step in range(1, horizon + 1):
            rows.append(
                {
                    "unique_id": "total_sales",
                    "horizon": horizon,
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(days=step),
                    "horizon_step": step,
                    "actual": 100.0 + step,
                    "model_name": "model_a",
                    "forecast": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_conformal_calibration_cutoffs_get_no_intervals() -> None:
    preds = build_preds(n_cutoffs=6)

    out = add_conformal_intervals(preds)

    cutoffs = sorted(out["cutoff"].unique())
    calibration = out[out["cutoff"].isin(cutoffs[:2])]
    later = out[out["cutoff"].isin(cutoffs[2:])]
    assert calibration["p10"].isna().all()
    assert calibration["p50"].notna().all()
    assert later["p10"].notna().all()
    assert later["p975"].notna().all()


def test_conformal_residuals_come_from_earliest_cutoffs() -> None:
    preds = build_preds(n_cutoffs=6)
    cutoffs = sorted(preds["cutoff"].unique())
    # Distinct residuals per era: early cutoffs +10, later cutoffs +50.
    early = preds["cutoff"].isin(cutoffs[:2])
    preds.loc[early, "actual"] = preds.loc[early, "forecast"] + 10.0
    preds.loc[~early, "actual"] = preds.loc[~early, "forecast"] + 50.0

    out = add_conformal_intervals(preds)

    later = out[~out["cutoff"].isin(cutoffs[:2])]
    # All early residuals equal +10, so every quantile offset is +10.
    np.testing.assert_allclose(later["p90"] - later["forecast"], 10.0)
    np.testing.assert_allclose(later["p10"] - later["forecast"], 10.0)


def test_coverage_ignores_rows_without_intervals() -> None:
    group = pd.DataFrame(
        {
            "actual": [100.0, 100.0, 100.0, 100.0],
            "forecast": [100.0, 100.0, 100.0, 100.0],
            "p10": [90.0, 90.0, np.nan, np.nan],
            "p90": [110.0, 110.0, np.nan, np.nan],
            "p025": [80.0, 80.0, np.nan, np.nan],
            "p975": [120.0, 120.0, np.nan, np.nan],
        }
    )

    scores = score_group(group)

    assert scores["coverage_80"] == 1.0
    assert scores["coverage_95"] == 1.0
