"""Shared forecast metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score_group(group: pd.DataFrame) -> pd.Series:
    actual = group["actual"].to_numpy()
    forecast = group["forecast"].to_numpy()
    err = forecast - actual
    denom_abs = np.sum(np.abs(actual))
    denom = np.sum(actual)
    p10 = group["p10"] if "p10" in group else pd.Series(np.nan, index=group.index)
    p90 = group["p90"] if "p90" in group else pd.Series(np.nan, index=group.index)
    p025 = group["p025"] if "p025" in group else pd.Series(np.nan, index=group.index)
    p975 = group["p975"] if "p975" in group else pd.Series(np.nan, index=group.index)
    width80 = p90 - p10
    width95 = p975 - p025
    return pd.Series(
        {
            "rows": len(group),
            "actual_sum": np.sum(actual),
            "forecast_sum": np.sum(forecast),
            "wape": np.sum(np.abs(err)) / denom_abs if denom_abs else np.nan,
            "bias_pct": np.sum(err) / denom if abs(denom) > 1e-9 else np.nan,
            "bias_abs": np.sum(err),
            "mae": mean_absolute_error(actual, forecast),
            "rmse": math.sqrt(mean_squared_error(actual, forecast)),
            # Coverage is computed only over rows that have intervals, so rows
            # with point forecasts but no intervals (e.g. conformal calibration
            # cutoffs) are excluded rather than counted as misses.
            "coverage_80": float(((actual >= p10) & (actual <= p90))[p10.notna() & p90.notna()].mean()) if (p10.notna() & p90.notna()).any() else np.nan,
            "coverage_95": float(((actual >= p025) & (actual <= p975))[p025.notna() & p975.notna()].mean()) if (p025.notna() & p975.notna()).any() else np.nan,
            "avg_width_80": float(width80.mean()) if width80.notna().any() else np.nan,
            "avg_width_95": float(width95.mean()) if width95.notna().any() else np.nan,
            "runtime_seconds": group["runtime_seconds"].sum() if "runtime_seconds" in group else 0.0,
        }
    )
