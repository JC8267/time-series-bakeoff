"""Runnable daily sales forecasting bakeoff.

This script implements the first practical version of the benchmark described
in time_series_forecasting_bakeoff_plan.md using dependencies that are commonly
available in the local Python environment.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from metrics import score_group
from experiment_registry import append_experiment

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _STATSMODELS_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - import availability is recorded at runtime
    ExponentialSmoothing = None
    SARIMAX = None
    _STATSMODELS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, MSTL
    _STATSFORECAST_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    StatsForecast = None
    AutoARIMA = None
    AutoETS = None
    MSTL = None
    _STATSFORECAST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


SEED = 42
DATE_COL = "transaction_date"
SALES_COL = "sales"
PRIMARY_HORIZONS = (28, 91)
MIN_TRAIN_DAYS = 730
MIN_TRAIN_WEEKS = 104
DAILY_GRAIN = "daily"
WEEKLY_GRAIN = "weekly"
DEFAULT_WEEKLY_FREQ = "W-WED"
OPENINGS_DF: pd.DataFrame | None = None

# Records every model that silently degraded to a baseline, with the reason.
# Keyed by model_name -> reason. A non-empty dict at the end of a run means the
# reported metrics for those models are NOT the intended model and must not be
# trusted as a real result.
DEGRADED_MODELS: dict[str, str] = {}
_WARNED_FALLBACKS: set[str] = set()


def warn_fallback(model_name: str, fallback_to: str, reason: str) -> np.ndarray | None:
    """Loudly record that ``model_name`` could not run and fell back to a baseline.

    Prints a WARNING to stderr once per (model, reason) and records the
    degradation in DEGRADED_MODELS so it surfaces in the manifest and final
    summary. Returns None so callers can keep their existing return expression.
    """
    DEGRADED_MODELS[model_name] = f"fell back to {fallback_to} ({reason})"
    key = f"{model_name}:{reason}"
    if key not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(key)
        import sys

        print(
            f"WARNING [bakeoff]: model '{model_name}' did NOT run -- "
            f"fell back to '{fallback_to}'. Reason: {reason}. "
            f"Its reported metrics are the baseline's, not '{model_name}'.",
            file=sys.stderr,
            flush=True,
        )
    return None


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def normalize_grain(grain: str) -> str:
    grain = grain.lower().strip()
    if grain in {"daily", "day", "d"}:
        return DAILY_GRAIN
    if grain in {"weekly", "week", "w"}:
        return WEEKLY_GRAIN
    raise ValueError(f"Unsupported grain: {grain}")


def freq_for_grain(grain: str, weekly_freq: str = DEFAULT_WEEKLY_FREQ) -> str:
    return "D" if normalize_grain(grain) == DAILY_GRAIN else weekly_freq


def horizon_delta(horizon: int, grain: str) -> pd.Timedelta:
    return pd.Timedelta(days=horizon) if normalize_grain(grain) == DAILY_GRAIN else pd.Timedelta(weeks=horizon)


def annual_lag_for_grain(grain: str) -> int:
    return 364 if normalize_grain(grain) == DAILY_GRAIN else 52


def short_lag_for_grain(grain: str) -> int:
    return 7 if normalize_grain(grain) == DAILY_GRAIN else 1


def short_window_for_grain(grain: str) -> int:
    return 28 if normalize_grain(grain) == DAILY_GRAIN else 4


def arcsinh(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.arcsinh(np.asarray(x, dtype=float))


def inv_arcsinh(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.sinh(np.asarray(x, dtype=float))


def load_sales(
    path: Path,
    fill_missing: str = "interpolate",
    audit_path: Path | None = None,
    grain: str = DAILY_GRAIN,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> pd.DataFrame:
    grain = normalize_grain(grain)
    raw = pd.read_csv(path)
    if DATE_COL not in raw.columns or SALES_COL not in raw.columns:
        raise ValueError(f"{path} must contain {DATE_COL!r} and {SALES_COL!r}.")

    df = raw[[DATE_COL, SALES_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce")
    if df[SALES_COL].isna().any():
        raise ValueError("sales contains non-numeric or missing values.")

    df = df.groupby(DATE_COL, as_index=False)[SALES_COL].sum()
    full_dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    df = df.set_index(DATE_COL).reindex(full_dates)
    missing_mask = df[SALES_COL].isna()
    missing_days = int(missing_mask.sum())
    if missing_days:
        audit = pd.DataFrame({"ds": df.index[missing_mask], "fill_strategy": fill_missing})
        if audit_path is not None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit.to_csv(audit_path, index=False)
        if fill_missing == "interpolate":
            df[SALES_COL] = df[SALES_COL].interpolate("time").ffill().bfill()
        elif fill_missing == "zero":
            df[SALES_COL] = df[SALES_COL].fillna(0.0)
        elif fill_missing == "error":
            dates = ", ".join(str(d.date()) for d in audit["ds"].head(10))
            raise ValueError(f"Missing sales dates found ({missing_days}). First dates: {dates}")
        elif fill_missing == "leave_missing":
            pass
        else:
            raise ValueError(f"Unsupported fill_missing strategy: {fill_missing}")
    df["data_gap_filled_flag"] = missing_mask.astype(int)
    if df[SALES_COL].isna().any():
        raise ValueError("sales still contains missing values after fill strategy.")
    if grain == WEEKLY_GRAIN:
        weekly = (
            df.resample(weekly_freq)
            .agg(
                y=(SALES_COL, "sum"),
                data_gap_filled_flag=("data_gap_filled_flag", "sum"),
                days_in_period=(SALES_COL, "size"),
            )
            .reset_index(names="ds")
        )
        weekly = weekly[weekly["days_in_period"] == 7].drop(columns=["days_in_period"]).reset_index(drop=True)
        df = weekly
    else:
        df.index.name = "ds"
        df = df.reset_index().rename(columns={SALES_COL: "y"})
    df["unique_id"] = "total_sales"
    df.attrs["missing_days_filled"] = missing_days
    df.attrs["grain"] = grain
    df.attrs["freq"] = freq_for_grain(grain, weekly_freq)
    return df[["unique_id", "ds", "y", "data_gap_filled_flag"]]


def load_openings(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    raw = pd.read_csv(path)
    date_renames = {"Opening_date": "open_date", "opening_date": "open_date", "Open_date": "open_date"}
    openings = raw.rename(columns={k: v for k, v in date_renames.items() if k in raw.columns}).copy()
    # Use exactly one unit-id source; renaming both Name and BU would create
    # duplicate unit_id columns and break downstream column selection.
    if "unit_id" not in openings.columns:
        for candidate in ["BU", "Name"]:
            if candidate in openings.columns:
                openings = openings.rename(columns={candidate: "unit_id"})
                break
    if "open_date" not in openings.columns:
        raise ValueError(f"{path} must contain an opening date column.")
    if "unit_id" not in openings.columns:
        openings["unit_id"] = np.arange(len(openings)).astype(str)
    openings["open_date"] = pd.to_datetime(openings["open_date"])
    openings = openings.dropna(subset=["open_date"]).sort_values("open_date").reset_index(drop=True)
    return openings


def add_opening_features(frame: pd.DataFrame, openings: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    feature_cols = [
        "new_unit_opened_flag",
        "days_since_latest_opening",
        "num_units_open",
        "num_new_units_last_30_days",
        "num_new_units_last_60_days",
        "num_new_units_last_90_days",
        "num_new_units_last_180_days",
        "grand_opening_0_30_days",
        "ramp_31_90_days",
        "ramp_91_180_days",
        "mature_180_plus_days",
        "post_opening_ramp_flag",
        "grand_opening_window_flag",
    ]
    if openings is None or openings.empty:
        for col in feature_cols:
            out[col] = 0
        out["days_since_latest_opening"] = 9999
        return out

    dates = pd.to_datetime(out["ds"])
    open_dates = pd.to_datetime(openings["open_date"]).sort_values().to_numpy(dtype="datetime64[ns]")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    opened_counts = np.searchsorted(open_dates, date_values, side="right")
    latest_idx = opened_counts - 1
    has_opening = latest_idx >= 0
    latest_dates = np.where(has_opening, open_dates[np.clip(latest_idx, 0, len(open_dates) - 1)], np.datetime64("NaT"))
    days_since = np.where(
        has_opening,
        ((date_values - latest_dates).astype("timedelta64[D]").astype(float)),
        9999.0,
    )

    out["new_unit_opened_flag"] = np.isin(date_values, open_dates).astype(int)
    out["days_since_latest_opening"] = days_since
    out["num_units_open"] = opened_counts
    for window in [30, 60, 90, 180]:
        lower = date_values - np.timedelta64(window, "D")
        prior_counts = np.searchsorted(open_dates, lower, side="left")
        out[f"num_new_units_last_{window}_days"] = opened_counts - prior_counts
    out["grand_opening_0_30_days"] = ((days_since >= 0) & (days_since <= 30)).astype(int)
    out["ramp_31_90_days"] = ((days_since >= 31) & (days_since <= 90)).astype(int)
    out["ramp_91_180_days"] = ((days_since >= 91) & (days_since <= 180)).astype(int)
    out["mature_180_plus_days"] = ((days_since > 180) & (days_since < 9999)).astype(int)
    out["post_opening_ramp_flag"] = ((days_since >= 0) & (days_since <= 180)).astype(int)
    out["grand_opening_window_flag"] = out["grand_opening_0_30_days"]
    return out


def add_calendar_features(frame: pd.DataFrame, openings: pd.DataFrame | None = None) -> pd.DataFrame:
    out = frame.copy()
    ds = pd.to_datetime(out["ds"])
    iso = ds.dt.isocalendar()
    fiscal_start_month = 9

    out["dayofweek"] = ds.dt.dayofweek
    out["dayofmonth"] = ds.dt.day
    out["dayofyear"] = ds.dt.dayofyear
    out["weekofyear"] = iso.week.astype(int)
    out["month"] = ds.dt.month
    out["quarter"] = ds.dt.quarter
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = ds.dt.is_month_start.astype(int)
    out["is_month_end"] = ds.dt.is_month_end.astype(int)
    out["is_quarter_start"] = ds.dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = ds.dt.is_quarter_end.astype(int)
    out["fiscal_year"] = ds.dt.year + (ds.dt.month >= fiscal_start_month).astype(int)
    out["fiscal_month"] = ((ds.dt.month - fiscal_start_month) % 12) + 1
    out["fiscal_week"] = (((ds - pd.to_datetime((ds.dt.year - (ds.dt.month < fiscal_start_month).astype(int)).astype(str) + "-09-01")).dt.days) // 7 + 1).clip(lower=1)
    return add_opening_features(out, openings)


def build_cutoffs(df: pd.DataFrame, horizon: int, max_cutoffs: int | None, grain: str = DAILY_GRAIN) -> list[pd.Timestamp]:
    grain = normalize_grain(grain)
    if grain == WEEKLY_GRAIN:
        first_idx = MIN_TRAIN_WEEKS - 1
        last_idx = len(df) - horizon - 1
        if first_idx > last_idx:
            raise ValueError(f"Not enough history for weekly horizon={horizon}.")
        step = 1 if horizon <= 4 else 4
        cutoffs = [pd.Timestamp(c).normalize() for c in df["ds"].iloc[first_idx : last_idx + 1 : step]]
        if max_cutoffs:
            cutoffs = cutoffs[-max_cutoffs:]
        return cutoffs

    start = df["ds"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    end = df["ds"].max() - pd.Timedelta(days=horizon)
    if start > end:
        raise ValueError(f"Not enough history for horizon={horizon}.")
    freq = "7D" if horizon <= 28 else "MS"
    cutoffs = list(pd.date_range(start, end, freq=freq))
    if not cutoffs or cutoffs[0] > start:
        cutoffs.insert(0, start)
    if max_cutoffs:
        cutoffs = cutoffs[-max_cutoffs:]
    return [pd.Timestamp(c).normalize() for c in cutoffs]


def make_lag_features(
    history: pd.DataFrame,
    future_dates: pd.Series,
    recursive_values: list[float],
    openings: pd.DataFrame | None = None,
    grain: str = DAILY_GRAIN,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            history[["ds", "y"]],
            pd.DataFrame({"ds": future_dates, "y": recursive_values}),
        ],
        ignore_index=True,
    )
    combined = add_calendar_features(combined, openings)
    if normalize_grain(grain) == WEEKLY_GRAIN:
        lags = [1, 2, 3, 4, 8, 13, 26, 52]
        mean_windows = [4, 8, 13, 26, 52]
        std_windows = [4, 8, 13]
        minmax_windows = [4, 13]
    else:
        lags = [1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 364]
        mean_windows = [7, 14, 28, 56, 91]
        std_windows = [7, 14, 28]
        minmax_windows = [7, 28]
    for lag in lags:
        combined[f"lag_{lag}"] = combined["y"].shift(lag)
    for window in mean_windows:
        combined[f"roll_mean_{window}"] = combined["y"].shift(1).rolling(window).mean()
    for window in std_windows:
        combined[f"roll_std_{window}"] = combined["y"].shift(1).rolling(window).std()
    for window in minmax_windows:
        combined[f"roll_min_{window}"] = combined["y"].shift(1).rolling(window).min()
        combined[f"roll_max_{window}"] = combined["y"].shift(1).rolling(window).max()
    combined["expanding_mean"] = combined["y"].shift(1).expanding().mean()
    return combined


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"unique_id", "ds", "y"}
    return [c for c in frame.columns if c not in blocked]


def target_date_features(dates: pd.Series, openings: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"ds": pd.to_datetime(dates), "y": 0.0})
    features = add_calendar_features(frame, openings)
    return features.drop(columns=["ds", "y"], errors="ignore").add_prefix("target_")


def predict_sklearn_recursive(
    model_factory: Callable[[], object],
    train: pd.DataFrame,
    dates: pd.Series,
    openings: pd.DataFrame | None = None,
    grain: str = DAILY_GRAIN,
) -> np.ndarray:
    train_features = make_lag_features(train, pd.Series([], dtype="datetime64[ns]"), [], openings, grain)
    train_features["target"] = arcsinh(train_features["y"])
    train_features = train_features.dropna()
    cols = feature_columns(train_features.drop(columns=["target"]))
    model = model_factory()
    model.fit(train_features[cols], train_features["target"])

    preds: list[float] = []
    dates = dates.reset_index(drop=True)
    for step_idx, ds in enumerate(dates):
        frame = make_lag_features(train, dates.iloc[: step_idx + 1], preds + [np.nan], openings, grain)
        row = frame.iloc[[-1]][cols]
        if row.isna().any().any():
            missing_cols = row.columns[row.isna().any()].tolist()
            raise ValueError(f"Recursive feature row contains NaN before prediction at {pd.Timestamp(ds).date()}: {missing_cols}")
        pred = float(inv_arcsinh(model.predict(row))[0])
        preds.append(max(0.0, pred))
    return np.asarray(preds)


def predict_sklearn_direct(
    model_factory: Callable[[], object],
    train: pd.DataFrame,
    dates: pd.Series,
    openings: pd.DataFrame | None = None,
    grain: str = DAILY_GRAIN,
) -> np.ndarray:
    grain = normalize_grain(grain)
    base = make_lag_features(train, pd.Series([], dtype="datetime64[ns]"), [], openings, grain)
    base_cols = feature_columns(base)
    dates = dates.reset_index(drop=True)
    preds: list[float] = []
    for step, target_ds in enumerate(dates, start=1):
        train_step = base.copy()
        target_features = target_date_features(train_step["ds"] + horizon_delta(step, grain), openings)
        train_step = pd.concat([train_step.reset_index(drop=True), target_features.reset_index(drop=True)], axis=1)
        train_step["target"] = arcsinh(train_step["y"].shift(-step))
        train_step = train_step.dropna()
        cols = base_cols + list(target_features.columns)
        if train_step.empty:
            preds.append(float(train["y"].tail(28).mean()))
            continue
        model = model_factory()
        model.fit(train_step[cols], train_step["target"])

        latest = base.iloc[[-1]].copy()
        latest_target = target_date_features(pd.Series([target_ds]), openings)
        row = pd.concat([latest[base_cols].reset_index(drop=True), latest_target.reset_index(drop=True)], axis=1)
        if row.isna().any().any():
            missing_cols = row.columns[row.isna().any()].tolist()
            raise ValueError(f"Direct feature row contains NaN before prediction at {pd.Timestamp(target_ds).date()}: {missing_cols}")
        pred = float(inv_arcsinh(model.predict(row[cols]))[0])
        preds.append(max(0.0, pred))
    return np.asarray(preds)


def seasonal_naive(train: pd.DataFrame, horizon: int, lag: int) -> np.ndarray:
    values = seasonal_values(train, horizon, lag)
    if values is None:
        return np.repeat(train["y"].to_numpy()[-1], horizon)
    return values


def gregorian_easter(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def thanksgiving_date(year: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=11, day=1)
    return first + pd.Timedelta(days=(3 - first.dayofweek) % 7 + 21)


def memorial_day(year: int) -> pd.Timestamp:
    may31 = pd.Timestamp(year=year, month=5, day=31)
    return may31 - pd.Timedelta(days=(may31.dayofweek - 0) % 7)


def labor_day(year: int) -> pd.Timestamp:
    sep1 = pd.Timestamp(year=year, month=9, day=1)
    return sep1 + pd.Timedelta(days=(0 - sep1.dayofweek) % 7)


def holiday_anchor_aware_source_dates(target_dates: pd.Series, default_lag: int = 364) -> pd.Series:
    """Map each target date to last year's source date by holiday anchor type."""
    dates = pd.to_datetime(target_dates).dt.normalize()
    source_dates = []
    for date in dates:
        date_anchored = (date.month == 12 and date.day >= 18) or (date.month == 1 and date.day <= 3) or (date.month == 7 and 1 <= date.day <= 7)
        if date_anchored:
            source_dates.append(date - pd.DateOffset(years=1))
            continue

        anchors = [
            (thanksgiving_date(date.year), 7),
            (gregorian_easter(date.year), 7),
            (memorial_day(date.year), 3),
            (labor_day(date.year), 3),
        ]
        if any(abs((date - anchor).days) <= radius for anchor, radius in anchors):
            source_dates.append(date - pd.Timedelta(days=364))
        else:
            source_dates.append(date - pd.Timedelta(days=default_lag))
    return pd.Series(pd.to_datetime(source_dates), index=dates.index)


def holiday_anchor_aware_seasonal_naive(train: pd.DataFrame, target_dates: pd.Series, default_lag: int = 364) -> np.ndarray:
    dates = pd.to_datetime(target_dates).dt.normalize()
    source_dates = holiday_anchor_aware_source_dates(dates, default_lag)
    lookup = train.assign(ds=pd.to_datetime(train["ds"]).dt.normalize()).set_index("ds")["y"]
    values = source_dates.map(lookup).to_numpy(dtype=float)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = seasonal_naive(train, len(dates), default_lag)[missing]
    return values


def seasonal_values(train: pd.DataFrame, horizon: int, lag: int) -> np.ndarray | None:
    y = train["y"].to_numpy()
    if len(y) < lag:
        return None
    # Tile the last full season so horizons longer than the lag repeat the
    # seasonal cycle instead of flatlining at the final training value.
    idx = len(y) - lag + (np.arange(horizon) % lag)
    return y[idx]


def level_scale_ratio(train: pd.DataFrame, lag: int, window: int, damping: float = 1.0) -> float:
    y = train["y"].to_numpy(dtype=float)
    if window <= 0 or len(y) < lag + window:
        return 1.0
    current = y[-window:].sum()
    prior = y[-lag - window : -lag].sum()
    if abs(prior) <= 1e-9:
        return 1.0
    ratio = current / prior
    if not np.isfinite(ratio):
        return 1.0
    return max(0.0, 1.0 + (ratio - 1.0) * damping)


def level_scaled_seasonal_naive(train: pd.DataFrame, horizon: int, lag: int, window: int, damping: float = 1.0) -> np.ndarray:
    return seasonal_naive(train, horizon, lag) * level_scale_ratio(train, lag, window, damping)


def seasonal_average(train: pd.DataFrame, horizon: int, base_lag: int, years: int = 2) -> np.ndarray:
    forecasts = []
    weights = []
    for year in range(1, years + 1):
        values = seasonal_values(train, horizon, base_lag * year)
        if values is None:
            continue
        forecasts.append(values.astype(float))
        weights.append(float(years - year + 1))
    if not forecasts:
        return seasonal_naive(train, horizon, base_lag)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    return np.average(np.vstack(forecasts), axis=0, weights=weights_arr)


def daily_forecast_horizon_days(daily_train: pd.DataFrame, weekly_dates: pd.Series) -> int:
    dates = pd.to_datetime(weekly_dates).dt.normalize()
    cutoff = pd.Timestamp(daily_train["ds"].max()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    horizon_days = int((end - cutoff).days)
    if horizon_days <= 0:
        raise ValueError("weekly_dates must be after the daily training cutoff.")
    return horizon_days


def reaggregate_daily_forecast(daily_train: pd.DataFrame, weekly_dates: pd.Series, daily_forecast: np.ndarray, weekly_freq: str = DEFAULT_WEEKLY_FREQ) -> np.ndarray:
    dates = pd.to_datetime(weekly_dates).dt.normalize()
    cutoff = pd.Timestamp(daily_train["ds"].max()).normalize()
    future_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=len(daily_forecast), freq="D")
    daily_frame = pd.DataFrame(
        {
            "ds": future_dates,
            "forecast": daily_forecast,
        }
    )
    weekly = daily_frame.set_index("ds")["forecast"].resample(weekly_freq).sum()
    missing = dates[~dates.isin(weekly.index)]
    if len(missing):
        first_missing = ", ".join(str(d.date()) for d in missing[:5])
        raise ValueError(f"Daily reaggregation could not produce weekly forecasts for: {first_missing}")
    return weekly.reindex(dates).to_numpy()


def daily_reagg_seasonal_naive(daily_train: pd.DataFrame, weekly_dates: pd.Series, lag: int = 364, weekly_freq: str = DEFAULT_WEEKLY_FREQ) -> np.ndarray:
    horizon_days = daily_forecast_horizon_days(daily_train, weekly_dates)
    forecast = seasonal_naive(daily_train, horizon_days, lag)
    return reaggregate_daily_forecast(daily_train, weekly_dates, forecast, weekly_freq)


def daily_reagg_holiday_anchor_aware(
    daily_train: pd.DataFrame,
    weekly_dates: pd.Series,
    default_lag: int = 365,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    horizon_days = daily_forecast_horizon_days(daily_train, weekly_dates)
    cutoff = pd.Timestamp(daily_train["ds"].max()).normalize()
    daily_dates = pd.Series(pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D"))
    forecast = holiday_anchor_aware_seasonal_naive(daily_train, daily_dates, default_lag)
    return reaggregate_daily_forecast(daily_train, weekly_dates, forecast, weekly_freq)


def daily_reagg_level_scaled_seasonal_naive(
    daily_train: pd.DataFrame,
    weekly_dates: pd.Series,
    lag: int = 364,
    window: int = 56,
    damping: float = 1.0,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    horizon_days = daily_forecast_horizon_days(daily_train, weekly_dates)
    forecast = level_scaled_seasonal_naive(daily_train, horizon_days, lag, window, damping)
    return reaggregate_daily_forecast(daily_train, weekly_dates, forecast, weekly_freq)


def daily_reagg_seasonal_average(
    daily_train: pd.DataFrame,
    weekly_dates: pd.Series,
    base_lag: int = 364,
    years: int = 2,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    horizon_days = daily_forecast_horizon_days(daily_train, weekly_dates)
    forecast = seasonal_average(daily_train, horizon_days, base_lag, years)
    return reaggregate_daily_forecast(daily_train, weekly_dates, forecast, weekly_freq)


def daily_train_through_weekly_cutoff(daily_history: pd.DataFrame, weekly_train: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(weekly_train["ds"].max()).normalize()
    return daily_history[pd.to_datetime(daily_history["ds"]).dt.normalize() <= cutoff].copy()


def daily_reagg_seasonal_naive_from_full_history(
    daily_history: pd.DataFrame,
    weekly_train: pd.DataFrame,
    weekly_dates: pd.Series,
    lag: int = 364,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    daily_train = daily_train_through_weekly_cutoff(daily_history, weekly_train)
    return daily_reagg_seasonal_naive(daily_train, weekly_dates, lag, weekly_freq)


def daily_reagg_holiday_anchor_aware_from_full_history(
    daily_history: pd.DataFrame,
    weekly_train: pd.DataFrame,
    weekly_dates: pd.Series,
    default_lag: int = 365,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    daily_train = daily_train_through_weekly_cutoff(daily_history, weekly_train)
    return daily_reagg_holiday_anchor_aware(daily_train, weekly_dates, default_lag, weekly_freq)


def daily_reagg_level_scaled_seasonal_naive_from_full_history(
    daily_history: pd.DataFrame,
    weekly_train: pd.DataFrame,
    weekly_dates: pd.Series,
    lag: int = 364,
    window: int = 56,
    damping: float = 1.0,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    daily_train = daily_train_through_weekly_cutoff(daily_history, weekly_train)
    return daily_reagg_level_scaled_seasonal_naive(daily_train, weekly_dates, lag, window, damping, weekly_freq)


def daily_reagg_seasonal_average_from_full_history(
    daily_history: pd.DataFrame,
    weekly_train: pd.DataFrame,
    weekly_dates: pd.Series,
    base_lag: int = 364,
    years: int = 2,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
) -> np.ndarray:
    daily_train = daily_train_through_weekly_cutoff(daily_history, weekly_train)
    return daily_reagg_seasonal_average(daily_train, weekly_dates, base_lag, years, weekly_freq)


def window_average(train: pd.DataFrame, horizon: int, window: int) -> np.ndarray:
    return np.repeat(train["y"].tail(window).mean(), horizon)


def same_weekday_average(train: pd.DataFrame, dates: pd.Series, weeks: int = 8) -> np.ndarray:
    hist = train.copy()
    hist["dow"] = hist["ds"].dt.dayofweek
    preds = []
    for ds in dates:
        sample = hist.loc[hist["dow"] == ds.dayofweek, "y"].tail(weeks)
        preds.append(float(sample.mean() if len(sample) else hist["y"].tail(28).mean()))
    return np.asarray(preds)


def ets_forecast(train: pd.DataFrame, horizon: int, grain: str = DAILY_GRAIN) -> np.ndarray:
    grain = normalize_grain(grain)
    seasonal_periods = 7 if grain == DAILY_GRAIN else 52
    fallback_window = short_window_for_grain(grain)
    if ExponentialSmoothing is None:
        warn_fallback("statsmodels_ets_weekly", f"window_average_{fallback_window}", _STATSMODELS_IMPORT_ERROR or "statsmodels import unavailable")
        return window_average(train, horizon, fallback_window)
    y = train.set_index("ds")["y"].asfreq(freq_for_grain(grain))
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated")
        return np.maximum(0.0, np.asarray(model.fit(optimized=True).forecast(horizon)))
    except Exception as exc:
        warn_fallback("statsmodels_ets_weekly", f"window_average_{fallback_window}", f"fit/predict error: {type(exc).__name__}: {exc}")
        return window_average(train, horizon, fallback_window)


def sarimax_forecast(train: pd.DataFrame, horizon: int, grain: str = DAILY_GRAIN) -> np.ndarray:
    grain = normalize_grain(grain)
    seasonal_period = 7 if grain == DAILY_GRAIN else 52
    fallback_lag = short_lag_for_grain(grain)
    if SARIMAX is None:
        warn_fallback("statsmodels_sarimax_weekly", f"seasonal_naive_lag_{fallback_lag}", _STATSMODELS_IMPORT_ERROR or "statsmodels import unavailable")
        return seasonal_naive(train, horizon, fallback_lag)
    y = train.set_index("ds")["y"].asfreq(freq_for_grain(grain))
    try:
        model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 1, seasonal_period), trend="c", enforce_stationarity=False, enforce_invertibility=False)
        return np.maximum(0.0, np.asarray(model.fit(disp=False, maxiter=75).forecast(horizon)))
    except Exception as exc:
        warn_fallback("statsmodels_sarimax_weekly", f"seasonal_naive_lag_{fallback_lag}", f"fit/predict error: {type(exc).__name__}: {exc}")
        return seasonal_naive(train, horizon, fallback_lag)


def statsforecast_forecast(train: pd.DataFrame, horizon: int, model_name: str, grain: str = DAILY_GRAIN) -> np.ndarray:
    grain = normalize_grain(grain)
    annual_lag = annual_lag_for_grain(grain)
    if StatsForecast is None:
        warn_fallback(model_name, f"seasonal_naive_lag_{annual_lag}", _STATSFORECAST_IMPORT_ERROR or "statsforecast import unavailable")
        return seasonal_naive(train, horizon, annual_lag)
    if grain == WEEKLY_GRAIN:
        season_lengths = [52] if len(train) >= MIN_TRAIN_WEEKS else [1]
        base_season_length = 52
    else:
        season_lengths = [7, 365] if len(train) >= 730 else [7]
        base_season_length = 7
    try:
        # season_length controls the seasonality the forecaster itself models.
        # Standalone AutoARIMA/AutoETS use weekly (7). MSTL deseasonalizes first,
        # so its trend_forecaster MUST be non-seasonal (season_length=1) or
        # statsforecast raises "Trend forecaster should not adjust seasonal models."
        auto_arima = lambda season_length=7, alias=None: AutoARIMA(
            season_length=season_length,
            stepwise=True,
            approximation=True,
            nmodels=12,
            max_p=3,
            max_q=3,
            max_P=1,
            max_Q=1,
            max_order=4,
            alias=alias or "AutoARIMA",
        )
        if model_name == "statsforecast_auto_arima":
            model = auto_arima(base_season_length, model_name)
        elif model_name == "statsforecast_auto_ets":
            model = AutoETS(season_length=base_season_length, alias=model_name)
        elif model_name == "statsforecast_mstl_auto_arima":
            model = MSTL(
                season_length=season_lengths,
                trend_forecaster=auto_arima(season_length=1),
                alias=model_name,
            )
        elif model_name == "statsforecast_mstl_auto_ets":
            # model="ZZN" => auto error/trend, NO seasonal component (MSTL handles seasonality).
            model = MSTL(season_length=season_lengths, trend_forecaster=AutoETS(season_length=1, model="ZZN"), alias=model_name)
        else:
            raise ValueError(model_name)

        sf_train = train[["unique_id", "ds", "y"]].copy()
        sf = StatsForecast(models=[model], freq=freq_for_grain(grain), n_jobs=1)
        sf.fit(sf_train)
        fcst = sf.predict(h=horizon).reset_index()
        forecast = pd.to_numeric(fcst[model_name], errors="coerce").to_numpy()
        if len(forecast) != horizon or np.isnan(forecast).any():
            raise ValueError(f"StatsForecast produced invalid forecast for {model_name}.")
        return np.maximum(0.0, forecast)
    except Exception as exc:
        warn_fallback(model_name, f"seasonal_naive_lag_{annual_lag}", f"fit/predict error: {type(exc).__name__}: {exc}")
        return seasonal_naive(train, horizon, annual_lag)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    refit_cadence: str
    predict: Callable[[pd.DataFrame, pd.Series], np.ndarray]


def model_specs(horizon: int, openings: pd.DataFrame | None = None, grain: str = DAILY_GRAIN) -> list[ModelSpec]:
    grain = normalize_grain(grain)
    opening_features = openings if openings is not None else OPENINGS_DF
    annual_lag = annual_lag_for_grain(grain)
    short_lag = short_lag_for_grain(grain)
    short_window = short_window_for_grain(grain)
    scale_window_long = 8 if grain == WEEKLY_GRAIN else 56
    medium_window = 12 if grain == WEEKLY_GRAIN else 7
    long_window = 26 if grain == WEEKLY_GRAIN else 28
    specs = [
        ModelSpec(f"seasonal_naive_lag_{short_lag}", "every_cutoff", lambda train, dates: seasonal_naive(train, len(dates), short_lag)),
        ModelSpec(f"seasonal_naive_lag_{annual_lag}", "every_cutoff", lambda train, dates: seasonal_naive(train, len(dates), annual_lag)),
        ModelSpec(
            f"seasonal_naive_scaled_lag_{annual_lag}_w{short_window}",
            "every_cutoff",
            lambda train, dates: level_scaled_seasonal_naive(train, len(dates), annual_lag, short_window),
        ),
        ModelSpec(
            f"seasonal_naive_scaled_lag_{annual_lag}_w{scale_window_long}_damped",
            "every_cutoff",
            lambda train, dates: level_scaled_seasonal_naive(train, len(dates), annual_lag, scale_window_long, 0.5),
        ),
        ModelSpec(
            f"seasonal_average_lag_{annual_lag}_y2",
            "every_cutoff",
            lambda train, dates: seasonal_average(train, len(dates), annual_lag, 2),
        ),
        ModelSpec(
            f"seasonal_average_lag_{annual_lag}_y3",
            "every_cutoff",
            lambda train, dates: seasonal_average(train, len(dates), annual_lag, 3),
        ),
        ModelSpec(f"window_average_{medium_window}", "every_cutoff", lambda train, dates: window_average(train, len(dates), medium_window)),
        ModelSpec(f"window_average_{long_window}", "every_cutoff", lambda train, dates: window_average(train, len(dates), long_window)),
        ModelSpec("statsmodels_ets_weekly", "every_cutoff", lambda train, dates: ets_forecast(train, len(dates), grain)),
        ModelSpec("statsmodels_sarimax_weekly", "every_cutoff", lambda train, dates: sarimax_forecast(train, len(dates), grain)),
        ModelSpec("statsforecast_auto_arima", "every_cutoff_auto_classical", lambda train, dates: statsforecast_forecast(train, len(dates), "statsforecast_auto_arima", grain)),
        ModelSpec("statsforecast_auto_ets", "every_cutoff_auto_classical", lambda train, dates: statsforecast_forecast(train, len(dates), "statsforecast_auto_ets", grain)),
        ModelSpec("statsforecast_mstl_auto_arima", "every_cutoff_mstl_auto_classical", lambda train, dates: statsforecast_forecast(train, len(dates), "statsforecast_mstl_auto_arima", grain)),
        ModelSpec("statsforecast_mstl_auto_ets", "every_cutoff_mstl_auto_classical", lambda train, dates: statsforecast_forecast(train, len(dates), "statsforecast_mstl_auto_ets", grain)),
        ModelSpec(
            "sklearn_hist_gradient_boosting",
            "every_cutoff_recursive",
            lambda train, dates: predict_sklearn_recursive(
                lambda: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=SEED),
                train,
                dates,
                opening_features,
                grain,
            ),
        ),
        ModelSpec(
            "sklearn_random_forest",
            "every_cutoff_recursive",
            lambda train, dates: predict_sklearn_recursive(
                lambda: RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=1),
                train,
                dates,
                opening_features,
                grain,
            ),
        ),
        ModelSpec(
            "sklearn_extra_trees",
            "every_cutoff_recursive",
            lambda train, dates: predict_sklearn_recursive(
                lambda: ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=1),
                train,
                dates,
                opening_features,
                grain,
            ),
        ),
        ModelSpec(
            "sklearn_ridge_lags",
            "every_cutoff_recursive",
            lambda train, dates: predict_sklearn_recursive(lambda: Ridge(alpha=10.0), train, dates, opening_features, grain),
        ),
        ModelSpec(
            "sklearn_hist_gradient_boosting_direct",
            "every_cutoff_direct_multistep",
            lambda train, dates: predict_sklearn_direct(
                lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, l2_regularization=0.05, random_state=SEED),
                train,
                dates,
                opening_features,
                grain,
            ),
        ),
    ]
    if grain == DAILY_GRAIN:
        specs.insert(4, ModelSpec("same_weekday_average_8w", "every_cutoff", lambda train, dates: same_weekday_average(train, dates, 8)))
    else:
        specs.insert(2, ModelSpec(f"window_average_{short_window}", "every_cutoff", lambda train, dates: window_average(train, len(dates), short_window)))
    if LGBMRegressor is not None:
        specs.append(
            ModelSpec(
                "mlforecast_style_lightgbm",
                "every_cutoff_recursive",
                lambda train, dates: predict_sklearn_recursive(
                    lambda: LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=SEED,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                    train,
                    dates,
                    opening_features,
                    grain,
                ),
            )
        )
    if CatBoostRegressor is not None:
        specs.append(
            ModelSpec(
                "mlforecast_style_catboost",
                "every_cutoff_recursive",
                lambda train, dates: predict_sklearn_recursive(
                    lambda: CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.03,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=SEED,
                        thread_count=1,
                        verbose=False,
                    ),
                    train,
                    dates,
                    opening_features,
                    grain,
                ),
            )
        )
    return specs


def add_conformal_intervals(preds: pd.DataFrame) -> pd.DataFrame:
    # Residuals come from the EARLIEST third of cutoffs and intervals are emitted
    # only for cutoffs after that calibration window, so reported coverage is
    # never scored on the rows that produced the residuals (and never uses
    # residuals from folds later than the row being scored). Calibration-cutoff
    # rows keep point forecasts only.
    out = preds.copy()
    out[["p10", "p50", "p90", "p025", "p975"]] = np.nan
    out["p50"] = out["forecast"]
    for (_, model_name), group in out.groupby(["horizon", "model_name"]):
        cutoffs = sorted(group["cutoff"].unique())
        if len(cutoffs) < 3:
            continue
        calibration_cutoffs = cutoffs[: max(2, len(cutoffs) // 3)]
        residuals = group.loc[group["cutoff"].isin(calibration_cutoffs)].copy()
        residuals["residual"] = residuals["actual"] - residuals["forecast"]
        target = group.loc[~group["cutoff"].isin(calibration_cutoffs)]
        for h_step, step_group in target.groupby("horizon_step"):
            step_resid = residuals.loc[residuals["horizon_step"] == h_step, "residual"]
            if len(step_resid) < 2:
                step_resid = residuals["residual"]
            qs = np.quantile(step_resid, [0.025, 0.10, 0.90, 0.975])
            idx = step_group.index
            out.loc[idx, "p025"] = out.loc[idx, "forecast"] + qs[0]
            out.loc[idx, "p10"] = out.loc[idx, "forecast"] + qs[1]
            out.loc[idx, "p90"] = out.loc[idx, "forecast"] + qs[2]
            out.loc[idx, "p975"] = out.loc[idx, "forecast"] + qs[3]
    return out


def build_ensembles(preds: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for horizon, hdf in preds.groupby("horizon"):
        metrics = hdf.groupby("model_name").apply(score_group, include_groups=False).sort_values("wape")
        candidates = list(metrics.head(top_n).index)
        candidate_selection_note = "diagnostic_same_cutoffs_topn_by_wape"
        for (cutoff, ds, h_step), group in hdf[hdf["model_name"].isin(candidates)].groupby(["cutoff", "ds", "horizon_step"]):
            actual = float(group["actual"].iloc[0])
            values = group.set_index("model_name")["forecast"]
            top3 = [m for m in candidates[:3] if m in values.index]
            top5 = [m for m in candidates[:5] if m in values.index]
            inv = 1 / np.square(metrics.loc[top5, "wape"].clip(lower=1e-6))
            weights = inv / inv.sum()
            ensembles = {
                "ensemble_top3_mean": values.loc[top3].mean(),
                "ensemble_top5_mean": values.loc[top5].mean(),
                "ensemble_top5_median": values.loc[top5].median(),
                "ensemble_top5_inverse_wape": float((values.loc[top5] * weights).sum()),
            }
            for name, forecast in ensembles.items():
                rows.append(
                    {
                        "unique_id": "total_sales",
                        "horizon": horizon,
                        "cutoff": cutoff,
                        "ds": ds,
                        "horizon_step": h_step,
                        "actual": actual,
                        "model_name": name,
                        "forecast": max(0.0, float(forecast)),
                        "runtime_seconds": 0.0,
                        "refit_cadence": "derived_from_base_predictions",
                        "hyperparams_json": json.dumps({"candidates": top5, "candidate_selection": candidate_selection_note}),
                        "library_version": "local",
                    }
                )
    return pd.DataFrame(rows)


def run_backtests(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
    max_cutoffs: int | None,
    openings: pd.DataFrame | None = None,
    model_names: set[str] | None = None,
    grain: str = DAILY_GRAIN,
) -> pd.DataFrame:
    grain = normalize_grain(grain)
    all_rows = []
    for horizon in horizons:
        cutoffs = build_cutoffs(df, horizon, max_cutoffs, grain)
        specs = model_specs(horizon, openings, grain)
        if model_names:
            specs = [spec for spec in specs if spec.name in model_names]
        for cutoff in cutoffs:
            train = df[df["ds"] <= cutoff].copy()
            test = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon_delta(horizon, grain))].copy()
            if len(test) != horizon:
                raise ValueError(f"Cutoff {cutoff.date()} has {len(test)} test rows, expected {horizon}.")
            dates = test["ds"].reset_index(drop=True)
            for spec in specs:
                started = time.perf_counter()
                forecast = spec.predict(train, dates)
                runtime = time.perf_counter() - started
                for i, (row, pred) in enumerate(zip(test.itertuples(index=False), forecast), start=1):
                    all_rows.append(
                        {
                            "unique_id": row.unique_id,
                            "horizon": horizon,
                            "cutoff": cutoff,
                            "ds": row.ds,
                            "horizon_step": i,
                            "actual": float(row.y),
                            "model_name": spec.name,
                            "forecast": float(pred),
                            "runtime_seconds": runtime,
                            "refit_cadence": spec.refit_cadence,
                            "hyperparams_json": "{}",
                            "library_version": "local",
                        }
                    )
    preds = pd.DataFrame(all_rows)
    ensembles = build_ensembles(preds)
    return add_conformal_intervals(pd.concat([preds, ensembles], ignore_index=True))


def write_metrics(preds: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    by_cutoff = preds.groupby(["horizon", "cutoff", "model_name"], dropna=False).apply(score_group, include_groups=False).reset_index()
    by_horizon = preds.groupby(["horizon", "model_name"], dropna=False).apply(score_group, include_groups=False).reset_index()
    # ensemble_* candidates were selected by WAPE on these same folds, so their
    # metrics are optimistically biased; flag them so the leaderboard is honest.
    by_horizon["is_diagnostic_ensemble"] = by_horizon["model_name"].str.startswith("ensemble")
    by_month = preds.assign(month=preds["ds"].dt.to_period("M").astype(str)).groupby(["horizon", "month", "model_name"], dropna=False).apply(score_group, include_groups=False).reset_index()
    by_horizon.to_csv(output_dir / "model_metrics_by_horizon.csv", index=False)
    by_cutoff.to_csv(output_dir / "model_metrics_by_cutoff.csv", index=False)
    by_month.to_csv(output_dir / "model_metrics_by_month.csv", index=False)
    by_horizon[by_horizon["model_name"].str.startswith("ensemble")].to_csv(output_dir / "ensemble_results.csv", index=False)
    return {"by_cutoff": by_cutoff, "by_horizon": by_horizon, "by_month": by_month}


def plot_outputs(df: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(exist_ok=True)

    for horizon in sorted(preds["horizon"].unique()):
        hmetrics = metrics[metrics["horizon"] == horizon].sort_values("wape")
        plt.figure(figsize=(11, 6))
        plt.barh(hmetrics["model_name"], hmetrics["wape"])
        plt.gca().invert_yaxis()
        plt.xlabel("WAPE")
        plt.title(f"WAPE by model, horizon {horizon}")
        plt.tight_layout()
        plt.savefig(chart_dir / f"wape_by_model_h{horizon}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(hmetrics["wape"], hmetrics["bias_pct"])
        for _, row in hmetrics.iterrows():
            plt.annotate(row["model_name"], (row["wape"], row["bias_pct"]), fontsize=8)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("WAPE")
        plt.ylabel("Bias pct")
        plt.title(f"WAPE vs bias, horizon {horizon}")
        plt.tight_layout()
        plt.savefig(chart_dir / f"wape_vs_bias_h{horizon}.png", dpi=150)
        plt.close()

    primary_horizon = 91 if 91 in set(preds["horizon"]) else int(preds["horizon"].max())
    eligible = metrics.loc[metrics["horizon"].eq(primary_horizon) & ~metrics["model_name"].str.startswith("ensemble")]
    best_model = eligible.sort_values("wape").iloc[0]["model_name"]
    model_preds = preds[(preds["horizon"] == primary_horizon) & (preds["model_name"] == best_model)].copy()
    latest_cutoff = model_preds["cutoff"].max()
    latest = model_preds[model_preds["cutoff"] == latest_cutoff]
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Actual history", color="black", linewidth=1.2)
    plt.plot(latest["ds"], latest["forecast"], label=f"{best_model} forecast", color="#1f77b4", linewidth=2)
    if latest["p10"].notna().any():
        plt.fill_between(latest["ds"], latest["p10"], latest["p90"], color="#1f77b4", alpha=0.18, label="80% interval")
        plt.fill_between(latest["ds"], latest["p025"], latest["p975"], color="#1f77b4", alpha=0.10, label="95% interval")
    plt.title(f"Actual vs forecast, {best_model}, cutoff {latest_cutoff.date()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "actual_vs_best_forecast.png", dpi=150)
    plt.close()

    residuals = model_preds.copy()
    residuals["error"] = residuals["forecast"] - residuals["actual"]
    plt.figure(figsize=(12, 5))
    plt.scatter(residuals["ds"], residuals["error"], s=10, alpha=0.55)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Residuals over time, {best_model}, horizon {primary_horizon}")
    plt.tight_layout()
    plt.savefig(chart_dir / "residuals_over_time.png", dpi=150)
    plt.close()


def write_future_forecast(df: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path, grain: str = DAILY_GRAIN) -> None:
    grain = normalize_grain(grain)
    horizon = 91 if 91 in set(preds["horizon"]) else int(preds["horizon"].max())
    # ensemble_* rows are excluded: their candidates were selected on these same
    # folds (leaky ranking), and their varying aggregations (median, weighted)
    # were previously misreproduced here as a plain mean.
    eligible = metrics[(metrics["horizon"] == horizon) & ~metrics["model_name"].str.startswith("ensemble")]
    best_model = eligible.sort_values("wape").iloc[0]["model_name"]
    base_models = {spec.name: spec for spec in model_specs(horizon, OPENINGS_DF, grain)}
    future_dates = pd.Series(pd.date_range(df["ds"].max() + horizon_delta(1, grain), periods=horizon, freq=freq_for_grain(grain)))
    forecast = base_models[best_model].predict(df, future_dates)
    model_name = best_model

    residual_source = preds[(preds["horizon"] == horizon) & (preds["model_name"] == model_name)].copy()
    residual_source["residual"] = residual_source["actual"] - residual_source["forecast"]
    rows = []
    for i, (ds, pred) in enumerate(zip(future_dates, forecast), start=1):
        resid = residual_source.loc[residual_source["horizon_step"] == i, "residual"]
        if len(resid) < 3:
            resid = residual_source["residual"]
        qs = np.quantile(resid, [0.025, 0.10, 0.90, 0.975])
        rows.append(
            {
                "unique_id": "total_sales",
                "ds": ds,
                "model_name": model_name,
                "forecast": max(0.0, float(pred)),
                "p10": max(0.0, float(pred + qs[1])),
                "p50": max(0.0, float(pred)),
                "p90": max(0.0, float(pred + qs[2])),
                "p025": max(0.0, float(pred + qs[0])),
                "p975": max(0.0, float(pred + qs[3])),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "training_cutoff": df["ds"].max(),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "final_forecast.csv", index=False)


def write_manifest(df: pd.DataFrame, preds: pd.DataFrame, output_dir: Path, input_path: Path, grain: str = DAILY_GRAIN) -> None:
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "seed": SEED,
        "python": platform.python_version(),
        "input_file": str(input_path),
        "date_min": str(df["ds"].min().date()),
        "date_max": str(df["ds"].max().date()),
        "rows": int(len(df)),
        "grain": normalize_grain(grain),
        "freq": df.attrs.get("freq", freq_for_grain(grain)),
        "missing_days_filled": int(df.attrs.get("missing_days_filled", 0)),
        "horizons": sorted(preds["horizon"].unique().tolist()),
        "models": sorted(preds["model_name"].unique().tolist()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_bakeoff.py", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--max-cutoffs", type=int, default=None, help="Limit to the most recent N cutoffs per horizon for faster smoke tests.")
    parser.add_argument("--opening-info", default=None, help="Optional unit-opening CSV with Opening_date/open_date.")
    parser.add_argument("--missing-date-strategy", choices=["interpolate", "zero", "error"], default="interpolate")
    parser.add_argument("--models", default="", help="Optional comma-separated ModelSpec names to run.")
    parser.add_argument("--grain", choices=[DAILY_GRAIN, WEEKLY_GRAIN], default=DAILY_GRAIN)
    parser.add_argument("--weekly-freq", default=DEFAULT_WEEKLY_FREQ, help="Pandas weekly frequency used when --grain weekly.")
    args = parser.parse_args()

    global OPENINGS_DF
    set_seeds()
    input_path = Path(args.sales_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    OPENINGS_DF = load_openings(Path(args.opening_info)) if args.opening_info else None
    df = load_sales(input_path, fill_missing=args.missing_date_strategy, audit_path=output_dir / "missing_sales_dates_audit.csv", grain=args.grain, weekly_freq=args.weekly_freq)
    if OPENINGS_DF is not None:
        feature_dates = add_opening_features(pd.DataFrame({"ds": pd.date_range(df["ds"].min(), df["ds"].max(), freq=freq_for_grain(args.grain, args.weekly_freq))}), OPENINGS_DF)
        feature_dates.to_csv(output_dir / "opening_features_by_date.csv", index=False)
    model_names = {m.strip() for m in args.models.split(",") if m.strip()} or None
    preds = run_backtests(df, horizons, args.max_cutoffs, OPENINGS_DF, model_names, args.grain)
    preds.to_csv(output_dir / "model_backtest_predictions.csv", index=False)
    try:
        preds.to_parquet(output_dir / "model_backtest_predictions.parquet", index=False)
    except Exception:
        pass

    metrics = write_metrics(preds, output_dir)
    write_future_forecast(df, preds, metrics["by_horizon"], output_dir, args.grain)
    plot_outputs(df, preds, metrics["by_horizon"], output_dir)
    write_manifest(df, preds, output_dir, input_path, args.grain)
    if OPENINGS_DF is not None:
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["opening_info_file"] = args.opening_info
        manifest["opening_count"] = int(len(OPENINGS_DF))
        manifest["opening_features_used_for_ml_models"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Surface any models that silently degraded to a baseline so a broken run
    # can never masquerade as a clean one.
    (output_dir / "model_fallbacks.json").write_text(json.dumps(DEGRADED_MODELS, indent=2), encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["degraded_models"] = DEGRADED_MODELS
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    honest = metrics["by_horizon"][~metrics["by_horizon"]["model_name"].str.startswith("ensemble")]
    best = honest.sort_values(["horizon", "wape"]).groupby("horizon").head(1)
    print(best[["horizon", "model_name", "wape", "bias_pct", "mae", "rmse"]].to_string(index=False))

    if DEGRADED_MODELS:
        import sys

        print("\n" + "=" * 72, file=sys.stderr)
        print(f"WARNING: {len(DEGRADED_MODELS)} model(s) DID NOT RUN and fell back to a baseline:", file=sys.stderr)
        for name, reason in DEGRADED_MODELS.items():
            print(f"  - {name}: {reason}", file=sys.stderr)
        print("Their metrics are the baseline's. Do not treat them as real results.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)


if __name__ == "__main__":
    main()
