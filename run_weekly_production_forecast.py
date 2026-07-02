"""Package weekly production forecasts from validated weekly-grain rules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from run_bakeoff import (
    DEFAULT_WEEKLY_FREQ,
    daily_reagg_level_scaled_seasonal_naive_from_full_history,
    daily_reagg_seasonal_average_from_full_history,
    daily_reagg_seasonal_naive_from_full_history,
    load_sales,
    seasonal_naive,
    set_seeds,
)


QUANTILE_LEVELS = [0.025, 0.10, 0.90, 0.975]


def weekly_segment(ds: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ds)
    return np.where(dates.dt.month.isin([11, 12]), "holiday_weeks", "rest_weeks")


def grid_path_for_horizon(horizon: int) -> Path:
    path = Path(f"outputs_weekly_multifold_{horizon}w_daily_reagg_pool") / "multifold_grid_predictions.csv"
    if not path.exists():
        raise ValueError(f"{path} is missing; run daily-reaggregation weekly validation first.")
    return path


def trend_seasonal_grid_path_for_horizon(horizon: int) -> Path:
    path = Path(f"outputs_weekly_multifold_{horizon}w_trend_seasonal_pool") / "multifold_grid_predictions.csv"
    if not path.exists():
        raise ValueError(f"{path} is missing; run weekly trend/seasonal validation first.")
    return path


def trend_seasonal_pool_members(horizon: int) -> list[str]:
    manifest_path = Path(f"outputs_weekly_multifold_{horizon}w_trend_seasonal_pool") / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} is missing; cannot confirm the 26w challenger pool.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(manifest["pool"])


def parameter_free_member_forecast(
    name: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    dates: pd.Series,
    horizon: int,
    weekly_freq: str,
) -> np.ndarray:
    if name == "seasonal_naive_lag_52":
        return seasonal_naive(weekly, horizon, 52)
    if name == "daily_reagg_seasonal_naive_lag_365":
        return daily_reagg_seasonal_naive_from_full_history(daily, weekly, dates, lag=365, weekly_freq=weekly_freq)
    if name.startswith("daily_reagg_scaled_seasonal_naive_lag_"):
        parts = name.split("_")
        lag = int(parts[6])
        window = int(parts[7].removeprefix("w"))
        damping = 0.5 if parts[-1] == "damped" else 1.0
        return daily_reagg_level_scaled_seasonal_naive_from_full_history(daily, weekly, dates, lag, window, damping, weekly_freq)
    if name.startswith("daily_reagg_seasonal_average_lag_"):
        parts = name.split("_")
        lag = int(parts[5])
        years = int(parts[6].removeprefix("y"))
        return daily_reagg_seasonal_average_from_full_history(daily, weekly, dates, lag, years, weekly_freq)
    raise ValueError(f"{name} is not a supported parameter-free weekly challenger member.")


def trend_seasonal_pool_median_forecast(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    dates: pd.Series,
    horizon: int,
    weekly_freq: str,
) -> np.ndarray:
    members = trend_seasonal_pool_members(horizon)
    forecasts = [parameter_free_member_forecast(name, daily, weekly, dates, horizon, weekly_freq) for name in members]
    return np.median(np.vstack(forecasts), axis=0)


def residual_quantiles(grid_path: Path, model_name: str, interval_windows: int) -> dict:
    preds = pd.read_csv(grid_path, parse_dates=["cutoff", "ds"])
    model_preds = preds[preds["model_name"].eq(model_name)].copy()
    if model_preds.empty and model_name == "pool_median":
        manifest = json.loads((grid_path.parent / "manifest.json").read_text(encoding="utf-8"))
        pool = list(manifest["pool"])
        wide = preds[preds["model_name"].isin(pool)].pivot_table(index=["cutoff", "ds", "actual"], columns="model_name", values="forecast").reset_index()
        missing = [name for name in pool if name not in wide.columns]
        if missing:
            raise ValueError(f"{grid_path} is missing pool member(s): {missing}")
        model_preds = wide[["cutoff", "ds", "actual"]].copy()
        model_preds["forecast"] = wide[pool].median(axis=1)
    if model_preds.empty:
        raise ValueError(f"{grid_path} has no rows for {model_name}.")
    recent_cutoffs = sorted(model_preds["cutoff"].unique())[-interval_windows:]
    recent = model_preds[model_preds["cutoff"].isin(recent_cutoffs)].copy()
    recent["segment"] = weekly_segment(recent["ds"])
    recent["residual"] = recent["actual"] - recent["forecast"]

    quantiles = {"global": np.quantile(recent["residual"].to_numpy(), QUANTILE_LEVELS), "counts": {}}
    for segment, group in recent.groupby("segment"):
        quantiles[segment] = np.quantile(group["residual"].to_numpy(), QUANTILE_LEVELS)
        quantiles["counts"][segment] = len(group)
    quantiles["n_windows"] = len(recent_cutoffs)
    return quantiles


def quantiles_for_segment(quantiles: dict, segment: str, interval_mode: str, shrinkage_k: float) -> np.ndarray:
    if interval_mode == "segment":
        return quantiles.get(segment, quantiles["global"])
    if interval_mode != "shrunken":
        return quantiles["global"]
    segment_qs = quantiles.get(segment)
    n_segment = quantiles.get("counts", {}).get(segment, 0)
    if segment_qs is None or n_segment == 0:
        return quantiles["global"]
    if shrinkage_k < 0:
        raise ValueError("shrinkage_k must be non-negative.")
    weight = n_segment / (n_segment + shrinkage_k) if n_segment + shrinkage_k else 1.0
    return weight * segment_qs + (1.0 - weight) * quantiles["global"]


def rows_for_model(
    model_name: str,
    dates: pd.Series,
    forecast: np.ndarray,
    quantiles: dict,
    training_cutoff: pd.Timestamp,
    interval_mode: str = "global",
    shrinkage_k: float = 30.0,
) -> list[dict]:
    rows = []
    segments = weekly_segment(dates)
    for ds, pred, segment in zip(dates, forecast, segments):
        qs = quantiles_for_segment(quantiles, segment, interval_mode, shrinkage_k)
        interval_note = f"{interval_mode}_residual_quantiles_recent_{quantiles['n_windows']}_grid_windows"
        if interval_mode == "segment":
            interval_note = f"{interval_note}_{segment}"
        if interval_mode == "shrunken":
            interval_note = f"{interval_note}_k{shrinkage_k:g}_{segment}"
        rows.append(
            {
                "unique_id": "total_sales",
                "ds": ds,
                "model_name": model_name,
                "forecast": max(0.0, float(pred)),
                "p025": max(0.0, float(pred + qs[0])),
                "p10": max(0.0, float(pred + qs[1])),
                "p50": max(0.0, float(pred)),
                "p90": max(0.0, float(pred + qs[2])),
                "p975": max(0.0, float(pred + qs[3])),
                "interval_method": interval_note,
                "weekly_segment": segment,
                "training_cutoff": training_cutoff,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--horizons", default="4,12,26,52")
    parser.add_argument("--weekly-freq", default=DEFAULT_WEEKLY_FREQ)
    parser.add_argument("--interval-windows", type=int, default=8)
    parser.add_argument("--interval-mode", choices=["global", "segment", "shrunken"], default="global")
    parser.add_argument("--shrinkage-k", type=float, default=30.0)
    parser.add_argument("--output-dir", default="outputs_weekly_production_forecast")
    args = parser.parse_args()

    set_seeds()
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    daily = load_sales(Path(args.sales_csv))
    weekly = load_sales(Path(args.sales_csv), grain="weekly", weekly_freq=args.weekly_freq)
    training_cutoff = pd.Timestamp(weekly["ds"].max())

    rows = []
    interval_meta = {}
    for horizon in horizons:
        dates = pd.Series(pd.date_range(training_cutoff + pd.Timedelta(weeks=1), periods=horizon, freq=args.weekly_freq))
        final_model = f"final_{horizon}w_daily_reagg_seasonal_naive_lag_365"
        benchmark_model = f"benchmark_{horizon}w_seasonal_naive_lag_52"
        final_forecast = daily_reagg_seasonal_naive_from_full_history(daily, weekly, dates, lag=365, weekly_freq=args.weekly_freq)
        benchmark_forecast = seasonal_naive(weekly, horizon, 52)

        grid_path = grid_path_for_horizon(horizon)
        final_qs = residual_quantiles(grid_path, "daily_reagg_seasonal_naive_lag_365", args.interval_windows)
        benchmark_qs = residual_quantiles(grid_path, "seasonal_naive_lag_52", args.interval_windows)
        interval_meta[str(horizon)] = {
            "grid": str(grid_path),
            "n_windows": int(final_qs["n_windows"]),
            "shrinkage_k": args.shrinkage_k if args.interval_mode == "shrunken" else None,
            "segment_definition": "W-WED rows with week-ending month in Nov/Dec are holiday_weeks; all others rest_weeks",
        }
        rows.extend(rows_for_model(final_model, dates, final_forecast, final_qs, training_cutoff, args.interval_mode, args.shrinkage_k))
        rows.extend(rows_for_model(benchmark_model, dates, benchmark_forecast, benchmark_qs, training_cutoff, args.interval_mode, args.shrinkage_k))
        if horizon == 26:
            challenger_model = "challenger_26w_pool_median_trend_seasonal"
            challenger_forecast = trend_seasonal_pool_median_forecast(daily, weekly, dates, horizon, args.weekly_freq)
            challenger_grid = trend_seasonal_grid_path_for_horizon(horizon)
            challenger_qs = residual_quantiles(challenger_grid, "pool_median", args.interval_windows)
            interval_meta[str(horizon)]["challenger_grid"] = str(challenger_grid)
            rows.extend(rows_for_model(challenger_model, dates, challenger_forecast, challenger_qs, training_cutoff, args.interval_mode, args.shrinkage_k))

    out = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "weekly_production_forecast.csv", index=False)

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": str(training_cutoff.date()),
        "horizons": horizons,
        "weekly_freq": args.weekly_freq,
        "models": sorted(out["model_name"].unique().tolist()),
        "interval_windows": args.interval_windows,
        "interval_mode": args.interval_mode,
        "shrinkage_k": args.shrinkage_k if args.interval_mode == "shrunken" else None,
        "interval_sources": interval_meta,
        "basis": "Weekly daily-reaggregation validation; see model_cards.md and model_usage_plan.md.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_weekly_production_forecast.py", manifest)

    summary = out.groupby("model_name").agg(rows=("ds", "size"), start=("ds", "min"), end=("ds", "max"), forecast_sum=("forecast", "sum")).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
