"""Multi-fold nested final validation for the production stack decision.

Evaluates K rolling final-holdout folds at a given horizon. For each fold,
convex-stack and inverse-WAPE weights are refit using only backtest windows
that end on or before that fold's cutoff (nested validation), then scored on
the fold alongside the standalone pool models. This answers whether the
learned stack beats the grain-specific seasonal naive consistently, not just
on the single latest holdout window that phase 5 validates.

Run summary reports relative metrics only (WAPE, bias pct); absolute values
stay in the prediction CSVs under the gitignored output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_bakeoff
from run_bakeoff import (
    DEGRADED_MODELS,
    DAILY_GRAIN,
    DEFAULT_WEEKLY_FREQ,
    MIN_TRAIN_DAYS,
    MIN_TRAIN_WEEKS,
    SEED,
    WEEKLY_GRAIN,
    annual_lag_for_grain,
    daily_reagg_holiday_anchor_aware_from_full_history,
    daily_reagg_level_scaled_seasonal_naive_from_full_history,
    daily_reagg_seasonal_average_from_full_history,
    daily_reagg_seasonal_naive_from_full_history,
    daily_forecast_horizon_days,
    daily_train_through_weekly_cutoff,
    ets_forecast,
    freq_for_grain,
    horizon_delta,
    load_openings,
    load_sales,
    normalize_grain,
    predict_sklearn_recursive,
    reaggregate_daily_forecast,
    holiday_anchor_aware_seasonal_naive,
    seasonal_naive,
    set_seeds,
    statsforecast_forecast,
    window_average,
)
from run_phase3 import optimize_weights, wape
from experiment_registry import append_experiment


DEFAULT_POOL = [
    "seasonal_naive_lag_364",
    "timesfm_2_5",
    "chronos_bolt_small",
    "statsforecast_mstl_auto_arima",
    "statsforecast_mstl_auto_ets",
]
WEEKLY_DEFAULT_POOL = [
    "seasonal_naive_lag_52",
    "daily_reagg_seasonal_naive_lag_365",
    "timesfm_2_5",
    "chronos_bolt_small",
    "statsforecast_mstl_auto_arima",
    "statsforecast_mstl_auto_ets",
]
MIN_INNER_CUTOFFS = 3


def default_grid_step_for_grain(grain: str) -> int:
    return 4 if normalize_grain(grain) == WEEKLY_GRAIN else 28


def benchmark_for_grain(grain: str) -> str:
    return f"seasonal_naive_lag_{annual_lag_for_grain(grain)}"


def build_grid(df: pd.DataFrame, horizon: int, step: int, grain: str) -> list[pd.Timestamp]:
    grain = normalize_grain(grain)
    if grain == WEEKLY_GRAIN:
        first_idx = MIN_TRAIN_WEEKS - 1
        last_idx = len(df) - horizon - 1
        if first_idx > last_idx:
            raise ValueError(f"Not enough history for weekly horizon={horizon}.")
        return [pd.Timestamp(c).normalize() for c in df["ds"].iloc[first_idx : last_idx + 1 : step]]

    last = df["ds"].max() - pd.Timedelta(days=horizon)
    first = df["ds"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    if last < first:
        raise ValueError(f"Not enough history for horizon={horizon}.")
    cutoffs = []
    cutoff = last
    while cutoff >= first:
        cutoffs.append(cutoff)
        cutoff -= pd.Timedelta(days=step)
    return sorted(cutoffs)


def future_dates(train: pd.DataFrame, horizon: int, grain: str, weekly_freq: str) -> pd.Series:
    return pd.Series(pd.date_range(train["ds"].max() + horizon_delta(1, grain), periods=horizon, freq=freq_for_grain(grain, weekly_freq)))


def make_predictors(
    pool: list[str],
    horizon: int,
    timesfm_context: int,
    chronos_context: int,
    openings: pd.DataFrame | None = None,
    grain: str = DAILY_GRAIN,
    weekly_freq: str = DEFAULT_WEEKLY_FREQ,
    daily_history: pd.DataFrame | None = None,
) -> dict:
    grain = normalize_grain(grain)
    predictors = {}
    for name in pool:
        if name == "seasonal_naive_holiday_anchor_aware":
            if grain != DAILY_GRAIN:
                raise ValueError(f"{name} is only valid with --grain daily.")
            predictors[name] = lambda train, h: holiday_anchor_aware_seasonal_naive(train, future_dates(train, h, grain, weekly_freq), default_lag=364)
        elif name == "daily_reagg_seasonal_naive_holiday_anchor_aware":
            if grain != WEEKLY_GRAIN:
                raise ValueError(f"{name} is only valid with --grain weekly.")
            if daily_history is None:
                raise ValueError(f"{name} requires daily history.")
            predictors[name] = lambda train, h, daily=daily_history: daily_reagg_holiday_anchor_aware_from_full_history(
                daily,
                train,
                future_dates(train, h, grain, weekly_freq),
                default_lag=365,
                weekly_freq=weekly_freq,
            )
        elif name.startswith("seasonal_naive_lag_"):
            lag = int(name.rsplit("_", 1)[1])
            predictors[name] = lambda train, h, lag=lag: seasonal_naive(train, h, lag)
        elif name.startswith("daily_reagg_scaled_seasonal_naive_lag_"):
            if grain != WEEKLY_GRAIN:
                raise ValueError(f"{name} is only valid with --grain weekly.")
            if daily_history is None:
                raise ValueError(f"{name} requires daily history.")
            parts = name.split("_")
            lag = int(parts[6])
            window = int(parts[7].removeprefix("w"))
            damping = 0.5 if parts[-1] == "damped" else 1.0
            predictors[name] = lambda train, h, daily=daily_history, lag=lag, window=window, damping=damping: daily_reagg_level_scaled_seasonal_naive_from_full_history(
                daily,
                train,
                future_dates(train, h, grain, weekly_freq),
                lag,
                window,
                damping,
                weekly_freq,
            )
        elif name.startswith("daily_reagg_seasonal_average_lag_"):
            if grain != WEEKLY_GRAIN:
                raise ValueError(f"{name} is only valid with --grain weekly.")
            if daily_history is None:
                raise ValueError(f"{name} requires daily history.")
            parts = name.split("_")
            lag = int(parts[5])
            years = int(parts[6].removeprefix("y"))
            predictors[name] = lambda train, h, daily=daily_history, lag=lag, years=years: daily_reagg_seasonal_average_from_full_history(
                daily,
                train,
                future_dates(train, h, grain, weekly_freq),
                lag,
                years,
                weekly_freq,
            )
        elif name.startswith("daily_reagg_seasonal_naive_lag_"):
            if grain != WEEKLY_GRAIN:
                raise ValueError(f"{name} is only valid with --grain weekly.")
            if daily_history is None:
                raise ValueError(f"{name} requires daily history.")
            lag = int(name.rsplit("_", 1)[1])
            predictors[name] = lambda train, h, daily=daily_history, lag=lag: daily_reagg_seasonal_naive_from_full_history(
                daily,
                train,
                future_dates(train, h, grain, weekly_freq),
                lag,
                weekly_freq,
            )
        elif name.startswith("window_average_"):
            window = int(name.rsplit("_", 1)[1])
            predictors[name] = lambda train, h, window=window: window_average(train, h, window)
        elif name == "statsmodels_ets_weekly":
            predictors[name] = lambda train, h: ets_forecast(train, h, grain)
        elif name == "sklearn_extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor

            predictors[name] = lambda train, h: predict_sklearn_recursive(
                lambda: ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=1),
                train,
                future_dates(train, h, grain, weekly_freq),
                openings,
                grain,
            )
        elif name.startswith("statsforecast_"):
            predictors[name] = lambda train, h, n=name: statsforecast_forecast(train, h, n, grain)
        elif name == "timesfm_2_5":
            from run_prophet_timesfm_benchmark import forecast_timesfm, load_timesfm

            model = load_timesfm(timesfm_context, horizon)
            predictors[name] = lambda train, h, m=model: forecast_timesfm(m, train, h, timesfm_context)["forecast"].to_numpy()
        elif name == "chronos_bolt_small":
            from run_prophet_timesfm_benchmark import forecast_chronos_bolt, load_chronos_bolt

            model = load_chronos_bolt()
            predictors[name] = lambda train, h, m=model: forecast_chronos_bolt(m, train, h, chronos_context)["forecast"].to_numpy()
        elif name == "chronos2":
            from run_prophet_timesfm_benchmark import forecast_chronos2, load_chronos2

            model = load_chronos2()
            predictors[name] = lambda train, h, m=model: forecast_chronos2(m, train, h, chronos_context)["forecast"].to_numpy()
        elif name == "daily_reagg_chronos2":
            if grain != WEEKLY_GRAIN:
                raise ValueError(f"{name} is only valid with --grain weekly.")
            if daily_history is None:
                raise ValueError(f"{name} requires daily history.")
            from run_prophet_timesfm_benchmark import forecast_chronos2, load_chronos2

            model = load_chronos2()

            def predict_daily_reagg_chronos2(train, h, m=model, daily=daily_history):
                weekly_dates = future_dates(train, h, grain, weekly_freq)
                daily_train = daily_train_through_weekly_cutoff(daily, train)
                horizon_days = daily_forecast_horizon_days(daily_train, weekly_dates)
                daily_forecast = forecast_chronos2(m, daily_train, horizon_days, chronos_context)["forecast"].to_numpy()
                return reaggregate_daily_forecast(daily_train, weekly_dates, daily_forecast, weekly_freq)

            predictors[name] = predict_daily_reagg_chronos2
        else:
            specs = {spec.name: spec for spec in run_bakeoff.model_specs(horizon, openings, grain)}
            if name not in specs:
                raise ValueError(f"Unsupported pool model: {name}")
            predictors[name] = lambda train, h, spec=specs[name]: spec.predict(train, future_dates(train, h, grain, weekly_freq))
    return predictors


def run_grid(df: pd.DataFrame, cutoffs: list[pd.Timestamp], horizon: int, predictors: dict, grain: str) -> pd.DataFrame:
    rows = []
    for i, cutoff in enumerate(cutoffs, start=1):
        train = df[df["ds"] <= cutoff].copy()
        test = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon_delta(horizon, grain))].copy()
        if len(test) != horizon:
            raise ValueError(f"Cutoff {cutoff.date()} has {len(test)} test rows, expected {horizon}.")
        for model_name, predict in predictors.items():
            started = time.perf_counter()
            forecast = np.maximum(0.0, np.asarray(predict(train, horizon), dtype=float))
            runtime = time.perf_counter() - started
            for step, (row, pred) in enumerate(zip(test.itertuples(index=False), forecast), start=1):
                rows.append(
                    {
                        "unique_id": row.unique_id,
                        "horizon": horizon,
                        "cutoff": cutoff,
                        "window_end": cutoff + horizon_delta(horizon, grain),
                        "ds": row.ds,
                        "horizon_step": step,
                        "actual": float(row.y),
                        "model_name": model_name,
                        "forecast": float(pred),
                        "runtime_seconds": runtime,
                    }
                )
        print(f"[grid] cutoff {cutoff.date()} done ({i}/{len(cutoffs)})", flush=True)
    return pd.DataFrame(rows)


def make_wide(preds: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    wide = preds.pivot_table(index=["cutoff", "ds", "horizon_step", "actual"], columns="model_name", values="forecast", aggfunc="first").reset_index()
    return wide.dropna(subset=[m for m in pool if m in wide.columns])


def fold_scores(fold_wide: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    actual = fold_wide["actual"].to_numpy()
    denom = actual.sum()
    scores = {}
    for col in columns:
        forecast = fold_wide[col].to_numpy()
        scores[col] = {
            "wape": wape(actual, forecast),
            "bias_pct": float((forecast - actual).sum() / denom) if abs(denom) > 1e-9 else np.nan,
        }
    return scores


def run_folds(grid_preds: pd.DataFrame, pool: list[str], fold_cutoffs: list[pd.Timestamp], horizon: int, grain: str, benchmark: str) -> tuple[pd.DataFrame, dict]:
    wide = make_wide(grid_preds, pool)
    available = [m for m in pool if m in wide.columns]
    if benchmark not in available:
        raise ValueError(f"Benchmark {benchmark} missing from grid predictions.")

    metric_rows = []
    fold_metadata = {}
    for cutoff in fold_cutoffs:
        window_end = cutoff + horizon_delta(horizon, grain)
        inner = wide[wide["cutoff"] + horizon_delta(horizon, grain) <= cutoff]
        inner_cutoffs = sorted(inner["cutoff"].unique())
        if len(inner_cutoffs) < MIN_INNER_CUTOFFS:
            raise ValueError(
                f"Fold {cutoff.date()} has only {len(inner_cutoffs)} inner cutoffs for weight fitting "
                f"(need >= {MIN_INNER_CUTOFFS}). Reduce --n-folds or extend history."
            )

        convex_weights = optimize_weights(inner["actual"].to_numpy(), inner[available].to_numpy())
        inner_wapes = {m: wape(inner["actual"].to_numpy(), inner[m].to_numpy()) for m in available}
        inv = np.asarray([1.0 / max(inner_wapes[m], 1e-6) ** 2 for m in available])
        inv_weights = inv / inv.sum()

        fold = wide[wide["cutoff"] == cutoff].copy()
        if fold.empty:
            raise ValueError(f"No grid predictions found for fold cutoff {cutoff.date()}.")
        fold["nested_convex_stack"] = fold[available].to_numpy() @ convex_weights
        fold["nested_inverse_wape_stack"] = fold[available].to_numpy() @ inv_weights
        fold["pool_median"] = fold[available].median(axis=1)

        scored_columns = available + ["nested_convex_stack", "nested_inverse_wape_stack", "pool_median"]
        scores = fold_scores(fold, scored_columns)
        for model_name, vals in scores.items():
            metric_rows.append(
                {
                    "fold_cutoff": cutoff,
                    "window_start": cutoff + pd.Timedelta(days=1),
                    "window_end": window_end,
                    "model_name": model_name,
                    "wape": vals["wape"],
                    "bias_pct": vals["bias_pct"],
                    "wape_vs_benchmark": vals["wape"] - scores[benchmark]["wape"],
                }
            )
        fold_metadata[str(cutoff.date())] = {
            "inner_cutoffs": [str(pd.Timestamp(c).date()) for c in inner_cutoffs],
            "inner_wapes": inner_wapes,
            "convex_stack_weights": dict(zip(available, convex_weights.tolist())),
            "inverse_wape_weights": dict(zip(available, inv_weights.tolist())),
        }
    return pd.DataFrame(metric_rows), fold_metadata


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in fold_metrics.groupby("model_name"):
        diffs = group["wape_vs_benchmark"]
        rows.append(
            {
                "model_name": model_name,
                "folds": len(group),
                "mean_wape": group["wape"].mean(),
                "median_wape": group["wape"].median(),
                "mean_bias_pct": group["bias_pct"].mean(),
                "wins_vs_benchmark": int((diffs < 0).sum()),
                "mean_wape_diff_vs_benchmark": diffs.mean(),
                "max_wape_diff_vs_benchmark": diffs.max(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_wape").reset_index(drop=True)


def weekly_segment(ds: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ds)
    return np.where(dates.dt.month.isin([11, 12]), "holiday_weeks", "rest_weeks")


def segment_metrics(grid_preds: pd.DataFrame, fold_cutoffs: list[pd.Timestamp], model_names: list[str], benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_preds = grid_preds[grid_preds["cutoff"].isin(fold_cutoffs)].copy()
    fold_preds["segment"] = weekly_segment(fold_preds["ds"])
    metric_rows = []
    for (cutoff, segment), group in fold_preds.groupby(["cutoff", "segment"]):
        segment_scores = {}
        for model_name, model_group in group.groupby("model_name"):
            if model_name not in model_names:
                continue
            actual = model_group["actual"].to_numpy()
            forecast = model_group["forecast"].to_numpy()
            denom = actual.sum()
            segment_scores[model_name] = {
                "wape": wape(actual, forecast),
                "bias_pct": float((forecast - actual).sum() / denom) if abs(denom) > 1e-9 else np.nan,
                "rows": int(len(model_group)),
            }
        if benchmark not in segment_scores:
            continue
        for model_name, vals in segment_scores.items():
            metric_rows.append(
                {
                    "fold_cutoff": cutoff,
                    "segment": segment,
                    "model_name": model_name,
                    "rows": vals["rows"],
                    "wape": vals["wape"],
                    "bias_pct": vals["bias_pct"],
                    "wape_vs_benchmark": vals["wape"] - segment_scores[benchmark]["wape"],
                }
            )
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        return metrics, metrics

    summary_rows = []
    for (segment, model_name), group in metrics.groupby(["segment", "model_name"]):
        diffs = group["wape_vs_benchmark"]
        summary_rows.append(
            {
                "segment": segment,
                "model_name": model_name,
                "fold_segments": len(group),
                "mean_wape": group["wape"].mean(),
                "median_wape": group["wape"].median(),
                "mean_bias_pct": group["bias_pct"].mean(),
                "wins_vs_benchmark": int((diffs < 0).sum()),
                "mean_wape_diff_vs_benchmark": diffs.mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["segment", "mean_wape"]).reset_index(drop=True)
    return metrics, summary


def fold_overlap_metadata(fold_cutoffs: list[pd.Timestamp], horizon: int, grid_step: int, grain: str) -> dict:
    unit_days = 7 if normalize_grain(grain) == WEEKLY_GRAIN else 1
    overlap = max(0, horizon - grid_step)
    if fold_cutoffs:
        span_units = int((max(fold_cutoffs) - min(fold_cutoffs)).days / unit_days) + horizon
        effective_windows = span_units / horizon
    else:
        span_units = 0
        effective_windows = 0.0
    return {
        "holdout_window_units": horizon,
        "grid_step_units": grid_step,
        "adjacent_overlap_units": overlap,
        "adjacent_overlap_pct": overlap / horizon if horizon else 0.0,
        "fold_span_units": span_units,
        "effective_independent_windows_approx": effective_windows,
    }


def write_outputs(grid_preds: pd.DataFrame, fold_metrics: pd.DataFrame, summary: pd.DataFrame, fold_metadata: dict, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_preds.to_csv(output_dir / "multifold_grid_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "multifold_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "multifold_summary.csv", index=False)
    (output_dir / "multifold_fold_weights.json").write_text(json.dumps(fold_metadata, indent=2), encoding="utf-8")

    n_folds = fold_metrics["fold_cutoff"].nunique()
    segment_summary = pd.DataFrame()
    if args.grain == WEEKLY_GRAIN:
        segment_detail, segment_summary = segment_metrics(
            grid_preds,
            [pd.Timestamp(c) for c in sorted(fold_metrics["fold_cutoff"].unique())],
            sorted(fold_metrics["model_name"].unique().tolist()),
            benchmark_for_grain(args.grain),
        )
        segment_detail.to_csv(output_dir / "multifold_segment_metrics.csv", index=False)
        segment_summary.to_csv(output_dir / "multifold_segment_summary.csv", index=False)
    overlap = fold_overlap_metadata(
        [pd.Timestamp(c) for c in sorted(fold_metrics["fold_cutoff"].unique())],
        args.horizon,
        args.grid_step,
        args.grain,
    )
    lines = [
        f"# Multi-Fold Nested Final Validation (horizon {args.horizon})",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        f"Grain: {args.grain}. Frequency: {args.weekly_freq if args.grain == WEEKLY_GRAIN else 'D'}.",
        f"Folds: {n_folds} rolling final-holdout windows, cutoffs stepped {args.grid_step} {'weeks' if args.grain == WEEKLY_GRAIN else 'days'}.",
        f"Adjacent holdout overlap: {overlap['adjacent_overlap_units']} {'weeks' if args.grain == WEEKLY_GRAIN else 'days'} "
        f"({overlap['adjacent_overlap_pct']:.0%}); effective independent windows ~= {overlap['effective_independent_windows_approx']:.2f}.",
        "Stack weights are refit per fold using only backtest windows ending on or before the fold cutoff.",
        "",
        "## Summary Across Folds",
        "",
        "| Model | Mean WAPE | Median WAPE | Mean bias pct | Wins vs seasonal naive | Mean WAPE diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.mean_wape:.4f} | {row.median_wape:.4f} | {row.mean_bias_pct:.4f} "
            f"| {row.wins_vs_benchmark}/{row.folds} | {row.mean_wape_diff_vs_benchmark:+.4f} |"
        )
    lines.extend(["", "## Per-Fold WAPE", "", "| Fold cutoff | " + " | ".join(summary["model_name"]) + " |", "|---|" + "---:|" * len(summary)])
    fold_wide = fold_metrics.pivot_table(index="fold_cutoff", columns="model_name", values="wape")
    for cutoff, row in fold_wide.iterrows():
        lines.append(f"| {pd.Timestamp(cutoff).date()} | " + " | ".join(f"{row[m]:.4f}" for m in summary["model_name"]) + " |")
    if not segment_summary.empty:
        lines.extend(
            [
                "",
                "## Holiday Segment Summary",
                "",
                "Segments are based on W-WED week-ending month: November/December are `holiday_weeks`; all others are `rest_weeks`.",
                "",
                "| Segment | Model | Mean WAPE | Median WAPE | Mean bias pct | Wins vs seasonal naive | Mean WAPE diff |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in segment_summary.itertuples(index=False):
            lines.append(
                f"| {row.segment} | {row.model_name} | {row.mean_wape:.4f} | {row.median_wape:.4f} | "
                f"{row.mean_bias_pct:.4f} | {row.wins_vs_benchmark}/{row.fold_segments} | {row.mean_wape_diff_vs_benchmark:+.4f} |"
            )
    if DEGRADED_MODELS:
        lines.extend(["", "## WARNING: Degraded Models", ""])
        for name, reason in DEGRADED_MODELS.items():
            lines.append(f"- `{name}`: {reason}")
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon": args.horizon,
        "grain": args.grain,
        "freq": args.weekly_freq if args.grain == WEEKLY_GRAIN else "D",
        "n_folds": n_folds,
        "grid_step": args.grid_step,
        "grid_step_unit": "weeks" if args.grain == WEEKLY_GRAIN else "days",
        "opening_info_file": args.opening_info,
        "pool": sorted(grid_preds["model_name"].unique().tolist()),
        "grid_cutoffs": [str(pd.Timestamp(c).date()) for c in sorted(grid_preds["cutoff"].unique())],
        "fold_cutoffs": [str(pd.Timestamp(c).date()) for c in sorted(fold_metrics["fold_cutoff"].unique())],
        "degraded_models": DEGRADED_MODELS,
        "horizons": [args.horizon],
        "models": sorted(fold_metrics["model_name"].unique().tolist()),
        "fold_overlap": overlap,
        "segment_metrics": "multifold_segment_summary.csv" if args.grain == WEEKLY_GRAIN else None,
        "segment_definition": "weekly W-WED rows with week-ending month in Nov/Dec are holiday_weeks; all others rest_weeks"
        if args.grain == WEEKLY_GRAIN
        else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_multifold_final_validation.py", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--horizon", type=int, default=91)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--grid-step", type=int, default=None)
    parser.add_argument("--pool", default=None, help="Comma-separated candidate models; must include the seasonal naive benchmark.")
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--chronos-context", type=int, default=2048)
    parser.add_argument("--max-grid-cutoffs", type=int, default=None, help="Limit grid to the most recent N cutoffs (smoke tests).")
    parser.add_argument("--opening-info", default=None, help="Optional unit-opening CSV; required for opening-aware pool models like sklearn_extra_trees.")
    parser.add_argument("--grain", choices=[DAILY_GRAIN, WEEKLY_GRAIN], default=DAILY_GRAIN)
    parser.add_argument("--weekly-freq", default=DEFAULT_WEEKLY_FREQ, help="Pandas weekly frequency used when --grain weekly.")
    parser.add_argument("--output-dir", default="outputs_multifold_91d_validation")
    args = parser.parse_args()

    set_seeds()
    if args.grid_step is None:
        args.grid_step = default_grid_step_for_grain(args.grain)
    if args.grid_step <= 0:
        raise ValueError("--grid-step must be positive.")
    default_pool = WEEKLY_DEFAULT_POOL if args.grain == WEEKLY_GRAIN else DEFAULT_POOL
    pool_text = args.pool if args.pool is not None else ",".join(default_pool)
    pool = [m.strip() for m in pool_text.split(",") if m.strip()]
    benchmark = benchmark_for_grain(args.grain)
    if benchmark not in pool:
        raise ValueError(f"Pool must include the benchmark {benchmark}.")
    openings = load_openings(Path(args.opening_info)) if args.opening_info else None
    df = load_sales(Path(args.sales_csv), grain=args.grain, weekly_freq=args.weekly_freq)
    grid = build_grid(df, args.horizon, args.grid_step, args.grain)
    if args.max_grid_cutoffs:
        grid = grid[-args.max_grid_cutoffs :]
    fold_cutoffs = grid[-args.n_folds :]
    print(f"Grid: {len(grid)} cutoffs from {grid[0].date()} to {grid[-1].date()}; folds: {[str(c.date()) for c in fold_cutoffs]}", flush=True)

    daily_history = (
        load_sales(Path(args.sales_csv))
        if args.grain == WEEKLY_GRAIN and any(name.startswith("daily_reagg_") for name in pool)
        else None
    )
    predictors = make_predictors(
        pool,
        args.horizon,
        args.timesfm_context,
        args.chronos_context,
        openings,
        args.grain,
        args.weekly_freq,
        daily_history,
    )
    grid_preds = run_grid(df, grid, args.horizon, predictors, args.grain)
    # Persist the expensive grid immediately so a fold-stage failure cannot lose it.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_preds.to_csv(output_dir / "multifold_grid_predictions.csv", index=False)
    fold_metrics, fold_metadata = run_folds(grid_preds, pool, fold_cutoffs, args.horizon, args.grain, benchmark)
    summary = summarize(fold_metrics)
    write_outputs(grid_preds, fold_metrics, summary, fold_metadata, Path(args.output_dir), args)
    print(summary.to_string(index=False))

    if DEGRADED_MODELS:
        print("\nWARNING: degraded models present; their grid forecasts are baseline fallbacks:", file=sys.stderr)
        for name, reason in DEGRADED_MODELS.items():
            print(f"  - {name}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
