"""Multi-fold nested final validation for the production stack decision.

Evaluates K rolling final-holdout folds at a given horizon. For each fold,
convex-stack and inverse-WAPE weights are refit using only backtest windows
that end on or before that fold's cutoff (nested validation), then scored on
the fold alongside the standalone pool models. This answers whether the
learned stack beats `seasonal_naive_lag_364` consistently, not just on the
single latest holdout window that phase 5 validates.

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
    MIN_TRAIN_DAYS,
    ets_forecast,
    load_sales,
    seasonal_naive,
    set_seeds,
    statsforecast_forecast,
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
BENCHMARK = "seasonal_naive_lag_364"
MIN_INNER_CUTOFFS = 3


def build_grid(df: pd.DataFrame, horizon: int, step: int) -> list[pd.Timestamp]:
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


def make_predictors(pool: list[str], horizon: int, timesfm_context: int, chronos_context: int) -> dict:
    predictors = {}
    for name in pool:
        if name == "seasonal_naive_lag_364":
            predictors[name] = lambda train, h: seasonal_naive(train, h, 364)
        elif name == "statsmodels_ets_weekly":
            predictors[name] = lambda train, h: ets_forecast(train, h)
        elif name.startswith("statsforecast_"):
            predictors[name] = lambda train, h, n=name: statsforecast_forecast(train, h, n)
        elif name == "timesfm_2_5":
            from run_prophet_timesfm_benchmark import forecast_timesfm, load_timesfm

            model = load_timesfm(timesfm_context, horizon)
            predictors[name] = lambda train, h, m=model: forecast_timesfm(m, train, h, timesfm_context)["forecast"].to_numpy()
        elif name == "chronos_bolt_small":
            from run_prophet_timesfm_benchmark import forecast_chronos_bolt, load_chronos_bolt

            model = load_chronos_bolt()
            predictors[name] = lambda train, h, m=model: forecast_chronos_bolt(m, train, h, chronos_context)["forecast"].to_numpy()
        else:
            raise ValueError(f"Unsupported pool model: {name}")
    return predictors


def run_grid(df: pd.DataFrame, cutoffs: list[pd.Timestamp], horizon: int, predictors: dict) -> pd.DataFrame:
    rows = []
    for i, cutoff in enumerate(cutoffs, start=1):
        train = df[df["ds"] <= cutoff].copy()
        test = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=horizon))].copy()
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
                        "window_end": cutoff + pd.Timedelta(days=horizon),
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


def run_folds(grid_preds: pd.DataFrame, pool: list[str], fold_cutoffs: list[pd.Timestamp], horizon: int) -> tuple[pd.DataFrame, dict]:
    wide = make_wide(grid_preds, pool)
    available = [m for m in pool if m in wide.columns]
    if BENCHMARK not in available:
        raise ValueError(f"Benchmark {BENCHMARK} missing from grid predictions.")

    metric_rows = []
    fold_metadata = {}
    for cutoff in fold_cutoffs:
        window_end = cutoff + pd.Timedelta(days=horizon)
        inner = wide[wide["cutoff"] + pd.Timedelta(days=horizon) <= cutoff]
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
                    "wape_vs_benchmark": vals["wape"] - scores[BENCHMARK]["wape"],
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


def write_outputs(grid_preds: pd.DataFrame, fold_metrics: pd.DataFrame, summary: pd.DataFrame, fold_metadata: dict, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_preds.to_csv(output_dir / "multifold_grid_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "multifold_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "multifold_summary.csv", index=False)
    (output_dir / "multifold_fold_weights.json").write_text(json.dumps(fold_metadata, indent=2), encoding="utf-8")

    n_folds = fold_metrics["fold_cutoff"].nunique()
    lines = [
        f"# Multi-Fold Nested Final Validation (horizon {args.horizon})",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        f"Folds: {n_folds} rolling final-holdout windows, cutoffs stepped {args.grid_step} days.",
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
    if DEGRADED_MODELS:
        lines.extend(["", "## WARNING: Degraded Models", ""])
        for name, reason in DEGRADED_MODELS.items():
            lines.append(f"- `{name}`: {reason}")
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon": args.horizon,
        "n_folds": n_folds,
        "grid_step_days": args.grid_step,
        "pool": sorted(grid_preds["model_name"].unique().tolist()),
        "grid_cutoffs": [str(pd.Timestamp(c).date()) for c in sorted(grid_preds["cutoff"].unique())],
        "fold_cutoffs": [str(pd.Timestamp(c).date()) for c in sorted(fold_metrics["fold_cutoff"].unique())],
        "degraded_models": DEGRADED_MODELS,
        "horizons": [args.horizon],
        "models": sorted(fold_metrics["model_name"].unique().tolist()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_multifold_final_validation.py", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--horizon", type=int, default=91)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--grid-step", type=int, default=28)
    parser.add_argument("--pool", default=",".join(DEFAULT_POOL), help="Comma-separated candidate models; must include the seasonal naive benchmark.")
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--chronos-context", type=int, default=2048)
    parser.add_argument("--max-grid-cutoffs", type=int, default=None, help="Limit grid to the most recent N cutoffs (smoke tests).")
    parser.add_argument("--output-dir", default="outputs_multifold_91d_validation")
    args = parser.parse_args()

    set_seeds()
    pool = [m.strip() for m in args.pool.split(",") if m.strip()]
    if BENCHMARK not in pool:
        raise ValueError(f"Pool must include the benchmark {BENCHMARK}.")
    df = load_sales(Path(args.sales_csv))
    grid = build_grid(df, args.horizon, args.grid_step)
    if args.max_grid_cutoffs:
        grid = grid[-args.max_grid_cutoffs :]
    fold_cutoffs = grid[-args.n_folds :]
    print(f"Grid: {len(grid)} cutoffs from {grid[0].date()} to {grid[-1].date()}; folds: {[str(c.date()) for c in fold_cutoffs]}", flush=True)

    predictors = make_predictors(pool, args.horizon, args.timesfm_context, args.chronos_context)
    grid_preds = run_grid(df, grid, args.horizon, predictors)
    fold_metrics, fold_metadata = run_folds(grid_preds, pool, fold_cutoffs, args.horizon)
    summary = summarize(fold_metrics)
    write_outputs(grid_preds, fold_metrics, summary, fold_metadata, Path(args.output_dir), args)
    print(summary.to_string(index=False))

    if DEGRADED_MODELS:
        print("\nWARNING: degraded models present; their grid forecasts are baseline fallbacks:", file=sys.stderr)
        for name, reason in DEGRADED_MODELS.items():
            print(f"  - {name}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
