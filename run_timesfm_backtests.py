"""Generate rolling foundation-model backtest predictions for ensemble phases."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import build_cutoffs, load_sales, score_group, set_seeds
from run_prophet_timesfm_benchmark import forecast_chronos_bolt, forecast_timesfm, load_chronos_bolt, load_timesfm


def append_rows(rows: list[dict], test: pd.DataFrame, fcst: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, model_name: str, runtime: float, hyperparams: str, library_version: str) -> None:
    for i, (actual_row, fcst_row) in enumerate(zip(test.itertuples(index=False), fcst.itertuples(index=False)), start=1):
        rows.append(
            {
                "unique_id": actual_row.unique_id,
                "horizon": horizon,
                "cutoff": cutoff,
                "ds": actual_row.ds,
                "horizon_step": i,
                "actual": float(actual_row.y),
                "model_name": model_name,
                "forecast": max(0.0, float(fcst_row.forecast)),
                "p025": max(0.0, float(fcst_row.p025)),
                "p10": max(0.0, float(fcst_row.p10)),
                "p50": max(0.0, float(fcst_row.p50)),
                "p90": max(0.0, float(fcst_row.p90)),
                "p975": max(0.0, float(fcst_row.p975)),
                "runtime_seconds": runtime,
                "refit_cadence": "zero_shot_each_cutoff",
                "hyperparams_json": hyperparams,
                "library_version": library_version,
            }
        )


def run_backtests(
    sales_csv: Path,
    horizons: tuple[int, ...],
    max_cutoffs: int,
    max_context: int,
    chronos_context: int,
    models: list[str],
    chronos_model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    df = load_sales(sales_csv)
    try:
        tfm = load_timesfm(max_context, max(horizons)) if "timesfm_2_5" in models else None
    except Exception as exc:
        print(f"Skipping TimesFM: {exc}")
        tfm = None
    try:
        chronos = load_chronos_bolt(chronos_model_id) if "chronos_bolt_small" in models else None
    except Exception as exc:
        print(f"Skipping Chronos-Bolt: {exc}")
        chronos = None
    rows = []
    for horizon in horizons:
        for cutoff in build_cutoffs(df, horizon, max_cutoffs):
            train = df[df["ds"] <= cutoff].copy()
            test = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=horizon))].copy()
            if tfm is not None:
                started = time.perf_counter()
                fcst = forecast_timesfm(tfm, train, horizon, max_context)
                runtime = time.perf_counter() - started
                append_rows(rows, test, fcst, cutoff, horizon, "timesfm_2_5", runtime, f'{{"max_context": {max_context}}}', "timesfm_2.5")
            if chronos is not None:
                started = time.perf_counter()
                fcst = forecast_chronos_bolt(chronos, train, horizon, chronos_context)
                runtime = time.perf_counter() - started
                append_rows(
                    rows,
                    test,
                    fcst,
                    cutoff,
                    horizon,
                    "chronos_bolt_small",
                    runtime,
                    f'{{"max_context": {chronos_context}, "model_id": "{chronos_model_id}"}}',
                    "chronos_bolt",
                )
    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds, pd.DataFrame()
    metrics = (
        preds.groupby(["horizon", "model_name"], dropna=False)
        .apply(score_group, include_groups=False)
        .reset_index()
        .sort_values(["horizon", "wape"])
    )
    return preds, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--max-cutoffs", type=int, default=6)
    parser.add_argument("--tfm-max-context", type=int, default=1024)
    parser.add_argument("--chronos-max-context", type=int, default=2048)
    parser.add_argument("--chronos-model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--models", default="timesfm_2_5,chronos_bolt_small")
    parser.add_argument("--output-dir", default="outputs_timesfm_backtests")
    args = parser.parse_args()
    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    preds, metrics = run_backtests(Path(args.sales_csv), horizons, args.max_cutoffs, args.tfm_max_context, args.chronos_max_context, models, args.chronos_model_id)
    preds.to_csv(out / "foundation_backtest_predictions.csv", index=False)
    metrics.to_csv(out / "foundation_backtest_metrics.csv", index=False)
    preds.to_csv(out / "timesfm_backtest_predictions.csv", index=False)
    metrics.to_csv(out / "timesfm_backtest_metrics.csv", index=False)
    try:
        preds.to_parquet(out / "timesfm_backtest_predictions.parquet", index=False)
    except Exception:
        pass
    print(metrics[["horizon", "model_name", "wape", "bias_pct", "mae", "rmse", "coverage_80", "coverage_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
