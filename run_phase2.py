"""Phase 2 advanced forecasting challengers.

Runs heavier models against the same rolling-origin structure as phase 1:
AutoGluon-TimeSeries plus compact NeuralForecast NHITS/NBEATS configurations.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import (
    SEED,
    add_conformal_intervals,
    add_opening_features,
    build_cutoffs,
    score_group,
    set_seeds,
    load_sales,
    load_openings,
)


def autogluon_predict(train: pd.DataFrame, horizon: int, path: Path, time_limit: int, openings: pd.DataFrame | None = None) -> pd.DataFrame:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    train_with_features = add_opening_features(train, openings)
    known_covariates = [
        c for c in train_with_features.columns
        if c not in {"unique_id", "ds", "y"} and (
            c.startswith("num_")
            or c.endswith("_flag")
            or c.startswith("days_since")
            or c.startswith("grand_opening")
            or c.startswith("ramp_")
            or c.startswith("mature_")
        )
    ]
    ag_train = train_with_features.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
    ts_train = TimeSeriesDataFrame.from_data_frame(ag_train, id_column="item_id", timestamp_column="timestamp")
    future = pd.DataFrame(
        {
            "unique_id": "total_sales",
            "ds": pd.date_range(train["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"),
        }
    )
    future = add_opening_features(future, openings)
    known_covariates_df = TimeSeriesDataFrame.from_data_frame(
        future.rename(columns={"unique_id": "item_id", "ds": "timestamp"})[["item_id", "timestamp"] + known_covariates],
        id_column="item_id",
        timestamp_column="timestamp",
    )
    predictor = TimeSeriesPredictor(
        target="target",
        known_covariates_names=known_covariates or None,
        prediction_length=horizon,
        freq="D",
        eval_metric="WAPE",
        path=path,
        quantile_levels=[0.025, 0.10, 0.50, 0.90, 0.975],
        verbosity=1,
    )
    predictor.fit(
        ts_train,
        presets="medium_quality",
        time_limit=time_limit,
        random_seed=SEED,
        enable_ensemble=True,
        verbosity=1,
    )
    fcst = predictor.predict(ts_train, known_covariates=known_covariates_df if known_covariates else None).reset_index()
    fcst = fcst.rename(columns={"item_id": "unique_id", "timestamp": "ds"})
    fcst["forecast"] = fcst["mean"] if "mean" in fcst else fcst["0.5"]
    for src, dst in [("0.025", "p025"), ("0.1", "p10"), ("0.5", "p50"), ("0.9", "p90"), ("0.975", "p975")]:
        fcst[dst] = fcst[src] if src in fcst else np.nan
    return fcst[["unique_id", "ds", "forecast", "p025", "p10", "p50", "p90", "p975"]]


def neural_predict(train: pd.DataFrame, horizon: int, model_name: str, max_steps: int) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS

    input_size = min(max(2 * horizon, 56), min(364, len(train) - horizon - 1))
    common = {
        "h": horizon,
        "input_size": input_size,
        "max_steps": max_steps,
        "random_seed": SEED,
        "scaler_type": "robust",
        "batch_size": 16,
        "windows_batch_size": 128,
        "enable_progress_bar": False,
        "logger": False,
        "accelerator": "cpu",
    }
    if model_name == "neuralforecast_nhits":
        model = NHITS(alias=model_name, **common)
    elif model_name == "neuralforecast_nbeats":
        model = NBEATS(alias=model_name, **common)
    else:
        raise ValueError(model_name)

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train[["unique_id", "ds", "y"]])
    fcst = nf.predict().reset_index()
    fcst = fcst.rename(columns={model_name: "forecast"})
    fcst[["p025", "p10", "p50", "p90", "p975"]] = np.nan
    fcst["p50"] = fcst["forecast"]
    return fcst[["unique_id", "ds", "forecast", "p025", "p10", "p50", "p90", "p975"]]


def evaluate_phase2(df: pd.DataFrame, horizons: tuple[int, ...], max_cutoffs: int, autogluon_time_limit: int, neural_steps: int, output_dir: Path, openings: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    ag_model_dir = output_dir / "autogluon_models"
    for horizon in horizons:
        cutoffs = build_cutoffs(df, horizon, max_cutoffs)
        for cutoff in cutoffs:
            train = df[df["ds"] <= cutoff].copy()
            test = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=horizon))].copy()
            model_jobs = [
                ("autogluon_medium_quality", lambda: autogluon_predict(train, horizon, ag_model_dir / f"h{horizon}_{cutoff:%Y%m%d}", autogluon_time_limit, openings), "refit_every_cutoff_time_limited"),
                ("neuralforecast_nhits", lambda: neural_predict(train, horizon, "neuralforecast_nhits", neural_steps), "refit_every_cutoff_cpu"),
                ("neuralforecast_nbeats", lambda: neural_predict(train, horizon, "neuralforecast_nbeats", neural_steps), "refit_every_cutoff_cpu"),
            ]
            for model_name, predict_fn, cadence in model_jobs:
                started = time.perf_counter()
                try:
                    fcst = predict_fn()
                    status = "ok"
                    error = ""
                except Exception as exc:
                    fcst = pd.DataFrame()
                    status = "failed"
                    error = repr(exc)
                runtime = time.perf_counter() - started
                if status != "ok":
                    rows.append(
                        {
                            "unique_id": "total_sales",
                            "horizon": horizon,
                            "cutoff": cutoff,
                            "ds": pd.NaT,
                            "horizon_step": np.nan,
                            "actual": np.nan,
                            "model_name": model_name,
                            "forecast": np.nan,
                            "p025": np.nan,
                            "p10": np.nan,
                            "p50": np.nan,
                            "p90": np.nan,
                            "p975": np.nan,
                            "runtime_seconds": runtime,
                            "refit_cadence": cadence,
                            "hyperparams_json": json.dumps({"status": status, "error": error}),
                            "library_version": "phase2",
                        }
                    )
                    continue
                merged = test[["unique_id", "ds", "y"]].merge(fcst, on=["unique_id", "ds"], how="left")
                for i, row in enumerate(merged.itertuples(index=False), start=1):
                    rows.append(
                        {
                            "unique_id": row.unique_id,
                            "horizon": horizon,
                            "cutoff": cutoff,
                            "ds": row.ds,
                            "horizon_step": i,
                            "actual": float(row.y),
                            "model_name": model_name,
                            "forecast": max(0.0, float(row.forecast)),
                            "p025": getattr(row, "p025"),
                            "p10": getattr(row, "p10"),
                            "p50": getattr(row, "p50"),
                            "p90": getattr(row, "p90"),
                            "p975": getattr(row, "p975"),
                            "runtime_seconds": runtime,
                            "refit_cadence": cadence,
                            "hyperparams_json": json.dumps({"status": status}),
                            "library_version": "phase2",
                        }
                    )
    preds = pd.DataFrame(rows)
    good = preds[preds["forecast"].notna()].copy()
    failed = preds[preds["forecast"].isna()].copy()
    if not good.empty:
        good = add_conformal_intervals(good)
    return pd.concat([good, failed], ignore_index=True)


def write_outputs(preds: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_dir / "phase2_backtest_predictions.csv", index=False)
    try:
        preds.to_parquet(output_dir / "phase2_backtest_predictions.parquet", index=False)
    except Exception:
        pass
    good = preds[preds["forecast"].notna()].copy()
    if good.empty:
        metrics = pd.DataFrame()
    else:
        metrics = good.groupby(["horizon", "model_name"], dropna=False).apply(score_group, include_groups=False).reset_index()
        metrics = metrics.sort_values(["horizon", "wape"])
    metrics.to_csv(output_dir / "phase2_metrics_by_horizon.csv", index=False)
    failures = preds[preds["forecast"].isna()][["horizon", "cutoff", "model_name", "runtime_seconds", "hyperparams_json"]]
    failures.to_csv(output_dir / "phase2_failures.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(preds)),
        "successful_rows": int(len(good)),
        "failed_rows": int(len(failures)),
        "models": sorted(preds["model_name"].dropna().unique().tolist()),
    }
    (output_dir / "phase2_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--output-dir", default="outputs_phase2")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--max-cutoffs", type=int, default=2)
    parser.add_argument("--autogluon-time-limit", type=int, default=90)
    parser.add_argument("--neural-steps", type=int, default=100)
    parser.add_argument("--opening-info", default=None)
    args = parser.parse_args()

    set_seeds()
    output_dir = Path(args.output_dir)
    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    df = load_sales(Path(args.sales_csv))
    openings = load_openings(Path(args.opening_info)) if args.opening_info else None
    preds = evaluate_phase2(df, horizons, args.max_cutoffs, args.autogluon_time_limit, args.neural_steps, output_dir, openings)
    write_outputs(preds, output_dir)
    metrics_path = output_dir / "phase2_metrics_by_horizon.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty:
            print(metrics.sort_values(["horizon", "wape"]).groupby("horizon").head(5)[["horizon", "model_name", "wape", "bias_pct", "mae", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
