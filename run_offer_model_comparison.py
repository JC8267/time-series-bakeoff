"""Compare model-card candidates on the offer forecast horizon."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_multifold_final_validation import make_predictors
from run_offer_production_forecast import load_offer_series


BASELINE = "seasonal_naive_lag_364"
TIMESFM = "timesfm_2_5"
BLEND = "blend_snl75_timesfm25"
POOL = [
    BASELINE,
    "seasonal_naive_scaled_lag_364_w56_damped",
    "sklearn_extra_trees",
    "sklearn_hist_gradient_boosting",
    "statsforecast_mstl_auto_ets",
    TIMESFM,
]


def score(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    error = forecast - actual
    return {
        "wape": float(np.abs(error).sum() / np.abs(actual).sum()),
        "mae": float(np.abs(error).mean()),
        "bias_pct": float(error.sum() / actual.sum()),
    }


def add_blend(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["offer", "cutoff", "ds", "actual", "sample"]
    wide = predictions[predictions["model_name"].isin([BASELINE, TIMESFM])].pivot(index=keys, columns="model_name", values="forecast")
    blend = wide.reset_index()
    blend["model_name"] = BLEND
    blend["forecast"] = 0.75 * blend[BASELINE] + 0.25 * blend[TIMESFM]
    return pd.concat([predictions, blend[keys + ["model_name", "forecast"]]], ignore_index=True)


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["offer", "cutoff", "sample", "model_name"]):
        rows.append(dict(zip(["offer", "cutoff", "sample", "model_name"], keys)) | score(group["actual"].to_numpy(), group["forecast"].to_numpy()))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="by_offer/raw_data.csv")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--forecast-end", default="2026-08-31")
    parser.add_argument("--output-dir", default="by_offer/outputs_model_comparison_123d")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    forecast_end = pd.Timestamp(args.forecast_end)
    horizon = (forecast_end - cutoff).days
    wide, actuals, _, _ = load_offer_series(Path(args.input), cutoff)
    actuals = actuals.reindex(columns=wide.columns)
    actuals["Total"] = actuals.sum(axis=1, min_count=3)
    series = {offer: wide[offer] for offer in wide.columns}
    series["Total"] = wide.sum(axis=1)

    predictors = make_predictors(POOL, horizon, 1024, 2048)
    last_fold_cutoff = cutoff - pd.Timedelta(days=horizon)
    fold_cutoffs = [last_fold_cutoff - pd.Timedelta(days=28 * k) for k in reversed(range(5))]
    records = []
    for offer, history in series.items():
        frame = pd.DataFrame({"unique_id": offer, "ds": history.index, "y": history.to_numpy()})
        for fold_cutoff in fold_cutoffs:
            train = frame[frame["ds"] <= fold_cutoff]
            test = frame[(frame["ds"] > fold_cutoff) & (frame["ds"] <= fold_cutoff + pd.Timedelta(days=horizon))]
            for model_name, predict in predictors.items():
                started = time.perf_counter()
                forecast = np.asarray(predict(train, horizon), dtype=float)
                print(f"{offer} | {fold_cutoff.date()} | {model_name} | {time.perf_counter() - started:.1f}s", flush=True)
                records.extend(
                    {
                        "offer": offer,
                        "cutoff": fold_cutoff,
                        "ds": ds,
                        "actual": actual,
                        "sample": "rolling_backtest",
                        "model_name": model_name,
                        "forecast": max(0.0, float(prediction)),
                    }
                    for ds, actual, prediction in zip(test["ds"], test["y"], forecast)
                )

        train = frame[frame["ds"] <= cutoff]
        forecast_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
        realized = actuals[offer].loc[cutoff + pd.Timedelta(days=1) : forecast_end].dropna()
        for model_name, predict in predictors.items():
            forecast = pd.Series(np.asarray(predict(train, horizon), dtype=float), index=forecast_dates).loc[realized.index]
            records.extend(
                {
                    "offer": offer,
                    "cutoff": cutoff,
                    "ds": ds,
                    "actual": actual,
                    "sample": "realized_holdout",
                    "model_name": model_name,
                    "forecast": max(0.0, float(prediction)),
                }
                for ds, actual, prediction in zip(realized.index, realized, forecast)
            )

    predictions = add_blend(pd.DataFrame(records))
    metrics = fold_metrics(predictions)
    rolling = metrics[metrics["sample"].eq("rolling_backtest")]
    baseline = rolling[rolling["model_name"].eq(BASELINE)][["offer", "cutoff", "wape"]].rename(columns={"wape": "baseline_wape"})
    rolling = rolling.merge(baseline, on=["offer", "cutoff"])
    summary = (
        rolling.groupby(["offer", "model_name"])
        .agg(
            mean_wape=("wape", "mean"),
            median_wape=("wape", "median"),
            mean_mae=("mae", "mean"),
            mean_bias_pct=("bias_pct", "mean"),
            wins_vs_baseline=("wape", lambda x: int((x < rolling.loc[x.index, "baseline_wape"]).sum())),
            folds=("wape", "size"),
        )
        .reset_index()
    )
    realized_summary = metrics[metrics["sample"].eq("realized_holdout")].drop(columns=["sample"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rolling.drop(columns="baseline_wape").to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    realized_summary.to_csv(output_dir / "realized_holdout_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": str(cutoff.date()),
        "horizon_days": horizon,
        "fold_cutoffs": [str(date.date()) for date in fold_cutoffs],
        "models": POOL + [BLEND],
        "realized_holdout_end": str(actuals["Total"].loc[:forecast_end].dropna().index.max().date()),
        "outputs": ["fold_metrics.csv", "comparison_summary.csv", "realized_holdout_summary.csv"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[summary["offer"].eq("Total")].sort_values("mean_wape").to_string(index=False))


if __name__ == "__main__":
    main()
