"""Create reconciled offer forecasts with the naive/TimesFM ensemble."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import score_group
from run_multifold_final_validation import make_predictors
from run_offer_production_forecast import load_offer_series


MODEL_NAME = "blend_snl75_timesfm25"
NAIVE_WEIGHT = 0.75
QUANTILES = [0.025, 0.10, 0.90, 0.975]


def ensemble_forecast(history: pd.Series, cutoff: pd.Timestamp, horizon: int, predictors: dict) -> np.ndarray:
    train = history.loc[:cutoff]
    frame = pd.DataFrame({"unique_id": history.name, "ds": train.index, "y": train.to_numpy()})
    naive = np.asarray(predictors["seasonal_naive_lag_364"](frame, horizon), dtype=float)
    timesfm = np.asarray(predictors["timesfm_2_5"](frame, horizon), dtype=float)
    return np.maximum(0.0, NAIVE_WEIGHT * naive + (1.0 - NAIVE_WEIGHT) * timesfm)


def backtest_predictions(history: pd.Series, cutoff: pd.Timestamp, horizon: int, predictors: dict) -> pd.DataFrame:
    frames = []
    for k in range(1, 9):
        bt_cutoff = cutoff - pd.Timedelta(days=28 * k)
        actual = history.loc[bt_cutoff + pd.Timedelta(days=1) : bt_cutoff + pd.Timedelta(days=horizon)]
        forecast = ensemble_forecast(history, bt_cutoff, horizon, predictors)[: len(actual)]
        frames.append(
            pd.DataFrame(
                {"cutoff": bt_cutoff, "ds": actual.index, "actual": actual.to_numpy(), "forecast": forecast}
            )
        )
    return pd.concat(frames, ignore_index=True)


def forecast_rows(offer: str, dates: pd.DatetimeIndex, p50: np.ndarray, backtests: pd.DataFrame, cutoff: pd.Timestamp) -> list[dict]:
    qs = np.quantile(backtests["actual"] - backtests["forecast"], QUANTILES)
    return [
        {
            "offer": offer,
            "ds": ds,
            "model_name": MODEL_NAME,
            "p025": max(0.0, float(prediction + qs[0])),
            "p10": max(0.0, float(prediction + qs[1])),
            "p50": float(prediction),
            "p90": max(0.0, float(prediction + qs[2])),
            "p975": max(0.0, float(prediction + qs[3])),
            "interval_method": f"own_residual_quantiles_ensemble_{len(backtests)}_backtest_days",
            "training_cutoff": cutoff,
        }
        for ds, prediction in zip(dates, p50)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="by_offer/raw_data.csv")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--forecast-end", default="2026-08-31")
    parser.add_argument("--output-dir", default="by_offer/outputs_forecast_naive_timesfm_cutoff_20260430")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    forecast_end = pd.Timestamp(args.forecast_end)
    horizon = (forecast_end - cutoff).days
    wide, actuals, audit, excluded_rows = load_offer_series(Path(args.input), cutoff)
    if len(wide.columns) != 3:
        raise ValueError(f"Expected 3 offers, found {len(wide.columns)}: {list(wide.columns)}")

    predictors = make_predictors(["seasonal_naive_lag_364", "timesfm_2_5"], horizon, 1024, 2048)
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    points = {}
    backtests = {}
    for offer in wide.columns:
        history = wide[offer].rename(offer)
        points[offer] = ensemble_forecast(history, cutoff, horizon, predictors)
        backtests[offer] = backtest_predictions(history, cutoff, horizon, predictors)

    points["Total"] = np.sum(list(points.values()), axis=0)
    total_backtests = pd.concat(backtests.values()).groupby(["cutoff", "ds"], as_index=False)[["actual", "forecast"]].sum()
    backtests["Total"] = total_backtests

    rows = []
    summaries = []
    for offer in [*wide.columns, "Total"]:
        rows.extend(forecast_rows(offer, dates, points[offer], backtests[offer], cutoff))
        summaries.append({"offer": offer, "model_name": MODEL_NAME, **score_group(backtests[offer]).to_dict()})

    forecast = pd.DataFrame(rows)
    actuals = actuals.reindex(columns=wide.columns)
    actuals["Total"] = actuals.sum(axis=1, min_count=3)
    actuals = actuals.rename_axis(index="ds", columns="offer").stack().rename("actual").reset_index()
    forecast = forecast.merge(actuals, on=["offer", "ds"], how="left")
    forecast = forecast[
        ["offer", "ds", "actual", "model_name", "p025", "p10", "p50", "p90", "p975", "interval_method", "training_cutoff"]
    ]

    p50 = forecast.pivot(index="ds", columns="offer", values="p50")
    np.testing.assert_allclose(p50["Total"], p50.drop(columns="Total").sum(axis=1))
    actual_pivot = forecast.pivot(index="ds", columns="offer", values="actual")
    complete = actual_pivot.drop(columns="Total").notna().all(axis=1)
    np.testing.assert_allclose(actual_pivot.loc[complete, "Total"], actual_pivot.loc[complete].drop(columns="Total").sum(axis=1))
    if actual_pivot.loc[~complete, "Total"].notna().any():
        raise AssertionError("Total actual is populated for an incomplete offer date.")
    if not ((forecast["p025"] <= forecast["p10"]) & (forecast["p10"] <= forecast["p50"]) &
            (forecast["p50"] <= forecast["p90"]) & (forecast["p90"] <= forecast["p975"])).all():
        raise AssertionError("Forecast intervals are not ordered.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "offer_forecast.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "backtest_summary.csv", index=False)
    audit.to_csv(output_dir / "missing_dates_audit.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "training_cutoff": str(cutoff.date()),
        "forecast_start": str(dates.min().date()),
        "forecast_end": str(dates.max().date()),
        "horizon_days": horizon,
        "offers": list(wide.columns),
        "model": MODEL_NAME,
        "naive_weight": NAIVE_WEIGHT,
        "timesfm_weight": 1.0 - NAIVE_WEIGHT,
        "intervals": "global residual quantiles from 8 rolling ensemble backtest windows",
        "total": "sum of the three offer forecasts; intervals calibrated on summed offer residuals",
        "missing_dates_filled": len(audit),
        "actual_rows": {offer: int(count) for offer, count in forecast.groupby("offer")["actual"].count().items()},
        "post_cutoff_rows_excluded_from_training": excluded_rows,
        "outputs": ["offer_forecast.csv", "backtest_summary.csv", "missing_dates_audit.csv"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries)[["offer", "wape", "mae", "bias_pct"]].to_string(index=False))
    print(f"forecast rows: {len(forecast)} | dates: {dates.min().date()} to {dates.max().date()}")


if __name__ == "__main__":
    main()
