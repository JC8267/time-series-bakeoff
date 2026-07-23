"""Create reconciled offer and total production forecasts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import score_group
from run_market_bakeoff import naive_forecast
from run_market_production_forecast import aggregate_with_intervals


REQUIRED_COLUMNS = {"transaction_date", "sales", "range_offer"}


def load_offer_series(path: Path, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    raw = pd.read_csv(path)
    if not REQUIRED_COLUMNS.issubset(raw.columns):
        raise ValueError(f"{path} must contain {sorted(REQUIRED_COLUMNS)}.")

    raw["transaction_date"] = pd.to_datetime(raw["transaction_date"], errors="coerce")
    raw["sales"] = pd.to_numeric(raw["sales"], errors="coerce")
    if raw[["transaction_date", "sales", "range_offer"]].isna().any().any():
        raise ValueError("Offer data contains a missing or invalid date, sales value, or offer.")

    excluded_rows = int((raw["transaction_date"] > cutoff).sum())
    daily = raw.groupby(["transaction_date", "range_offer"], as_index=False)["sales"].sum()
    actuals = daily.pivot(index="transaction_date", columns="range_offer", values="sales")
    history = daily[daily["transaction_date"] <= cutoff]
    wide = history.pivot(index="transaction_date", columns="range_offer", values="sales")
    wide = wide.reindex(pd.date_range(wide.index.min(), cutoff, freq="D"))

    missing = wide.isna().stack()
    audit = missing[missing].rename("missing").reset_index().rename(columns={"level_0": "ds"})
    audit["fill_strategy"] = "interpolate"
    wide = wide.interpolate("time").ffill().bfill()
    if wide.isna().any().any():
        raise ValueError("Offer data still contains missing values after interpolation.")
    return wide, actuals, audit, excluded_rows


def backtest_predictions(series: pd.Series, horizon: int, cutoff: pd.Timestamp, n_windows: int) -> pd.DataFrame:
    frames = []
    for k in range(1, n_windows + 1):
        bt_cutoff = cutoff - pd.Timedelta(days=28 * k)
        train = series.loc[:bt_cutoff].to_numpy()
        actual = series.loc[bt_cutoff + pd.Timedelta(days=1) : bt_cutoff + pd.Timedelta(days=horizon)]
        forecast = naive_forecast(train, horizon)[: len(actual)]
        frames.append(pd.DataFrame({"actual": actual.to_numpy(), "forecast": forecast}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="by_offer/raw_data.csv")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--forecast-end", default="2026-08-31")
    parser.add_argument("--output-dir", default="by_offer/outputs_forecast_cutoff_20260430")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    forecast_end = pd.Timestamp(args.forecast_end)
    horizon = (forecast_end - cutoff).days
    if horizon <= 0:
        raise ValueError("forecast-end must be after cutoff.")

    wide, actuals, audit, excluded_rows = load_offer_series(Path(args.input), cutoff)
    if len(wide.columns) != 3:
        raise ValueError(f"Expected 3 offers, found {len(wide.columns)}: {list(wide.columns)}")

    rows = []
    summaries = []
    series = {offer: wide[offer] for offer in wide.columns}
    series["Total"] = wide.sum(axis=1)
    for offer, history in series.items():
        rows.extend(aggregate_with_intervals(history, horizon, cutoff, 8, "seasonal_naive_lag_364", offer))
        metrics = score_group(backtest_predictions(history, horizon, cutoff, 8))
        summaries.append({"offer": offer, "model_name": "seasonal_naive_lag_364", **metrics.to_dict()})

    forecast = pd.DataFrame(rows).rename(columns={"scope": "offer"})
    forecast["training_cutoff"] = cutoff
    actuals = actuals.reindex(columns=wide.columns)
    actuals["Total"] = actuals.sum(axis=1, min_count=3)
    actuals = actuals.rename_axis(index="ds", columns="offer").stack().rename("actual").reset_index()
    forecast = forecast.merge(actuals, on=["offer", "ds"], how="left")
    forecast = forecast[
        ["offer", "ds", "actual", "model_name", "p025", "p10", "p50", "p90", "p975", "interval_method", "training_cutoff"]
    ]

    pivot = forecast.pivot(index="ds", columns="offer", values="p50")
    np.testing.assert_allclose(pivot["Total"], pivot.drop(columns="Total").sum(axis=1))
    actual_pivot = forecast.pivot(index="ds", columns="offer", values="actual")
    complete_actuals = actual_pivot.drop(columns="Total").notna().all(axis=1)
    np.testing.assert_allclose(
        actual_pivot.loc[complete_actuals, "Total"], actual_pivot.loc[complete_actuals].drop(columns="Total").sum(axis=1)
    )
    if actual_pivot.loc[~complete_actuals, "Total"].notna().any():
        raise AssertionError("Total actual is populated for an incomplete offer date.")
    if not ((forecast["p025"] <= forecast["p10"]) & (forecast["p10"] <= forecast["p50"]) &
            (forecast["p50"] <= forecast["p90"]) & (forecast["p90"] <= forecast["p975"])).all():
        raise AssertionError("Forecast intervals are not ordered.")
    if forecast["ds"].min() != cutoff + pd.Timedelta(days=1) or forecast["ds"].max() != forecast_end:
        raise AssertionError("Forecast date range does not match the request.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "offer_forecast.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "backtest_summary.csv", index=False)
    audit.to_csv(output_dir / "missing_dates_audit.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "training_cutoff": str(cutoff.date()),
        "forecast_start": str((cutoff + pd.Timedelta(days=1)).date()),
        "forecast_end": str(forecast_end.date()),
        "horizon_days": horizon,
        "offers": list(wide.columns),
        "model": "seasonal_naive_lag_364",
        "intervals": "global residual quantiles from 8 rolling backtest windows",
        "total_intervals": "directly calibrated on the summed offer series",
        "missing_dates_filled": len(audit),
        "actual_rows": {offer: int(count) for offer, count in forecast.groupby("offer")["actual"].count().items()},
        "post_cutoff_rows_excluded_from_training": excluded_rows,
        "outputs": ["offer_forecast.csv", "backtest_summary.csv", "missing_dates_audit.csv"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(pd.DataFrame(summaries)[["offer", "wape", "bias_pct"]].to_string(index=False))
    print(f"forecast rows: {len(forecast)} | dates: {forecast['ds'].min().date()} to {forecast['ds'].max().date()}")


if __name__ == "__main__":
    main()
