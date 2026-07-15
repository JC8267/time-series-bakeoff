"""Create reconciled weekly offer forecasts with the validated winner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import DEFAULT_WEEKLY_FREQ, daily_reagg_seasonal_naive_from_full_history
from run_offer_production_forecast import load_offer_series
from run_weekly_production_forecast import residual_quantiles, rows_for_model


MODEL_NAME = "daily_reagg_seasonal_naive_lag_365"


def weekly_history(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.DataFrame({"unique_id": series.name, "ds": series.index, "y": series.to_numpy()})
    weekly = (
        daily.set_index("ds")["y"]
        .resample(DEFAULT_WEEKLY_FREQ)
        .agg(["sum", "size"])
        .query("size == 7")
        .rename(columns={"sum": "y"})
        .drop(columns="size")
        .reset_index()
    )
    weekly["unique_id"] = series.name
    return daily, weekly


def complete_weekly_actuals(series: pd.Series, dates: pd.Series) -> pd.Series:
    daily_dates = pd.date_range(dates.min() - pd.Timedelta(days=6), dates.max(), freq="D")
    weekly = series.reindex(daily_dates).resample(DEFAULT_WEEKLY_FREQ).agg(["sum", "count"])
    return weekly["sum"].where(weekly["count"].eq(7)).reindex(dates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="by_offer/raw_data.csv")
    parser.add_argument("--cutoff", default="2026-04-30")
    parser.add_argument("--forecast-end", default="2026-08-31")
    parser.add_argument("--comparison-dir", default="by_offer/outputs_weekly_model_comparison_17w")
    parser.add_argument("--output-dir", default="by_offer/outputs_weekly_forecast_winner_cutoff_20260430")
    args = parser.parse_args()

    source_cutoff = pd.Timestamp(args.cutoff)
    forecast_end = pd.Timestamp(args.forecast_end)
    wide, actuals, audit, excluded_rows = load_offer_series(Path(args.input), source_cutoff)
    if len(wide.columns) != 3:
        raise ValueError(f"Expected 3 offers, found {len(wide.columns)}: {list(wide.columns)}")

    series = {offer: wide[offer].rename(offer) for offer in wide.columns}
    series["Total"] = wide.sum(axis=1).rename("Total")
    histories = {offer: weekly_history(history) for offer, history in series.items()}
    training_cutoff = histories["Total"][1]["ds"].max()
    dates = pd.Series(pd.date_range(training_cutoff + pd.Timedelta(weeks=1), forecast_end, freq=DEFAULT_WEEKLY_FREQ))

    rows = []
    comparison_dir = Path(args.comparison_dir)
    for offer, (daily, weekly) in histories.items():
        forecast = daily_reagg_seasonal_naive_from_full_history(
            daily, weekly, dates, lag=365, weekly_freq=DEFAULT_WEEKLY_FREQ
        )
        grid = comparison_dir / offer / "multifold_grid_predictions.csv"
        quantiles = residual_quantiles(grid, MODEL_NAME, 8)
        offer_rows = rows_for_model(MODEL_NAME, dates, forecast, quantiles, training_cutoff)
        for row in offer_rows:
            row["offer"] = offer
        rows.extend(offer_rows)

    forecast = pd.DataFrame(rows).drop(columns=["unique_id", "forecast"])
    actuals = actuals.reindex(columns=wide.columns)
    actuals["Total"] = actuals.sum(axis=1, min_count=3)
    weekly_actuals = pd.concat(
        {offer: complete_weekly_actuals(actuals[offer], dates) for offer in [*wide.columns, "Total"]},
        names=["offer", "ds"],
    ).rename("actual").reset_index()
    forecast = forecast.merge(weekly_actuals, on=["offer", "ds"], how="left")
    forecast = forecast[
        ["offer", "ds", "actual", "model_name", "p025", "p10", "p50", "p90", "p975", "interval_method", "training_cutoff"]
    ]

    p50 = forecast.pivot(index="ds", columns="offer", values="p50")
    np.testing.assert_allclose(p50["Total"], p50.drop(columns="Total").sum(axis=1))
    actual_pivot = forecast.pivot(index="ds", columns="offer", values="actual")
    complete = actual_pivot.drop(columns="Total").notna().all(axis=1)
    np.testing.assert_allclose(actual_pivot.loc[complete, "Total"], actual_pivot.loc[complete].drop(columns="Total").sum(axis=1))
    if actual_pivot.loc[~complete, "Total"].notna().any():
        raise AssertionError("Total actual is populated for an incomplete offer week.")
    if not ((forecast["p025"] <= forecast["p10"]) & (forecast["p10"] <= forecast["p50"]) &
            (forecast["p50"] <= forecast["p90"]) & (forecast["p90"] <= forecast["p975"])).all():
        raise AssertionError("Forecast intervals are not ordered.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "weekly_offer_forecast.csv", index=False)
    audit.to_csv(output_dir / "missing_dates_audit.csv", index=False)
    winner_summary = pd.read_csv(comparison_dir / "weekly_all_model_summary.csv")
    winner_summary = winner_summary[winner_summary["model_name"].eq(MODEL_NAME)]
    winner_summary.to_csv(output_dir / "backtest_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "source_cutoff": str(source_cutoff.date()),
        "weekly_training_cutoff": str(training_cutoff.date()),
        "forecast_start": str(dates.min().date()),
        "forecast_end": str(dates.max().date()),
        "horizon_weeks": len(dates),
        "weekly_freq": DEFAULT_WEEKLY_FREQ,
        "offers": list(wide.columns),
        "model": MODEL_NAME,
        "intervals": "global residual quantiles from the most recent 8 weekly grid windows",
        "total": "sum of the three offer forecasts; intervals calibrated on Total residuals",
        "actual_rows": {offer: int(count) for offer, count in forecast.groupby("offer")["actual"].count().items()},
        "missing_dates_filled": len(audit),
        "post_cutoff_rows_excluded_from_training": excluded_rows,
        "outputs": ["weekly_offer_forecast.csv", "backtest_summary.csv", "missing_dates_audit.csv"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(forecast.groupby("offer").agg(rows=("ds", "size"), actuals=("actual", "count"), p50_sum=("p50", "sum")).to_string())


if __name__ == "__main__":
    main()
