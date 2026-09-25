"""Forecast AMZN units by item, day, and non-remote channel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"transaction_date", "item_no", "units", "sales_channel"}
MODEL_BY_CHANNEL = {
    "Online": "timesfm_2_5",
    "Store": "seasonal_naive_scaled_lag_364_w56_damped",
}


def load_histories(
    path: Path, cutoff: pd.Timestamp
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict]:
    raw = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), dtype={"item_no": "string"})
    raw["transaction_date"] = pd.to_datetime(raw["transaction_date"], errors="coerce")
    raw["units"] = pd.to_numeric(raw["units"], errors="coerce")
    if raw[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Input contains a missing or invalid date, item, units value, or channel.")

    source_min = raw["transaction_date"].min()
    source_max = raw["transaction_date"].max()
    if not source_min <= cutoff <= source_max:
        raise ValueError(f"Cutoff must be between {source_min.date()} and {source_max.date()}.")

    remote = raw["sales_channel"].str.casefold().eq("remote")
    after_cutoff = raw["transaction_date"].gt(cutoff)
    kept = raw.loc[~remote & ~after_cutoff].copy()
    unexpected = sorted(set(kept["sales_channel"]) - set(MODEL_BY_CHANNEL))
    if unexpected:
        raise ValueError(f"No production model configured for channels: {unexpected}")

    daily = (
        kept.groupby(["transaction_date", "item_no", "sales_channel"], as_index=False, observed=True)["units"]
        .sum()
        .sort_values(["sales_channel", "item_no", "transaction_date"])
    )
    histories: dict[str, dict[str, np.ndarray]] = {}
    for (channel, item), group in daily.groupby(["sales_channel", "item_no"], sort=True, observed=True):
        dates = pd.date_range(group["transaction_date"].min(), cutoff, freq="D")
        values = group.set_index("transaction_date")["units"].reindex(dates, fill_value=0.0).to_numpy(dtype=float)
        histories.setdefault(channel, {})[str(item)] = values

    channel_histories = {}
    for channel, group in daily.groupby("sales_channel", sort=True, observed=True):
        dates = pd.date_range(group["transaction_date"].min(), cutoff, freq="D")
        values = group.groupby("transaction_date", observed=True)["units"].sum().reindex(dates, fill_value=0.0)
        channel_histories[channel] = values.to_numpy(dtype=float)

    audit = {
        "source_rows": int(len(raw)),
        "source_date_min": str(source_min.date()),
        "source_date_max": str(source_max.date()),
        "remote_rows_excluded": int(remote.sum()),
        "post_cutoff_nonremote_rows_excluded": int((after_cutoff & ~remote).sum()),
        "negative_unit_rows_retained": int((kept["units"] < 0).sum()),
    }
    return histories, channel_histories, audit


def store_forecast(y: np.ndarray, horizon: int) -> tuple[np.ndarray, str]:
    if len(y) < 364:
        value = max(0.0, float(y[-min(len(y), 28) :].mean()))
        return np.repeat(value, horizon), "window_average_28_fallback"

    base = y[len(y) - 364 + (np.arange(horizon) % 364)]
    model = "seasonal_naive_lag_364"
    if len(y) >= 420:
        current = y[-56:].sum()
        prior = y[-420:-364].sum()
        scale = max(0.0, 1.0 + 0.5 * (current / prior - 1.0)) if prior > 0 else 1.0
        base = base * scale
        model = "seasonal_naive_scaled_lag_364_w56_damped"
    return np.maximum(0.0, base), model


def store_intervals(y: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residuals = y[364:] - y[:-364] if len(y) > 364 else y - y.mean()
    if len(residuals) < 20:
        recent = y[-min(len(y), 56) :]
        residuals = recent - recent.mean()
    q10, q90 = np.quantile(residuals, [0.1, 0.9])
    p10 = np.minimum(forecast, np.maximum(0.0, forecast + q10))
    p90 = np.maximum(forecast, forecast + q90)
    return p10, p90


def timesfm_forecasts(
    histories: dict[str, np.ndarray],
    horizon: int,
    max_context: int,
    batch_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    from run_prophet_timesfm_benchmark import load_timesfm

    items = sorted(histories)
    model = load_timesfm(max_context, horizon)
    forecasts = []
    lower = []
    upper = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        contexts = []
        for item in batch_items:
            y = histories[item][-max_context:].astype(float)
            if len(y) < max_context:
                y = np.pad(y, (max_context - len(y), 0), mode="edge")
            contexts.append(y)
        point, quantiles = model.forecast(horizon=horizon, inputs=contexts)
        batch = np.asarray(point, dtype=float)[:, :horizon]
        if batch.shape != (len(batch_items), horizon) or not np.isfinite(batch).all():
            raise ValueError(f"TimesFM returned invalid forecasts for batch starting at item {batch_items[0]}.")
        quantiles = np.asarray(quantiles, dtype=float)
        if (
            quantiles.ndim != 3
            or quantiles.shape[0] != len(batch_items)
            or quantiles.shape[1] < horizon
            or quantiles.shape[2] < 10
        ):
            raise ValueError("TimesFM did not return the quantiles needed for 80% prediction intervals.")
        batch = np.maximum(0.0, batch)
        p10 = np.maximum(0.0, quantiles[:, :horizon, 1])
        p90 = np.maximum(0.0, quantiles[:, :horizon, 9])
        forecasts.append(batch)
        lower.append(np.minimum(batch, p10))
        upper.append(np.maximum(batch, p90))
        print(f"TimesFM: {min(start + batch_size, len(items))}/{len(items)} Online items", flush=True)
    return items, np.vstack(forecasts), np.vstack(lower), np.vstack(upper)


def allocate_to_items(
    histories: dict[str, np.ndarray],
    channel_values: np.ndarray,
    channel_p10: np.ndarray,
    channel_p90: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
    items = sorted(histories)
    patterns = np.zeros((len(items), 364), dtype=float)
    fallback_series = 0
    for row, item in enumerate(items):
        y = histories[item]
        if len(y) >= 364:
            patterns[row] = np.maximum(0.0, y[-364:])
        else:
            patterns[row] = max(0.0, float(y[-min(len(y), 28) :].mean()))
            fallback_series += 1

    weights = patterns[:, np.arange(len(channel_values)) % 364]
    totals = weights.sum(axis=0)
    if (totals <= 0).any():
        recent = np.asarray(
            [max(0.0, float(histories[item][-min(len(histories[item]), 28) :].mean())) for item in items],
            dtype=float,
        )
        if recent.sum() <= 0:
            raise ValueError("Cannot allocate a positive channel forecast because every item weight is zero.")
        weights[:, totals <= 0] = recent[:, None]
        totals = weights.sum(axis=0)
    shares = weights / totals
    return (
        items,
        shares * channel_values,
        shares * channel_p10,
        shares * channel_p90,
        fallback_series,
    )


def monthly_values(values: np.ndarray, dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    periods = dates.to_period("M")
    unique = periods.unique()
    month_starts = unique.to_timestamp()
    days = np.asarray([(periods == period).sum() for period in unique], dtype=int)
    monthly = np.column_stack([values[:, periods == period].sum(axis=1) for period in unique])
    return month_starts, monthly, days


def write_outputs(
    output_dir: Path,
    dates: pd.DatetimeIndex,
    channel_forecasts: dict[str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str], str]],
) -> tuple[dict, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "daily_item_channel_forecast.csv"
    monthly_path = output_dir / "monthly_item_channel_forecast.csv"
    totals_daily_path = output_dir / "daily_channel_totals.csv"
    totals_monthly_path = output_dir / "monthly_channel_totals.csv"

    daily_parts = []
    monthly_parts = []
    total_daily_parts = []
    total_monthly_parts = []
    month_starts: pd.DatetimeIndex | None = None
    forecast_days: np.ndarray | None = None

    for channel, (items, values, p10, p90, models, interval_method) in channel_forecasts.items():
        values = np.round(values, 4)
        p10 = np.round(p10, 4)
        p90 = np.round(p90, 4)
        n_items, horizon = values.shape
        daily_parts.append(
            pd.DataFrame(
                {
                    "transaction_date": np.tile(dates, n_items),
                    "item_no": np.repeat(items, horizon),
                    "sales_channel": channel,
                    "forecast_units": values.ravel(),
                    "p10_units": p10.ravel(),
                    "p90_units": p90.ravel(),
                    "model_name": np.repeat(models, horizon),
                    "interval_method": interval_method,
                }
            )
        )

        month_starts, by_month, forecast_days = monthly_values(values, dates)
        _, p10_by_month, _ = monthly_values(p10, dates)
        _, p90_by_month, _ = monthly_values(p90, dates)
        n_months = len(month_starts)
        monthly_parts.append(
            pd.DataFrame(
                {
                    "forecast_month": np.tile(month_starts, n_items),
                    "item_no": np.repeat(items, n_months),
                    "sales_channel": channel,
                    "forecast_units": np.round(by_month, 4).ravel(),
                    "p10_units": np.round(p10_by_month, 4).ravel(),
                    "p90_units": np.round(p90_by_month, 4).ravel(),
                    "forecast_days": np.tile(forecast_days, n_items),
                    "model_name": np.repeat(models, n_months),
                    "interval_method": interval_method,
                }
            )
        )
        total_daily_parts.append(
            pd.DataFrame(
                {
                    "transaction_date": dates,
                    "sales_channel": channel,
                    "forecast_units": np.round(values.sum(axis=0), 4),
                    "p10_units": np.round(p10.sum(axis=0), 4),
                    "p90_units": np.round(p90.sum(axis=0), 4),
                }
            )
        )
        total_monthly_parts.append(
            pd.DataFrame(
                {
                    "forecast_month": month_starts,
                    "sales_channel": channel,
                    "forecast_units": np.round(by_month.sum(axis=0), 4),
                    "p10_units": np.round(p10_by_month.sum(axis=0), 4),
                    "p90_units": np.round(p90_by_month.sum(axis=0), 4),
                    "forecast_days": forecast_days,
                }
            )
        )

    daily = pd.concat(daily_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    channel_daily = pd.concat(total_daily_parts, ignore_index=True)
    channel_monthly = pd.concat(total_monthly_parts, ignore_index=True)

    value_columns = ["forecast_units", "p10_units", "p90_units"]
    grand_daily = channel_daily.groupby("transaction_date", as_index=False)[value_columns].sum()
    grand_daily["sales_channel"] = "Total"
    grand_monthly = channel_monthly.groupby(["forecast_month", "forecast_days"], as_index=False)[value_columns].sum()
    grand_monthly["sales_channel"] = "Total"
    totals_daily = pd.concat([channel_daily, grand_daily], ignore_index=True)
    totals_monthly = pd.concat([channel_monthly, grand_monthly], ignore_index=True)

    daily.to_csv(daily_path, index=False, float_format="%.4f")
    monthly.to_csv(monthly_path, index=False, float_format="%.4f")
    totals_daily.to_csv(totals_daily_path, index=False, float_format="%.4f")
    totals_monthly.to_csv(totals_monthly_path, index=False, float_format="%.4f")

    daily_check = (
        daily.groupby(["transaction_date", "sales_channel"], as_index=False)[value_columns]
        .sum()
        .merge(channel_daily, on=["transaction_date", "sales_channel"], suffixes=("_items", "_total"))
    )
    monthly_check = (
        monthly.groupby(["forecast_month", "sales_channel"], as_index=False)[value_columns]
        .sum()
        .merge(channel_monthly, on=["forecast_month", "sales_channel"], suffixes=("_items", "_total"))
    )
    reconciliation = {
        "max_daily_item_to_channel_difference": float(
            max(
                (daily_check[f"{column}_items"] - daily_check[f"{column}_total"]).abs().max()
                for column in value_columns
            )
        ),
        "max_monthly_item_to_channel_difference": float(
            max(
                (monthly_check[f"{column}_items"] - monthly_check[f"{column}_total"]).abs().max()
                for column in value_columns
            )
        ),
    }
    outputs = {
        "daily_item_channel_forecast": daily_path.name,
        "monthly_item_channel_forecast": monthly_path.name,
        "daily_channel_totals": totals_daily_path.name,
        "monthly_channel_totals": totals_monthly_path.name,
    }
    return outputs, reconciliation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="amzn/item_level.csv")
    parser.add_argument("--cutoff", required=True, help="Last complete date to train on.")
    parser.add_argument("--forecast-end", default="2028-08-31")
    parser.add_argument("--output-dir", default="outputs_amzn_item_forecast")
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--timesfm-batch-size", type=int, default=64)
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    forecast_end = pd.Timestamp(args.forecast_end)
    horizon = (forecast_end - cutoff).days
    if horizon <= 0:
        raise ValueError("--forecast-end must be after --cutoff.")
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), forecast_end, freq="D")

    histories, channel_histories, audit = load_histories(Path(args.input), cutoff)
    _, online_total, online_total_p10, online_total_p90 = timesfm_forecasts(
        {"Online": channel_histories["Online"]}, horizon, args.timesfm_context, 1
    )
    online_items, online_values, online_p10, online_p90, online_share_fallbacks = allocate_to_items(
        histories["Online"], online_total[0], online_total_p10[0], online_total_p90[0]
    )
    store_total, store_model = store_forecast(channel_histories["Store"], horizon)
    store_total_p10, store_total_p90 = store_intervals(channel_histories["Store"], store_total)
    store_items, store_values, store_p10, store_p90, store_share_fallbacks = allocate_to_items(
        histories["Store"], store_total, store_total_p10, store_total_p90
    )
    channel_forecasts = {
        "Online": (
            online_items,
            online_values,
            online_p10,
            online_p90,
            ["top_down_timesfm_2_5_annual_item_share"] * len(online_items),
            "timesfm_native_q10_q90_allocated_by_item_share",
        ),
        "Store": (
            store_items,
            store_values,
            store_p10,
            store_p90,
            [f"top_down_{store_model}_annual_item_share"] * len(store_items),
            "channel_annual_residual_q10_q90_allocated_by_item_share",
        ),
    }

    output_dir = Path(args.output_dir)
    outputs, reconciliation = write_outputs(output_dir, dates, channel_forecasts)
    if max(reconciliation.values()) > 1e-6:
        raise AssertionError(f"Forecast rollups do not reconcile: {reconciliation}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        **audit,
        "training_cutoff": str(cutoff.date()),
        "forecast_start": str(dates.min().date()),
        "forecast_end": str(dates.max().date()),
        "horizons": [horizon],
        "channels": list(channel_forecasts),
        "item_channel_series": {channel: len(values[0]) for channel, values in channel_forecasts.items()},
        "models": sorted({model for values in channel_forecasts.values() for model in values[4]}),
        "production_model_by_channel": MODEL_BY_CHANNEL,
        "prediction_interval": {
            "coverage": 0.8,
            "columns": ["p10_units", "p90_units"],
            "methods": {channel: values[5] for channel, values in channel_forecasts.items()},
            "monthly_rollup": "sum_of_daily_channel_bounds_allocated_by_item_share",
        },
        "item_allocation": {
            "method": "same_day_annual_nonnegative_unit_share",
            "short_history_fallback": "recent_28_day_average_unit_share",
            "fallback_series": {
                "Online": online_share_fallbacks,
                "Store": store_share_fallbacks,
            },
        },
        "outputs": outputs,
        "reconciliation": reconciliation,
        "caveats": [
            "Requested horizon exceeds the 365-day validation window; long-range values are extrapolations.",
            "Item p10/p90 values allocate channel-level uncertainty and are planning ranges, not item-calibrated coverage claims.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
