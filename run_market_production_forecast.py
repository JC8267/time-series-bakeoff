"""Package production forecasts for the trade-area (market-level) pipeline.

Implements the 2026-06-10 market-bakeoff verdicts (run_market_bakeoff.py):

- Per-market forecasts: TimesFM 2.5 batched across markets (won 77-78% of
  markets vs per-market seasonal naive). Emitted as point forecasts with each
  market's mean backtest WAPE attached as a quality indicator -- read a high
  `backtest_wape` as "this market is hard to forecast", not as confidence.
- Aggregate forecasts: DIRECT seasonal naive on the bu-total and the
  like-for-like aggregate (mature markets, physical unit open >= 365 days at
  the cutoff), with residual-quantile intervals from leak-free rolling
  backtest windows computed in-script.

Scope note (DATA.md): the bu-level panel covers ~88% of daily_sales_data.csv.
These aggregates are NOT comparable to run_production_forecast.py outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from run_bakeoff import set_seeds
from run_market_bakeoff import load_panel, naive_forecast, timesfm_batched


QUANTILE_LEVELS = [0.025, 0.10, 0.90, 0.975]


def aggregate_with_intervals(series: pd.Series, horizon: int, cutoff: pd.Timestamp, n_windows: int, model_label: str, scope: str) -> list[dict]:
    y = series.loc[:cutoff].to_numpy()
    forecast = naive_forecast(y, horizon)

    residuals = []
    for k in range(1, n_windows + 1):
        bt_cutoff = cutoff - pd.Timedelta(days=28 * k)
        bt_y = series.loc[:bt_cutoff].to_numpy()
        if len(bt_y) < 364:
            break
        bt_fcst = naive_forecast(bt_y, horizon)
        bt_actual = series.loc[bt_cutoff + pd.Timedelta(days=1) : bt_cutoff + pd.Timedelta(days=horizon)].to_numpy()
        n = min(len(bt_fcst), len(bt_actual))
        residuals.extend((bt_actual[:n] - bt_fcst[:n]).tolist())
    qs = np.quantile(residuals, QUANTILE_LEVELS) if len(residuals) >= 50 else np.array([np.nan] * 4)

    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    rows = []
    for ds, pred in zip(dates, forecast):
        rows.append(
            {
                "scope": scope,
                "ds": ds,
                "model_name": model_label,
                "horizon": horizon,
                "forecast": max(0.0, float(pred)),
                "p025": max(0.0, float(pred + qs[0])),
                "p10": max(0.0, float(pred + qs[1])),
                "p50": max(0.0, float(pred)),
                "p90": max(0.0, float(pred + qs[2])),
                "p975": max(0.0, float(pred + qs[3])),
                "interval_method": f"own_residual_quantiles_{len(residuals)}_backtest_days",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="outputs_market_prep/market_daily_panel.csv")
    parser.add_argument("--bakeoff-predictions", default="outputs_market_bakeoff/market_bakeoff_fold_predictions.csv")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--interval-windows", type=int, default=8)
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--output-dir", default="outputs_market_production_forecast")
    args = parser.parse_args()

    set_seeds()
    panel = load_panel(Path(args.panel))
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    cutoff = panel["ds"].max()

    quality = pd.DataFrame()
    bakeoff_path = Path(args.bakeoff_predictions)
    if bakeoff_path.exists():
        bp = pd.read_csv(bakeoff_path)
        tfm = bp[(bp["model_name"] == "market_timesfm") & (bp["abs_actual_sum"] > 0)].copy()
        tfm["wape"] = tfm["abs_err_sum"] / tfm["abs_actual_sum"]
        quality = tfm.groupby(["sub_market_no", "horizon"])["wape"].mean().rename("backtest_wape").reset_index()

    from run_prophet_timesfm_benchmark import load_timesfm

    timesfm_model = load_timesfm(args.timesfm_context, max(horizons))

    histories = {m: g[g["ds"] <= cutoff]["y"].to_numpy() for m, g in panel.groupby("sub_market_no")}
    meta = panel.groupby("sub_market_no").agg(open_date=("open_date", "first"), unit_type=("unit_type", "first")).reset_index()
    mature = set(meta.loc[meta["open_date"].notna() & (meta["open_date"] <= cutoff - pd.Timedelta(days=365)), "sub_market_no"])

    market_rows = []
    aggregate_rows = []
    for horizon in horizons:
        dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
        forecasts = timesfm_batched(timesfm_model, histories, horizon, args.timesfm_context)
        for market, fcst in forecasts.items():
            for ds, pred in zip(dates, fcst):
                market_rows.append(
                    {
                        "sub_market_no": market,
                        "ds": ds,
                        "horizon": horizon,
                        "model_name": "market_timesfm_production",
                        "forecast": max(0.0, float(pred)),
                    }
                )

        total_series = panel.groupby("ds")["y"].sum()
        lfl_series = panel[panel["sub_market_no"].isin(mature)].groupby("ds")["y"].sum()
        aggregate_rows.extend(aggregate_with_intervals(total_series, horizon, cutoff, args.interval_windows, "bu_total_seasonal_naive_production", "bu_total"))
        aggregate_rows.extend(aggregate_with_intervals(lfl_series, horizon, cutoff, args.interval_windows, "like_for_like_seasonal_naive_production", "like_for_like"))

    market_out = pd.DataFrame(market_rows)
    if not quality.empty:
        market_out = market_out.merge(quality, on=["sub_market_no", "horizon"], how="left")
    aggregate_out = pd.DataFrame(aggregate_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    market_out.to_csv(output_dir / "market_production_forecast.csv", index=False)
    aggregate_out.to_csv(output_dir / "aggregate_production_forecast.csv", index=False)

    recent_openings = meta[meta["open_date"].notna() & (meta["open_date"] > cutoff - pd.Timedelta(days=180))]
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": str(cutoff.date()),
        "horizons": horizons,
        "markets": int(market_out["sub_market_no"].nunique()),
        "mature_markets_in_lfl": len(mature),
        "models": sorted(set(market_out["model_name"]) | set(aggregate_out["model_name"])),
        "recently_opened_markets_still_ramping": {
            str(r.sub_market_no): str(r.open_date.date()) for r in recent_openings.itertuples(index=False)
        },
        "basis": "2026-06-10 market bakeoff verdicts (run_id 20260610T132950Z); scope is the bu-level panel (~88% of daily_sales_data.csv, see DATA.md)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_market_production_forecast.py", manifest)

    print(f"markets forecast: {market_out['sub_market_no'].nunique()} | cutoff: {cutoff.date()}")
    print(aggregate_out.groupby(["scope", "horizon"]).agg(rows=("ds", "size"), forecast_sum=("forecast", "sum")).reset_index().to_string(index=False))
    if manifest["recently_opened_markets_still_ramping"]:
        print("\nNote: markets opened <180 days ago (TimesFM sees only partial ramp):", ", ".join(manifest["recently_opened_markets_still_ramping"]))


if __name__ == "__main__":
    main()
