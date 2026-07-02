"""Package the production forecasts implied by the 2026-06-10 multi-fold verdicts.

Emits, for each requested horizon:

- `final_<h>d_seasonal_naive_production`: the recommended production forecast
  (tiled seasonal naive, lag 364).
- `challenger_<h>d_blend_snl75_tfm25`: the pre-registered 0.75 seasonal-naive /
  0.25 TimesFM blend, included so it can be tracked against production on
  future windows before any promotion (see model_usage_plan.md item 7).

Intervals are residual quantiles computed from each model's own backtest
residuals in the multifold grid predictions (most recent --interval-windows
cutoffs), so they reflect leak-free historical errors of the exact forecast
rule being shipped. Grids are discovered via outputs_multifold_*_validation/
manifests; run run_multifold_final_validation.py first for any new horizon.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from run_bakeoff import load_openings, load_sales, seasonal_naive, set_seeds


BLEND_NAIVE_WEIGHT = 0.75
QUANTILE_COLS = ["p025", "p10", "p50", "p90", "p975"]
QUANTILE_LEVELS = [0.025, 0.10, 0.90, 0.975]


def seasonal_segment(ds: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ds)
    return np.where(dates.dt.month.isin([11, 12]), "holiday_weeks", "rest_weeks")


def discover_grids() -> dict[int, Path]:
    grids = {}
    for manifest_path in sorted(Path(".").glob("outputs_multifold_*_validation/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        grid_path = manifest_path.parent / "multifold_grid_predictions.csv"
        if grid_path.exists():
            grids[int(manifest["horizon"])] = grid_path
    return grids


def blend_from_wide(wide: pd.DataFrame) -> np.ndarray:
    return BLEND_NAIVE_WEIGHT * wide["seasonal_naive_lag_364"].to_numpy() + (1 - BLEND_NAIVE_WEIGHT) * wide["timesfm_2_5"].to_numpy()


def residual_quantiles(grid_path: Path, interval_windows: int) -> dict[str, np.ndarray]:
    preds = pd.read_csv(grid_path, parse_dates=["cutoff", "ds"])
    wide = preds.pivot_table(index=["cutoff", "ds", "actual"], columns="model_name", values="forecast").reset_index()
    if "seasonal_naive_lag_364" not in wide.columns:
        raise ValueError(f"{grid_path} is missing seasonal_naive_lag_364 grid forecasts.")
    recent_cutoffs = sorted(wide["cutoff"].unique())[-interval_windows:]
    recent = wide[wide["cutoff"].isin(recent_cutoffs)]
    pairs = [("seasonal_naive", recent["seasonal_naive_lag_364"].to_numpy())]
    if "timesfm_2_5" in wide.columns:
        pairs.append(("blend", blend_from_wide(recent)))
    out = {}
    for name, forecast in pairs:
        temp = recent[["ds", "actual"]].copy()
        temp["forecast"] = forecast
        temp["segment"] = seasonal_segment(temp["ds"])
        temp["residual"] = temp["actual"] - temp["forecast"]
        model_quantiles = {"global": np.quantile(temp["residual"].to_numpy(), QUANTILE_LEVELS)}
        for segment, group in temp.groupby("segment"):
            model_quantiles[segment] = np.quantile(group["residual"].to_numpy(), QUANTILE_LEVELS)
        out[name] = model_quantiles
    out["n_windows"] = len(recent_cutoffs)
    return out


def build_horizon_rows(
    naive: np.ndarray,
    tfm: np.ndarray | None,
    horizon: int,
    dates: pd.Series,
    qs: dict,
    training_cutoff: pd.Timestamp,
    interval_mode: str = "global",
) -> list[dict]:
    """Production rows always; challenger rows only when a TimesFM forecast exists.

    The seasonal-naive production package must never depend on TimesFM being
    loadable -- pass tfm=None to skip the challenger.
    """
    interval_note = f"own_residual_quantiles_recent_{qs['n_windows']}_grid_windows"
    rows = forecast_rows(
        f"final_{horizon}d_seasonal_naive_production",
        dates,
        naive,
        qs["seasonal_naive"],
        training_cutoff,
        interval_note,
        interval_mode,
    )
    if tfm is not None and "blend" in qs:
        blend = BLEND_NAIVE_WEIGHT * naive + (1 - BLEND_NAIVE_WEIGHT) * tfm
        rows.extend(
            forecast_rows(
                f"challenger_{horizon}d_blend_snl75_tfm25",
                dates,
                blend,
                qs["blend"],
                training_cutoff,
                interval_note,
                interval_mode,
            )
        )
    return rows


def forecast_rows(
    model_name: str,
    dates: pd.Series,
    forecast: np.ndarray,
    qs: np.ndarray,
    training_cutoff: pd.Timestamp,
    interval_note: str,
    interval_mode: str = "global",
) -> list[dict]:
    rows = []
    segments = seasonal_segment(dates)
    for ds, pred, segment in zip(dates, forecast, segments):
        q_values = qs
        note = interval_note
        if isinstance(qs, dict):
            q_values = qs.get(segment, qs["global"]) if interval_mode == "segment" else qs["global"]
            note = f"{interval_note}_{interval_mode}"
            if interval_mode == "segment":
                note = f"{note}_{segment}"
        rows.append(
            {
                "unique_id": "total_sales",
                "ds": ds,
                "model_name": model_name,
                "forecast": max(0.0, float(pred)),
                "p025": max(0.0, float(pred + q_values[0])),
                "p10": max(0.0, float(pred + q_values[1])),
                "p50": max(0.0, float(pred)),
                "p90": max(0.0, float(pred + q_values[2])),
                "p975": max(0.0, float(pred + q_values[3])),
                "interval_method": note,
                "seasonal_segment": segment,
                "training_cutoff": training_cutoff,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--horizons", default="28,91,180,365")
    parser.add_argument("--interval-windows", type=int, default=8, help="Most recent multifold grid cutoffs used for residual quantiles.")
    parser.add_argument("--interval-mode", choices=["global", "segment"], default="global")
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--opening-info", default="opening_info.csv", help="Used only to flag planned openings inside forecast windows; no uplift is applied.")
    parser.add_argument("--output-dir", default="outputs_production_forecast")
    args = parser.parse_args()

    set_seeds()
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    df = load_sales(Path(args.sales_csv))
    training_cutoff = df["ds"].max()
    grids = discover_grids()
    missing = [h for h in horizons if h not in grids]
    if missing:
        raise ValueError(f"No multifold grid found for horizons {missing}; run run_multifold_final_validation.py first.")

    timesfm_model = None
    challenger_skipped_reason = None
    try:
        from run_prophet_timesfm_benchmark import load_timesfm

        timesfm_model = load_timesfm(args.timesfm_context, max(horizons))
    except Exception as exc:
        challenger_skipped_reason = f"TimesFM unavailable: {type(exc).__name__}: {exc}"
        print(f"WARNING [production]: {challenger_skipped_reason}. Emitting seasonal-naive production rows only.", flush=True)

    rows = []
    interval_meta = {}
    for horizon in horizons:
        qs = residual_quantiles(grids[horizon], args.interval_windows)
        interval_meta[str(horizon)] = {
            "grid": str(grids[horizon]),
            "n_windows": int(qs["n_windows"]),
        }
        dates = pd.Series(pd.date_range(training_cutoff + pd.Timedelta(days=1), periods=horizon, freq="D"))
        naive = seasonal_naive(df, horizon, 364)
        tfm = None
        if timesfm_model is not None:
            try:
                from run_prophet_timesfm_benchmark import forecast_timesfm

                tfm = forecast_timesfm(timesfm_model, df, horizon, args.timesfm_context)["forecast"].to_numpy()
            except Exception as exc:
                challenger_skipped_reason = f"TimesFM forecast failed at horizon {horizon}: {type(exc).__name__}: {exc}"
                print(f"WARNING [production]: {challenger_skipped_reason}.", flush=True)
        rows.extend(build_horizon_rows(naive, tfm, horizon, dates, qs, training_cutoff, args.interval_mode))

    out = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "production_forecast.csv", index=False)

    # Planned openings inside a forecast window are volume the seasonal-naive
    # rule structurally cannot anticipate. The aggregate event study found
    # per-unit lift unidentifiable, so no uplift is applied; the manifest
    # flags them as an explicit unquantified caveat instead.
    planned_openings: dict[str, list[str]] = {}
    openings = load_openings(Path(args.opening_info)) if args.opening_info else None
    if openings is not None:
        future = openings[openings["open_date"] > training_cutoff]
        for horizon in horizons:
            in_window = future[future["open_date"] <= training_cutoff + pd.Timedelta(days=horizon)]
            planned_openings[str(horizon)] = [f"{u} ({d.date()})" for u, d in zip(in_window["unit_id"], in_window["open_date"])]

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": str(training_cutoff.date()),
        "horizons": horizons,
        "models": sorted(out["model_name"].unique().tolist()),
        "blend_naive_weight": BLEND_NAIVE_WEIGHT,
        "interval_windows": args.interval_windows,
        "interval_mode": args.interval_mode,
        "interval_sources": interval_meta,
        "challenger_included": challenger_skipped_reason is None,
        "challenger_skipped_reason": challenger_skipped_reason,
        "planned_openings_not_in_forecast": planned_openings,
        "basis": "2026-06-10 multi-fold verdicts; see model_cards.md and model_usage_plan.md",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_production_forecast.py", manifest)

    summary = out.groupby("model_name").agg(rows=("ds", "size"), start=("ds", "min"), end=("ds", "max"), forecast_sum=("forecast", "sum")).reset_index()
    print(summary.to_string(index=False))
    flagged = {h: units for h, units in planned_openings.items() if units}
    if flagged:
        print("\nCAVEAT: planned openings inside forecast windows are NOT in these forecasts (no identifiable uplift):")
        for horizon, units in flagged.items():
            print(f"  {horizon}d: {', '.join(units)}")


if __name__ == "__main__":
    main()
