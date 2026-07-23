"""Package the production forecasts implied by the 2026-06-10 multi-fold verdicts.

Emits, for each requested horizon:

- `final_<h>d_seasonal_naive_production`: the recommended production forecast
  (tiled seasonal naive, lag 364).

The production CSV contains final rows only. With --include-challenger, a
separate diagnostic CSV also contains the pre-registered 0.75 seasonal-naive /
0.25 TimesFM blend for exact validated horizons.

Intervals are residual quantiles computed from each model's own backtest
residuals in the multifold grid predictions (most recent --interval-windows
cutoffs). Intermediate horizons use the first h steps of the smallest covering
validated grid, which is valid for the prefix-stable seasonal-naive production
rule. Grids are discovered via outputs_multifold_*_validation/ manifests.
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
MIN_HORIZON = 1
MAX_HORIZON = 365
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


def parse_horizons(text: str) -> list[int]:
    try:
        horizons = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--horizons must be a comma-separated list of integers.") from exc
    if not horizons:
        raise ValueError("--horizons must include at least one value.")
    invalid = sorted({h for h in horizons if not MIN_HORIZON <= h <= MAX_HORIZON})
    if invalid:
        raise ValueError(f"--horizons must be between {MIN_HORIZON} and {MAX_HORIZON} days; got {invalid}.")
    return sorted(set(horizons))


def resolve_grid(grids: dict[int, Path], horizon: int) -> tuple[int, Path]:
    covering = [grid_horizon for grid_horizon in grids if grid_horizon >= horizon]
    if not covering:
        raise ValueError(f"No multifold validation grid covers horizon {horizon}; need a grid of at least {horizon} days.")
    source_horizon = min(covering)
    return source_horizon, grids[source_horizon]


def grid_model_degradation(grid_path: Path, model_name: str) -> str | None:
    manifest_path = grid_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("degraded_models", {}).get(model_name)


def blend_from_wide(wide: pd.DataFrame) -> np.ndarray:
    return BLEND_NAIVE_WEIGHT * wide["seasonal_naive_lag_364"].to_numpy() + (1 - BLEND_NAIVE_WEIGHT) * wide["timesfm_2_5"].to_numpy()


def residual_quantiles(grid_path: Path, interval_windows: int, horizon: int | None = None) -> dict[str, np.ndarray]:
    if interval_windows <= 0:
        raise ValueError("--interval-windows must be positive.")
    preds = pd.read_csv(grid_path, parse_dates=["cutoff", "ds"])
    if horizon is not None:
        if "horizon_step" not in preds:
            raise ValueError(f"{grid_path} is missing horizon_step.")
        preds = preds[preds["horizon_step"].le(horizon)].copy()
    if preds.empty:
        raise ValueError(f"{grid_path} has no calibration rows for horizon {horizon}.")
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
        model_quantiles = {"global": np.quantile(temp["residual"].to_numpy(), QUANTILE_LEVELS), "counts": {}}
        for segment, group in temp.groupby("segment"):
            model_quantiles[segment] = np.quantile(group["residual"].to_numpy(), QUANTILE_LEVELS)
            model_quantiles["counts"][segment] = len(group)
        out[name] = model_quantiles
    out["n_windows"] = len(recent_cutoffs)
    out["n_rows"] = len(recent)
    return out


def quantiles_for_segment(qs: dict, segment: str, interval_mode: str, shrinkage_k: float) -> np.ndarray:
    if interval_mode == "segment":
        return qs.get(segment, qs["global"])
    if interval_mode != "shrunken":
        return qs["global"]
    segment_qs = qs.get(segment)
    n_segment = qs.get("counts", {}).get(segment, 0)
    if segment_qs is None or n_segment == 0:
        return qs["global"]
    if shrinkage_k < 0:
        raise ValueError("shrinkage_k must be non-negative.")
    weight = n_segment / (n_segment + shrinkage_k) if n_segment + shrinkage_k else 1.0
    return weight * segment_qs + (1.0 - weight) * qs["global"]


def build_horizon_rows(
    naive: np.ndarray,
    tfm: np.ndarray | None,
    horizon: int,
    dates: pd.Series,
    qs: dict,
    training_cutoff: pd.Timestamp,
    run_id: str,
    interval_mode: str = "global",
    shrinkage_k: float = 30.0,
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
        horizon,
        training_cutoff,
        run_id,
        interval_note,
        interval_mode,
        shrinkage_k,
    )
    if tfm is not None and "blend" in qs:
        blend = BLEND_NAIVE_WEIGHT * naive + (1 - BLEND_NAIVE_WEIGHT) * tfm
        rows.extend(
            forecast_rows(
                f"challenger_{horizon}d_blend_snl75_tfm25",
                dates,
                blend,
                qs["blend"],
                horizon,
                training_cutoff,
                run_id,
                interval_note,
                interval_mode,
                shrinkage_k,
            )
        )
    return rows


def forecast_rows(
    model_name: str,
    dates: pd.Series,
    forecast: np.ndarray,
    qs: np.ndarray,
    horizon: int,
    training_cutoff: pd.Timestamp,
    run_id: str,
    interval_note: str,
    interval_mode: str = "global",
    shrinkage_k: float = 30.0,
) -> list[dict]:
    rows = []
    segments = seasonal_segment(dates)
    for horizon_step, (ds, pred, segment) in enumerate(zip(dates, forecast, segments), start=1):
        q_values = qs
        note = interval_note
        if isinstance(qs, dict):
            q_values = quantiles_for_segment(qs, segment, interval_mode, shrinkage_k)
            note = f"{interval_note}_{interval_mode}"
            if interval_mode == "segment":
                note = f"{note}_{segment}"
            if interval_mode == "shrunken":
                note = f"{note}_k{shrinkage_k:g}_{segment}"
        rows.append(
            {
                "run_id": run_id,
                "unique_id": "total_sales",
                "ds": ds,
                "forecast_origin": training_cutoff,
                "grain": "daily",
                "horizon": horizon,
                "horizon_step": horizon_step,
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
    parser.add_argument("--interval-mode", choices=["global", "segment", "shrunken"], default="global")
    parser.add_argument("--shrinkage-k", type=float, default=30.0)
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--include-challenger", action="store_true", help="Write an exact-horizon TimesFM blend to diagnostic_forecast.csv.")
    parser.add_argument("--opening-info", default="opening_info.csv", help="Used only to flag planned openings inside forecast windows; no uplift is applied.")
    parser.add_argument("--output-dir", default="outputs_production_forecast")
    args = parser.parse_args()

    set_seeds()
    horizons = parse_horizons(args.horizons)
    df = load_sales(Path(args.sales_csv))
    training_cutoff = df["ds"].max()
    grids = discover_grids()
    calibrations = {}
    interval_meta = {}
    challenger_skipped_reasons: dict[str, str] = {}
    for horizon in horizons:
        source_horizon, grid_path = resolve_grid(grids, horizon)
        qs = residual_quantiles(grid_path, args.interval_windows, horizon)
        calibrations[horizon] = qs
        interval_meta[str(horizon)] = {
            "grid": str(grid_path),
            "source_horizon": source_horizon,
            "requested_horizon": horizon,
            "n_windows": int(qs["n_windows"]),
            "n_rows": int(qs["n_rows"]),
        }
        if args.include_challenger and source_horizon != horizon:
            challenger_skipped_reasons[str(horizon)] = "challenger requires an exact-horizon validation grid"
        elif args.include_challenger and (reason := grid_model_degradation(grid_path, "timesfm_2_5")):
            challenger_skipped_reasons[str(horizon)] = f"TimesFM validation was degraded: {reason}"
        elif args.include_challenger and "blend" not in qs:
            challenger_skipped_reasons[str(horizon)] = "exact validation grid has no TimesFM forecasts"

    timesfm_model = None
    challenger_horizons = [h for h in horizons if args.include_challenger and str(h) not in challenger_skipped_reasons]
    if challenger_horizons:
        try:
            from run_prophet_timesfm_benchmark import load_timesfm

            timesfm_model = load_timesfm(args.timesfm_context, max(challenger_horizons))
        except Exception as exc:
            reason = f"TimesFM unavailable: {type(exc).__name__}: {exc}"
            challenger_skipped_reasons.update({str(h): reason for h in challenger_horizons})
            print(f"WARNING [production]: {reason}. Emitting seasonal-naive production rows only.", flush=True)

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    max_horizon = max(horizons)
    all_dates = pd.Series(pd.date_range(training_cutoff + pd.Timedelta(days=1), periods=max_horizon, freq="D"))
    naive_path = seasonal_naive(df, max_horizon, 364)
    production_rows = []
    challenger_rows = []
    for horizon in horizons:
        qs = calibrations[horizon]
        dates = all_dates.iloc[:horizon]
        naive = naive_path[:horizon]
        tfm = None
        if timesfm_model is not None and horizon in challenger_horizons:
            try:
                from run_prophet_timesfm_benchmark import forecast_timesfm

                tfm = forecast_timesfm(timesfm_model, df, horizon, args.timesfm_context)["forecast"].to_numpy()
            except Exception as exc:
                reason = f"TimesFM forecast failed: {type(exc).__name__}: {exc}"
                challenger_skipped_reasons[str(horizon)] = reason
                print(f"WARNING [production]: horizon {horizon}: {reason}.", flush=True)
        horizon_rows = build_horizon_rows(
            naive,
            tfm,
            horizon,
            dates,
            qs,
            training_cutoff,
            run_id,
            args.interval_mode,
            args.shrinkage_k,
        )
        production_rows.extend(row for row in horizon_rows if row["model_name"].startswith("final_"))
        challenger_rows.extend(row for row in horizon_rows if row["model_name"].startswith("challenger_"))

    out = pd.DataFrame(production_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "production_forecast.csv", index=False)
    diagnostic_path = output_dir / "diagnostic_forecast.csv"
    if challenger_rows:
        diagnostic = pd.DataFrame(production_rows + challenger_rows)
        diagnostic.to_csv(diagnostic_path, index=False)
    else:
        diagnostic_path.unlink(missing_ok=True)

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
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "training_cutoff": str(training_cutoff.date()),
        "grain": "daily",
        "horizons": horizons,
        "models": sorted(set(out["model_name"]).union(row["model_name"] for row in challenger_rows)),
        "production_models": sorted(out["model_name"].unique().tolist()),
        "diagnostic_models": sorted(set(out["model_name"]).union(row["model_name"] for row in challenger_rows)) if challenger_rows else [],
        "challenger_models": sorted({row["model_name"] for row in challenger_rows}),
        "artifacts": {
            "production_forecast": "production_forecast.csv",
            "diagnostic_forecast": "diagnostic_forecast.csv" if challenger_rows else None,
        },
        "blend_naive_weight": BLEND_NAIVE_WEIGHT,
        "interval_windows": args.interval_windows,
        "interval_mode": args.interval_mode,
        "shrinkage_k": args.shrinkage_k if args.interval_mode == "shrunken" else None,
        "interval_sources": interval_meta,
        "challenger_requested": args.include_challenger,
        "challenger_included": bool(challenger_rows),
        "challenger_included_horizons": sorted({int(row["horizon"]) for row in challenger_rows}),
        "challenger_skipped_reasons": challenger_skipped_reasons,
        "planned_openings_not_in_forecast": planned_openings,
        "basis": "2026-06-10 multi-fold verdicts; see model_cards.md and model_usage_plan.md",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_production_forecast.py", manifest)

    summary = out.groupby("model_name").agg(rows=("ds", "size"), start=("ds", "min"), end=("ds", "max"), forecast_sum=("forecast", "sum")).reset_index()
    print(summary.to_string(index=False))
    if challenger_rows:
        print(f"Diagnostic forecast: {diagnostic_path}")
    flagged = {h: units for h, units in planned_openings.items() if units}
    if flagged:
        print("\nCAVEAT: planned openings inside forecast windows are NOT in these forecasts (no identifiable uplift):")
        for horizon, units in flagged.items():
            print(f"  {horizon}d: {', '.join(units)}")


if __name__ == "__main__":
    main()
