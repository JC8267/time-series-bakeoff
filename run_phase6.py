"""Phase 6 production forecast package.

Trains the final selected ensemble inputs on all available history and writes
forward forecasts with calibrated intervals.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_bakeoff
from run_bakeoff import load_sales, model_specs, set_seeds
from run_bakeoff import load_openings
from run_phase2 import autogluon_predict
from run_prophet_timesfm_benchmark import forecast_chronos_bolt, forecast_timesfm, load_chronos_bolt, load_timesfm
from experiment_registry import append_experiment


BASE_MODELS = [
    "seasonal_naive_lag_364",
    "mlforecast_style_lightgbm",
    "mlforecast_style_catboost",
    "sklearn_hist_gradient_boosting",
    "sklearn_extra_trees",
    "statsforecast_auto_arima",
    "statsforecast_auto_ets",
    "statsforecast_mstl_auto_arima",
    "statsforecast_mstl_auto_ets",
    "statsmodels_ets_weekly",
    "statsmodels_sarimax_weekly",
    "autogluon_medium_quality",
    "timesfm_2_5",
    "chronos_bolt_small",
]

DEFAULT_FINAL_MODELS = {
    28: [
        {"kind": "base", "source": "sklearn_extra_trees", "name": "final_28d_extra_trees_opening_timesfm_pool"},
    ],
    91: [
        {"kind": "convex_stack", "name": "final_91d_convex_stack_opening_timesfm_pool"},
    ],
    180: [
        {"kind": "base", "source": "seasonal_naive_lag_364", "name": "final_180d_seasonal_naive_benchmark"},
        {"kind": "base", "source": "sklearn_hist_gradient_boosting", "name": "final_180d_hist_gradient_boosting_opening_sensitive"},
    ],
    365: [
        {"kind": "base", "source": "seasonal_naive_lag_364", "name": "final_365d_seasonal_naive_benchmark"},
        {"kind": "base", "source": "mlforecast_style_lightgbm", "name": "final_365d_lightgbm_opening_sensitive"},
    ],
}


def future_frame(last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": "total_sales",
            "ds": pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D"),
            "y": np.nan,
        }
    )


def required_base_models(horizon: int, final_config: dict[int, list[dict]], phase3_metadata: dict) -> list[str]:
    needed: list[str] = []
    meta = phase3_metadata.get(str(horizon), {})
    for item in final_config.get(horizon, []):
        kind = item["kind"]
        if kind == "base":
            needed.append(item["source"])
        elif kind == "top5_median":
            needed.extend(meta.get("inverse_wape_models", []))
        elif kind == "convex_stack":
            needed.extend(meta.get("convex_stack_weights", {}).keys())
        elif kind == "ridge_stack":
            needed.extend(meta.get("ridge_coefficients", {}).keys())
    ordered = [m for m in BASE_MODELS if m in set(needed)]
    return ordered or BASE_MODELS


def forecast_base_models(
    train: pd.DataFrame,
    horizon: int,
    output_dir: Path,
    autogluon_time_limit: int,
    openings: pd.DataFrame | None = None,
    timesfm_model=None,
    chronos_model=None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    future = future_frame(train["ds"].max(), horizon)
    dates = future["ds"].reset_index(drop=True)
    spec_lookup = {spec.name: spec for spec in model_specs(horizon, openings)}
    rows = []
    for model_name in (model_names or BASE_MODELS):
        started = time.perf_counter()
        if model_name == "autogluon_medium_quality":
            try:
                fcst = autogluon_predict(
                    train,
                    horizon,
                    output_dir / "autogluon_models" / f"h{horizon}_production",
                    autogluon_time_limit,
                    openings,
                )
                forecast = future[["unique_id", "ds"]].merge(fcst, on=["unique_id", "ds"], how="left")["forecast"].to_numpy()
            except Exception as exc:
                print(f"Skipping AutoGluon in phase 6 horizon {horizon}: {exc}")
                continue
        elif model_name == "timesfm_2_5":
            if timesfm_model is None:
                continue
            fcst = forecast_timesfm(timesfm_model, train, horizon, 1024)
            forecast = fcst["forecast"].to_numpy()
        elif model_name == "chronos_bolt_small":
            if chronos_model is None:
                continue
            fcst = forecast_chronos_bolt(chronos_model, train, horizon, 2048)
            forecast = fcst["forecast"].to_numpy()
        else:
            if model_name not in spec_lookup:
                continue
            forecast = spec_lookup[model_name].predict(train, dates)
        runtime = time.perf_counter() - started
        for i, (row, pred) in enumerate(zip(future.itertuples(index=False), forecast), start=1):
            rows.append(
                {
                    "unique_id": row.unique_id,
                    "horizon": horizon,
                    "training_cutoff": train["ds"].max(),
                    "ds": row.ds,
                    "horizon_step": i,
                    "model_name": model_name,
                    "forecast": max(0.0, float(pred)),
                    "runtime_seconds": runtime,
                }
            )
    return pd.DataFrame(rows)


def build_final_ensembles(base: pd.DataFrame, phase3_metadata: dict, horizon: int, final_config: dict[int, list[dict]]) -> pd.DataFrame:
    wide = base.pivot_table(
        index=["unique_id", "horizon", "training_cutoff", "ds", "horizon_step"],
        columns="model_name",
        values="forecast",
        aggfunc="first",
    ).reset_index()
    meta = phase3_metadata.get(str(horizon), {})
    frames = []

    for item in final_config.get(horizon, []):
        kind = item["kind"]
        final_model = item["name"]
        if kind == "base":
            source = item["source"]
            if source not in wide.columns:
                continue
            values = wide[source]
            model_meta = {"kind": kind, "source": source}
        elif kind == "top5_median":
            top5 = [m for m in meta.get("inverse_wape_models", []) if m in wide.columns]
            if not top5:
                continue
            values = wide[top5].median(axis=1)
            model_meta = {"kind": kind, "models": top5, "phase3_metadata": meta}
        elif kind == "convex_stack":
            convex_models = [m for m in meta.get("convex_stack_weights", {}).keys() if m in wide.columns]
            if not convex_models:
                continue
            weights = np.asarray([meta["convex_stack_weights"][m] for m in convex_models])
            weights = weights / weights.sum() if weights.sum() else np.ones(len(convex_models)) / len(convex_models)
            values = wide[convex_models].to_numpy() @ weights
            model_meta = {"kind": kind, "models": convex_models, "weights": dict(zip(convex_models, weights.tolist())), "phase3_metadata": meta}
        elif kind == "ridge_stack":
            ridge_models = list(meta.get("ridge_coefficients", {}).keys())
            if not ridge_models:
                continue
            missing_ridge = [m for m in ridge_models if m not in wide.columns]
            if missing_ridge:
                raise ValueError(
                    f"{final_model} horizon {horizon}: base forecasts missing for {missing_ridge}. "
                    "Ridge coefficients cannot be applied with terms dropped."
                )
            coefs = np.asarray([meta["ridge_coefficients"][m] for m in ridge_models])
            values = np.maximum(0.0, wide[ridge_models].to_numpy() @ coefs + float(meta.get("ridge_intercept", 0.0)))
            model_meta = {"kind": kind, "models": ridge_models, "phase3_metadata": meta}
        else:
            raise ValueError(f"Unsupported final model kind: {kind}")

        out = wide[["unique_id", "horizon", "training_cutoff", "ds", "horizon_step"]].copy()
        out["model_name"] = final_model
        out["forecast"] = values
        out["created_at"] = datetime.now(timezone.utc).isoformat()
        out["ensemble_metadata_json"] = json.dumps(model_meta)
        frames.append(out)

    if not frames:
        raise ValueError(f"No final models could be built for horizon {horizon}.")
    return pd.concat(frames, ignore_index=True)


def residual_quantiles(phase4: pd.DataFrame, horizon: int, source_model: str) -> np.ndarray:
    subset = phase4[
        (phase4["horizon"] == horizon)
        & (phase4["source_model_name"] == source_model)
        & (phase4["interval_method"] == "global_residual_quantile")
    ].copy()
    if subset.empty:
        return np.zeros(4)
    residuals = subset["actual"] - subset["forecast"]
    return np.quantile(residuals, [0.025, 0.10, 0.90, 0.975])


def apply_intervals(final: pd.DataFrame, phase4_path: Path) -> pd.DataFrame:
    phase4 = pd.read_csv(phase4_path, parse_dates=["cutoff", "ds"])
    source_map = {
        (28, "final_28d_extra_trees_opening_timesfm_pool"): "sklearn_extra_trees",
        (91, "final_91d_convex_stack_opening_timesfm_pool"): "phase3_heldout_cutoffs_convex_stack",
        (180, "final_180d_seasonal_naive_benchmark"): "seasonal_naive_lag_364",
        (180, "final_180d_hist_gradient_boosting_opening_sensitive"): "sklearn_hist_gradient_boosting",
        (180, "final_180d_convex_stack_stats_chronos_pool"): "phase3_heldout_cutoffs_convex_stack",
        (365, "final_365d_seasonal_naive_benchmark"): "seasonal_naive_lag_364",
        (365, "final_365d_lightgbm_opening_sensitive"): "mlforecast_style_lightgbm",
        (365, "final_365d_hist_gradient_boosting_opening_sensitive"): "sklearn_hist_gradient_boosting",
        (365, "final_365d_convex_stack_stats_chronos_pool"): "phase3_heldout_cutoffs_convex_stack",
    }
    out = final.copy()
    for (horizon, model_name), group in out.groupby(["horizon", "model_name"]):
        source = source_map.get((int(horizon), model_name), model_name)
        qs = residual_quantiles(phase4, int(horizon), source)
        idx = group.index
        out.loc[idx, "p50"] = out.loc[idx, "forecast"]
        out.loc[idx, "p025"] = (out.loc[idx, "forecast"] + qs[0]).clip(lower=0)
        out.loc[idx, "p10"] = (out.loc[idx, "forecast"] + qs[1]).clip(lower=0)
        out.loc[idx, "p90"] = (out.loc[idx, "forecast"] + qs[2]).clip(lower=0)
        out.loc[idx, "p975"] = (out.loc[idx, "forecast"] + qs[3]).clip(lower=0)
        out.loc[idx, "interval_source_model"] = source
        out.loc[idx, "interval_method"] = "global_residual_quantile"
    return out


def write_outputs(base: pd.DataFrame, final: pd.DataFrame, output_dir: Path, history: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_dir / "phase6_base_model_future_forecasts.csv", index=False)
    final.to_csv(output_dir / "phase6_final_forecast.csv", index=False)
    try:
        final.to_parquet(output_dir / "phase6_final_forecast.parquet", index=False)
    except Exception:
        pass
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": str(history["ds"].max().date()),
        "history_start": str(history["ds"].min().date()),
        "history_rows": int(len(history)),
        "forecast_rows": int(len(final)),
        "models": sorted(final["model_name"].unique().tolist()),
        "horizons": sorted(int(h) for h in final["horizon"].unique()),
    }
    (output_dir / "phase6_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_phase6.py", manifest)

    summary = [
        "# Phase 6 Production Forecast Package",
        "",
        f"Created at: {manifest['created_at']}",
        f"Training cutoff: {manifest['training_cutoff']}",
        "",
        "Selected production forecasts:",
        "",
        "- 28-day: `final_28d_extra_trees_opening_timesfm_pool`, selected from final holdout WAPE.",
        "- 91-day: `final_91d_convex_stack_opening_timesfm_pool`, selected from final holdout WAPE after adding TimesFM to the ensemble pool.",
        "- 180/365-day: both a pure accuracy seasonal benchmark and an opening-sensitive model are packaged when requested.",
        "",
        "Generated artifacts:",
        "",
        "- `phase6_final_forecast.csv`",
        "- `phase6_final_forecast.parquet`",
        "- `phase6_base_model_future_forecasts.csv`",
        "- `phase6_manifest.json`",
        "",
        "Forecast date ranges:",
    ]
    for row in final.groupby("model_name").agg(start=("ds", "min"), end=("ds", "max"), rows=("ds", "size")).reset_index().itertuples(index=False):
        summary.append(f"- `{row.model_name}`: {pd.Timestamp(row.start).date()} to {pd.Timestamp(row.end).date()} ({row.rows} rows)")
    (output_dir / "run_summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--phase3-metadata", default="outputs_phase3/phase3_metadata.json")
    parser.add_argument("--phase4-predictions", default="outputs_phase4/phase4_calibrated_predictions.csv")
    parser.add_argument("--output-dir", default="outputs_phase6")
    parser.add_argument("--autogluon-time-limit", type=int, default=180)
    parser.add_argument("--opening-info", default=None)
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--final-model-config-json", default=None, help="Optional JSON mapping horizon to final model specs.")
    args = parser.parse_args()

    set_seeds()
    output_dir = Path(args.output_dir)
    run_bakeoff.OPENINGS_DF = load_openings(Path(args.opening_info)) if args.opening_info else None
    history = load_sales(Path(args.sales_csv))
    metadata = json.loads(Path(args.phase3_metadata).read_text(encoding="utf-8"))
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if args.final_model_config_json:
        config_path = Path(args.final_model_config_json)
        config_raw = config_path.read_text(encoding="utf-8") if config_path.exists() else args.final_model_config_json
        final_config = {int(k): v for k, v in json.loads(config_raw).items()}
    else:
        final_config = DEFAULT_FINAL_MODELS
    needed_by_horizon = {h: required_base_models(h, final_config, metadata) for h in horizons}
    needs_timesfm = any("timesfm_2_5" in models for models in needed_by_horizon.values())
    needs_chronos = any("chronos_bolt_small" in models for models in needed_by_horizon.values())
    try:
        timesfm_model = load_timesfm(1024, max(horizons)) if needs_timesfm else None
    except Exception as exc:
        print(f"Skipping TimesFM in phase 6: {exc}")
        timesfm_model = None
    try:
        chronos_model = load_chronos_bolt() if needs_chronos else None
    except Exception as exc:
        print(f"Skipping Chronos-Bolt in phase 6: {exc}")
        chronos_model = None

    base_frames = []
    final_frames = []
    for horizon in horizons:
        base = forecast_base_models(
            history,
            horizon,
            output_dir,
            args.autogluon_time_limit,
            run_bakeoff.OPENINGS_DF,
            timesfm_model,
            chronos_model,
            needed_by_horizon[horizon],
        )
        final = build_final_ensembles(base, metadata, horizon, final_config)
        base_frames.append(base)
        final_frames.append(final)
    base_all = pd.concat(base_frames, ignore_index=True)
    final_all = apply_intervals(pd.concat(final_frames, ignore_index=True), Path(args.phase4_predictions))
    write_outputs(base_all, final_all, output_dir, history)
    print(final_all.groupby("model_name").agg(start=("ds", "min"), end=("ds", "max"), rows=("ds", "size"), forecast_sum=("forecast", "sum")).to_string())


if __name__ == "__main__":
    main()
