"""Phase 5 final holdout validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_bakeoff
from run_bakeoff import load_sales, model_specs, score_group, set_seeds
from run_bakeoff import load_openings
from run_phase2 import autogluon_predict
from run_prophet_timesfm_benchmark import forecast_chronos_bolt, forecast_timesfm, load_chronos_bolt, load_timesfm


SELECTED_MODELS = {
    28: [
        "seasonal_naive_lag_364",
        "sklearn_extra_trees",
        "chronos_bolt_small",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "statsforecast_auto_arima",
        "statsforecast_auto_ets",
        "statsmodels_ets_weekly",
        "statsmodels_sarimax_weekly",
        "phase5_top5_median",
        "phase5_ridge_stack",
    ],
    91: [
        "seasonal_naive_lag_364",
        "autogluon_medium_quality",
        "timesfm_2_5",
        "chronos_bolt_small",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "statsforecast_auto_arima",
        "statsforecast_auto_ets",
        "statsmodels_ets_weekly",
        "statsmodels_sarimax_weekly",
        "phase5_convex_stack",
        "phase5_ridge_stack",
    ],
    180: [
        "seasonal_naive_lag_364",
        "sklearn_hist_gradient_boosting",
        "mlforecast_style_lightgbm",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "chronos_bolt_small",
        "phase5_convex_stack",
        "phase5_ridge_stack",
    ],
    365: [
        "seasonal_naive_lag_364",
        "mlforecast_style_lightgbm",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "chronos_bolt_small",
        "sklearn_hist_gradient_boosting",
        "phase5_convex_stack",
        "phase5_ridge_stack",
    ],
}


def latest_full_holdout(df: pd.DataFrame, horizon: int) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    max_date = df["ds"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff].copy()
    test = df[(df["ds"] > cutoff) & (df["ds"] <= max_date)].copy()
    if len(test) != horizon:
        raise ValueError(f"Expected {horizon} holdout rows, got {len(test)}.")
    return cutoff, train, test


def base_forecasts(train: pd.DataFrame, test: pd.DataFrame, horizon: int, output_dir: Path, autogluon_time_limit: int, openings: pd.DataFrame | None = None, timesfm_model=None, chronos_model=None) -> pd.DataFrame:
    dates = test["ds"].reset_index(drop=True)
    needed = {
        "seasonal_naive_lag_364",
        "mlforecast_style_lightgbm",
        "mlforecast_style_catboost",
        "sklearn_hist_gradient_boosting",
        "sklearn_extra_trees",
        "autogluon_medium_quality",
        "timesfm_2_5",
        "chronos_bolt_small",
        "statsforecast_auto_arima",
        "statsforecast_auto_ets",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "statsmodels_ets_weekly",
        "statsmodels_sarimax_weekly",
    }
    spec_lookup = {spec.name: spec for spec in model_specs(horizon, openings)}
    rows = []
    for model_name in sorted(needed):
        started = time.perf_counter()
        if model_name == "autogluon_medium_quality":
            try:
                fcst = autogluon_predict(
                    train,
                    horizon,
                    output_dir / "autogluon_models" / f"h{horizon}_final",
                    autogluon_time_limit,
                    openings,
                )
                forecast = test[["unique_id", "ds", "y"]].merge(fcst, on=["unique_id", "ds"], how="left")["forecast"].to_numpy()
            except Exception as exc:
                print(f"Skipping AutoGluon in phase 5 horizon {horizon}: {exc}")
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
        for i, (row, pred) in enumerate(zip(test.itertuples(index=False), forecast), start=1):
            rows.append(
                {
                    "unique_id": row.unique_id,
                    "horizon": horizon,
                    "cutoff": train["ds"].max(),
                    "ds": row.ds,
                    "horizon_step": i,
                    "actual": float(row.y),
                    "model_name": model_name,
                    "forecast": max(0.0, float(pred)),
                    "runtime_seconds": runtime,
                    "refit_cadence": "phase5_final_holdout_refit",
                    "hyperparams_json": "{}",
                    "library_version": "phase5",
                }
            )
    return pd.DataFrame(rows)


def make_ensembles(base: pd.DataFrame, metadata: dict, horizon: int) -> pd.DataFrame:
    if str(horizon) not in metadata:
        return pd.DataFrame(columns=base.columns)
    wide = base.pivot_table(
        index=["unique_id", "horizon", "cutoff", "ds", "horizon_step", "actual"],
        columns="model_name",
        values="forecast",
        aggfunc="first",
    ).reset_index()
    meta = metadata[str(horizon)]
    rows = []

    top5 = [m for m in meta["inverse_wape_models"] if m in wide.columns]
    wide["phase5_top5_median"] = wide[top5].median(axis=1)

    convex_models = [m for m in meta["convex_stack_weights"].keys() if m in wide.columns]
    convex_weights = np.asarray([meta["convex_stack_weights"][m] for m in convex_models])
    convex_weights = convex_weights / convex_weights.sum() if convex_weights.sum() else np.ones(len(convex_models)) / len(convex_models)
    wide["phase5_convex_stack"] = wide[convex_models].to_numpy() @ convex_weights

    ridge_models = list(meta["ridge_coefficients"].keys())
    missing_ridge = [m for m in ridge_models if m not in wide.columns]
    if missing_ridge:
        raise ValueError(
            f"phase5_ridge_stack horizon {horizon}: base forecasts missing for {missing_ridge}. "
            "Ridge coefficients cannot be applied with terms dropped; ensure these models run in "
            "phase 5 or rebuild the phase 3 metadata without them."
        )
    ridge_coef = np.asarray([meta["ridge_coefficients"][m] for m in ridge_models])
    wide["phase5_ridge_stack"] = np.maximum(0.0, wide[ridge_models].to_numpy() @ ridge_coef + float(meta["ridge_intercept"]))

    for model_name in ["phase5_top5_median", "phase5_convex_stack", "phase5_ridge_stack"]:
        out = wide[["unique_id", "horizon", "cutoff", "ds", "horizon_step", "actual"]].copy()
        out["model_name"] = model_name
        out["forecast"] = wide[model_name]
        out["runtime_seconds"] = 0.0
        out["refit_cadence"] = "phase5_derived_from_refit_base_models"
        out["hyperparams_json"] = json.dumps(meta)
        out["library_version"] = "phase5"
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def residual_offsets(phase4: pd.DataFrame, horizon: int, source_model: str, method: str) -> dict[str, np.ndarray]:
    subset = phase4[
        (phase4["horizon"] == horizon)
        & (phase4["source_model_name"] == source_model)
        & (phase4["interval_method"] == method)
    ].copy()
    if subset.empty:
        # No calibration residuals for this model => degenerate zero-width intervals.
        # This silently masked the TimesFM 91d interval bug, so make it LOUD: a model
        # is being scored with no interval source. Add it to phase4 DEFAULT_MODELS.
        print(
            f"WARNING [phase5]: no phase4 residuals for source_model='{source_model}' "
            f"method='{method}' horizon={horizon}. Intervals will be ZERO-WIDTH "
            f"(coverage 0). Add '{source_model}' to run_phase4.py DEFAULT_MODELS "
            f"(and ensure its predictions are in phase4 inputs).",
            file=sys.stderr,
            flush=True,
        )
        return {"global": np.zeros(4)}
    subset["residual"] = subset["actual"] - subset["forecast"]
    return {"global": np.quantile(subset["residual"], [0.025, 0.10, 0.90, 0.975])}


def apply_intervals(preds: pd.DataFrame, phase4_path: Path) -> pd.DataFrame:
    phase4 = pd.read_csv(phase4_path, parse_dates=["cutoff", "ds"])
    mapping = {
        (28, "phase5_top5_median"): ("phase3_heldout_cutoffs_top5_median", "global_residual_quantile"),
        (28, "phase5_ridge_stack"): ("phase3_all_cutoffs_ridge_stack", "global_residual_quantile"),
        (28, "seasonal_naive_lag_364"): ("seasonal_naive_lag_364", "global_residual_quantile"),
        (28, "sklearn_extra_trees"): ("sklearn_extra_trees", "global_residual_quantile"),
        (91, "phase5_convex_stack"): ("phase3_heldout_cutoffs_convex_stack", "global_residual_quantile"),
        (91, "phase5_ridge_stack"): ("phase3_all_cutoffs_ridge_stack", "global_residual_quantile"),
        (91, "seasonal_naive_lag_364"): ("seasonal_naive_lag_364", "global_residual_quantile"),
        (91, "autogluon_medium_quality"): ("autogluon_medium_quality", "global_residual_quantile"),
    }
    out = preds.copy()
    out[["p025", "p10", "p50", "p90", "p975"]] = np.nan
    for (horizon, model_name), group in out.groupby(["horizon", "model_name"]):
        source_model, method = mapping.get((int(horizon), model_name), (model_name, "global_residual_quantile"))
        qs = residual_offsets(phase4, int(horizon), source_model, method)["global"]
        idx = group.index
        out.loc[idx, "p50"] = out.loc[idx, "forecast"]
        out.loc[idx, "p025"] = (out.loc[idx, "forecast"] + qs[0]).clip(lower=0.0)
        out.loc[idx, "p10"] = (out.loc[idx, "forecast"] + qs[1]).clip(lower=0.0)
        out.loc[idx, "p90"] = (out.loc[idx, "forecast"] + qs[2]).clip(lower=0.0)
        out.loc[idx, "p975"] = (out.loc[idx, "forecast"] + qs[3]).clip(lower=0.0)
        out.loc[idx, "interval_source_model"] = source_model
        out.loc[idx, "interval_method"] = method
    return out


def interval_score(actual: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    width = upper - lower
    return width + (2 / alpha) * (lower - actual).clip(lower=0) + (2 / alpha) * (actual - upper).clip(lower=0)


def holdout_metrics(group: pd.DataFrame) -> pd.Series:
    metrics = score_group(group)
    metrics["wis_80"] = interval_score(group["actual"], group["p10"], group["p90"], 0.20).mean()
    metrics["wis_95"] = interval_score(group["actual"], group["p025"], group["p975"], 0.05).mean()
    return metrics


def run_phase5(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_sales(Path(args.sales_csv))
    metadata = json.loads(Path(args.phase3_metadata).read_text(encoding="utf-8"))
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    try:
        timesfm_model = load_timesfm(1024, max(horizons))
    except Exception as exc:
        print(f"Skipping TimesFM in phase 5: {exc}")
        timesfm_model = None
    try:
        chronos_model = load_chronos_bolt()
    except Exception as exc:
        print(f"Skipping Chronos-Bolt in phase 5: {exc}")
        chronos_model = None

    prediction_frames = []
    for horizon in horizons:
        cutoff, train, test = latest_full_holdout(df, horizon)
        base = base_forecasts(train, test, horizon, output_dir, args.autogluon_time_limit, run_bakeoff.OPENINGS_DF, timesfm_model, chronos_model)
        ensembles = make_ensembles(base, metadata, horizon)
        keep = SELECTED_MODELS.get(horizon, [])
        combined = pd.concat([base, ensembles], ignore_index=True)
        if keep:
            combined = combined[combined["model_name"].isin(keep)].copy()
        combined["holdout_start"] = test["ds"].min()
        combined["holdout_end"] = test["ds"].max()
        prediction_frames.append(combined)

    preds = pd.concat(prediction_frames, ignore_index=True)
    preds = apply_intervals(preds, Path(args.phase4_predictions))
    metrics = (
        preds.groupby(["horizon", "model_name"], dropna=False)
        .apply(holdout_metrics, include_groups=False)
        .reset_index()
        .sort_values(["horizon", "wape"])
    )
    return preds, metrics


def write_outputs(preds: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    preds.to_csv(output_dir / "phase5_holdout_predictions.csv", index=False)
    try:
        preds.to_parquet(output_dir / "phase5_holdout_predictions.parquet", index=False)
    except Exception:
        pass
    metrics.to_csv(output_dir / "phase5_holdout_metrics.csv", index=False)
    lines = [
        "# Phase 5 Final Holdout Summary",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Final holdout results:",
        "",
        "| Horizon | Model | WAPE | Bias pct | Coverage 80 | Coverage 95 | WIS 80 | WIS 95 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["horizon", "wape"]).itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {row.model_name} | {row.wape:.6f} | {row.bias_pct:.6f} | {row.coverage_80:.3f} | {row.coverage_95:.3f} | {row.wis_80:,.0f} | {row.wis_95:,.0f} |"
        )
    lines.extend(
        [
            "",
            "Decision read:",
            "",
            "- Use WAPE and bias for point-forecast selection.",
            "- Use coverage and weighted interval score for interval selection.",
            "- This run validates only the latest full 28-day and 91-day observed windows.",
        ]
    )
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(preds)),
        "models": sorted(preds["model_name"].unique().tolist()),
        "horizons": sorted(int(h) for h in preds["horizon"].unique()),
    }
    (output_dir / "phase5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--phase3-metadata", default="outputs_phase3/phase3_metadata.json")
    parser.add_argument("--phase4-predictions", default="outputs_phase4/phase4_calibrated_predictions.csv")
    parser.add_argument("--output-dir", default="outputs_phase5")
    parser.add_argument("--autogluon-time-limit", type=int, default=180)
    parser.add_argument("--opening-info", default=None)
    parser.add_argument("--horizons", default="28,91")
    args = parser.parse_args()

    run_bakeoff.OPENINGS_DF = load_openings(Path(args.opening_info)) if args.opening_info else None
    preds, metrics = run_phase5(args)
    write_outputs(preds, metrics, Path(args.output_dir))
    print(metrics[["horizon", "model_name", "wape", "bias_pct", "coverage_80", "coverage_95", "wis_80", "wis_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
