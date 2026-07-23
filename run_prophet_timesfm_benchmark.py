"""Benchmark Prophet + TimesFM 2.5 and simple ensembles against holdout windows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import load_sales, score_group, set_seeds

if not hasattr(np, "float_"):
    np.float_ = np.float64


def load_timesfm(max_context: int, max_horizon: int):
    import torch
    from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

    torch.set_float32_matmul_precision("high")
    model = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    model.compile(
        ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )
    return model


def load_chronos_bolt(model_id: str = "amazon/chronos-bolt-small"):
    import torch
    from chronos import ChronosBoltPipeline

    try:
        return ChronosBoltPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    except TypeError:
        return ChronosBoltPipeline.from_pretrained(model_id)


def load_chronos2(model_id: str = "amazon/chronos-2"):
    import torch
    from chronos import Chronos2Pipeline

    try:
        return Chronos2Pipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    except TypeError:
        return Chronos2Pipeline.from_pretrained(model_id)


def forecast_timesfm(model, train: pd.DataFrame, horizon: int, max_context: int) -> pd.DataFrame:
    y = train["y"].astype(float).to_numpy()
    ctx = y[-min(len(y), max_context) :]
    if len(ctx) < max_context:
        ctx = np.pad(ctx, (max_context - len(ctx), 0), mode="edge")
    point, quantiles = model.forecast(horizon=horizon, inputs=[ctx])
    forecast = np.asarray(point)[0][:horizon]
    out = pd.DataFrame({"forecast": forecast})
    q = np.asarray(quantiles)[0] if quantiles is not None else None
    if q is not None and q.ndim >= 2 and q.shape[1] >= 10:
        out["p10"] = q[:horizon, 1]
        out["p50"] = q[:horizon, 5] if q.shape[1] > 5 else forecast
        out["p90"] = q[:horizon, 9]
        out["p025"] = q[:horizon, 0]
        out["p975"] = q[:horizon, -1]
    else:
        out[["p025", "p10", "p50", "p90", "p975"]] = np.nan
        out["p50"] = forecast
    return out


def forecast_chronos2(model, train: pd.DataFrame, horizon: int, max_context: int) -> pd.DataFrame:
    context = train[["ds", "y"]].rename(columns={"ds": "timestamp", "y": "target"}).copy()
    context["item_id"] = "sales"
    pred = model.predict_df(
        context[["item_id", "timestamp", "target"]],
        prediction_length=horizon,
        quantile_levels=[0.025, 0.1, 0.5, 0.9, 0.975],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        context_length=min(max_context, len(context)),
        cross_learning=False,
    ).sort_values("timestamp")
    out = pd.DataFrame(
        {
            "forecast": pred["0.5"].to_numpy(),
            "p025": pred["0.025"].to_numpy(),
            "p10": pred["0.1"].to_numpy(),
            "p50": pred["0.5"].to_numpy(),
            "p90": pred["0.9"].to_numpy(),
            "p975": pred["0.975"].to_numpy(),
        }
    )
    for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
    return out


def forecast_chronos_bolt(model, train: pd.DataFrame, horizon: int, max_context: int) -> pd.DataFrame:
    import torch

    y = train["y"].astype(float).to_numpy()
    ctx = y[-min(len(y), max_context) :]
    context = torch.tensor(ctx, dtype=torch.float32)
    quantiles = model.predict([context], prediction_length=horizon)[0].detach().cpu().numpy()
    q10 = quantiles[0, :horizon]
    q50 = quantiles[4, :horizon] if quantiles.shape[0] > 4 else np.nanmedian(quantiles, axis=0)[:horizon]
    q90 = quantiles[-1, :horizon]
    spread = q90 - q10
    out = pd.DataFrame(
        {
            "forecast": q50,
            "p10": q10,
            "p50": q50,
            "p90": q90,
            "p025": q50 - 1.4 * spread / 2,
            "p975": q50 + 1.4 * spread / 2,
        }
    )
    for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
    return out


def forecast_prophet(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    from prophet import Prophet

    dfp = train[["ds", "y"]].copy()
    use_additive = (dfp["y"] <= 0).any()
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive" if use_additive else "multiplicative",
        interval_width=0.80,
    )
    model.fit(dfp)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    fcst = model.predict(future).tail(horizon)
    out = pd.DataFrame(
        {
            "forecast": fcst["yhat"].to_numpy(),
            "p10": fcst["yhat_lower"].to_numpy(),
            "p90": fcst["yhat_upper"].to_numpy(),
        }
    )
    spread = out["p90"] - out["p10"]
    out["p50"] = out["forecast"]
    out["p025"] = out["forecast"] - 1.4 * spread / 2
    out["p975"] = out["forecast"] + 1.4 * spread / 2
    return out


def latest_holdout(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff].copy()
    test = df[(df["ds"] > cutoff) & (df["ds"] <= df["ds"].max())].copy()
    return train, test


def build_rows(test: pd.DataFrame, model_name: str, fcst: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, runtime: float) -> pd.DataFrame:
    out = test[["unique_id", "ds", "y"]].rename(columns={"y": "actual"}).reset_index(drop=True)
    out["horizon"] = horizon
    out["cutoff"] = cutoff
    out["horizon_step"] = np.arange(1, len(out) + 1)
    out["model_name"] = model_name
    for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
        out[col] = fcst[col].to_numpy()[: len(out)] if col in fcst else np.nan
    for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0)
    out["runtime_seconds"] = runtime
    return out


def weighted_ensemble(prophet: pd.DataFrame, timesfm: pd.DataFrame, weights: tuple[float, float]) -> pd.DataFrame:
    wp, wt = weights
    out = prophet.copy()
    for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
        out[col] = wp * prophet[col].to_numpy() + wt * timesfm[col].to_numpy()
    return out


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seeds()
    df = load_sales(Path(args.sales_csv))
    horizons = [int(x) for x in args.horizons.split(",")]
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        tfm = load_timesfm(args.tfm_max_context, max(horizons)) if "timesfm_2_5" in requested_models else None
    except Exception as exc:
        print(f"Skipping TimesFM: {exc}")
        tfm = None
    try:
        chronos = load_chronos_bolt(args.chronos_model_id) if "chronos_bolt_small" in requested_models else None
    except Exception as exc:
        print(f"Skipping Chronos-Bolt: {exc}")
        chronos = None
    try:
        chronos2 = load_chronos2(args.chronos2_model_id) if "chronos2" in requested_models else None
    except Exception as exc:
        print(f"Skipping Chronos-2: {exc}")
        chronos2 = None
    all_rows = []

    for horizon in horizons:
        train, test = latest_holdout(df, horizon)
        cutoff = train["ds"].max()

        timesfm_fcst = None
        timesfm_runtime = 0.0
        if tfm is not None:
            started = time.perf_counter()
            timesfm_fcst = forecast_timesfm(tfm, train, horizon, args.tfm_max_context)
            timesfm_runtime = time.perf_counter() - started

        chronos_fcst = None
        chronos_runtime = 0.0
        if chronos is not None:
            started = time.perf_counter()
            chronos_fcst = forecast_chronos_bolt(chronos, train, horizon, args.chronos_max_context)
            chronos_runtime = time.perf_counter() - started

        chronos2_fcst = None
        chronos2_runtime = 0.0
        if chronos2 is not None:
            started = time.perf_counter()
            chronos2_fcst = forecast_chronos2(chronos2, train, horizon, args.chronos2_max_context)
            chronos2_runtime = time.perf_counter() - started

        started = time.perf_counter()
        prophet_fcst = forecast_prophet(train, horizon)
        prophet_runtime = time.perf_counter() - started

        models = {"prophet": (prophet_fcst, prophet_runtime)}
        if timesfm_fcst is not None:
            models["timesfm_2_5"] = (timesfm_fcst, timesfm_runtime)
            models["prophet_timesfm_mean"] = (weighted_ensemble(prophet_fcst, timesfm_fcst, (0.5, 0.5)), 0.0)
            models["prophet_timesfm_weighted_30_70"] = (weighted_ensemble(prophet_fcst, timesfm_fcst, (0.3, 0.7)), 0.0)
            models["prophet_timesfm_weighted_70_30"] = (weighted_ensemble(prophet_fcst, timesfm_fcst, (0.7, 0.3)), 0.0)
        if chronos_fcst is not None:
            models["chronos_bolt_small"] = (chronos_fcst, chronos_runtime)
            models["prophet_chronos_bolt_mean"] = (weighted_ensemble(prophet_fcst, chronos_fcst, (0.5, 0.5)), 0.0)
        if chronos2_fcst is not None:
            models["chronos2"] = (chronos2_fcst, chronos2_runtime)
            models["prophet_chronos2_mean"] = (weighted_ensemble(prophet_fcst, chronos2_fcst, (0.5, 0.5)), 0.0)
        if timesfm_fcst is not None and chronos_fcst is not None:
            foundation = prophet_fcst.copy()
            for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
                foundation[col] = np.nanmean(np.vstack([timesfm_fcst[col], chronos_fcst[col]]), axis=0)
            models["timesfm_chronos_bolt_mean"] = (foundation, 0.0)
        if timesfm_fcst is not None and chronos2_fcst is not None:
            foundation = prophet_fcst.copy()
            for col in ["forecast", "p025", "p10", "p50", "p90", "p975"]:
                foundation[col] = np.nanmean(np.vstack([timesfm_fcst[col], chronos2_fcst[col]]), axis=0)
            models["timesfm_chronos2_mean"] = (foundation, 0.0)
        for name, (fcst, runtime) in models.items():
            all_rows.append(build_rows(test, name, fcst, cutoff, horizon, runtime))

    preds = pd.concat(all_rows, ignore_index=True)
    metrics = (
        preds.groupby(["horizon", "model_name"], dropna=False)
        .apply(score_group, include_groups=False)
        .reset_index()
        .sort_values(["horizon", "wape"])
    )
    return preds, metrics


def write_outputs(preds: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_dir / "prophet_timesfm_holdout_predictions.csv", index=False)
    metrics.to_csv(output_dir / "prophet_timesfm_holdout_metrics.csv", index=False)
    try:
        preds.to_parquet(output_dir / "prophet_timesfm_holdout_predictions.parquet", index=False)
    except Exception:
        pass
    lines = [
        "# Prophet + TimesFM 2.5 Holdout Benchmark",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Horizon | Model | WAPE | Bias pct | MAE | RMSE | Coverage 80 | Coverage 95 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {row.model_name} | {row.wape:.6f} | {row.bias_pct:.6f} | {row.mae:,.0f} | {row.rmse:,.0f} | {row.coverage_80:.3f} | {row.coverage_95:.3f} |"
        )
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "rows": int(len(preds))}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--tfm-max-context", type=int, default=1024)
    parser.add_argument("--chronos-max-context", type=int, default=2048)
    parser.add_argument("--chronos-model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--chronos2-max-context", type=int, default=8192)
    parser.add_argument("--chronos2-model-id", default="amazon/chronos-2")
    parser.add_argument("--models", default="timesfm_2_5,chronos_bolt_small,prophet")
    parser.add_argument("--output-dir", default="outputs_prophet_timesfm")
    args = parser.parse_args()
    preds, metrics = run(args)
    write_outputs(preds, metrics, Path(args.output_dir))
    print(metrics[["horizon", "model_name", "wape", "bias_pct", "mae", "rmse", "coverage_80", "coverage_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
