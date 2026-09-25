"""Registered NeuralProphet screen on existing daily forecast grids."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from run_bakeoff import load_sales, set_seeds
from run_new_model_screen import registered_baseline, read_registration, score_predictions, sha256_file, validate_forecast


REGISTRATION = Path("neuralprophet_screen_registration.json")


def neuralprophet_predict(train: pd.DataFrame, horizon: int, spec: dict) -> np.ndarray:
    from neuralprophet import NeuralProphet, set_log_level

    set_log_level("ERROR")
    model = NeuralProphet(
        n_lags=spec["n_lags"],
        n_forecasts=horizon if spec["n_lags"] else 1,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=spec["epochs"],
        batch_size=spec["batch_size"],
        learning_rate=spec["learning_rate"],
    )
    model.fit(train[["ds", "y"]], freq="D", progress=None)
    future = model.make_future_dataframe(train[["ds", "y"]], periods=horizon, n_historic_predictions=False)
    predicted = model.predict(future)
    dates = pd.date_range(train["ds"].max() + pd.Timedelta(days=1), periods=horizon)
    future_rows = predicted.set_index("ds").reindex(dates)
    values = [future_rows.at[date, f"yhat{step if spec['n_lags'] else 1}"] for step, date in enumerate(dates, 1)]
    return validate_forecast(values, horizon)


def screen(registration: dict, sales: pd.DataFrame, name: str, horizon: int, smoke: bool) -> tuple[pd.DataFrame, dict]:
    spec = registration["candidates"][name]
    started = time.perf_counter()
    rows = []
    try:
        baseline = registered_baseline(registration, horizon, smoke)
        cutoffs = registration.get("evaluation_cutoffs", {}).get(str(horizon))
        if cutoffs:
            baseline = baseline[baseline["cutoff"].isin(pd.to_datetime(cutoffs))]
            if set(baseline["cutoff"]) != set(pd.to_datetime(cutoffs if not smoke else cutoffs[-1:])):
                raise ValueError("Evaluation cutoffs do not match the registered grid.")
        for cutoff, window in baseline.groupby("cutoff", sort=True):
            train = sales[sales["ds"].le(cutoff)]
            run_started = time.perf_counter()
            forecast = neuralprophet_predict(train, horizon, spec)
            runtime = time.perf_counter() - run_started
            expected = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon)
            if not pd.DatetimeIndex(window["ds"]).equals(expected):
                raise ValueError("Candidate dates do not match the registered window.")
            for row, value in zip(window.itertuples(index=False), forecast):
                rows.append({"horizon": horizon, "cutoff": cutoff, "ds": row.ds, "actual": row.actual,
                             "seasonal_naive_lag_364": row.seasonal_naive_lag_364, "timesfm_2_5": row.timesfm_2_5,
                             spec["standalone_name"]: value, "runtime_seconds": runtime})
        return pd.DataFrame(rows), {"status": "smoke_passed" if smoke else "completed", "runtime_seconds": time.perf_counter() - started}
    except Exception as exc:
        return pd.DataFrame(), {"status": "blocked" if isinstance(exc, (ImportError, ModuleNotFoundError)) else "failed",
                                "runtime_seconds": time.perf_counter() - started, "error_type": type(exc).__name__, "error": str(exc)}


def segment_score(predictions: pd.DataFrame, spec: dict) -> pd.DataFrame:
    frame = predictions.copy()
    frame[spec["blend_name"]] = 0.75 * frame["seasonal_naive_lag_364"] + 0.25 * frame[spec["standalone_name"]]
    frame["blend_snl75_tfm25"] = 0.75 * frame["seasonal_naive_lag_364"] + 0.25 * frame["timesfm_2_5"]
    frame["segment"] = np.where(frame["ds"].dt.month.isin([11, 12]), "nov_dec", "rest")
    rows = []
    for (horizon, segment), window in frame.groupby(["horizon", "segment"]):
        denom = window["actual"].abs().sum()
        for name in ("seasonal_naive_lag_364", "blend_snl75_tfm25", spec["standalone_name"], spec["blend_name"]):
            error = window[name] - window["actual"]
            rows.append({"horizon": horizon, "segment": segment, "model_name": name, "days": len(window),
                         "wape": error.abs().sum() / denom, "bias_pct": error.sum() / denom})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_neuralprophet_screen"))
    args = parser.parse_args()
    set_seeds()
    registration = read_registration(args.registration)
    installed = version("neuralprophet")
    if any(spec["package"] != f"neuralprophet=={installed}" for spec in registration["candidates"].values()):
        raise ValueError(f"Installed NeuralProphet {installed} differs from the frozen registration.")
    sales = load_sales(Path(registration["data"]["path"]))
    horizons = [int(h) for h in args.horizons.split(",")]
    if not {str(h) for h in horizons}.issubset(registration["windows"]):
        raise ValueError("Only registered horizons may be scored.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, folds, summaries, segments, statuses = [], [], [], [], {}
    for horizon in horizons:
        for name, spec in registration["candidates"].items():
            frame, status = screen(registration, sales, name, horizon, args.smoke)
            statuses[f"{name}_{horizon}d"] = status
            if frame.empty:
                continue
            fold, summary = score_predictions(frame, spec)
            predictions.append(frame)
            folds.append(fold)
            summaries.append(summary)
            segments.append(segment_score(frame, spec))
    artifacts = {}
    for filename, frames in (("predictions.csv", predictions), ("fold_metrics.csv", folds),
                             ("scorecard.csv", summaries), ("segment_scorecard.csv", segments)):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(args.output_dir / filename, index=False)
            artifacts[filename.removesuffix(".csv")] = filename
    now = datetime.now(timezone.utc)
    manifest = {"run_id": now.strftime("%Y%m%dT%H%M%SZ"), "created_at": now.isoformat(),
                "run_mode": "integration_smoke" if args.smoke else "registered_exploratory_screen",
                "registration": str(args.registration), "registration_sha256": sha256_file(args.registration),
                "horizons": horizons, "model_status": statuses, "no_fallback_predictions": True,
                "production_defaults_changed": False, "environment": {"python": platform.python_version(), "platform": platform.platform()},
                "neuralprophet_version": installed, "artifacts": artifacts}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if predictions and not args.smoke:
        append_experiment(args.output_dir, Path(__file__).name, manifest, "Registered NeuralProphet exploratory screen")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
