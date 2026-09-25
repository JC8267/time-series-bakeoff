"""Pre-registered TiRex-2/TabPFN-TS screen on the existing 28/91-day grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from run_bakeoff import load_sales, set_seeds


REGISTRATION = Path("new_model_screen_registration.json")
NAIVE = "seasonal_naive_lag_364"
TIMESFM = "timesfm_2_5"
TIMESFM_BLEND = "blend_snl75_tfm25"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registration(path: Path = REGISTRATION) -> dict:
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration["status"] != "frozen_before_scoring":
        raise ValueError("Registration must be frozen_before_scoring.")
    data = registration["data"]
    if sha256_file(Path(data["path"])) != data["sha256"]:
        raise ValueError("Sales snapshot hash differs from the pre-registration.")
    for window in registration["windows"].values():
        if sha256_file(Path(window["baseline_grid"])) != window["baseline_grid_sha256"]:
            raise ValueError(f"Baseline grid hash differs: {window['baseline_grid']}")
    return registration


def registered_baseline(registration: dict, horizon: int, smoke: bool = False) -> pd.DataFrame:
    window = registration["windows"][str(horizon)]
    raw = pd.read_csv(window["baseline_grid"], parse_dates=["cutoff", "ds"])
    raw = raw[raw["model_name"].isin([NAIVE, TIMESFM])]
    wide = raw.pivot(index=["cutoff", "ds", "actual"], columns="model_name", values="forecast").reset_index()
    expected = [pd.Timestamp(value) for value in window["cutoffs"]]
    found = sorted(wide["cutoff"].unique())
    if found != expected or wide[[NAIVE, TIMESFM]].isna().any().any():
        raise ValueError(f"Baseline grid does not exactly match registered {horizon}-day windows.")
    if smoke:
        wide = wide[wide["cutoff"].eq(expected[-1])]
    return wide.sort_values(["cutoff", "ds"]).reset_index(drop=True)


def tirex_predict(model, values: np.ndarray, horizon: int, context_limit: int) -> np.ndarray:
    import torch
    from tirex2 import TimeseriesType

    context = torch.as_tensor(values[-context_limit:], dtype=torch.float32).unsqueeze(0)
    series = TimeseriesType(target=context, past_covariates=None, future_covariates=None)
    forecast = np.asarray(model.forecast([series], prediction_length=horizon, output_type="numpy")[0])
    levels = np.asarray(model.quantiles.detach().cpu(), dtype=float)
    median = forecast[0, int(np.flatnonzero(np.isclose(levels, 0.5))[0]), :]
    return validate_forecast(median, horizon)


def tabpfn_predict(pipeline, train: pd.DataFrame, horizon: int, context_limit: int) -> np.ndarray:
    context = train.tail(context_limit).rename(columns={"ds": "timestamp", "y": "target"})
    predicted = pipeline.predict_df(context[["timestamp", "target"]], prediction_length=horizon, quantiles=[0.5])
    return validate_forecast(predicted["target"].to_numpy(), horizon)


def tabfm_feature_row(series: pd.Series, first_date: pd.Timestamp, origin: pd.Timestamp, target: pd.Timestamp, step: int) -> dict:
    return {
        "horizon_step": step,
        "trend_days": (target - first_date).days,
        "day_of_week": target.day_name(),
        "month": target.month_name(),
        "day_of_month": target.day,
        "day_of_year": target.dayofyear,
        "lag_364": series.at[target - pd.Timedelta(364, unit="D")],
        "origin_last_28_mean": series.loc[origin - pd.Timedelta(27, unit="D") : origin].mean(),
        "origin_last_91_mean": series.loc[origin - pd.Timedelta(90, unit="D") : origin].mean(),
    }


def tabfm_examples(train: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    series = train.set_index("ds")["y"].sort_index()
    expected = pd.date_range(series.index.min(), series.index.max())
    if not series.index.equals(expected):
        raise ValueError("TabFM requires a complete daily training series.")
    first_date, cutoff = series.index.min(), series.index.max()
    first_origin = first_date + pd.Timedelta(363, unit="D")
    last_origin = cutoff - pd.Timedelta(horizon, unit="D")
    rows, targets = [], []
    for origin in pd.date_range(first_origin, last_origin):
        for step in range(1, horizon + 1):
            target = origin + pd.Timedelta(step, unit="D")
            rows.append(tabfm_feature_row(series, first_date, origin, target, step))
            targets.append(series.at[target])
    future = [
        tabfm_feature_row(series, first_date, cutoff, cutoff + pd.Timedelta(step, unit="D"), step)
        for step in range(1, horizon + 1)
    ]
    if not rows:
        raise ValueError("Insufficient history for TabFM's registered feature recipe.")
    return pd.DataFrame(rows), np.asarray(targets, dtype=float), pd.DataFrame(future)


def tabfm_predict(model_and_class, train: pd.DataFrame, horizon: int, spec: dict) -> np.ndarray:
    model, regressor_class = model_and_class
    X_train, y_train, X_future = tabfm_examples(train, horizon)
    regressor = regressor_class(
        model=model,
        n_estimators=spec["n_estimators"],
        max_num_rows=spec["context_rows"],
        batch_size=spec["batch_size"],
        random_state=spec["seed"],
    )
    regressor.fit(X_train, y_train)
    return validate_forecast(regressor.predict(X_future), horizon)


def validate_forecast(values, horizon: int) -> np.ndarray:
    forecast = np.asarray(values, dtype=float).reshape(-1)
    if len(forecast) != horizon or not np.isfinite(forecast).all():
        raise ValueError(f"Expected {horizon} finite predictions, got shape={forecast.shape}.")
    return np.maximum(0.0, forecast)


def load_tirex(spec: dict, cache_dir: Path):
    from huggingface_hub import snapshot_download
    from tirex2 import load_model

    repo, revision = spec["checkpoint"].split("@", 1)
    local = snapshot_download(repo, revision=revision, cache_dir=cache_dir)
    return load_model(local, device="cpu")


def load_tabpfn(spec: dict, cache_dir: Path):
    from huggingface_hub import snapshot_download
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    repo, revision = spec["checkpoint"].split("@", 1)
    local = Path(snapshot_download(repo, revision=revision, local_dir=cache_dir / repo.replace("/", "--")))
    checkpoint = local / "tabpfn-v3.5-20260909.safetensors"
    return TabPFNTSPipeline(
        max_context_length=spec["context_limit"],
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_output_selection="median",
        tabpfn_model_config={"model_path": str(checkpoint)},
    )


def load_tabfm(spec: dict, cache_dir: Path):
    import torch
    from tabfm import TabFMRegressor
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    repo, revision = spec["checkpoint"].split("@", 1)
    checkpoint = cache_dir / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Pinned TabFM checkpoint is not present locally: {checkpoint}")
    if spec["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The registered TabFM screen requires CUDA.")
    model = tabfm_v1_0_0.load(model_type="regression", checkpoint_path=str(checkpoint), device=spec["device"])
    return model, TabFMRegressor


def make_predictor(name: str, spec: dict, cache_dir: Path, allow_tabpfn: bool, allow_tabfm: bool):
    if name == "tabpfn_ts" and not allow_tabpfn:
        raise PermissionError("TabPFN-3.5 license acceptance/commercial-license review is required before local evaluation.")
    if name == "tirex_2":
        model = load_tirex(spec, cache_dir)
        return lambda train, horizon: tirex_predict(model, train["y"].to_numpy(), horizon, spec["context_limit"])
    if name == "tabfm_google":
        if not allow_tabfm:
            raise PermissionError("TabFM's non-commercial weights require explicit research-evaluation acceptance.")
        model_and_class = load_tabfm(spec, cache_dir)
        return lambda train, horizon: tabfm_predict(model_and_class, train, horizon, spec)
    pipeline = load_tabpfn(spec, cache_dir)
    return lambda train, horizon: tabpfn_predict(pipeline, train, horizon, spec["context_limit"])


def run_model(
    name: str,
    spec: dict,
    registration: dict,
    sales: pd.DataFrame,
    horizons: list[int],
    smoke: bool,
    cache_dir: Path,
    allow_tabpfn: bool,
    allow_tabfm: bool,
) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    try:
        predictor = make_predictor(name, spec, cache_dir, allow_tabpfn, allow_tabfm)
        rows = []
        for horizon in horizons:
            baseline = registered_baseline(registration, horizon, smoke)
            for cutoff, window in baseline.groupby("cutoff", sort=True):
                train = sales[sales["ds"].le(cutoff)]
                forecast_started = time.perf_counter()
                forecast = predictor(train, horizon)
                runtime = time.perf_counter() - forecast_started
                if not pd.DatetimeIndex(window["ds"]).equals(pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(1, unit="D"), periods=horizon)):
                    raise ValueError("Candidate dates do not exactly match the registered window.")
                for row, value in zip(window.itertuples(index=False), forecast):
                    rows.append(
                        {
                            "horizon": horizon,
                            "cutoff": cutoff,
                            "ds": row.ds,
                            "actual": row.actual,
                            NAIVE: getattr(row, NAIVE),
                            TIMESFM: getattr(row, TIMESFM),
                            spec["standalone_name"]: value,
                            "runtime_seconds": runtime,
                        }
                    )
        return pd.DataFrame(rows), {"status": "smoke_passed" if smoke else "completed", "runtime_seconds": time.perf_counter() - started}
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "blocked" if isinstance(exc, (ImportError, ModuleNotFoundError, PermissionError)) else "failed",
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def score_predictions(predictions: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    standalone, blend = spec["standalone_name"], spec["blend_name"]
    predictions = predictions.copy()
    predictions[TIMESFM_BLEND] = 0.75 * predictions[NAIVE] + 0.25 * predictions[TIMESFM]
    predictions[blend] = 0.75 * predictions[NAIVE] + 0.25 * predictions[standalone]
    fold_rows = []
    models = [NAIVE, TIMESFM_BLEND, standalone, blend]
    for (horizon, cutoff), window in predictions.groupby(["horizon", "cutoff"]):
        denom = np.abs(window["actual"]).sum()
        for model in models:
            errors = window[model] - window["actual"]
            fold_rows.append(
                {
                    "horizon": horizon,
                    "cutoff": cutoff,
                    "model_name": model,
                    "wape": np.abs(errors).sum() / denom,
                    "bias_pct": errors.sum() / denom,
                    "runtime_seconds": window["runtime_seconds"].iloc[0] if model in (standalone, blend) else 0.0,
                }
            )
    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    for horizon, horizon_folds in folds.groupby("horizon"):
        wide = horizon_folds.pivot(index="cutoff", columns="model_name", values="wape")
        bias = horizon_folds.groupby("model_name")["bias_pct"].mean()
        candidate_diff = wide[blend] - wide[NAIVE]
        timesfm_diff = wide[TIMESFM_BLEND] - wide[NAIVE]
        bias_ok = spec.get("bias_guardrail") == "report_only" or abs(bias[blend]) <= abs(bias[NAIVE])
        keep = (
            wide[blend].mean() < min(wide[NAIVE].mean(), wide[TIMESFM_BLEND].mean())
            and bias_ok
            and candidate_diff.max() <= timesfm_diff.max()
        )
        for model in models:
            diff = wide[model] - wide[NAIVE]
            summary_rows.append(
                {
                    "horizon": horizon,
                    "model_name": model,
                    "windows": len(wide),
                    "mean_wape": wide[model].mean(),
                    "median_wape": wide[model].median(),
                    "mean_bias_pct": bias[model],
                    "wins_vs_naive": int((diff < 0).sum()),
                    "mean_wape_diff_vs_naive": diff.mean(),
                    "worst_wape_diff_vs_naive": diff.max(),
                    "verdict": "keep_for_fresh_confirmation" if model == blend and keep else "reject" if model == blend else "comparator",
                }
            )
    return folds, pd.DataFrame(summary_rows)


def package_versions(names: list[str]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--models", default="tirex_2,tabpfn_ts")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-tabpfn-noncommercial-evaluation", action="store_true")
    parser.add_argument("--allow-tabfm-noncommercial-evaluation", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "new_model_screen")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_new_model_screen"))
    args = parser.parse_args()

    set_seeds()
    registration = read_registration(args.registration)
    sales = load_sales(Path(registration["data"]["path"]))
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    horizons = [int(value) for value in args.horizons.split(",")]
    if not set(models).issubset(registration["candidates"]):
        raise ValueError(f"Models must be registered: {models}")
    if not {str(value) for value in horizons}.issubset(registration["windows"]):
        raise ValueError(f"Only registered horizons may be scored: {sorted(registration['windows'])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, fold_metrics, summaries, statuses = [], [], [], {}
    for name in models:
        frame, status = run_model(
            name,
            registration["candidates"][name],
            registration,
            sales,
            horizons,
            args.smoke,
            args.cache_dir,
            args.allow_tabpfn_noncommercial_evaluation,
            args.allow_tabfm_noncommercial_evaluation,
        )
        statuses[name] = status
        if not frame.empty:
            folds, summary = score_predictions(frame, registration["candidates"][name])
            predictions.append(frame)
            fold_metrics.append(folds)
            summaries.append(summary)

    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "predictions.csv", index=False)
        pd.concat(fold_metrics, ignore_index=True).to_csv(args.output_dir / "fold_metrics.csv", index=False)
        pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "scorecard.csv", index=False)
    created_at = datetime.now(timezone.utc)
    manifest = {
        "run_id": created_at.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": created_at.isoformat(),
        "run_mode": "integration_smoke" if args.smoke else "registered_exploratory_screen",
        "registration": str(args.registration),
        "registration_sha256": sha256_file(args.registration),
        "horizons": horizons,
        "models": models,
        "model_status": statuses,
        "no_fallback_predictions": True,
        "production_defaults_changed": False,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "package_versions": package_versions(
            ["tirex-2", "tabpfn-time-series", "tabpfn", "tabfm", "torch", "triton-windows", "flashrnn", "xlstm", "numpy", "pandas", "scikit-learn"]
        ),
        "artifacts": {
            "predictions": "predictions.csv" if predictions else None,
            "fold_metrics": "fold_metrics.csv" if fold_metrics else None,
            "scorecard": "scorecard.csv" if summaries else None,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if predictions and not args.smoke:
        append_experiment(args.output_dir, Path(__file__).name, manifest, "Pre-registered TiRex-2/TabPFN-TS exploratory screen")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
