"""Leak-free residual ridge covariate harness for seasonal-naive champions."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_registry import append_experiment
from metrics import score_group
from run_bakeoff import (
    DEFAULT_WEEKLY_FREQ,
    add_opening_features,
    daily_forecast_horizon_days,
    gregorian_easter,
    load_openings,
    load_sales,
    reaggregate_daily_forecast,
    seasonal_naive,
    set_seeds,
    thanksgiving_date,
)


ALPHAS = (0.1, 1.0, 10.0, 100.0)
DAILY_CHAMPION = "seasonal_naive_lag_364"
WEEKLY_CHAMPION = "daily_reagg_seasonal_naive_lag_365"


def signed_days_to_nearest(dates: pd.Series, anchor_func) -> np.ndarray:
    out = []
    for ds in pd.to_datetime(dates).dt.normalize():
        anchors = [anchor_func(ds.year + offset) for offset in (-1, 0, 1)]
        deltas = [(ds - anchor).days for anchor in anchors]
        out.append(min(deltas, key=abs))
    return np.asarray(out, dtype=int)


def add_banded_distance_features(frame: pd.DataFrame, prefix: str, distances: np.ndarray) -> None:
    clipped = np.clip(distances, -21, 21)
    frame[f"{prefix}_days"] = clipped
    bands = {
        "pre_8_21": clipped <= -8,
        "pre_1_7": (clipped >= -7) & (clipped <= -1),
        "day": clipped == 0,
        "post_1_7": (clipped >= 1) & (clipped <= 7),
        "post_8_21": clipped >= 8,
    }
    for name, mask in bands.items():
        frame[f"{prefix}_{name}"] = mask.astype(float)


def holiday_features(dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"ds": pd.to_datetime(dates).dt.normalize()})
    add_banded_distance_features(out, "easter", signed_days_to_nearest(out["ds"], gregorian_easter))
    add_banded_distance_features(out, "thanksgiving", signed_days_to_nearest(out["ds"], thanksgiving_date))
    add_banded_distance_features(out, "black_friday", signed_days_to_nearest(out["ds"], lambda year: thanksgiving_date(year) + pd.Timedelta(days=1)))
    return out.drop(columns=["ds"])


def load_covariates(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    raw = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "name", "value"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    wide = raw.pivot_table(index="date", columns="name", values="value", aggfunc="last").reset_index()
    return wide.rename(columns={"date": "ds"})


def feature_matrix(dates: pd.Series, feature_set: str, openings: pd.DataFrame | None, covariates: pd.DataFrame | None) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates).dt.normalize()).reset_index(drop=True)
    pieces = [holiday_features(dates)]
    if feature_set in {"f1f2", "f1f2f3"}:
        if openings is None:
            raise ValueError("Feature set F2 requires --opening-info.")
        opening_frame = add_opening_features(pd.DataFrame({"ds": dates}), openings).drop(columns=["ds"])
        pieces.append(opening_frame.reset_index(drop=True))
    if feature_set == "f1f2f3":
        if covariates is None:
            raise ValueError("Feature set F3 requires --covariate-csv.")
        cov = pd.DataFrame({"ds": dates}).merge(covariates, on="ds", how="left").drop(columns=["ds"]).fillna(0.0)
        cov = cov.add_prefix("covariate_")
        pieces.append(cov.reset_index(drop=True))
    return pd.concat(pieces, axis=1).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_model(x: pd.DataFrame, y: pd.Series, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, y)
    return model


def choose_alpha(train_rows: pd.DataFrame, feature_set: str, openings: pd.DataFrame | None, covariates: pd.DataFrame | None) -> float:
    cutoffs = sorted(pd.to_datetime(train_rows["cutoff"]).unique())
    scores = {alpha: [] for alpha in ALPHAS}
    for cutoff in cutoffs:
        inner_train = train_rows[train_rows["window_end"].le(cutoff)].copy()
        validation = train_rows[train_rows["cutoff"].eq(cutoff)].copy()
        if inner_train.empty or validation.empty:
            continue
        x_train = feature_matrix(inner_train["ds"], feature_set, openings, covariates)
        y_train = inner_train["actual"] - inner_train["base_forecast"]
        x_val = feature_matrix(validation["ds"], feature_set, openings, covariates)
        for alpha in ALPHAS:
            model = fit_model(x_train, y_train, alpha)
            forecast = np.maximum(0.0, validation["base_forecast"].to_numpy() + model.predict(x_val))
            denom = validation["actual"].abs().sum()
            if denom:
                scores[alpha].append(float(np.abs(validation["actual"].to_numpy() - forecast).sum() / denom))
    mean_scores = {alpha: np.mean(values) for alpha, values in scores.items() if values}
    if not mean_scores:
        return 10.0
    return min(mean_scores, key=mean_scores.get)


def residual_predictions_for_rows(
    base_rows: pd.DataFrame,
    fold_cutoffs: list[pd.Timestamp],
    feature_sets: list[str],
    openings: pd.DataFrame | None,
    covariates: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    for cutoff in fold_cutoffs:
        train_rows = base_rows[base_rows["window_end"].le(cutoff)].copy()
        test_rows = base_rows[base_rows["cutoff"].eq(cutoff)].copy()
        if train_rows.empty or test_rows.empty:
            continue
        for feature_set in feature_sets:
            started = time.perf_counter()
            alpha = choose_alpha(train_rows, feature_set, openings, covariates)
            model = fit_model(
                feature_matrix(train_rows["ds"], feature_set, openings, covariates),
                train_rows["actual"] - train_rows["base_forecast"],
                alpha,
            )
            residual = model.predict(feature_matrix(test_rows["ds"], feature_set, openings, covariates))
            forecast = np.maximum(0.0, test_rows["base_forecast"].to_numpy() + residual)
            runtime = time.perf_counter() - started
            for row, pred in zip(test_rows.itertuples(index=False), forecast):
                record = row._asdict()
                record["forecast"] = float(pred)
                record["model_name"] = f"residual_ridge_{feature_set}"
                record["runtime_seconds"] = runtime
                record["alpha"] = alpha
                rows.append(record)
    return pd.DataFrame(rows)


def daily_base_rows(horizon: int, manifest: dict) -> pd.DataFrame:
    grid_path = Path(f"outputs_multifold_{horizon}d_validation") / "multifold_grid_predictions.csv"
    if not grid_path.exists():
        raise ValueError(f"{grid_path} is missing.")
    grid = pd.read_csv(grid_path, parse_dates=["cutoff", "window_end", "ds"])
    base = grid[grid["model_name"].eq(DAILY_CHAMPION)].copy()
    base = base.rename(columns={"forecast": "base_forecast"})
    base["base_model_name"] = DAILY_CHAMPION
    return base


def weekly_daily_base_rows(sales: pd.DataFrame, horizon: int, manifest: dict, weekly_freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = [pd.Timestamp(c) for c in manifest["grid_cutoffs"]]
    sales_lookup = sales.set_index("ds")["y"]
    daily_rows = []
    weekly_base_rows = []
    for cutoff in cutoffs:
        weekly_dates = pd.Series(pd.date_range(cutoff + pd.Timedelta(weeks=1), periods=horizon, freq=weekly_freq))
        daily_train = sales[sales["ds"].le(cutoff)].copy()
        horizon_days = daily_forecast_horizon_days(daily_train, weekly_dates)
        daily_dates = pd.Series(pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D"))
        base_forecast = seasonal_naive(daily_train, horizon_days, 365)
        actual = sales_lookup.reindex(daily_dates).to_numpy(dtype=float)
        if np.isnan(actual).any():
            continue
        adjusted_daily = pd.DataFrame({"ds": daily_dates, "actual": actual, "base_forecast": base_forecast})
        adjusted_daily["cutoff"] = cutoff
        adjusted_daily["window_end"] = weekly_dates.max()
        adjusted_daily["base_model_name"] = WEEKLY_CHAMPION
        adjusted_daily["horizon"] = horizon
        adjusted_daily["horizon_step"] = np.arange(1, len(adjusted_daily) + 1)
        daily_rows.append(adjusted_daily)

        actual_weekly = adjusted_daily.set_index("ds")["actual"].resample(weekly_freq).sum().reindex(weekly_dates).to_numpy()
        base_weekly = reaggregate_daily_forecast(daily_train, weekly_dates, base_forecast, weekly_freq)
        for step, (ds, actual_y, base_y) in enumerate(zip(weekly_dates, actual_weekly, base_weekly), start=1):
            weekly_base_rows.append(
                {
                    "unique_id": "total_sales",
                    "horizon": horizon,
                    "cutoff": cutoff,
                    "window_end": weekly_dates.max(),
                    "ds": ds,
                    "horizon_step": step,
                    "actual": float(actual_y),
                    "base_forecast": float(base_y),
                    "base_model_name": WEEKLY_CHAMPION,
                }
            )
    return pd.concat(daily_rows, ignore_index=True), pd.DataFrame(weekly_base_rows)


def weekly_predictions_from_daily(
    daily_base: pd.DataFrame,
    weekly_base: pd.DataFrame,
    fold_cutoffs: list[pd.Timestamp],
    feature_sets: list[str],
    openings: pd.DataFrame | None,
    covariates: pd.DataFrame | None,
    weekly_freq: str,
) -> pd.DataFrame:
    daily_preds = residual_predictions_for_rows(daily_base, fold_cutoffs, feature_sets, openings, covariates)
    weekly_rows = []
    for (cutoff, model_name), group in daily_preds.groupby(["cutoff", "model_name"]):
        weekly_forecast = group.set_index("ds")["forecast"].resample(weekly_freq).sum()
        base = weekly_base[weekly_base["cutoff"].eq(cutoff)].copy()
        alpha = group["alpha"].iloc[0]
        for row in base.itertuples(index=False):
            weekly_rows.append(
                {
                    "unique_id": "total_sales",
                    "horizon": row.horizon,
                    "cutoff": row.cutoff,
                    "window_end": row.window_end,
                    "ds": row.ds,
                    "horizon_step": row.horizon_step,
                    "actual": row.actual,
                    "model_name": model_name,
                    "forecast": float(max(0.0, weekly_forecast.loc[row.ds])),
                    "base_forecast": row.base_forecast,
                    "base_model_name": row.base_model_name,
                    "runtime_seconds": float(group["runtime_seconds"].sum()),
                    "alpha": alpha,
                }
            )
    return pd.DataFrame(weekly_rows)


def fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (cutoff, model_name), group in preds.groupby(["cutoff", "model_name"]):
        scores = score_group(group)
        frames.append({"fold_cutoff": cutoff, "model_name": model_name, "wape": scores["wape"], "bias_pct": scores["bias_pct"], "rows": int(scores["rows"])})
    return pd.DataFrame(frames)


def summary_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    summary = preds.groupby("model_name", group_keys=False).apply(score_group, include_groups=False).reset_index()
    keep = ["model_name", "rows", "wape", "bias_pct", "mae", "rmse", "runtime_seconds"]
    return summary[keep]


def write_run(run_dir: Path, preds: pd.DataFrame, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(run_dir / "multifold_grid_predictions.csv", index=False)
    fold_metrics(preds).to_csv(run_dir / "multifold_fold_metrics.csv", index=False)
    summary_metrics(preds).to_csv(run_dir / "multifold_summary.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_daily(horizon: int, output_dir: Path, feature_sets: list[str], openings: pd.DataFrame | None, covariates: pd.DataFrame | None) -> Path:
    source_manifest_path = Path(f"outputs_multifold_{horizon}d_validation") / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    base = daily_base_rows(horizon, manifest)
    fold_cutoffs = [pd.Timestamp(c) for c in manifest["fold_cutoffs"]]
    residual = residual_predictions_for_rows(base, fold_cutoffs, feature_sets, openings, covariates)
    champion = base[base["cutoff"].isin(fold_cutoffs)].copy()
    champion["model_name"] = DAILY_CHAMPION
    champion["forecast"] = champion["base_forecast"]
    champion["runtime_seconds"] = 0.0
    champion["alpha"] = np.nan
    preds = pd.concat([champion, residual], ignore_index=True)
    run_dir = output_dir / f"daily_{horizon}d"
    run_manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon,
        "horizons": [horizon],
        "grain": "daily",
        "grid_step": manifest.get("grid_step", 28),
        "models": sorted(preds["model_name"].unique().tolist()),
        "base_model": DAILY_CHAMPION,
        "feature_sets": feature_sets,
        "leakage_checklist": {
            "features_known_at_cutoff": True,
            "ridge_alpha_and_scaler_fit_inside_fold": True,
            "no_target_derived_covariates": True,
        },
    }
    write_run(run_dir, preds, run_manifest)
    return run_dir


def run_weekly(
    horizon: int,
    sales: pd.DataFrame,
    output_dir: Path,
    feature_sets: list[str],
    openings: pd.DataFrame | None,
    covariates: pd.DataFrame | None,
    weekly_freq: str,
) -> Path:
    source_manifest_path = Path(f"outputs_weekly_multifold_{horizon}w_daily_reagg_pool") / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    daily_base, weekly_base = weekly_daily_base_rows(sales, horizon, manifest, weekly_freq)
    fold_cutoffs = [pd.Timestamp(c) for c in manifest["fold_cutoffs"]]
    residual = weekly_predictions_from_daily(daily_base, weekly_base, fold_cutoffs, feature_sets, openings, covariates, weekly_freq)
    champion = weekly_base[weekly_base["cutoff"].isin(fold_cutoffs)].copy()
    champion["model_name"] = WEEKLY_CHAMPION
    champion["forecast"] = champion["base_forecast"]
    champion["runtime_seconds"] = 0.0
    champion["alpha"] = np.nan
    preds = pd.concat([champion, residual], ignore_index=True)
    run_dir = output_dir / f"weekly_{horizon}w"
    run_manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon,
        "horizons": [horizon],
        "grain": "weekly",
        "freq": weekly_freq,
        "grid_step": manifest.get("grid_step", 4),
        "models": sorted(preds["model_name"].unique().tolist()),
        "base_model": WEEKLY_CHAMPION,
        "feature_sets": feature_sets,
        "weekly_method": "daily residual ridge forecasts reaggregated to W-WED weeks",
        "leakage_checklist": {
            "features_known_at_cutoff": True,
            "ridge_alpha_and_scaler_fit_inside_fold": True,
            "no_target_derived_covariates": True,
        },
    }
    write_run(run_dir, preds, run_manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--opening-info", default="opening_info.csv")
    parser.add_argument("--covariate-csv")
    parser.add_argument("--daily-horizons", default="28,91,180,365")
    parser.add_argument("--weekly-horizons", default="4,12,26,52")
    parser.add_argument("--feature-sets", default="f1,f1f2")
    parser.add_argument("--weekly-freq", default=DEFAULT_WEEKLY_FREQ)
    parser.add_argument("--output-dir", default="outputs_residual_covariate_model")
    args = parser.parse_args()

    set_seeds()
    output_dir = Path(args.output_dir)
    openings = load_openings(Path(args.opening_info)) if args.opening_info else None
    covariates = load_covariates(Path(args.covariate_csv)) if args.covariate_csv else None
    feature_sets = [name.strip().lower() for name in args.feature_sets.split(",") if name.strip()]
    daily_horizons = [int(h.strip()) for h in args.daily_horizons.split(",") if h.strip()]
    weekly_horizons = [int(h.strip()) for h in args.weekly_horizons.split(",") if h.strip()]
    sales = load_sales(Path(args.sales_csv))

    run_dirs = []
    for horizon in daily_horizons:
        run_dirs.append(run_daily(horizon, output_dir, feature_sets, openings, covariates))
    for horizon in weekly_horizons:
        run_dirs.append(run_weekly(horizon, sales, output_dir, feature_sets, openings, covariates, args.weekly_freq))

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "daily_horizons": daily_horizons,
        "weekly_horizons": weekly_horizons,
        "feature_sets": feature_sets,
        "covariate_csv": args.covariate_csv,
        "run_dirs": [str(path) for path in run_dirs],
        "outputs": ["per-horizon multifold_grid_predictions.csv, multifold_fold_metrics.csv, multifold_summary.csv"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_residual_covariate_model.py", manifest)
    for run_dir in run_dirs:
        summary = pd.read_csv(run_dir / "multifold_summary.csv")
        print(f"\n{run_dir}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
