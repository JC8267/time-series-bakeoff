"""Leak-free cumulative-error interval calibration for long daily forecasts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment


QUANTILES = (0.025, 0.10, 0.50, 0.90, 0.975)
WINDOWS = {"recent8": 8, "all_prior": None}


def parse_run(text: str) -> tuple[str, Path, str]:
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError("Run must be label|run_dir|model_name.")
    return parts[0], Path(parts[1]), parts[2]


def _select_unique_id(preds: pd.DataFrame) -> pd.DataFrame:
    if "unique_id" not in preds or preds["unique_id"].nunique() <= 1:
        return preds
    if "total_sales" in set(preds["unique_id"].astype(str)):
        return preds[preds["unique_id"].astype(str).eq("total_sales")].copy()
    raise ValueError("Grid contains multiple unique_id values; use a total_sales grid.")


def build_cumulative_paths(preds: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Return one cumulative actual/forecast/error path per cutoff."""
    required = {"cutoff", "ds", "horizon_step", "actual", "forecast"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"Grid is missing columns: {sorted(missing)}")
    preds = _select_unique_id(preds.copy())
    if horizon is None:
        horizons = sorted(preds["horizon"].dropna().astype(int).unique()) if "horizon" in preds else []
        if len(horizons) != 1:
            raise ValueError("Specify horizon when the grid contains multiple horizons.")
        horizon = horizons[0]
    if "horizon" in preds:
        preds = preds[preds["horizon"].astype(int).eq(horizon)].copy()
    if preds.empty:
        raise ValueError(f"No grid rows for horizon {horizon}.")

    preds["cutoff"] = pd.to_datetime(preds["cutoff"])
    preds["ds"] = pd.to_datetime(preds["ds"])
    preds["horizon_step"] = pd.to_numeric(preds["horizon_step"], errors="raise").astype(int)
    preds["actual"] = pd.to_numeric(preds["actual"], errors="raise")
    preds["forecast"] = pd.to_numeric(preds["forecast"], errors="raise")
    rows: list[pd.DataFrame] = []
    expected_steps = set(range(1, horizon + 1))
    for cutoff, group in preds.groupby("cutoff", sort=True):
        steps = set(group["horizon_step"])
        if steps != expected_steps or len(group) != horizon:
            raise ValueError(f"Cutoff {cutoff.date()} does not contain exactly steps 1..{horizon}.")
        group = group.sort_values("horizon_step").copy()
        group["actual_cumulative"] = group["actual"].cumsum()
        group["forecast_cumulative"] = group["forecast"].cumsum()
        group["cumulative_error"] = group["actual_cumulative"] - group["forecast_cumulative"]
        group["cumulative_relative_error"] = np.divide(
            group["cumulative_error"],
            group["forecast_cumulative"],
            out=np.full(len(group), np.nan, dtype=float),
            where=np.abs(group["forecast_cumulative"].to_numpy(dtype=float)) > 1e-12,
        )
        rows.append(
            group[
                [
                    "cutoff",
                    "ds",
                    "horizon_step",
                    "actual_cumulative",
                    "forecast_cumulative",
                    "cumulative_error",
                    "cumulative_relative_error",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["cutoff", "horizon_step"]).reset_index(drop=True)


def _interval(forecast: float, quantiles: np.ndarray, method: str) -> list[float]:
    if method == "additive":
        values = forecast + quantiles
    elif method == "relative":
        values = forecast * (1.0 + quantiles)
    else:
        raise ValueError(f"Unknown method: {method}")
    return [max(0.0, float(value)) for value in values]


def rolling_intervals(paths: pd.DataFrame, min_calibration_windows: int = 4) -> pd.DataFrame:
    """Apply observable prior-cutoff cumulative residual quantiles at every step."""
    if min_calibration_windows < 1:
        raise ValueError("min_calibration_windows must be positive.")
    cutoffs = sorted(paths["cutoff"].drop_duplicates())
    rows: list[dict[str, object]] = []
    for idx, cutoff in enumerate(cutoffs):
        prior = cutoffs[:idx]
        current = paths[paths["cutoff"].eq(cutoff)]
        for current_row in current.itertuples(index=False):
            eligible = paths[
                paths["cutoff"].isin(prior)
                & paths["horizon_step"].eq(current_row.horizon_step)
                & paths["ds"].le(cutoff)
            ].sort_values("cutoff")
            for window_name, window_size in WINDOWS.items():
                for method, error_column in (("additive", "cumulative_error"), ("relative", "cumulative_relative_error")):
                    calibration = eligible.dropna(subset=[error_column])
                    if window_size is not None:
                        calibration = calibration.tail(window_size)
                    values = calibration[error_column]
                    if len(values) < min_calibration_windows:
                        continue
                    quantiles = np.quantile(values.to_numpy(dtype=float), QUANTILES)
                    interval = _interval(float(current_row.forecast_cumulative), quantiles, method)
                    rows.append(
                        {
                            "cutoff": cutoff,
                            "ds": current_row.ds,
                            "horizon_step": current_row.horizon_step,
                            "actual_cumulative": current_row.actual_cumulative,
                            "forecast_cumulative": current_row.forecast_cumulative,
                            "cumulative_error": current_row.cumulative_error,
                            "cumulative_relative_error": current_row.cumulative_relative_error,
                            "calibration_method": method,
                            "calibration_window": window_name,
                            "calibration_count": len(values),
                            "calibration_cutoff_start": calibration["cutoff"].min(),
                            "calibration_cutoff_end": calibration["cutoff"].max(),
                            "p025": interval[0],
                            "p10": interval[1],
                            "p50": interval[2],
                            "p90": interval[3],
                            "p975": interval[4],
                            "covered_80": interval[1] <= current_row.actual_cumulative <= interval[3],
                            "covered_95": interval[0] <= current_row.actual_cumulative <= interval[4],
                            "width_80": interval[3] - interval[1],
                            "width_95": interval[4] - interval[0],
                        }
                    )
    if not rows:
        raise ValueError("No rolling intervals produced; reduce min_calibration_windows.")
    return pd.DataFrame(rows)


def summarize_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["calibration_method", "calibration_window", "horizon_step"]
    summary = (
        intervals.groupby(group_columns, as_index=False)
        .agg(
            rows=("cutoff", "size"),
            eval_cutoffs=("cutoff", "nunique"),
            calibration_count_min=("calibration_count", "min"),
            calibration_count_max=("calibration_count", "max"),
            coverage_80=("covered_80", "mean"),
            coverage_95=("covered_95", "mean"),
            avg_width_80=("width_80", "mean"),
            avg_width_95=("width_95", "mean"),
            mean_cumulative_error=("cumulative_error", "mean"),
            mean_cumulative_relative_error=("cumulative_relative_error", "mean"),
        )
        .sort_values(group_columns)
    )
    return summary.reset_index(drop=True)


def load_run(label: str, run_dir: Path, model_name: str, horizon: int | None = None) -> tuple[pd.DataFrame, int]:
    grid_path = run_dir / "multifold_grid_predictions.csv"
    if not grid_path.exists():
        raise ValueError(f"Missing {grid_path}")
    preds = pd.read_csv(grid_path)
    preds = preds[preds["model_name"].eq(model_name)].copy()
    if preds.empty:
        raise ValueError(f"{grid_path} has no rows for {model_name}.")
    paths = build_cumulative_paths(preds, horizon)
    paths.insert(0, "run_label", label)
    paths.insert(1, "model_name", model_name)
    paths.insert(2, "horizon", horizon or int(preds["horizon"].iloc[0]))
    return paths, int(paths["horizon"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Run as label|run_dir|model_name.")
    parser.add_argument("--min-calibration-windows", type=int, default=4)
    parser.add_argument("--horizon", type=int, help="Use when a source grid contains multiple horizons.")
    parser.add_argument("--output-dir", default="outputs_cumulative_error_calibration")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_frames: list[pd.DataFrame] = []
    run_manifest = []
    for run_text in args.run:
        label, run_dir, model_name = parse_run(run_text)
        paths, horizon = load_run(label, run_dir, model_name, args.horizon)
        path_frames.append(paths)
        run_manifest.append({"label": label, "run_dir": str(run_dir), "model_name": model_name, "horizon": horizon, "cutoffs": int(paths["cutoff"].nunique())})

    paths = pd.concat(path_frames, ignore_index=True)
    intervals = []
    for label, group in paths.groupby("run_label", sort=False):
        result = rolling_intervals(group, args.min_calibration_windows)
        result.insert(0, "run_label", label)
        result.insert(1, "model_name", group["model_name"].iloc[0])
        result.insert(2, "horizon", group["horizon"].iloc[0])
        intervals.append(result)
    interval_df = pd.concat(intervals, ignore_index=True)
    summaries = []
    for (label, model_name, horizon), group in interval_df.groupby(["run_label", "model_name", "horizon"], sort=False):
        current_summary = summarize_intervals(group)
        current_summary.insert(0, "run_label", label)
        current_summary.insert(1, "model_name", model_name)
        current_summary.insert(2, "horizon", horizon)
        summaries.append(current_summary)
    summary = pd.concat(summaries, ignore_index=True)
    terminal = summary[summary["horizon_step"].eq(summary["horizon"])].copy()

    paths.to_csv(output_dir / "cumulative_error_paths.csv", index=False)
    interval_df.to_csv(output_dir / "cumulative_interval_paths.csv", index=False)
    summary.to_csv(output_dir / "cumulative_interval_summary.csv", index=False)
    terminal.to_csv(output_dir / "terminal_interval_summary.csv", index=False)
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": run_manifest,
        "horizons": sorted({int(value) for value in paths["horizon"].unique()}),
        "models": sorted(paths["model_name"].unique()),
        "min_calibration_windows": args.min_calibration_windows,
        "calibration_windows": list(WINDOWS),
        "methods": ["additive", "relative"],
        "quantiles": list(QUANTILES),
        "outputs": [
            "cumulative_error_paths.csv",
            "cumulative_interval_paths.csv",
            "cumulative_interval_summary.csv",
            "terminal_interval_summary.csv",
        ],
        "notes": [
            "Each evaluation row uses only prior-cutoff residuals whose ds is on or before the evaluation cutoff.",
            "Recent-window selection and minimum-count enforcement happen after the residual availability filter.",
            "Cutoffs are overlapping rolling windows and are not independent annual observations.",
            "Relative error is cumulative_error / forecast_cumulative.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    append_experiment(output_dir, "calibrate_cumulative_error_paths.py", manifest)
    print(terminal.to_string(index=False))


if __name__ == "__main__":
    main()
