"""Backtest residual-quantile interval coverage for forecast grids."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from metrics import score_group


QUANTILE_LEVELS = [0.025, 0.10, 0.90, 0.975]


def seasonal_segment(ds: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ds)
    return np.where(dates.dt.month.isin([11, 12]), "holiday_weeks", "rest_weeks")


def parse_run(text: str) -> tuple[str, Path, str, str]:
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 4:
        raise ValueError("Run must be label|run_dir|model_name|mode, where mode is global, segment, or shrunken.")
    if parts[3] not in {"global", "segment", "shrunken"}:
        raise ValueError("mode must be global, segment, or shrunken.")
    return parts[0], Path(parts[1]), parts[2], parts[3]


def quantiles_for_group(calibration: pd.DataFrame, mode: str, segment: str | None = None, shrinkage_k: float = 30.0) -> np.ndarray:
    residuals = calibration["actual"].to_numpy(dtype=float) - calibration["forecast"].to_numpy(dtype=float)
    global_qs = np.quantile(residuals, QUANTILE_LEVELS)
    if mode == "global" or segment is None or "seasonal_segment" not in calibration:
        return global_qs
    segment_calibration = calibration[calibration["seasonal_segment"].eq(segment)]
    if segment_calibration.empty:
        return global_qs
    segment_residuals = segment_calibration["actual"].to_numpy(dtype=float) - segment_calibration["forecast"].to_numpy(dtype=float)
    segment_qs = np.quantile(segment_residuals, QUANTILE_LEVELS)
    if mode == "segment":
        return segment_qs
    if shrinkage_k < 0:
        raise ValueError("shrinkage_k must be non-negative.")
    weight = len(segment_calibration) / (len(segment_calibration) + shrinkage_k) if len(segment_calibration) + shrinkage_k else 1.0
    return weight * segment_qs + (1.0 - weight) * global_qs


def add_rolling_intervals(preds: pd.DataFrame, interval_windows: int, min_calibration_windows: int, mode: str, shrinkage_k: float = 30.0) -> pd.DataFrame:
    rows = []
    preds = preds.sort_values(["cutoff", "ds"]).copy()
    preds["seasonal_segment"] = seasonal_segment(preds["ds"])
    cutoffs = sorted(preds["cutoff"].unique())
    for idx, cutoff in enumerate(cutoffs):
        if idx < min_calibration_windows:
            continue
        cal_cutoffs = cutoffs[max(0, idx - interval_windows) : idx]
        calibration = preds[preds["cutoff"].isin(cal_cutoffs)].copy()
        current = preds[preds["cutoff"].eq(cutoff)].copy()
        if calibration.empty or current.empty:
            continue
        global_qs = quantiles_for_group(calibration, "global")
        for row in current.itertuples(index=False):
            if mode in {"segment", "shrunken"}:
                qs = quantiles_for_group(calibration, mode, row.seasonal_segment, shrinkage_k)
            else:
                qs = global_qs
            record = row._asdict()
            forecast = float(record["forecast"])
            record["p025"] = max(0.0, forecast + float(qs[0]))
            record["p10"] = max(0.0, forecast + float(qs[1]))
            record["p50"] = forecast
            record["p90"] = max(0.0, forecast + float(qs[2]))
            record["p975"] = max(0.0, forecast + float(qs[3]))
            record["interval_mode"] = mode
            record["interval_shrinkage_k"] = shrinkage_k if mode == "shrunken" else np.nan
            record["calibration_windows"] = len(cal_cutoffs)
            rows.append(record)
    return pd.DataFrame(rows)


def coverage_for_run(
    label: str,
    run_dir: Path,
    model_name: str,
    mode: str,
    interval_windows: int,
    min_calibration_windows: int,
    shrinkage_k: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_path = run_dir / "multifold_grid_predictions.csv"
    if not grid_path.exists():
        raise ValueError(f"Missing {grid_path}")
    preds = pd.read_csv(grid_path, parse_dates=["cutoff", "ds"])
    preds = preds[preds["model_name"].eq(model_name)].copy()
    if preds.empty:
        raise ValueError(f"{grid_path} has no rows for {model_name}.")
    intervals = add_rolling_intervals(preds, interval_windows, min_calibration_windows, mode, shrinkage_k)
    if intervals.empty:
        raise ValueError(f"No interval backtest rows for {label}; reduce --min-calibration-windows.")
    intervals.insert(0, "run_label", label)
    summary = score_group(intervals).to_frame().T
    summary.insert(0, "run_label", label)
    summary.insert(1, "model_name", model_name)
    summary.insert(2, "interval_mode", mode)
    summary.insert(3, "eval_cutoffs", intervals["cutoff"].nunique())
    summary.insert(4, "interval_windows", interval_windows)
    summary.insert(5, "interval_shrinkage_k", shrinkage_k if mode == "shrunken" else np.nan)
    by_segment = intervals.groupby("seasonal_segment", group_keys=False).apply(score_group, include_groups=False).reset_index()
    by_segment.insert(0, "run_label", label)
    by_segment.insert(1, "model_name", model_name)
    by_segment.insert(2, "interval_mode", mode)
    by_segment.insert(3, "interval_shrinkage_k", shrinkage_k if mode == "shrunken" else np.nan)
    return summary, by_segment


def write_report(summary: pd.DataFrame, by_segment: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Interval Coverage Backtest",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Run | Model | Mode | Eval cutoffs | Rows | Coverage 80 | Coverage 95 | Avg width 80 | Avg width 95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.run_label} | {row.model_name} | {row.interval_mode} | {row.eval_cutoffs} | {int(row.rows)} | "
            f"{row.coverage_80:.3f} | {row.coverage_95:.3f} | {row.avg_width_80:.0f} | {row.avg_width_95:.0f} |"
        )
    lines.extend(["", "## By Segment", ""])
    lines.append("| Run | Segment | Mode | Rows | Coverage 80 | Coverage 95 | Avg width 80 | Avg width 95 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in by_segment.itertuples(index=False):
        lines.append(
            f"| {row.run_label} | {row.seasonal_segment} | {row.interval_mode} | {int(row.rows)} | "
            f"{row.coverage_80:.3f} | {row.coverage_95:.3f} | {row.avg_width_80:.0f} | {row.avg_width_95:.0f} |"
        )
    (output_dir / "interval_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Run as label|run_dir|model_name|global_or_segment.")
    parser.add_argument("--interval-windows", type=int, default=8)
    parser.add_argument("--min-calibration-windows", type=int, default=4)
    parser.add_argument("--shrinkage-k", type=float, default=30.0)
    parser.add_argument("--output-dir", default="outputs_interval_coverage_backtest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    segment_summaries = []
    for run_text in args.run:
        label, run_dir, model_name, mode = parse_run(run_text)
        summary, by_segment = coverage_for_run(label, run_dir, model_name, mode, args.interval_windows, args.min_calibration_windows, args.shrinkage_k)
        summaries.append(summary)
        segment_summaries.append(by_segment)
    summary_df = pd.concat(summaries, ignore_index=True)
    segment_df = pd.concat(segment_summaries, ignore_index=True)
    summary_df.to_csv(output_dir / "interval_coverage_summary.csv", index=False)
    segment_df.to_csv(output_dir / "interval_coverage_by_segment.csv", index=False)
    write_report(summary_df, segment_df, output_dir)
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": args.run,
        "interval_windows": args.interval_windows,
        "min_calibration_windows": args.min_calibration_windows,
        "shrinkage_k": args.shrinkage_k,
        "outputs": ["interval_coverage_summary.csv", "interval_coverage_by_segment.csv", "interval_coverage_report.md"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_interval_coverage_backtest.py", manifest)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
