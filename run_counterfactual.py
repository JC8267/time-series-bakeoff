"""Counterfactual forecast comparison utility.

Compares an actual/planned scenario forecast against a no-event scenario
forecast and writes incremental sales summaries.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_forecast(path: Path, scenario_name: str, model_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    if model_filter:
        df = df[df["model_name"] == model_filter].copy()
    if df.empty:
        raise ValueError(f"No forecast rows found in {path} for model filter {model_filter!r}.")
    needed = {"ds", "model_name", "forecast"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    keep = ["ds", "model_name", "forecast"] + [c for c in ["p025", "p10", "p50", "p90", "p975"] if c in df.columns]
    out = df[keep].copy()
    out["scenario"] = scenario_name
    return out


def compare_scenarios(actual: pd.DataFrame, counterfactual: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["ds"]
    if actual["model_name"].nunique() == 1 and counterfactual["model_name"].nunique() == 1:
        actual_model = actual["model_name"].iloc[0]
        counterfactual_model = counterfactual["model_name"].iloc[0]
        left = actual.rename(columns={"model_name": "actual_model", "forecast": "actual_forecast"})
        right = counterfactual.rename(columns={"model_name": "counterfactual_model", "forecast": "counterfactual_forecast"})
        merged = left.merge(right, on=join_cols, how="inner", suffixes=("_actual", "_counterfactual"))
        merged["actual_model"] = actual_model
        merged["counterfactual_model"] = counterfactual_model
    else:
        left = actual.rename(columns={"forecast": "actual_forecast"})
        right = counterfactual.rename(columns={"forecast": "counterfactual_forecast"})
        merged = left.merge(right, on=["ds", "model_name"], how="inner", suffixes=("_actual", "_counterfactual"))
        merged["actual_model"] = merged["model_name"]
        merged["counterfactual_model"] = merged["model_name"]

    if merged.empty:
        raise ValueError("The two forecast files have no overlapping forecast dates/models.")
    merged["incremental_sales"] = merged["actual_forecast"] - merged["counterfactual_forecast"]
    merged["incremental_lift_pct"] = merged["incremental_sales"] / merged["counterfactual_forecast"].replace(0, pd.NA)

    interval_pairs = [
        ("p025_actual", "p975_counterfactual", "incremental_lower_95"),
        ("p975_actual", "p025_counterfactual", "incremental_upper_95"),
        ("p10_actual", "p90_counterfactual", "incremental_lower_80"),
        ("p90_actual", "p10_counterfactual", "incremental_upper_80"),
    ]
    for left_col, right_col, out_col in interval_pairs:
        if left_col in merged.columns and right_col in merged.columns:
            merged[out_col] = merged[left_col] - merged[right_col]
    return merged


def write_outputs(daily: pd.DataFrame, output_dir: Path, actual_path: Path, counterfactual_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "counterfactual_daily_incrementality.csv", index=False)
    try:
        daily.to_parquet(output_dir / "counterfactual_daily_incrementality.parquet", index=False)
    except Exception:
        pass

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actual_forecast_file": str(actual_path),
        "counterfactual_forecast_file": str(counterfactual_path),
        "start": str(daily["ds"].min().date()),
        "end": str(daily["ds"].max().date()),
        "rows": int(len(daily)),
        "actual_forecast_sum": float(daily["actual_forecast"].sum()),
        "counterfactual_forecast_sum": float(daily["counterfactual_forecast"].sum()),
        "incremental_sales_sum": float(daily["incremental_sales"].sum()),
        "incremental_lift_pct": float(daily["incremental_sales"].sum() / daily["counterfactual_forecast"].sum()) if daily["counterfactual_forecast"].sum() else None,
    }
    for col in ["incremental_lower_80", "incremental_upper_80", "incremental_lower_95", "incremental_upper_95"]:
        if col in daily.columns:
            summary[f"{col}_sum"] = float(daily[col].sum())

    pd.DataFrame([summary]).to_csv(output_dir / "counterfactual_summary.csv", index=False)
    (output_dir / "counterfactual_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Counterfactual Incrementality Summary",
        "",
        f"Created at: {summary['created_at']}",
        f"Date range: {summary['start']} to {summary['end']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Actual/planned scenario forecast | {summary['actual_forecast_sum']:,.0f} |",
        f"| Counterfactual forecast | {summary['counterfactual_forecast_sum']:,.0f} |",
        f"| Incremental sales | {summary['incremental_sales_sum']:,.0f} |",
        f"| Incremental lift | {summary['incremental_lift_pct']:.2%} |" if summary["incremental_lift_pct"] is not None else "| Incremental lift | n/a |",
    ]
    if "incremental_lower_80_sum" in summary:
        lines.append(f"| Incremental lower 80 | {summary['incremental_lower_80_sum']:,.0f} |")
        lines.append(f"| Incremental upper 80 | {summary['incremental_upper_80_sum']:,.0f} |")
    if "incremental_lower_95_sum" in summary:
        lines.append(f"| Incremental lower 95 | {summary['incremental_lower_95_sum']:,.0f} |")
        lines.append(f"| Incremental upper 95 | {summary['incremental_upper_95_sum']:,.0f} |")
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-forecast", required=True, help="Forecast CSV for the actual/planned scenario.")
    parser.add_argument("--counterfactual-forecast", required=True, help="Forecast CSV for the no-event scenario.")
    parser.add_argument("--output-dir", default="outputs_counterfactual")
    parser.add_argument("--actual-model", default=None, help="Optional model_name filter for actual/planned forecast.")
    parser.add_argument("--counterfactual-model", default=None, help="Optional model_name filter for counterfactual forecast.")
    args = parser.parse_args()

    actual_path = Path(args.actual_forecast)
    counterfactual_path = Path(args.counterfactual_forecast)
    actual = load_forecast(actual_path, "actual", args.actual_model)
    counterfactual = load_forecast(counterfactual_path, "counterfactual", args.counterfactual_model)
    daily = compare_scenarios(actual, counterfactual)
    write_outputs(daily, Path(args.output_dir), actual_path, counterfactual_path)

    summary = pd.read_csv(Path(args.output_dir) / "counterfactual_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

