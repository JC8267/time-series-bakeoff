"""Evaluate historical counterfactuals from paired backtest prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_predictions(path: Path, model_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["cutoff", "ds"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    df = df[(df["model_name"] == model_name) & (df["ds"].between(start, end))].copy()
    if df.empty:
        raise ValueError(f"No rows found in {path} for {model_name} between {start_date} and {end_date}.")
    latest_cutoff = df["cutoff"].max()
    return df[df["cutoff"] == latest_cutoff].copy()


def calibrate(actual_path: Path, counterfactual_path: Path, model_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    actual_model = load_predictions(actual_path, model_name, start_date, end_date)
    counterfactual = load_predictions(counterfactual_path, model_name, start_date, end_date)
    merged = actual_model[["ds", "actual", "forecast"]].rename(columns={"forecast": "actual_scenario_forecast"}).merge(
        counterfactual[["ds", "forecast"]].rename(columns={"forecast": "counterfactual_forecast"}),
        on="ds",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping dates in paired backtest predictions.")
    merged["observed_incremental_vs_counterfactual"] = merged["actual"] - merged["counterfactual_forecast"]
    merged["model_incremental_scenario_delta"] = merged["actual_scenario_forecast"] - merged["counterfactual_forecast"]
    merged["scenario_delta_recovery_error"] = merged["model_incremental_scenario_delta"] - merged["observed_incremental_vs_counterfactual"]
    return merged


def write_outputs(daily: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "historical_counterfactual_daily.csv", index=False)
    summary = {
        "model_name": model_name,
        "start": str(daily["ds"].min().date()),
        "end": str(daily["ds"].max().date()),
        "rows": int(len(daily)),
        "actual_sum": float(daily["actual"].sum()),
        "counterfactual_sum": float(daily["counterfactual_forecast"].sum()),
        "observed_incremental_sum": float(daily["observed_incremental_vs_counterfactual"].sum()),
        "model_incremental_sum": float(daily["model_incremental_scenario_delta"].sum()),
        "recovery_error_sum": float(daily["scenario_delta_recovery_error"].sum()),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "historical_counterfactual_summary.csv", index=False)
    lines = [
        "# Historical Counterfactual Calibration",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Actual sales | {summary['actual_sum']:,.0f} |",
        f"| Counterfactual forecast | {summary['counterfactual_sum']:,.0f} |",
        f"| Observed incremental vs counterfactual | {summary['observed_incremental_sum']:,.0f} |",
        f"| Model scenario delta | {summary['model_incremental_sum']:,.0f} |",
        f"| Recovery error | {summary['recovery_error_sum']:,.0f} |",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-backtest", required=True)
    parser.add_argument("--counterfactual-backtest", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default="outputs_counterfactual_calibration")
    args = parser.parse_args()

    daily = calibrate(Path(args.actual_backtest), Path(args.counterfactual_backtest), args.model_name, args.start_date, args.end_date)
    write_outputs(daily, Path(args.output_dir), args.model_name)
    print(pd.read_csv(Path(args.output_dir) / "historical_counterfactual_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()

