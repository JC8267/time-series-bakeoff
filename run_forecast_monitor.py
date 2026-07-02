"""Monitor realized production forecasts against refreshed actuals."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from metrics import score_group
from run_bakeoff import DEFAULT_WEEKLY_FREQ, load_sales, set_seeds


LOG_COLUMNS = [
    "logged_at",
    "source_run_id",
    "source_forecast_csv",
    "row_type",
    "model_name",
    "comparison_model",
    "evaluation_window",
    "rows",
    "wape",
    "bias_pct",
    "coverage_80",
    "coverage_95",
    "wape_diff_vs_final",
    "bias_diff_vs_final",
    "verdict",
]


def parse_model_horizon(model_name: str) -> tuple[int | None, str | None]:
    match = re.search(r"_(\d+)([dw])_", model_name)
    if not match:
        return None, None
    return int(match.group(1)), "weekly" if match.group(2) == "w" else "daily"


def infer_grain(forecast: pd.DataFrame) -> str:
    if "weekly_segment" in forecast.columns or forecast["model_name"].astype(str).str.contains(r"_\d+w_").any():
        return "weekly"
    return "daily"


def elapsed_bucket(ds: pd.Timestamp, training_cutoff: pd.Timestamp | None, grain: str) -> str:
    if training_cutoff is None or pd.isna(training_cutoff):
        return "realized"
    days = max(1, int((pd.Timestamp(ds) - pd.Timestamp(training_cutoff)).days))
    if grain == "weekly":
        weeks = max(1, round(days / 7))
        for limit in (4, 12, 26, 52):
            if weeks <= limit:
                return f"<= {limit}w"
        return "> 52w"
    for limit in (28, 91, 180, 365):
        if days <= limit:
            return f"<= {limit}d"
    return "> 365d"


def load_manifest(forecast_csv: Path) -> dict[str, Any]:
    manifest_path = forecast_csv.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def realized_forecasts(forecast_csv: Path, sales_csv: Path, weekly_freq: str = DEFAULT_WEEKLY_FREQ) -> tuple[pd.DataFrame, dict[str, Any], str]:
    forecast = pd.read_csv(forecast_csv)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    if "training_cutoff" in forecast:
        forecast["training_cutoff"] = pd.to_datetime(forecast["training_cutoff"])
    grain = infer_grain(forecast)
    actuals = load_sales(sales_csv, grain=grain, weekly_freq=weekly_freq)[["ds", "y"]].rename(columns={"y": "actual"})
    realized = forecast.merge(actuals, on="ds", how="inner")
    realized = realized[realized["ds"].le(actuals["ds"].max())].copy()
    manifest = load_manifest(forecast_csv)
    return realized, manifest, grain


def model_summary_rows(realized: pd.DataFrame, source_run_id: str, forecast_csv: Path, grain: str) -> list[dict[str, Any]]:
    rows = []
    realized = realized.copy()
    realized["evaluation_window"] = [
        elapsed_bucket(row.ds, row.training_cutoff if "training_cutoff" in realized else None, grain)
        for row in realized.itertuples(index=False)
    ]
    for (model_name, window), group in realized.groupby(["model_name", "evaluation_window"], dropna=False):
        scores = score_group(group)
        rows.append(
            {
                "source_run_id": source_run_id,
                "source_forecast_csv": str(forecast_csv),
                "row_type": "model",
                "model_name": model_name,
                "comparison_model": "",
                "evaluation_window": window,
                "rows": int(scores["rows"]),
                "wape": float(scores["wape"]),
                "bias_pct": float(scores["bias_pct"]),
                "coverage_80": float(scores["coverage_80"]) if pd.notna(scores["coverage_80"]) else np.nan,
                "coverage_95": float(scores["coverage_95"]) if pd.notna(scores["coverage_95"]) else np.nan,
                "wape_diff_vs_final": np.nan,
                "bias_diff_vs_final": np.nan,
                "verdict": "",
            }
        )
    return rows


def final_model_for_challenger(model_name: str, model_names: set[str]) -> str | None:
    horizon, grain = parse_model_horizon(model_name)
    if horizon is None or grain is None:
        return None
    unit = "w" if grain == "weekly" else "d"
    prefix = f"final_{horizon}{unit}_"
    matches = [name for name in model_names if name.startswith(prefix)]
    return matches[0] if matches else None


def challenger_rows(realized: pd.DataFrame, source_run_id: str, forecast_csv: Path, grain: str) -> list[dict[str, Any]]:
    rows = []
    model_names = set(realized["model_name"].astype(str))
    challengers = sorted(name for name in model_names if name.startswith("challenger_"))
    for challenger in challengers:
        final_model = final_model_for_challenger(challenger, model_names)
        if final_model is None:
            continue
        pair = realized[realized["model_name"].isin([final_model, challenger])].copy()
        pair["evaluation_window"] = [
            elapsed_bucket(row.ds, row.training_cutoff if "training_cutoff" in pair else None, grain)
            for row in pair.itertuples(index=False)
        ]
        for window, group in pair.groupby("evaluation_window"):
            wide = group.pivot_table(index=["ds", "actual"], columns="model_name", values="forecast").dropna().reset_index()
            if wide.empty:
                continue
            final_group = pd.DataFrame({"actual": wide["actual"], "forecast": wide[final_model]})
            challenger_group = pd.DataFrame({"actual": wide["actual"], "forecast": wide[challenger]})
            final_scores = score_group(final_group)
            challenger_scores = score_group(challenger_group)
            wape_diff = float(challenger_scores["wape"] - final_scores["wape"])
            bias_diff = float(challenger_scores["bias_pct"] - final_scores["bias_pct"])
            verdict = "challenger_better" if wape_diff < 0 else "final_better" if wape_diff > 0 else "tie"
            rows.append(
                {
                    "source_run_id": source_run_id,
                    "source_forecast_csv": str(forecast_csv),
                    "row_type": "adjudication",
                    "model_name": challenger,
                    "comparison_model": final_model,
                    "evaluation_window": window,
                    "rows": int(challenger_scores["rows"]),
                    "wape": float(challenger_scores["wape"]),
                    "bias_pct": float(challenger_scores["bias_pct"]),
                    "coverage_80": np.nan,
                    "coverage_95": np.nan,
                    "wape_diff_vs_final": wape_diff,
                    "bias_diff_vs_final": bias_diff,
                    "verdict": verdict,
                }
            )
    return rows


def append_monitor_rows(log_csv: Path, rows: list[dict[str, Any]]) -> int:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(log_csv) if log_csv.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    key_cols = ["source_run_id", "row_type", "model_name", "comparison_model", "evaluation_window"]
    existing_keys = set(map(tuple, existing[key_cols].fillna("").astype(str).to_numpy())) if len(existing) else set()
    now = datetime.now(timezone.utc).isoformat()
    new_rows = []
    for row in rows:
        record = {col: row.get(col, "") for col in LOG_COLUMNS}
        record["logged_at"] = now
        key = tuple(str(record[col]) for col in key_cols)
        if key not in existing_keys:
            new_rows.append(record)
            existing_keys.add(key)
    if not new_rows:
        return 0
    new_df = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
    out = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(log_csv, index=False)
    return len(new_rows)


def grid_model_for_final(model_name: str) -> str | None:
    horizon, grain = parse_model_horizon(model_name)
    if horizon is None or grain is None:
        return None
    if grain == "weekly":
        return "daily_reagg_seasonal_naive_lag_365"
    return "seasonal_naive_lag_364"


def grid_path_for_final(model_name: str) -> Path | None:
    horizon, grain = parse_model_horizon(model_name)
    if horizon is None or grain is None:
        return None
    if grain == "weekly":
        return Path(f"outputs_weekly_multifold_{horizon}w_daily_reagg_pool") / "multifold_grid_predictions.csv"
    return Path(f"outputs_multifold_{horizon}d_validation") / "multifold_grid_predictions.csv"


def mean_abs_backtest_bias(model_name: str) -> float | None:
    grid_path = grid_path_for_final(model_name)
    grid_model = grid_model_for_final(model_name)
    if grid_path is None or grid_model is None or not grid_path.exists():
        return None
    preds = pd.read_csv(grid_path, parse_dates=["cutoff"])
    preds = preds[preds["model_name"].eq(grid_model)].copy()
    if preds.empty:
        return None
    bias = preds.groupby("cutoff", group_keys=False).apply(score_group, include_groups=False)["bias_pct"].abs()
    return float(bias.mean())


def cusum_series(bias_pct: pd.Series, slack: float) -> pd.DataFrame:
    if slack <= 0 or not np.isfinite(slack):
        slack = float(bias_pct.abs().mean()) or 0.001
    pos = 0.0
    neg = 0.0
    rows = []
    for idx, value in bias_pct.items():
        pos = max(0.0, pos + (float(value) - slack))
        neg = max(0.0, neg + (-float(value) - slack))
        rows.append({"ds": idx, "positive_cusum": pos, "negative_cusum": neg})
    return pd.DataFrame(rows)


def write_alarm_state(realized: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    states = []
    for model_name, group in realized[realized["model_name"].astype(str).str.startswith("final_")].groupby("model_name"):
        row_bias = group.sort_values("ds").set_index("ds").assign(row_bias_pct=lambda x: (x["forecast"] - x["actual"]) / x["actual"].replace(0, np.nan))["row_bias_pct"].dropna()
        if row_bias.empty:
            continue
        backtest_bias = mean_abs_backtest_bias(model_name)
        slack = 0.5 * backtest_bias if backtest_bias is not None else 0.5 * float(row_bias.abs().mean())
        series = cusum_series(row_bias, slack)
        threshold = 4.0 * slack
        alarm = bool(((series["positive_cusum"] > threshold) | (series["negative_cusum"] > threshold)).any())
        states.append(
            {
                "model_name": model_name,
                "slack": slack,
                "threshold": threshold,
                "backtest_mean_abs_bias_pct": backtest_bias,
                "positive_cusum": float(series["positive_cusum"].iloc[-1]),
                "negative_cusum": float(series["negative_cusum"].iloc[-1]),
                "alarm": alarm,
            }
        )
    state = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "constants": {"slack_multiplier": 0.5, "threshold_multiplier": 4.0},
        "models": states,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alarm_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-csv", required=True)
    parser.add_argument("--sales-csv", required=True)
    parser.add_argument("--log-csv", default="outputs_forecast_monitor/monitoring_log.csv")
    parser.add_argument("--weekly-freq", default=DEFAULT_WEEKLY_FREQ)
    args = parser.parse_args()

    set_seeds()
    forecast_csv = Path(args.forecast_csv)
    log_csv = Path(args.log_csv)
    realized, manifest, grain = realized_forecasts(forecast_csv, Path(args.sales_csv), args.weekly_freq)
    source_run_id = str(manifest.get("run_id") or forecast_csv.parent.name)
    if realized.empty:
        raise ValueError("No forecast rows are realized in the provided sales file.")

    rows = model_summary_rows(realized, source_run_id, forecast_csv, grain)
    rows.extend(challenger_rows(realized, source_run_id, forecast_csv, grain))
    appended = append_monitor_rows(log_csv, rows)
    alarm_state = write_alarm_state(realized, log_csv.parent)

    run_manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "forecast_csv": str(forecast_csv),
        "sales_csv": args.sales_csv,
        "source_run_id": source_run_id,
        "grain": grain,
        "realized_rows": int(len(realized)),
        "rows_appended": appended,
        "alarm_constants": alarm_state["constants"],
        "outputs": [str(log_csv), str(log_csv.parent / "alarm_state.json")],
    }
    (log_csv.parent / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    append_experiment(log_csv.parent, "run_forecast_monitor.py", run_manifest)

    print(f"Realized rows: {len(realized)}")
    print(f"Rows appended: {appended}")
    if any(model["alarm"] for model in alarm_state["models"]):
        print("WARNING [monitor]: bias CUSUM alarm fired. See alarm_state.json.")


if __name__ == "__main__":
    main()
