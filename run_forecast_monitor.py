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
    "horizon",
    "grain",
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
    if "grain" in forecast:
        grains = set(forecast["grain"].dropna().astype(str).str.lower())
        if len(grains) == 1:
            grain = next(iter(grains))
            if grain in {"daily", "weekly"}:
                return grain
        if grains:
            raise ValueError(f"Forecast contains unsupported or mixed grains: {sorted(grains)}")
    if "weekly_segment" in forecast.columns or forecast["model_name"].astype(str).str.contains(r"_\d+w_").any():
        return "weekly"
    return "daily"


def horizon_limits(forecast: pd.DataFrame, grain: str) -> list[int]:
    if "horizon" in forecast:
        limits = sorted({int(value) for value in pd.to_numeric(forecast["horizon"], errors="coerce").dropna() if int(value) > 0})
        if limits:
            return limits
    return [4, 12, 26, 52] if grain == "weekly" else [28, 91, 180, 365]


def elapsed_bucket(
    ds: pd.Timestamp,
    training_cutoff: pd.Timestamp | None,
    grain: str,
    limits: list[int] | None = None,
    horizon_step: int | None = None,
) -> str:
    limits = limits or ([4, 12, 26, 52] if grain == "weekly" else [28, 91, 180, 365])
    if horizon_step is not None and pd.notna(horizon_step):
        elapsed = max(1, int(horizon_step))
    elif training_cutoff is None or pd.isna(training_cutoff):
        return "realized"
    else:
        days = max(1, int((pd.Timestamp(ds) - pd.Timestamp(training_cutoff)).days))
        elapsed = max(1, round(days / 7)) if grain == "weekly" else days
    unit = "w" if grain == "weekly" else "d"
    for limit in limits:
        if elapsed <= limit:
            return f"<= {limit}{unit}"
    return f"> {limits[-1]}{unit}"


def monitor_dimensions(forecast: pd.DataFrame, default_grain: str) -> pd.DataFrame:
    out = forecast.copy()
    if "horizon" in out:
        out["monitor_horizon"] = pd.to_numeric(out["horizon"], errors="coerce")
    else:
        out["monitor_horizon"] = [parse_model_horizon(name)[0] for name in out["model_name"].astype(str)]
    if "grain" in out:
        out["monitor_grain"] = out["grain"].fillna(default_grain).astype(str).str.lower()
    else:
        out["monitor_grain"] = [
            parse_model_horizon(name)[1] or default_grain for name in out["model_name"].astype(str)
        ]
    return out


def add_evaluation_windows(forecast: pd.DataFrame, default_grain: str) -> pd.DataFrame:
    out = monitor_dimensions(forecast, default_grain)
    limits = horizon_limits(out, default_grain)
    out["evaluation_window"] = [
        elapsed_bucket(
            row.ds,
            getattr(row, "training_cutoff", None),
            row.monitor_grain,
            limits,
            getattr(row, "horizon_step", None),
        )
        for row in out.itertuples(index=False)
    ]
    return out


def load_manifest(forecast_csv: Path) -> dict[str, Any]:
    manifest_path = forecast_csv.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_forecast_csv(forecast_csv: Path) -> Path:
    """Use the packaged diagnostic file when monitoring a final-only package."""
    manifest = load_manifest(forecast_csv)
    artifacts = manifest.get("artifacts", {})
    production_name = artifacts.get("production_forecast")
    diagnostic_name = artifacts.get("diagnostic_forecast")
    if production_name and diagnostic_name and forecast_csv.name == Path(production_name).name:
        diagnostic_path = forecast_csv.parent / diagnostic_name
        if diagnostic_path.exists():
            return diagnostic_path
    return forecast_csv


def realized_forecasts(forecast_csv: Path, sales_csv: Path, weekly_freq: str = DEFAULT_WEEKLY_FREQ) -> tuple[pd.DataFrame, dict[str, Any], str]:
    forecast = pd.read_csv(forecast_csv)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    if "training_cutoff" in forecast:
        forecast["training_cutoff"] = pd.to_datetime(forecast["training_cutoff"])
    if "forecast_origin" in forecast:
        forecast["forecast_origin"] = pd.to_datetime(forecast["forecast_origin"])
    grain = infer_grain(forecast)
    actuals = load_sales(sales_csv, grain=grain, weekly_freq=weekly_freq)[["ds", "y"]].rename(columns={"y": "actual"})
    realized = forecast.merge(actuals, on="ds", how="inner")
    realized = realized[realized["ds"].le(actuals["ds"].max())].copy()
    manifest = load_manifest(forecast_csv)
    return realized, manifest, grain


def model_summary_rows(realized: pd.DataFrame, source_run_id: str, forecast_csv: Path, grain: str) -> list[dict[str, Any]]:
    rows = []
    realized = add_evaluation_windows(realized, grain)
    dimensions = ["model_name", "monitor_horizon", "monitor_grain", "evaluation_window"]
    for (model_name, horizon, row_grain, window), group in realized.groupby(dimensions, dropna=False):
        scores = score_group(group)
        rows.append(
            {
                "source_run_id": source_run_id,
                "source_forecast_csv": str(forecast_csv),
                "row_type": "model",
                "model_name": model_name,
                "comparison_model": "",
                "horizon": "" if pd.isna(horizon) else int(horizon),
                "grain": row_grain,
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


def final_model_for_challenger(
    model_name: str,
    model_names: set[str],
    horizon: int | None = None,
    grain: str | None = None,
) -> str | None:
    parsed_horizon, parsed_grain = parse_model_horizon(model_name)
    horizon = horizon or parsed_horizon
    grain = grain or parsed_grain
    if horizon is None or grain is None:
        return None
    unit = "w" if grain == "weekly" else "d"
    prefix = f"final_{horizon}{unit}_"
    matches = [name for name in model_names if name.startswith(prefix)]
    return matches[0] if matches else None


def challenger_rows(realized: pd.DataFrame, source_run_id: str, forecast_csv: Path, grain: str) -> list[dict[str, Any]]:
    rows = []
    realized = add_evaluation_windows(realized, grain)
    model_names = set(realized["model_name"].astype(str))
    challengers = sorted(name for name in model_names if name.startswith("challenger_"))
    for challenger in challengers:
        challenger_group = realized[realized["model_name"].eq(challenger)]
        horizon_values = challenger_group["monitor_horizon"].dropna()
        horizon = int(horizon_values.iloc[0]) if len(horizon_values) else None
        grain_values = challenger_group["monitor_grain"].dropna()
        row_grain = str(grain_values.iloc[0]) if len(grain_values) else grain
        final_model = final_model_for_challenger(challenger, model_names, horizon, row_grain)
        if final_model is None:
            continue
        pair = realized[realized["model_name"].isin([final_model, challenger])].copy()
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
                    "horizon": horizon if horizon is not None else "",
                    "grain": row_grain,
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


def monitor_row_key(record: dict[str, Any]) -> tuple[str, ...]:
    parsed_horizon, parsed_grain = parse_model_horizon(str(record.get("model_name", "")))
    horizon = record.get("horizon", "")
    if horizon == "" or pd.isna(horizon):
        horizon = parsed_horizon or ""
    if horizon != "":
        horizon = str(int(float(horizon)))
    grain = record.get("grain", "")
    if grain == "" or pd.isna(grain):
        grain = parsed_grain or ""
    values = {
        **record,
        "horizon": horizon,
        "grain": str(grain).lower(),
    }
    key_cols = ["source_run_id", "row_type", "model_name", "comparison_model", "horizon", "grain", "evaluation_window", "rows"]
    return tuple("" if pd.isna(values.get(col, "")) else str(values.get(col, "")) for col in key_cols)


def append_monitor_rows(log_csv: Path, rows: list[dict[str, Any]]) -> int:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(log_csv) if log_csv.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    existing = existing.reindex(columns=LOG_COLUMNS)
    existing_keys = {monitor_row_key(record) for record in existing.to_dict("records")}
    now = datetime.now(timezone.utc).isoformat()
    new_rows = []
    for row in rows:
        record = {col: row.get(col, "") for col in LOG_COLUMNS}
        record["logged_at"] = now
        key = monitor_row_key(record)
        if key not in existing_keys:
            new_rows.append(record)
            existing_keys.add(key)
    if not new_rows:
        return 0
    new_df = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
    out = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(log_csv, index=False)
    return len(new_rows)


def grid_model_for_final(model_name: str, grain: str | None = None) -> str | None:
    _, parsed_grain = parse_model_horizon(model_name)
    grain = grain or parsed_grain
    if grain is None:
        return None
    if grain == "weekly":
        return "daily_reagg_seasonal_naive_lag_365"
    return "seasonal_naive_lag_364"


def grid_path_for_final(
    model_name: str,
    horizon: int | None = None,
    grain: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    parsed_horizon, parsed_grain = parse_model_horizon(model_name)
    horizon = horizon or parsed_horizon
    grain = grain or parsed_grain
    if horizon is None or grain is None:
        return None
    interval_source = (manifest or {}).get("interval_sources", {}).get(str(horizon), {})
    if interval_source.get("grid"):
        return Path(interval_source["grid"])
    if grain == "weekly":
        return Path(f"outputs_weekly_multifold_{horizon}w_daily_reagg_pool") / "multifold_grid_predictions.csv"
    return Path(f"outputs_multifold_{horizon}d_validation") / "multifold_grid_predictions.csv"


def mean_abs_backtest_bias(
    model_name: str,
    horizon: int | None = None,
    grain: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> float | None:
    grid_path = grid_path_for_final(model_name, horizon, grain, manifest)
    grid_model = grid_model_for_final(model_name, grain)
    if grid_path is None or grid_model is None or not grid_path.exists():
        return None
    preds = pd.read_csv(grid_path, parse_dates=["cutoff"])
    preds = preds[preds["model_name"].eq(grid_model)].copy()
    if horizon is not None and "horizon_step" in preds:
        preds = preds[preds["horizon_step"].le(horizon)]
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


def write_alarm_state(realized: pd.DataFrame, output_dir: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    states = []
    final = monitor_dimensions(realized[realized["model_name"].astype(str).str.startswith("final_")], infer_grain(realized))
    for (model_name, horizon, grain), group in final.groupby(["model_name", "monitor_horizon", "monitor_grain"], dropna=False):
        row_bias = group.sort_values("ds").set_index("ds").assign(row_bias_pct=lambda x: (x["forecast"] - x["actual"]) / x["actual"].replace(0, np.nan))["row_bias_pct"].dropna()
        if row_bias.empty:
            continue
        horizon_value = None if pd.isna(horizon) else int(horizon)
        backtest_bias = mean_abs_backtest_bias(model_name, horizon_value, grain, manifest)
        slack = 0.5 * backtest_bias if backtest_bias is not None else 0.5 * float(row_bias.abs().mean())
        series = cusum_series(row_bias, slack)
        threshold = 4.0 * slack
        alarm = bool(((series["positive_cusum"] > threshold) | (series["negative_cusum"] > threshold)).any())
        states.append(
            {
                "model_name": model_name,
                "horizon": horizon_value,
                "grain": grain,
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
    requested_forecast_csv = Path(args.forecast_csv)
    forecast_csv = resolve_forecast_csv(requested_forecast_csv)
    log_csv = Path(args.log_csv)
    realized, manifest, grain = realized_forecasts(forecast_csv, Path(args.sales_csv), args.weekly_freq)
    row_run_ids = realized["run_id"].dropna().astype(str).unique().tolist() if "run_id" in realized else []
    source_run_id = str(manifest.get("run_id") or (row_run_ids[0] if len(row_run_ids) == 1 else forecast_csv.parent.name))
    if realized.empty:
        raise ValueError("No forecast rows are realized in the provided sales file.")

    rows = model_summary_rows(realized, source_run_id, forecast_csv, grain)
    rows.extend(challenger_rows(realized, source_run_id, forecast_csv, grain))
    appended = append_monitor_rows(log_csv, rows)
    alarm_state = write_alarm_state(realized, log_csv.parent, manifest)

    run_manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_forecast_csv": str(requested_forecast_csv),
        "forecast_csv": str(forecast_csv),
        "sales_csv": args.sales_csv,
        "source_run_id": source_run_id,
        "grain": grain,
        "horizons": sorted({int(value) for value in pd.to_numeric(realized.get("horizon"), errors="coerce").dropna()})
        if "horizon" in realized
        else [],
        "realized_rows": int(len(realized)),
        "rows_appended": appended,
        "alarm_constants": alarm_state["constants"],
        "outputs": [str(log_csv), str(log_csv.parent / "alarm_state.json")],
    }
    (log_csv.parent / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    append_experiment(log_csv.parent, "run_forecast_monitor.py", run_manifest)

    print(f"Realized rows: {len(realized)}")
    print(f"Rows appended: {appended}")
    if forecast_csv != requested_forecast_csv:
        print(f"Challenger diagnostics: {forecast_csv}")
    if any(model["alarm"] for model in alarm_state["models"]):
        print("WARNING [monitor]: bias CUSUM alarm fired. See alarm_state.json.")


if __name__ == "__main__":
    main()
