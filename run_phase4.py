"""Phase 4 interval calibration for selected models and ensembles."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import score_group


KEYS = ["unique_id", "horizon", "cutoff", "ds", "horizon_step"]
DEFAULT_MODELS = {
    28: [
        "phase3_all_cutoffs_ridge_stack",
        "phase3_heldout_cutoffs_top5_median",
        "sklearn_extra_trees",
        "seasonal_naive_lag_364",
        "timesfm_2_5",
        "chronos_bolt_small",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "statsforecast_auto_arima",
        "statsforecast_auto_ets",
        "statsmodels_ets_weekly",
        "statsmodels_sarimax_weekly",
    ],
    91: [
        "phase3_all_cutoffs_convex_stack",
        "phase3_heldout_cutoffs_convex_stack",
        "phase3_all_cutoffs_ridge_stack",
        "seasonal_naive_lag_364",
        "timesfm_2_5",
        "chronos_bolt_small",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "statsforecast_auto_arima",
        "statsforecast_auto_ets",
        "statsmodels_ets_weekly",
        "statsmodels_sarimax_weekly",
    ],
}


def load_candidates(phase1_path: Path, phase2_path: Path, phase3_path: Path, extra_paths: list[Path] | None = None) -> pd.DataFrame:
    frames = []
    for path in [phase1_path, phase2_path, phase3_path, *(extra_paths or [])]:
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["cutoff", "ds"]))
    if not frames:
        raise FileNotFoundError("No prediction files found.")
    preds = pd.concat(frames, ignore_index=True)
    preds = preds[preds["forecast"].notna()].copy()
    preds["actual"] = pd.to_numeric(preds["actual"], errors="coerce")
    preds["forecast"] = pd.to_numeric(preds["forecast"], errors="coerce")
    return preds.dropna(subset=["actual", "forecast"])


def select_predictions(preds: pd.DataFrame, model_config: dict[int, list[str]]) -> pd.DataFrame:
    selected = []
    for horizon, models in model_config.items():
        hdf = preds[(preds["horizon"] == horizon) & (preds["model_name"].isin(models))].copy()
        # Warn loudly if a whitelisted model produced no predictions for this horizon.
        # Silently dropping it would leave it without calibrated intervals downstream
        # (the failure mode that gave TimesFM zero-width 91d intervals).
        present = set(hdf["model_name"].unique())
        missing = [m for m in models if m not in present]
        if missing:
            print(
                f"WARNING [phase4]: horizon={horizon} requested models with NO predictions "
                f"in the inputs: {missing}. They will be skipped and get no calibrated "
                f"intervals. Check phase1/phase3/--extra-predictions inputs.",
                file=sys.stderr,
                flush=True,
            )
        selected.append(hdf)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def split_cutoffs(group: pd.DataFrame) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    cutoffs = sorted(pd.Timestamp(c) for c in group["cutoff"].dropna().unique())
    if len(cutoffs) <= 1:
        return cutoffs, cutoffs
    split_idx = max(1, len(cutoffs) // 2)
    return cutoffs[split_idx:], cutoffs[split_idx:]


def residual_quantiles(calibration: pd.DataFrame, horizon_step: int, quantile_levels: list[float]) -> np.ndarray:
    residuals = calibration.loc[calibration["horizon_step"] == horizon_step, "residual"]
    if len(residuals) < 3:
        residuals = calibration["residual"]
    if len(residuals) == 0:
        return np.repeat(0.0, len(quantile_levels))
    return np.quantile(residuals, quantile_levels)


def calibrate_model(group: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict]:
    out = group.copy()
    out["residual"] = out["actual"] - out["forecast"]
    calibration_cutoffs, evaluation_cutoffs = split_cutoffs(out)
    calibration = out[out["cutoff"].isin(calibration_cutoffs)].copy()
    evaluation = out[out["cutoff"].isin(evaluation_cutoffs)].copy()

    rows = []
    metadata = {
        "calibration_cutoffs": [str(c.date()) for c in calibration_cutoffs],
        "evaluation_cutoffs": [str(c.date()) for c in evaluation_cutoffs],
        "method": method,
    }

    for frame_name, frame in [("all_cutoffs", out), ("evaluation_cutoffs", evaluation)]:
        if frame.empty:
            continue
        calibrated = frame.copy()
        calibrated["calibration_subset"] = frame_name
        calibrated["source_model_name"] = calibrated["model_name"]
        calibrated["model_name"] = calibrated["model_name"] + f"__{method}"
        calibrated["p50"] = calibrated["forecast"]

        if method == "global_residual_quantile":
            qs = np.quantile(calibration["residual"], [0.025, 0.10, 0.90, 0.975]) if len(calibration) else np.zeros(4)
            calibrated["p025"] = calibrated["forecast"] + qs[0]
            calibrated["p10"] = calibrated["forecast"] + qs[1]
            calibrated["p90"] = calibrated["forecast"] + qs[2]
            calibrated["p975"] = calibrated["forecast"] + qs[3]
        elif method == "horizon_step_residual_quantile":
            p025 = []
            p10 = []
            p90 = []
            p975 = []
            for row in calibrated.itertuples(index=False):
                qs = residual_quantiles(calibration, int(row.horizon_step), [0.025, 0.10, 0.90, 0.975])
                p025.append(row.forecast + qs[0])
                p10.append(row.forecast + qs[1])
                p90.append(row.forecast + qs[2])
                p975.append(row.forecast + qs[3])
            calibrated["p025"] = p025
            calibrated["p10"] = p10
            calibrated["p90"] = p90
            calibrated["p975"] = p975
        elif method == "absolute_error_symmetric":
            abs_err = np.abs(calibration["residual"])
            if len(abs_err):
                q80 = np.quantile(abs_err, 0.80)
                q95 = np.quantile(abs_err, 0.95)
            else:
                q80 = q95 = 0.0
            calibrated["p10"] = calibrated["forecast"] - q80
            calibrated["p90"] = calibrated["forecast"] + q80
            calibrated["p025"] = calibrated["forecast"] - q95
            calibrated["p975"] = calibrated["forecast"] + q95
        else:
            raise ValueError(method)

        for col in ["p025", "p10", "p50", "p90", "p975"]:
            calibrated[col] = calibrated[col].clip(lower=0.0)
        calibrated["interval_method"] = method
        rows.append(calibrated)

    return pd.concat(rows, ignore_index=True), metadata


def weighted_interval_score(actual: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    width = upper - lower
    lower_penalty = (2.0 / alpha) * (lower - actual).clip(lower=0.0)
    upper_penalty = (2.0 / alpha) * (actual - upper).clip(lower=0.0)
    return width + lower_penalty + upper_penalty


def interval_metrics(group: pd.DataFrame) -> pd.Series:
    base = score_group(group)
    wis80 = weighted_interval_score(group["actual"], group["p10"], group["p90"], 0.20)
    wis95 = weighted_interval_score(group["actual"], group["p025"], group["p975"], 0.05)
    base["wis_80"] = wis80.mean()
    base["wis_95"] = wis95.mean()
    base["median_width_80"] = (group["p90"] - group["p10"]).median()
    base["median_width_95"] = (group["p975"] - group["p025"]).median()
    return base


def run_calibration(preds: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    metadata = {}
    for (horizon, model_name), group in preds.groupby(["horizon", "model_name"]):
        metadata_key = f"{horizon}:{model_name}"
        metadata[metadata_key] = {}
        for method in methods:
            calibrated, meta = calibrate_model(group, method)
            metadata[metadata_key][method] = meta
            frames.append(calibrated)
    return pd.concat(frames, ignore_index=True), metadata


def write_outputs(calibrated: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_dir / "phase4_calibrated_predictions.csv", index=False)
    try:
        calibrated.to_parquet(output_dir / "phase4_calibrated_predictions.parquet", index=False)
    except Exception:
        pass

    metrics = (
        calibrated.groupby(["calibration_subset", "horizon", "source_model_name", "interval_method", "model_name"], dropna=False)
        .apply(interval_metrics, include_groups=False)
        .reset_index()
        .sort_values(["calibration_subset", "horizon", "coverage_80", "wis_80"], ascending=[True, True, False, True])
    )
    metrics.to_csv(output_dir / "phase4_interval_metrics.csv", index=False)
    (output_dir / "phase4_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    eval_metrics = metrics[metrics["calibration_subset"] == "evaluation_cutoffs"].copy()
    best = eval_metrics.sort_values(["horizon", "wis_80"]).groupby("horizon").head(6)
    lines = [
        "# Phase 4 Interval Calibration Summary",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Best calibrated intervals on evaluation cutoffs, ranked by 80% weighted interval score:",
        "",
        "| Horizon | Source model | Method | WAPE | Coverage 80 | Width 80 | WIS 80 | Coverage 95 | Width 95 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {row.source_model_name} | {row.interval_method} | {row.wape:.6f} | {row.coverage_80:.3f} | {row.avg_width_80:,.0f} | {row.wis_80:,.0f} | {row.coverage_95:.3f} | {row.avg_width_95:,.0f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `horizon_step_residual_quantile` uses residual quantiles by forecast step where enough residuals exist, otherwise it falls back to all residuals for the model.",
            "- The current calibration set is small because prior phases used a limited number of cutoffs. Treat interval coverage as directional until more rolling cutoffs are generated.",
        ]
    )
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_model_config(raw: str | None) -> dict[int, list[str]]:
    if not raw:
        return DEFAULT_MODELS
    config_path = Path(raw)
    data = json.loads(config_path.read_text(encoding="utf-8") if config_path.exists() else raw)
    return {int(k): v for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-predictions", default="outputs/model_backtest_predictions.csv")
    parser.add_argument("--phase2-predictions", default="outputs_phase2/phase2_backtest_predictions.csv")
    parser.add_argument("--phase3-predictions", default="outputs_phase3/phase3_ensemble_predictions.csv")
    parser.add_argument("--extra-predictions", default="", help="Comma-separated additional prediction CSVs to include for interval calibration.")
    parser.add_argument("--output-dir", default="outputs_phase4")
    parser.add_argument("--model-config-json", default=None)
    parser.add_argument("--methods", default="horizon_step_residual_quantile,global_residual_quantile,absolute_error_symmetric")
    args = parser.parse_args()

    model_config = parse_model_config(args.model_config_json)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    extra_paths = [Path(p.strip()) for p in args.extra_predictions.split(",") if p.strip()]
    preds = load_candidates(Path(args.phase1_predictions), Path(args.phase2_predictions), Path(args.phase3_predictions), extra_paths)
    selected = select_predictions(preds, model_config)
    calibrated, metadata = run_calibration(selected, methods)
    write_outputs(calibrated, metadata, Path(args.output_dir))
    metrics = pd.read_csv(Path(args.output_dir) / "phase4_interval_metrics.csv")
    eval_metrics = metrics[metrics["calibration_subset"] == "evaluation_cutoffs"]
    print(eval_metrics.sort_values(["horizon", "wis_80"]).groupby("horizon").head(8)[["horizon", "source_model_name", "interval_method", "wape", "coverage_80", "avg_width_80", "wis_80", "coverage_95", "avg_width_95", "wis_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
