"""Phase 3 ensemble testing across phase-1 and phase-2 model predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from run_bakeoff import add_conformal_intervals, score_group


KEYS = ["unique_id", "horizon", "cutoff", "ds", "horizon_step"]


def load_predictions(phase1_path: Path, phase2_path: Path | None, extra_paths: list[Path] | None = None) -> pd.DataFrame:
    frames = [pd.read_csv(phase1_path, parse_dates=["cutoff", "ds"])]
    if phase2_path and phase2_path.exists():
        frames.append(pd.read_csv(phase2_path, parse_dates=["cutoff", "ds"]))
    for path in extra_paths or []:
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["cutoff", "ds"]))
    preds = pd.concat(frames, ignore_index=True)
    preds = preds[preds["forecast"].notna()].copy()
    preds = preds[~preds["model_name"].str.startswith("ensemble_", na=False)].copy()
    preds["actual"] = pd.to_numeric(preds["actual"], errors="coerce")
    preds["forecast"] = pd.to_numeric(preds["forecast"], errors="coerce")
    return preds.dropna(subset=["actual", "forecast"])


def model_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    return (
        preds.groupby(["horizon", "model_name"], dropna=False)
        .apply(score_group, include_groups=False)
        .reset_index()
        .sort_values(["horizon", "wape"])
    )


def select_candidates(metrics: pd.DataFrame, horizon: int, max_models: int) -> list[str]:
    hmetrics = metrics[metrics["horizon"] == horizon].sort_values("wape")
    preferred = [
        "seasonal_naive_lag_364",
        "timesfm_2_5",
        "chronos_bolt_small",
        "statsforecast_mstl_auto_arima",
        "statsforecast_mstl_auto_ets",
        "statsforecast_auto_arima",
        "statsforecast_auto_ets",
        "autogluon_medium_quality",
        "mlforecast_style_lightgbm",
        "mlforecast_style_catboost",
        "sklearn_extra_trees",
        "statsmodels_ets_weekly",
        "statsmodels_sarimax_weekly",
        "neuralforecast_nhits",
        "neuralforecast_nbeats",
    ]
    selected: list[str] = []
    available = set(hmetrics["model_name"])
    for name in preferred:
        if name in available and name not in selected:
            selected.append(name)
    for name in hmetrics["model_name"]:
        if name not in selected:
            selected.append(name)
        if len(selected) >= max_models:
            break
    return selected[:max_models]


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - forecast).sum() / denom) if denom else np.nan


def optimize_weights(actual: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[1]
    if n == 1:
        return np.ones(1)

    def objective(weights: np.ndarray) -> float:
        return wape(actual, matrix @ weights)

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    result = minimize(objective, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return np.ones(n) / n
    return result.x / result.x.sum()


def make_wide(preds: pd.DataFrame, horizon: int, candidates: list[str]) -> pd.DataFrame:
    hdf = preds[(preds["horizon"] == horizon) & (preds["model_name"].isin(candidates))].copy()
    base = hdf.groupby(KEYS, as_index=False)["actual"].first()
    wide = hdf.pivot_table(index=KEYS, columns="model_name", values="forecast", aggfunc="first").reset_index()
    wide = base.merge(wide, on=KEYS, how="inner")
    return wide.dropna(subset=candidates)


def build_ensembles(preds: pd.DataFrame, metrics: pd.DataFrame, max_models: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    metadata = {}
    for horizon in sorted(preds["horizon"].unique()):
        candidates = select_candidates(metrics, int(horizon), max_models)
        wide = make_wide(preds, int(horizon), candidates)
        if wide.empty:
            continue

        cutoffs = sorted(wide["cutoff"].unique())
        split_idx = max(1, len(cutoffs) // 2)
        train_cutoffs = cutoffs[:split_idx]
        eval_cutoffs = cutoffs[split_idx:] or cutoffs
        train = wide[wide["cutoff"].isin(train_cutoffs)].copy()
        eval_df = wide[wide["cutoff"].isin(eval_cutoffs)].copy()

        top3 = candidates[: min(3, len(candidates))]
        top5 = candidates[: min(5, len(candidates))]
        hmetrics = metrics[(metrics["horizon"] == horizon) & (metrics["model_name"].isin(top5))].set_index("model_name")
        inv = 1.0 / np.square(hmetrics.loc[top5, "wape"].clip(lower=1e-6).to_numpy())
        inv_weights = inv / inv.sum()

        stack_models = [m for m in top5 if m in train.columns]
        stack_weights = optimize_weights(train["actual"].to_numpy(), train[stack_models].to_numpy())
        ridge = Ridge(alpha=1.0, positive=True)
        ridge.fit(train[stack_models], train["actual"])

        metadata[str(horizon)] = {
            "candidates": candidates,
            "train_cutoffs": [str(pd.Timestamp(c).date()) for c in train_cutoffs],
            "eval_cutoffs": [str(pd.Timestamp(c).date()) for c in eval_cutoffs],
            "inverse_wape_models": top5,
            "inverse_wape_weights": dict(zip(top5, inv_weights.tolist())),
            "convex_stack_models": stack_models,
            "convex_stack_weights": dict(zip(stack_models, stack_weights.tolist())),
            "ridge_models": stack_models,
            "ridge_coefficients": dict(zip(stack_models, ridge.coef_.tolist())),
            "ridge_intercept": float(ridge.intercept_),
        }

        for frame_name, frame in [("all_cutoffs", wide), ("heldout_cutoffs", eval_df)]:
            ensemble_values = {
                f"phase3_{frame_name}_top3_mean": frame[top3].mean(axis=1).to_numpy(),
                f"phase3_{frame_name}_top5_mean": frame[top5].mean(axis=1).to_numpy(),
                f"phase3_{frame_name}_top5_median": frame[top5].median(axis=1).to_numpy(),
                f"phase3_{frame_name}_inverse_wape": frame[top5].to_numpy() @ inv_weights,
                f"phase3_{frame_name}_convex_stack": frame[stack_models].to_numpy() @ stack_weights,
                f"phase3_{frame_name}_ridge_stack": np.maximum(0.0, ridge.predict(frame[stack_models])),
            }
            for model_name, values in ensemble_values.items():
                out = frame[KEYS + ["actual"]].copy()
                out["model_name"] = model_name
                out["forecast"] = values
                out["runtime_seconds"] = 0.0
                out["refit_cadence"] = "derived_from_backtest_predictions"
                out["hyperparams_json"] = json.dumps(metadata[str(horizon)])
                out["library_version"] = "phase3"
                rows.append(out)
    if not rows:
        return pd.DataFrame(), metadata
    ensembles = pd.concat(rows, ignore_index=True)
    ensembles[["p025", "p10", "p50", "p90", "p975"]] = np.nan
    return add_conformal_intervals(ensembles), metadata


def write_outputs(base_preds: pd.DataFrame, ensembles: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = model_metrics(base_preds)
    ensemble_metrics = (
        ensembles.groupby(["horizon", "model_name"], dropna=False)
        .apply(score_group, include_groups=False)
        .reset_index()
        .sort_values(["horizon", "wape"])
        if not ensembles.empty
        else pd.DataFrame()
    )
    combined_metrics = pd.concat([metrics, ensemble_metrics], ignore_index=True).sort_values(["horizon", "wape"])
    ensembles.to_csv(output_dir / "phase3_ensemble_predictions.csv", index=False)
    try:
        ensembles.to_parquet(output_dir / "phase3_ensemble_predictions.parquet", index=False)
    except Exception:
        pass
    metrics.to_csv(output_dir / "phase3_base_model_metrics.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "phase3_ensemble_metrics.csv", index=False)
    combined_metrics.to_csv(output_dir / "phase3_combined_metrics.csv", index=False)
    (output_dir / "phase3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fair_rows = []
    for horizon_text, meta in metadata.items():
        horizon = int(horizon_text)
        candidates = meta["candidates"]
        for subset_name, cutoff_key in [("all_cutoffs", None), ("heldout_cutoffs", "eval_cutoffs")]:
            subset_ensembles = ensembles[
                (ensembles["horizon"] == horizon)
                & (ensembles["model_name"].str.startswith(f"phase3_{subset_name}_", na=False))
            ].copy()
            if subset_ensembles.empty:
                continue
            cutoffs = sorted(subset_ensembles["cutoff"].dropna().unique())
            subset_base = base_preds[
                (base_preds["horizon"] == horizon)
                & (base_preds["model_name"].isin(candidates))
                & (base_preds["cutoff"].isin(cutoffs))
            ].copy()
            fair = pd.concat([subset_base, subset_ensembles], ignore_index=True)
            fair_metrics = (
                fair.groupby(["horizon", "model_name"], dropna=False)
                .apply(score_group, include_groups=False)
                .reset_index()
            )
            fair_metrics["comparison_subset"] = subset_name
            fair_rows.append(fair_metrics)
    fair_metrics = pd.concat(fair_rows, ignore_index=True).sort_values(["comparison_subset", "horizon", "wape"]) if fair_rows else pd.DataFrame()
    fair_metrics.to_csv(output_dir / "phase3_fair_comparison_metrics.csv", index=False)

    best = fair_metrics.sort_values(["comparison_subset", "horizon", "wape"]).groupby(["comparison_subset", "horizon"]).head(8) if not fair_metrics.empty else combined_metrics.sort_values(["horizon", "wape"]).groupby("horizon").head(8)
    lines = [
        "# Phase 3 Ensemble Summary",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Best models by horizon and comparison subset:",
        "",
        "| Subset | Horizon | Model | WAPE | Bias pct | MAE | RMSE |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        subset = getattr(row, "comparison_subset", "combined")
        lines.append(
            f"| {subset} | {int(row.horizon)} | {row.model_name} | {row.wape:.6f} | {row.bias_pct:.6f} | {row.mae:,.0f} | {row.rmse:,.0f} |"
        )
    lines.extend(
        [
            "",
            "Decision read:",
            "",
            "Phase-3 ensembles are evaluated both on all available aligned cutoffs and on held-out later cutoffs after fitting stack weights on earlier cutoffs. The held-out rows are the more honest estimate for learned stacking methods.",
        ]
    )
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-predictions", default="outputs/model_backtest_predictions.csv")
    parser.add_argument("--phase2-predictions", default="outputs_phase2/phase2_backtest_predictions.csv")
    parser.add_argument("--extra-predictions", default="", help="Comma-separated additional prediction CSVs to include in ensemble testing.")
    parser.add_argument("--output-dir", default="outputs_phase3")
    parser.add_argument("--max-models", type=int, default=8)
    args = parser.parse_args()

    extra_paths = [Path(p.strip()) for p in args.extra_predictions.split(",") if p.strip()]
    base_preds = load_predictions(Path(args.phase1_predictions), Path(args.phase2_predictions), extra_paths)
    metrics = model_metrics(base_preds)
    ensembles, metadata = build_ensembles(base_preds, metrics, args.max_models)
    write_outputs(base_preds, ensembles, metadata, Path(args.output_dir))
    combined = pd.read_csv(Path(args.output_dir) / "phase3_combined_metrics.csv")
    print(combined.sort_values(["horizon", "wape"]).groupby("horizon").head(8)[["horizon", "model_name", "wape", "bias_pct", "mae", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
