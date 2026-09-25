"""Exploratory: parameter-free candidates vs seasonal naive on the multifold grids.

All candidates are deterministic functions of history (or fixed blends of
already-computed grid forecasts), so every grid cutoff is a valid evaluation
window: no weights are fit, hence no leakage. This is still an exploratory
sweep over a handful of variants; any winner must be pre-registered before
being promoted to production.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import load_sales
from experiment_registry import append_experiment


BENCHMARK_MODEL = "seasonal_naive_lag_364"
CHALLENGER_MODEL = "blend_snl75_tfm25"
DEFAULT_HOLDOUT_CUTOFF = "2026-04-01"


def growth_ratio(y: pd.Series, cutoff: pd.Timestamp, k: int, cap: float | None = None) -> float:
    recent = y.loc[cutoff - pd.Timedelta(days=k - 1) : cutoff].sum()
    year_ago = y.loc[cutoff - pd.Timedelta(days=364 + k - 1) : cutoff - pd.Timedelta(days=364)].sum()
    if year_ago <= 0:
        return 1.0
    r = recent / year_ago
    if cap is not None:
        r = min(max(r, 1.0 / cap), cap)
    return float(r)


def seasonal_values(y: pd.Series, dates: pd.DatetimeIndex, lag: int, cutoff: pd.Timestamp) -> np.ndarray:
    # Wrap references that land after the cutoff back a further season, mirroring
    # the tiling behavior of seasonal_naive (matters when horizon > lag).
    vals = []
    for d in dates:
        ref = d - pd.Timedelta(days=lag)
        while ref > cutoff:
            ref -= pd.Timedelta(days=lag)
        vals.append(y.get(ref, np.nan))
    return np.asarray(vals)


def evaluate(grid_path: Path, sales_csv: Path, horizon: int) -> pd.DataFrame:
    sales = load_sales(sales_csv)
    y = sales.set_index("ds")["y"]
    preds = pd.read_csv(grid_path, parse_dates=["cutoff", "ds"])
    wide = preds.pivot_table(index=["cutoff", "ds", "actual"], columns="model_name", values="forecast").reset_index()

    rows = []
    for cutoff, g in wide.groupby("cutoff"):
        g = g.sort_values("ds")
        dates = pd.DatetimeIndex(g["ds"])
        actual = g["actual"].to_numpy()
        snl = g[BENCHMARK_MODEL].to_numpy()
        snl728 = seasonal_values(y, dates, 728, cutoff)
        tfm = g["timesfm_2_5"].to_numpy() if "timesfm_2_5" in g else np.full(len(g), np.nan)

        candidates = {"seasonal_naive_lag_364": snl}
        for k in (28, 56, 91):
            r = growth_ratio(y, cutoff, k)
            candidates[f"growth_adj_snl_k{k}"] = r * snl
            r_cap = growth_ratio(y, cutoff, k, cap=1.15)
            candidates[f"growth_adj_snl_k{k}_cap15"] = r_cap * snl
        if not np.isnan(snl728).any():
            candidates["snl_364_728_mean"] = 0.5 * (snl + snl728)
            r56 = growth_ratio(y, cutoff, 56)
            candidates["growth_adj_two_year_mean"] = r56 * 0.5 * (snl + snl728)
        if not np.isnan(tfm).any():
            for w in (0.75, 0.62, 0.50):
                candidates[f"blend_snl{int(w*100)}_tfm{int((1-w)*100)}"] = w * snl + (1 - w) * tfm
            r56 = growth_ratio(y, cutoff, 56)
            candidates["blend_gadj56_50_tfm50"] = 0.5 * (r56 * snl) + 0.5 * tfm
            candidates["median_snl_gadj56_tfm"] = np.median(np.vstack([snl, r56 * snl, tfm]), axis=0)

        for name, forecast in candidates.items():
            if np.isnan(forecast).any():
                continue
            denom = np.abs(actual).sum()
            rows.append(
                {
                    "horizon": horizon,
                    "grid_path": str(grid_path),
                    "cutoff": cutoff,
                    "model_name": name,
                    "wape": np.abs(forecast - actual).sum() / denom,
                    "bias_pct": (forecast - actual).sum() / denom,
                }
            )
    return pd.DataFrame(rows)


def summarize(fold_metrics: pd.DataFrame, benchmark: str = BENCHMARK_MODEL) -> pd.DataFrame:
    bench = fold_metrics[fold_metrics["model_name"] == benchmark].set_index("cutoff")["wape"]
    rows = []
    for name, g in fold_metrics.groupby("model_name"):
        g = g.set_index("cutoff")
        common = g.index.intersection(bench.index)
        diffs = g.loc[common, "wape"] - bench.loc[common]
        rows.append(
            {
                "model_name": name,
                "windows": len(common),
                "mean_wape": g.loc[common, "wape"].mean(),
                "mean_bias_pct": g.loc[common, "bias_pct"].mean(),
                "wins_vs_naive": int((diffs < 0).sum()),
                "mean_diff": diffs.mean(),
                "worst_diff": diffs.max(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_wape").reset_index(drop=True)


def discover_grids() -> list[tuple[int, Path]]:
    grids = []
    for manifest_path in sorted(Path(".").glob("outputs_multifold_*_validation/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        grid_path = manifest_path.parent / "multifold_grid_predictions.csv"
        if grid_path.exists() and not manifest.get("degraded_models", {}).get("timesfm_2_5"):
            grids.append((int(manifest["horizon"]), grid_path))
    return sorted(grids)


def challenger_scorecard(fold_metrics: pd.DataFrame, holdout_cutoff: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for horizon in sorted(fold_metrics["horizon"].unique()):
        metrics = fold_metrics[fold_metrics["horizon"].eq(horizon)]
        naive_columns = ["cutoff", "wape"] + (["grid_path"] if "grid_path" in metrics else [])
        naive = metrics[metrics["model_name"].eq(BENCHMARK_MODEL)][naive_columns].rename(columns={"wape": "naive_wape"})
        challenger = metrics[metrics["model_name"].eq(CHALLENGER_MODEL)][["cutoff", "wape", "bias_pct"]].rename(
            columns={"wape": "challenger_wape", "bias_pct": "challenger_bias_pct"}
        )
        paired = naive.merge(challenger, on="cutoff", validate="one_to_one")
        selection_grid = f"outputs_multifold_{int(horizon)}d_validation"
        is_selection_grid = (
            paired["grid_path"].map(lambda value: Path(value).parent.name == selection_grid)
            if "grid_path" in paired
            else pd.Series(True, index=paired.index)
        )
        for evidence_split, sample in (
            ("exploration", paired[paired["cutoff"].lt(holdout_cutoff) & is_selection_grid]),
            ("holdout", paired[paired["cutoff"].ge(holdout_cutoff)]),
        ):
            diffs = sample["challenger_wape"] - sample["naive_wape"]
            effective_windows = (
                ((sample["cutoff"].max() - sample["cutoff"].min()).days + int(horizon)) / int(horizon)
                if len(sample)
                else 0.0
            )
            rows.append(
                {
                    "horizon": int(horizon),
                    "evidence_split": evidence_split,
                    "windows": int(len(sample)),
                    "effective_independent_windows_approx": effective_windows,
                    "first_cutoff": sample["cutoff"].min() if len(sample) else pd.NaT,
                    "last_cutoff": sample["cutoff"].max() if len(sample) else pd.NaT,
                    "naive_mean_wape": sample["naive_wape"].mean(),
                    "challenger_mean_wape": sample["challenger_wape"].mean(),
                    "challenger_mean_bias_pct": sample["challenger_bias_pct"].mean(),
                    "wins_vs_naive": int((diffs < 0).sum()),
                    "mean_wape_diff": diffs.mean(),
                    "worst_wape_diff": diffs.max(),
                    "status": "historical_support" if evidence_split == "exploration" else "tracking" if len(sample) else "awaiting_holdout",
                }
            )
    return pd.DataFrame(rows)


def write_scorecard_outputs(
    fold_metrics: pd.DataFrame,
    scorecard: pd.DataFrame,
    output_dir: Path,
    holdout_cutoff: pd.Timestamp,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.sort_values(["horizon", "cutoff", "model_name"]).to_csv(output_dir / "fold_metrics.csv", index=False)
    scorecard.to_csv(output_dir / "challenger_scorecard.csv", index=False)
    holdout_windows = int(scorecard.loc[scorecard["evidence_split"].eq("holdout"), "windows"].sum())
    created_at = datetime.now(timezone.utc)
    manifest = {
        "run_id": created_at.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": created_at.isoformat(),
        "horizons": sorted(int(value) for value in fold_metrics["horizon"].unique()),
        "models": [BENCHMARK_MODEL, CHALLENGER_MODEL],
        "challenger_weights": {BENCHMARK_MODEL: 0.75, "timesfm_2_5": 0.25},
        "holdout_cutoff_inclusive": str(holdout_cutoff.date()),
        "holdout_windows": holdout_windows,
        "promotion_status": "manual_review_required" if holdout_windows else "awaiting_new_holdout_windows",
        "artifacts": {"fold_metrics": "fold_metrics.csv", "challenger_scorecard": "challenger_scorecard.csv"},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(
        output_dir,
        "explore_naive_beaters.py",
        manifest,
        f"75/25 naive/TimesFM challenger: {manifest['promotion_status']} ({holdout_windows} holdout windows)",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--recent-n", type=int, default=6, help="Also summarize over only the most recent N windows.")
    parser.add_argument("--holdout-cutoff", default=DEFAULT_HOLDOUT_CUTOFF, help="First forecast origin not used to select the fixed 75/25 blend.")
    parser.add_argument("--output-dir", default="outputs_naive_timesfm_challenger")
    args = parser.parse_args()

    all_metrics = []
    for horizon, grid_path in discover_grids():
        fold_metrics = evaluate(grid_path, Path(args.sales_csv), horizon)
        all_metrics.append(fold_metrics)
        print(f"\n=== Horizon {horizon}: all {fold_metrics['cutoff'].nunique()} grid windows ===")
        print(summarize(fold_metrics).to_string(index=False))
        recent = sorted(fold_metrics["cutoff"].unique())[-args.recent_n :]
        print(f"\n=== Horizon {horizon}: most recent {len(recent)} windows ===")
        print(summarize(fold_metrics[fold_metrics["cutoff"].isin(recent)]).to_string(index=False))

    if not all_metrics:
        raise ValueError("No non-degraded TimesFM multifold validation grids found.")
    combined = pd.concat(all_metrics, ignore_index=True)
    holdout_cutoff = pd.Timestamp(args.holdout_cutoff)
    scorecard = challenger_scorecard(combined, holdout_cutoff)
    manifest = write_scorecard_outputs(combined, scorecard, Path(args.output_dir), holdout_cutoff)
    print("\n=== Fixed 75/25 challenger evidence ===")
    print(scorecard.to_string(index=False))
    print(f"Promotion status: {manifest['promotion_status']}")


if __name__ == "__main__":
    main()
