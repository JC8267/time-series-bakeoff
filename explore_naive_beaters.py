"""Exploratory: parameter-free candidates vs seasonal naive on the multifold grids.

All candidates are deterministic functions of history (or fixed blends of
already-computed grid forecasts), so every grid cutoff is a valid evaluation
window: no weights are fit, hence no leakage. This is still an exploratory
sweep over a handful of variants; any winner must be pre-registered before
being promoted to production.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import load_sales


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
        dates = pd.DatetimeIndex(g["ds"])
        actual = g["actual"].to_numpy()
        snl = seasonal_values(y, dates, 364, cutoff)
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
                    "cutoff": cutoff,
                    "model_name": name,
                    "wape": np.abs(forecast - actual).sum() / denom,
                    "bias_pct": (forecast - actual).sum() / denom,
                }
            )
    return pd.DataFrame(rows)


def summarize(fold_metrics: pd.DataFrame, benchmark: str = "seasonal_naive_lag_364") -> pd.DataFrame:
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
    import json

    grids = []
    for manifest_path in sorted(Path(".").glob("outputs_multifold_*_validation/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        grid_path = manifest_path.parent / "multifold_grid_predictions.csv"
        if grid_path.exists():
            grids.append((int(manifest["horizon"]), grid_path))
    return sorted(grids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--recent-n", type=int, default=6, help="Also summarize over only the most recent N windows.")
    args = parser.parse_args()

    for horizon, grid_path in discover_grids():
        fold_metrics = evaluate(grid_path, Path(args.sales_csv), horizon)
        print(f"\n=== Horizon {horizon}: all {fold_metrics['cutoff'].nunique()} grid windows ===")
        print(summarize(fold_metrics).to_string(index=False))
        recent = sorted(fold_metrics["cutoff"].unique())[-args.recent_n :]
        print(f"\n=== Horizon {horizon}: most recent {len(recent)} windows ===")
        print(summarize(fold_metrics[fold_metrics["cutoff"].isin(recent)]).to_string(index=False))


if __name__ == "__main__":
    main()
