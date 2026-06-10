"""Trade-area (sub-market) forecasting bakeoff.

Forecasts every market in the panel produced by prepare_market_level_data.py
over K leak-free rolling folds and scores three evaluation scopes:

- bottom_up_total: per-market forecasts summed vs the bu-total actuals,
  compared against DIRECT models fit on the bu-total series itself
- like_for_like: same comparison restricted to mature markets (physical unit
  open >= 365 days at the fold cutoff) -- the "Version A" benchmark
- per_market: median per-market WAPE and the share of markets where each
  model beats that market's own seasonal naive

Per-market candidates are parameter-free or pooled (no per-fold weight
fitting, so every fold is honest):

- market_seasonal_naive: lag-364 tiled, falling back to a 28-day window
  average for markets younger than a year (a dead market decays to ~0)
- market_growth_adj_naive: seasonal naive scaled by the market's trailing
  56-day YoY ratio
- market_window_avg_28
- market_timesfm: TimesFM 2.5 batched across all markets per cutoff
- pooled_lightgbm: one global direct model across (market, lead-day) rows
  with calendar, seasonal lags, recent level, days-since-open, sqm, and
  unit-type features; trained per fold on pseudo-cutoffs strictly before
  the fold cutoff (targets capped at the cutoff), so it is leak-free.
  Opening dates, sqm, and unit type are treated as KNOWN-FUTURE covariates:
  physical openings are planned years ahead, so days_since_open for dates
  after a historical cutoff reflects information genuinely available at
  that cutoff. If an opening date in complete_opening_info.csv was ever a
  surprise (not planned in advance), this assumption breaks for that market.

The bu-total file covers ~88% of daily_sales_data.csv (see DATA.md); totals
here are on the bu-total's own terms and are not comparable to the aggregate
bakeoff numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment
from run_bakeoff import SEED, arcsinh, inv_arcsinh, seasonal_naive, set_seeds

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


def load_panel(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, parse_dates=["ds", "open_date"])
    end = panel["ds"].max()
    frames = []
    for market, g in panel.groupby("sub_market_no"):
        idx = pd.date_range(g["ds"].min(), end, freq="D")
        s = g.set_index("ds").reindex(idx)
        s["y"] = s["y"].fillna(0.0)
        for col in ["sub_market_no", "open_date", "total_sqm", "unit_type", "has_physical"]:
            s[col] = g[col].iloc[0]
        s.index.name = "ds"
        frames.append(s.reset_index())
    return pd.concat(frames, ignore_index=True)


def build_cutoffs(panel: pd.DataFrame, horizon: int, n_folds: int, step: int) -> list[pd.Timestamp]:
    last = panel["ds"].max() - pd.Timedelta(days=horizon)
    return [last - pd.Timedelta(days=step * k) for k in reversed(range(n_folds))]


def market_history(g: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return g[g["ds"] <= cutoff]


def naive_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) >= 364:
        idx = len(y) - 364 + (np.arange(horizon) % 364)
        return y[idx]
    if len(y) == 0:
        return np.zeros(horizon)
    return np.repeat(y[-min(len(y), 28):].mean(), horizon)


def growth_adj_forecast(y: np.ndarray, horizon: int, k: int = 56) -> np.ndarray:
    base = naive_forecast(y, horizon)
    if len(y) < 364 + k:
        return base
    recent = y[-k:].sum()
    year_ago = y[-364 - k : -364].sum()
    if year_ago <= 0:
        return base
    return base * (recent / year_ago)


def window_avg_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(horizon)
    return np.repeat(y[-min(len(y), 28):].mean(), horizon)


def timesfm_batched(model, histories: dict[int, np.ndarray], horizon: int, max_context: int) -> dict[int, np.ndarray]:
    markets = list(histories.keys())
    contexts = []
    for m in markets:
        y = histories[m].astype(float)
        ctx = y[-min(len(y), max_context):]
        if len(ctx) < max_context:
            ctx = np.pad(ctx, (max_context - len(ctx), 0), mode="edge")
        contexts.append(ctx)
    point, _ = model.forecast(horizon=horizon, inputs=contexts)
    return {m: np.maximum(0.0, np.asarray(point[i])[:horizon]) for i, m in enumerate(markets)}


UNIT_TYPE_CODES = {
    "IKEA Store": 1,
    "IKEA Small Store (L)": 2,
    "IKEA Small Store (M)": 3,
    "IKEA Small Store (S)": 4,
    "IKEA Small Store (S) Lada": 5,
    "IKEA Plan and Order Point": 6,
    "IKEA Plan and Order Point XS": 7,
}
POOLED_FEATURES = [
    "lead", "dow", "month", "weekofyear", "y_lag364", "y_lag728",
    "recent_mean_7", "recent_mean_28", "days_since_open", "log_sqm", "unit_type_code",
]


def pooled_rows(panel: pd.DataFrame, pseudo_cutoff: pd.Timestamp, horizon: int, target_cap: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for market, g in panel.groupby("sub_market_no"):
        hist = market_history(g, pseudo_cutoff)
        if hist.empty:
            continue
        y = hist.set_index("ds")["y"]
        recent_7 = float(y.tail(7).mean())
        recent_28 = float(y.tail(28).mean())
        open_date = g["open_date"].iloc[0]
        sqm = g["total_sqm"].iloc[0]
        type_code = UNIT_TYPE_CODES.get(g["unit_type"].iloc[0], 0)
        future = g[(g["ds"] > pseudo_cutoff) & (g["ds"] <= min(pseudo_cutoff + pd.Timedelta(days=horizon), target_cap))]
        for row in future.itertuples(index=False):
            t = row.ds
            rows.append(
                {
                    "sub_market_no": market,
                    "ds": t,
                    "lead": (t - pseudo_cutoff).days,
                    "dow": t.dayofweek,
                    "month": t.month,
                    "weekofyear": int(t.isocalendar()[1]),
                    "y_lag364": y.get(t - pd.Timedelta(days=364), np.nan),
                    "y_lag728": y.get(t - pd.Timedelta(days=728), np.nan),
                    "recent_mean_7": recent_7,
                    "recent_mean_28": recent_28,
                    "days_since_open": (t - open_date).days if pd.notna(open_date) else -9999,
                    "log_sqm": float(np.log1p(sqm)) if pd.notna(sqm) else 0.0,
                    "unit_type_code": type_code,
                    "y": row.y,
                }
            )
    return pd.DataFrame(rows)


def pooled_lightgbm_forecast(panel: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, train_cutoffs: list[pd.Timestamp]) -> dict[int, np.ndarray] | None:
    if LGBMRegressor is None:
        print("WARNING [market-bakeoff]: lightgbm unavailable; skipping pooled_lightgbm.", file=sys.stderr)
        return None
    train = pd.concat([pooled_rows(panel, pc, horizon, target_cap=cutoff) for pc in train_cutoffs], ignore_index=True)
    if train.empty:
        return None
    model = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(train[POOLED_FEATURES], arcsinh(train["y"]))

    predict = pooled_rows(panel, cutoff, horizon, target_cap=cutoff + pd.Timedelta(days=horizon))
    preds = np.maximum(0.0, inv_arcsinh(model.predict(predict[POOLED_FEATURES])))
    predict = predict.assign(pred=preds)
    return {m: g.sort_values("ds")["pred"].to_numpy() for m, g in predict.groupby("sub_market_no")}


def run_fold(panel: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, timesfm_model, timesfm_context: int, pseudo_cutoffs: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_dates = pd.date_range(cutoff + pd.Timedelta(days=1), cutoff + pd.Timedelta(days=horizon), freq="D")
    histories = {}
    actuals = {}
    mature = set()
    # actual totals include markets with no pre-cutoff history (e.g. brand-new
    # trade areas appearing inside the test window); bottom-up forecasts
    # implicitly assign them zero, which is the honest bottom-up treatment.
    test_panel = panel[(panel["ds"] > cutoff) & (panel["ds"] <= cutoff + pd.Timedelta(days=horizon))]
    actual_total = test_panel.groupby("ds")["y"].sum().reindex(test_dates, fill_value=0.0).to_numpy()
    for market, g in panel.groupby("sub_market_no"):
        hist = market_history(g, cutoff)
        test = g[(g["ds"] > cutoff) & (g["ds"] <= cutoff + pd.Timedelta(days=horizon))].sort_values("ds")
        open_date = g["open_date"].iloc[0]
        if pd.notna(open_date) and (cutoff - open_date).days >= 365:
            mature.add(market)
        if hist.empty:
            continue
        histories[market] = hist["y"].to_numpy()
        actuals[market] = test.set_index("ds")["y"].reindex(test_dates, fill_value=0.0).to_numpy()
    actual_lfl = np.sum([a for m, a in actuals.items() if m in mature], axis=0) if mature else np.zeros(len(test_dates))

    forecasts: dict[str, dict[int, np.ndarray]] = {
        "market_seasonal_naive": {m: naive_forecast(y, horizon) for m, y in histories.items()},
        "market_growth_adj_naive": {m: growth_adj_forecast(y, horizon) for m, y in histories.items()},
        "market_window_avg_28": {m: window_avg_forecast(y, horizon) for m, y in histories.items()},
    }
    if timesfm_model is not None:
        forecasts["market_timesfm"] = timesfm_batched(timesfm_model, histories, horizon, timesfm_context)
    pooled = pooled_lightgbm_forecast(panel, cutoff, horizon, pseudo_cutoffs)
    if pooled is not None:
        forecasts["pooled_lightgbm"] = pooled

    market_rows = []
    total_rows = []
    for model_name, by_market in forecasts.items():
        total_fcst = np.zeros(len(test_dates))
        lfl_fcst = np.zeros(len(test_dates))
        for market, fcst in by_market.items():
            actual = actuals.get(market)
            if actual is None:
                continue
            fcst = np.asarray(fcst)[: len(test_dates)]
            if len(fcst) < len(test_dates):
                fcst = np.pad(fcst, (0, len(test_dates) - len(fcst)))
            total_fcst += fcst
            if market in mature:
                lfl_fcst += fcst
            market_rows.append(
                {
                    "fold_cutoff": cutoff,
                    "horizon": horizon,
                    "model_name": model_name,
                    "sub_market_no": market,
                    "mature": market in mature,
                    "actual_sum": float(actual.sum()),
                    "forecast_sum": float(fcst.sum()),
                    "abs_err_sum": float(np.abs(fcst - actual).sum()),
                    "abs_actual_sum": float(np.abs(actual).sum()),
                }
            )
        for scope, fcst_vec, act_vec in [("bottom_up_total", total_fcst, actual_total), ("like_for_like", lfl_fcst, actual_lfl)]:
            denom = np.abs(act_vec).sum()
            total_rows.append(
                {
                    "fold_cutoff": cutoff,
                    "horizon": horizon,
                    "model_name": model_name,
                    "scope": scope,
                    "wape": float(np.abs(fcst_vec - act_vec).sum() / denom) if denom else np.nan,
                    "bias_pct": float((fcst_vec - act_vec).sum() / act_vec.sum()) if act_vec.sum() else np.nan,
                    "markets": len(by_market),
                }
            )

    # Direct models on the aggregate series, for the bottom-up vs direct question.
    total_hist = panel[panel["ds"] <= cutoff].groupby("ds")["y"].sum().to_numpy()
    lfl_hist = (
        panel[(panel["ds"] <= cutoff) & (panel["sub_market_no"].isin(mature))].groupby("ds")["y"].sum().to_numpy()
    )
    for scope, hist, act_vec in [("bottom_up_total", total_hist, actual_total), ("like_for_like", lfl_hist, actual_lfl)]:
        fcst_vec = naive_forecast(hist, horizon)
        denom = np.abs(act_vec).sum()
        total_rows.append(
            {
                "fold_cutoff": cutoff,
                "horizon": horizon,
                "model_name": "direct_aggregate_seasonal_naive",
                "scope": scope,
                "wape": float(np.abs(fcst_vec - act_vec).sum() / denom) if denom else np.nan,
                "bias_pct": float((fcst_vec - act_vec).sum() / act_vec.sum()) if act_vec.sum() else np.nan,
                "markets": 0,
            }
        )
    return pd.DataFrame(market_rows), pd.DataFrame(total_rows)


def score_per_market(fold_preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cutoff, horizon, model_name), g in fold_preds.groupby(["fold_cutoff", "horizon", "model_name"]):
        per_market = g[g["abs_actual_sum"] > 0].copy()
        per_market["wape"] = per_market["abs_err_sum"] / per_market["abs_actual_sum"]
        naive = fold_preds[
            (fold_preds["fold_cutoff"] == cutoff)
            & (fold_preds["horizon"] == horizon)
            & (fold_preds["model_name"] == "market_seasonal_naive")
        ].set_index("sub_market_no")
        naive_wape = naive["abs_err_sum"] / naive["abs_actual_sum"]
        joined = per_market.join(naive_wape.rename("naive_wape"), on="sub_market_no")
        rows.append(
            {
                "fold_cutoff": cutoff,
                "horizon": horizon,
                "model_name": model_name,
                "scope": "per_market",
                "wape": float(per_market["wape"].median()),
                "bias_pct": np.nan,
                "markets": len(per_market),
                "beats_naive_share": float((joined["wape"] < joined["naive_wape"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["horizon", "scope", "model_name"], dropna=False)
        .agg(folds=("fold_cutoff", "nunique"), mean_wape=("wape", "mean"), mean_bias_pct=("bias_pct", "mean"), beats_naive_share=("beats_naive_share", "mean"))
        .reset_index()
        .sort_values(["horizon", "scope", "mean_wape"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="outputs_market_prep/market_daily_panel.csv")
    parser.add_argument("--horizons", default="28,91")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-step", type=int, default=28)
    parser.add_argument("--pseudo-cutoffs", type=int, default=12, help="Pseudo-cutoffs (28d apart, before each fold) for pooled model training.")
    parser.add_argument("--timesfm-context", type=int, default=1024)
    parser.add_argument("--skip-timesfm", action="store_true")
    parser.add_argument("--output-dir", default="outputs_market_bakeoff")
    args = parser.parse_args()

    set_seeds()
    panel = load_panel(Path(args.panel))
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    timesfm_model = None
    if not args.skip_timesfm:
        from run_prophet_timesfm_benchmark import load_timesfm

        try:
            timesfm_model = load_timesfm(args.timesfm_context, max(horizons))
        except Exception as exc:
            print(f"WARNING [market-bakeoff]: TimesFM unavailable ({exc}); continuing without it.", file=sys.stderr)

    all_market_rows = []
    all_total_rows = []
    for horizon in horizons:
        cutoffs = build_cutoffs(panel, horizon, args.n_folds, args.fold_step)
        for i, cutoff in enumerate(cutoffs, start=1):
            pseudo = [cutoff - pd.Timedelta(days=28 * (j + 1) + horizon) for j in range(args.pseudo_cutoffs)]
            pseudo = [p for p in pseudo if p > panel["ds"].min() + pd.Timedelta(days=364)]
            market_rows, total_rows = run_fold(panel, cutoff, horizon, timesfm_model, args.timesfm_context, pseudo)
            all_market_rows.append(market_rows)
            all_total_rows.append(total_rows)
            print(f"[market-bakeoff] horizon {horizon} fold {cutoff.date()} done ({i}/{len(cutoffs)})", flush=True)

    fold_preds = pd.concat(all_market_rows, ignore_index=True)
    scores = pd.concat([pd.concat(all_total_rows, ignore_index=True), score_per_market(fold_preds)], ignore_index=True)
    summary = summarize(scores)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_preds.to_csv(output_dir / "market_bakeoff_fold_predictions.csv", index=False)
    scores.to_csv(output_dir / "market_bakeoff_fold_scores.csv", index=False)
    summary.to_csv(output_dir / "market_bakeoff_summary.csv", index=False)

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel": str(args.panel),
        "horizons": horizons,
        "n_folds": args.n_folds,
        "models": sorted(fold_preds["model_name"].unique().tolist()),
        "markets": int(fold_preds["sub_market_no"].nunique()),
        "timesfm_included": timesfm_model is not None,
        "pooled_lightgbm_included": "pooled_lightgbm" in set(fold_preds["model_name"]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_market_bakeoff.py", manifest)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
