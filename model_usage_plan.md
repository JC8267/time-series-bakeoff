# Forecasting and Counterfactual Model Usage Plan

Last updated: 2026-06-25

This document describes which validated models and ensembles to use for different project types, and how to implement each workflow with the current bakeoff pipeline.

## Current Validated Winners

| Use case | Horizon | Recommended model | Why |
|---|---:|---|---|
| Short-term production forecast | 28 days | `seasonal_naive_lag_364` | Won 4 of 6 folds of the 2026-06-10 multi-fold validation (mean WAPE 0.0680 vs 0.0865 for Extra Trees, which failed badly on the December holiday window). `final_28d_extra_trees_opening_timesfm_pool` remains the opening-aware scenario model. |
| Medium-term production forecast | 91 days | `seasonal_naive_lag_364` | Won all 5 folds of the 2026-06-10 nested multi-fold validation (mean WAPE 0.0757 vs 0.0918 for the honestly refit convex stack). The earlier single-holdout stack win was contaminated by weight-fitting overlap. `final_91d_convex_stack_opening_timesfm_pool` is now a sensitivity/diagnostic model. |
| Long-horizon pure forecast benchmark | 180 days | `final_180d_seasonal_naive_benchmark` | Won all 5 folds of the 2026-06-10 multi-fold validation (mean WAPE 0.0638 vs 0.0879 for the nested stack). The stack's focused-run win was contaminated by a ~5-month weight-fitting overlap; the stack is now a sensitivity comparator. |
| Long-horizon pure forecast benchmark | 365 days | `final_365d_seasonal_naive_benchmark` | Won the single leak-free nested fold (WAPE 0.0568 vs 0.0806) and the 17-window sweep. The stack's focused-run win was contaminated by a ~11-month weight-fitting overlap; the stack is now a sensitivity comparator. |
| Long-horizon opening-sensitive forecast | 180 days | `final_180d_hist_gradient_boosting_opening_sensitive` | Opening-sensitive scenario model. It did not beat the seasonal/stack benchmarks on WAPE, so use it for counterfactual sensitivity, not pure accuracy claims. |
| Long-horizon opening-sensitive forecast | 365 days | `final_365d_hist_gradient_boosting_opening_sensitive` | Opening-sensitive scenario model. It did not beat the seasonal/stack benchmarks on WAPE, so use it for counterfactual sensitivity, not pure accuracy claims. |
| Weekly-grain forecast benchmark | 4, 12, 26, 52 weeks | `daily_reagg_seasonal_naive_lag_365` | Daily lag-365 forecasts summed into `W-WED` weeks beat native weekly `seasonal_naive_lag_52` at all four weekly horizons. Evidence is strongest at 4 and 12 weeks; 26 and 52 weeks remain overlapping-window reads. |

Latest rerun artifacts:

| Step | Output directory |
|---|---|
| Supporting phase-2 input | `outputs_opening_phase2_plus_timesfm` |
| 28/91 phase 1 | `outputs_opening_phase1_statsmodels_timesfm25_venv` |
| 28/91 foundation models | `outputs_opening_foundation_statsmodels_timesfm25_venv` |
| 28/91 phase 3 | `outputs_opening_phase3_statsmodels_timesfm25_venv` |
| 28/91 phase 4 | `outputs_opening_phase4_statsmodels_timesfm25_recalibrated_venv` |
| 28/91 phase 5 | `outputs_opening_phase5_statsmodels_timesfm25_recalibrated_venv` |
| 28/91 phase 6 | `outputs_opening_phase6_statsmodels_timesfm25_recalibrated_venv` |
| 180/365 focused validation | `outputs_opening_long_horizons_stats_chronos_venv` |
| 180/365 foundation models | `outputs_opening_foundation_long_stats_chronos_venv` |
| 180/365 phase 3 | `outputs_opening_phase3_long_stats_chronos_venv` |
| 180/365 phase 4 | `outputs_opening_phase4_long_stats_chronos_venv` |
| 180/365 phase 6 | `outputs_opening_phase6_long_stats_chronos_venv` |
| 91-day multi-fold nested validation | `outputs_multifold_91d_validation` (run_id `20260610T124857Z`; 4-opening original archived) |
| 28-day multi-fold nested validation | `outputs_multifold_28d_validation` (run_id `20260610T124426Z`; 4-opening original archived) |
| Opening lift event study | `outputs_opening_event_study` (`estimate_opening_lift_event_study.py`) |
| 180-day multi-fold nested validation | `outputs_multifold_180d_validation` (run_id `20260610T112337Z`) |
| 365-day multi-fold nested validation | `outputs_multifold_365d_validation` (run_id `20260610T113121Z`) |
| Daily seasonal trend/averaging pools | `outputs_multifold_28d_trend_seasonal_pool`, `outputs_multifold_91d_trend_seasonal_pool`, `outputs_multifold_180d_trend_seasonal_pool`, `outputs_multifold_365d_trend_seasonal_pool` |
| Daily 180d extended trend significance pool | `outputs_multifold_180d_trend_seasonal_extended_pool` |
| Weekly baseline bakeoff | `outputs_weekly_bakeoff_baselines_4_12_26_52` (weekly `W-WED`, horizons 4/12/26/52) |
| Weekly nested baseline pools | `outputs_weekly_multifold_4w_baseline_pool`, `outputs_weekly_multifold_12w_baseline_pool`, `outputs_weekly_multifold_26w_baseline_pool`, `outputs_weekly_multifold_52w_baseline_pool` |
| Weekly nested TimesFM pools | `outputs_weekly_multifold_4w_timesfm_pool`, `outputs_weekly_multifold_12w_timesfm_pool`, `outputs_weekly_multifold_26w_timesfm_pool`, `outputs_weekly_multifold_52w_timesfm_pool` |
| Weekly nested foundation pools | `outputs_weekly_multifold_4w_foundation_pool`, `outputs_weekly_multifold_12w_foundation_pool`, `outputs_weekly_multifold_26w_foundation_pool`, `outputs_weekly_multifold_52w_foundation_pool` |
| Weekly nested ML pools | `outputs_weekly_multifold_4w_ml_pool`, `outputs_weekly_multifold_12w_ml_pool`, `outputs_weekly_multifold_26w_ml_pool`, `outputs_weekly_multifold_52w_ml_pool` |
| Weekly nested daily-reaggregation pools | `outputs_weekly_multifold_4w_daily_reagg_pool`, `outputs_weekly_multifold_12w_daily_reagg_pool`, `outputs_weekly_multifold_26w_daily_reagg_pool`, `outputs_weekly_multifold_52w_daily_reagg_pool` |
| Weekly nested lag-365 foundation pools | `outputs_weekly_multifold_4w_daily_reagg_foundation_pool`, `outputs_weekly_multifold_12w_daily_reagg_foundation_pool`, `outputs_weekly_multifold_26w_daily_reagg_foundation_pool`, `outputs_weekly_multifold_52w_daily_reagg_foundation_pool` |
| Weekly seasonal trend/averaging pools | `outputs_weekly_multifold_4w_trend_seasonal_pool`, `outputs_weekly_multifold_12w_trend_seasonal_pool`, `outputs_weekly_multifold_26w_trend_seasonal_pool`, `outputs_weekly_multifold_52w_trend_seasonal_pool` |
| Weekly 26w extended trend significance pool | `outputs_weekly_multifold_26w_trend_seasonal_extended_pool` |
| Overlap-aware significance tests | `outputs_forecast_significance_tests` |
| Interval coverage backtest | `outputs_interval_coverage_backtest` |
| Weekly production package | `outputs_weekly_production_forecast` |

Artifact notes:

- The phase 4/5/6 rows above point to the TimesFM interval-fix reruns. They supersede `outputs_opening_phase4_statsmodels_timesfm25_venv`, `outputs_opening_phase5_statsmodels_timesfm25_venv`, and `outputs_opening_phase6_statsmodels_timesfm25_venv`, where standalone `timesfm_2_5` was missing from phase-4 interval calibration and got zero-width 91-day intervals. Point forecasts/WAPE are identical between the two; only standalone TimesFM interval metrics changed (91-day coverage_80 0.000 → 0.857, coverage_95 0.000 → 0.945).
- Older and scratch output directories were moved to `archive/run_outputs_2026-06-01/` during cleanup. The table above lists the retained root-level artifacts.

Latest key validation metrics:

| Horizon | Candidate | WAPE | Bias | Validation folds |
|---:|---|---:|---:|---:|
| 28 | `seasonal_naive_lag_364` | 0.0680 | 0.0025 | 6 non-overlapping folds (2026-06-10) |
| 28 | `sklearn_extra_trees` | 0.0892 | -0.0292 | 6 non-overlapping folds (2026-06-10, 10-opening rerun) |
| 28 | `sklearn_extra_trees` | 0.0314 | 0.0020 | 1 final holdout (clean but unrepresentative; matches the best of the 6 folds) |
| 28 | `phase5_ridge_stack` | 0.0327 | -0.0164 | 1 final holdout (contaminated; see caveat) |
| 91 | `seasonal_naive_lag_364` | 0.0757 | -0.0024 | 5 nested rolling folds (2026-06-10) |
| 91 | `nested_convex_stack` | 0.0903 | -0.0011 | 5 nested rolling folds (2026-06-10, 10-opening rerun) |
| 91 | `phase5_convex_stack` | 0.0507 | -0.0116 | 1 final holdout (contaminated; see caveat) |
| 91 | `phase5_ridge_stack` | 0.0522 | -0.0216 | 1 final holdout (contaminated; see caveat) |
| 180 | `seasonal_naive_lag_364` | 0.0638 | -0.0096 | 5 nested rolling folds (2026-06-10) |
| 180 | `nested_convex_stack` | 0.0879 | -0.0210 | 5 nested rolling folds (2026-06-10) |
| 180 | `phase3_heldout_cutoffs_inverse_wape` | 0.0645 | 0.0049 | 3 rolling cutoffs (contaminated; see caveat) |
| 365 | `seasonal_naive_lag_364` | 0.0568 | 0.0001 | 1 nested fold (2026-06-10) |
| 365 | `nested_convex_stack` | 0.0806 | -0.0415 | 1 nested fold (2026-06-10) |
| 365 | `phase3_heldout_cutoffs_convex_stack` | 0.0563 | -0.0080 | 3 rolling cutoffs (contaminated; see caveat) |

Seasonal trend/averaging experiment added 2026-06-25:

- Added level-scaled seasonal naive (`seasonal_naive_scaled_lag_*`) and recency-weighted multi-year seasonal average (`seasonal_average_lag_*`) candidates at daily and weekly grains.
- Standalone scaled and multi-year candidates did not beat the current daily or weekly champions. The 4-week/full scaling was especially volatile; the 8-week damped scaling was closer but still worse than the simple seasonal champion.
- Daily seasonal-only `nested_inverse_wape_stack` is worth tracking as a challenger at 91 and 180 days: 91d mean WAPE 0.0748 vs 0.0757 for `seasonal_naive_lag_364` (3/5 wins), and 180d mean WAPE 0.0611 vs 0.0638 (4/5 wins). Keep `seasonal_naive_lag_364` as production because the 91d gain is tiny, 91/180 folds overlap, and production packaging is still benchmark-first.
- Weekly `pool_median` of the lag-365 reaggregation and seasonal trend variants barely edged standalone lag-365 at 26w (0.0348 vs 0.0352, 3/5 wins) and 52w (0.03469 vs 0.03471, 1/2 wins). Treat this as a long-horizon sensitivity read, not a default change, because those folds are heavily overlapping and the improvement is within noise. At 4w and 12w, standalone `daily_reagg_seasonal_naive_lag_365` remains best.

Overlap-aware significance tests added 2026-06-25:

- Method: paired fold-window WAPE differences, Newey-West/HAC standard error with lag tied to the horizon/step overlap, and circular moving-block bootstrap with block length `ceil(horizon / grid_step)`. See `run_forecast_significance_tests.py` and `outputs_forecast_significance_tests`.
- Daily 180d `nested_inverse_wape_stack` vs `seasonal_naive_lag_364`: 14 extended nested windows, 11/14 wins, mean WAPE diff -0.00536, HAC p=0.147, block-bootstrap 95% CI [-0.01073, +0.00001]. Directionally useful, not statistically confirmed.
- Weekly 26w `pool_median` vs `daily_reagg_seasonal_naive_lag_365`: 14 extended nested windows, 12/14 wins, mean WAPE diff -0.00140, HAC p=0.017, block-bootstrap 95% CI [-0.00212, -0.00067]. This is the first overlap-adjusted evidence for a real weekly challenger, but the absolute gain is small and currently only validated at 26w.
- Weekly 26w `pool_median` vs native `seasonal_naive_lag_52`: 14/14 wins, mean WAPE diff -0.00294, HAC p=0.00046. Keep native weekly seasonal naive as benchmark only.

Interval coverage backtest added 2026-06-25:

- `run_interval_coverage_backtest.py` emulates production residual-quantile intervals using only prior cutoffs. Daily and weekly production scripts now support `--interval-mode global|segment`; default is `global` because global intervals had better overall coverage in backtest.
- Daily global interval coverage for `seasonal_naive_lag_364`: 80% coverage ranged 0.759-0.805 and 95% coverage ranged 0.941-0.962 across 28/91/180/365d. Segment conditioning improved daily holiday coverage at 180d (80% 0.654 -> 0.784; 95% 0.864 -> 0.941) but narrowed bands and reduced total 80% coverage at shorter horizons.
- Weekly global interval coverage for `daily_reagg_seasonal_naive_lag_365`: 80% coverage ranged 0.680-0.771 and 95% coverage ranged 0.880-0.959 across 4/12/26/52w. Segment conditioning badly undercovered 4w holiday rows (95% 0.471), so keep global weekly intervals as the default and use segment coverage as a diagnostic.

Weekly-grain validation added 2026-06-24 and extended 2026-06-25:

- Grain: complete `W-WED` weeks aggregated from `daily_sales_data.csv`; 221 weekly rows from 2022-01-12 through 2026-04-01.
- Horizons: 4, 12, 26, and 52 weekly rows.
- Full baseline grid: `outputs_weekly_bakeoff_baselines_4_12_26_52` with 114/27/23/17 cutoffs at 4/12/26/52 weeks. Same-fold diagnostic ensembles beat seasonal naive at all four horizons, but they are candidate-selected on the same grid and remain exploratory.
- Nested validation before daily reaggregation: baseline/window pools, broad foundation pools, and standalone ML tree models did not beat `seasonal_naive_lag_52` robustly. The narrow seasonal-naive/TimesFM pool median beat mean WAPE at 4 and 12 weeks, but the 4-week mean edge depended on the 2025-12-03 holiday fold, seasonal naive had the better 4-week median WAPE, and the challenger did not survive broader-pool comparisons.
- Daily-to-weekly reaggregation resolved the main weekly phase issue. `daily_reagg_seasonal_naive_lag_364` is exactly equivalent to `seasonal_naive_lag_52` under `W-WED` buckets, but `daily_reagg_seasonal_naive_lag_365` aligns by calendar date and beat `seasonal_naive_lag_52` at all four horizons.
- A broad foundation pool that included `daily_reagg_seasonal_naive_lag_365` did not find an ensemble that beat standalone lag-365 at any weekly horizon. The 52-week `pool_median` beat native weekly `seasonal_naive_lag_52`, but still trailed lag-365.
- Weekly nested folds were stepped 4 weeks. Read them as overlapping windows, not independent folds, except at 4 weeks: effective independent windows are about 5.00 at 4w, 2.33 at 12w, 1.62 at 26w, and 1.08 at 52w.
- Weekly ML pool coverage: `sklearn_hist_gradient_boosting`, `sklearn_hist_gradient_boosting_direct`, `sklearn_random_forest`, `sklearn_extra_trees`, `sklearn_ridge_lags`, and `mlforecast_style_lightgbm` were run with `opening_info.csv`. None beat `seasonal_naive_lag_52`. Ridge emitted ill-conditioned-matrix warnings and performed poorly at longer horizons.
- The retired fixed `seasonal_naive_holiday_aligned_lag_52` candidate was tried for Thanksgiving/Christmas/New-Year alignment. It worsened the holiday folds because the holiday anchor can jump a full `W-WED` bucket when the holiday weekday changes; keep it out of active pools.

| Weekly horizon | Candidate | Mean WAPE | Mean bias | Folds | Wins vs `seasonal_naive_lag_52` | Verdict |
|---:|---|---:|---:|---:|---:|---|
| 4 | `daily_reagg_seasonal_naive_lag_365` | 0.0386 | 0.0039 | 5 | 3/5 | Primary weekly forecast |
| 4 | `seasonal_naive_lag_52` | 0.0489 | 0.0045 | 5 | benchmark | Benchmark |
| 4 | `pool_median` of `seasonal_naive_lag_52` + `timesfm_2_5` | 0.0455 | 0.0029 | 5 | 2/5 | Track as challenger only |
| 4 | lag-365 foundation `nested_inverse_wape_stack` | 0.0492 | 0.0030 | 5 | 3/5 | Worse than standalone lag-365 |
| 12 | `daily_reagg_seasonal_naive_lag_365` | 0.0410 | -0.0032 | 5 | 4/5 | Primary weekly forecast |
| 12 | `seasonal_naive_lag_52` | 0.0505 | -0.0031 | 5 | benchmark | Benchmark |
| 12 | `pool_median` of `seasonal_naive_lag_52` + `timesfm_2_5` | 0.0472 | 0.0000 | 5 | 3/5 | Track as challenger only |
| 12 | lag-365 foundation `pool_median` | 0.0523 | -0.0039 | 5 | 2/5 | Worse than standalone lag-365 |
| 26 | `daily_reagg_seasonal_naive_lag_365` | 0.0352 | -0.0108 | 5 | 4/5 | Primary weekly forecast; overlapping windows |
| 26 | `seasonal_naive_lag_52` | 0.0397 | -0.0117 | 5 | benchmark | Benchmark; overlapping windows |
| 26 | lag-365 foundation `pool_median` | 0.0401 | -0.0177 | 5 | 1/5 | Worse than standalone lag-365 |
| 52 | `daily_reagg_seasonal_naive_lag_365` | 0.0347 | 0.0000 | 2 | 2/2 | Primary weekly forecast; heavily overlapping |
| 52 | `seasonal_naive_lag_52` | 0.0389 | -0.0001 | 2 | benchmark | Benchmark; heavily overlapping |
| 52 | lag-365 foundation `pool_median` | 0.0378 | -0.0146 | 2 | 1/2 | Beats seasonal, worse than lag-365 |

Holiday/rest split from `outputs_weekly_multifold_*w_daily_reagg_pool`:

| Horizon | Segment | `daily_reagg_lag_365` WAPE | `seasonal_naive_lag_52` WAPE | Read |
|---:|---|---:|---:|---|
| 4 | Holiday weeks | 0.0418 | 0.0638 | Main improvement; 2025-12-03 fold drops from 0.1048 to 0.0496 |
| 4 | Rest weeks | 0.0364 | 0.0390 | Small improvement |
| 12 | Holiday weeks | 0.0409 | 0.0619 | Main improvement |
| 12 | Rest weeks | 0.0427 | 0.0452 | Small improvement |
| 26 | Holiday weeks | 0.0356 | 0.0454 | Improvement, but overlapping read |
| 52 | Holiday weeks | 0.0398 | 0.0578 | Improvement, but heavily overlapping read |

Fold-count caveat:

- The latest 28/91 phase-1 and foundation reruns used 3 rolling cutoffs per horizon. Phase 3 stacking trained on the earlier aligned cutoff and evaluated on 2 held-out cutoffs. Phase 5 is a single final holdout window.
- The phase-3/phase-5 stack metrics are contaminated at every horizon: the weight-fitting windows overlap the evaluation windows by 19 days at 28d (fit cutoff 2026-02-23), 31 days at 91d (fit cutoff 2025-11-01), ~5 months at 180d (fit cutoff 2025-08-01), and ~11 months at 365d (fit cutoff 2025-02-01). Standalone models like `sklearn_extra_trees` have no weight fitting and are unaffected.
- The 2026-06-10 multi-fold nested validations (`run_multifold_final_validation.py`, weights refit per fold leak-free) supersede those numbers: `seasonal_naive_lag_364` won 4 of 6 folds at 28 days, all 5 folds at 91 and 180 days, and the single feasible fold at 365 days. The 91/180-day folds overlap (28-day steps), so they are not fully independent; the 28-day folds do not overlap; the 365-day result rests on 1 nested fold plus a 17-window parameter-free sweep.
- Weekly nested validations use a 4-week grid step. At 12/26/52 weeks, consecutive holdouts overlap by 8/22/48 weeks respectively; report them as overlapping windows with the effective independent-window approximation above, not as independent folds.
- The latest 180/365 validation used 3 rolling cutoffs per horizon. This is intentionally underpowered.
- Tests with fewer folds can flip the apparent winner back to `seasonal_naive_lag_364`. Do not compare a naive-only, fewer-fold run against an ensemble run unless the ensemble is rebuilt and scored on the same folds.
- When fold count is reduced, report the result as a sensitivity check rather than a replacement recommendation unless phase 3/phase 5 are rerun under the same fold design.
- Weekly `data_gap_filled_flag` is summed into weekly rows but not consumed by model scoring or weighting. Current weekly runs had `missing_days_filled = 0`; if future weekly data has interpolated days, add an explicit quality note or down-weight those weeks before treating the metrics as comparable.

Model additions in the latest rerun:

- StatsForecast `AutoARIMA`, `AutoETS`, `MSTL + AutoARIMA`, and `MSTL + AutoETS`.
- Chronos-Bolt small as a second foundation model beside TimesFM.
- Statsmodels `ExponentialSmoothing` and `SARIMAX` now run without fallback in the latest Statsmodels-fixed rerun. They did not improve the 28-day or 91-day production winners.
- Phase 4 was rerun with standalone `timesfm_2_5` included in interval calibration. Corrected 91-day standalone TimesFM coverage is 0.857 at the 80% level and 0.945 at the 95% level; the 91-day production stack coverage remains 0.857 at 80% and 0.956 at 95%.
- TimesFM loader fallback for the current `huggingface-hub` behavior that forwards `proxies` into the TimesFM constructor.
- Direct dependency pins for `huggingface-hub==0.35.3` and `httpx==0.27.2` to keep TimesFM and Chronos loading in the project venv.

## Market-Level (Trade-Area) Forecasting

Added 2026-06-10 from the bu-level bakeoff (`run_market_bakeoff.py`, run_id `20260610T132950Z`, `outputs_market_bakeoff/`, 97 markets, 5 folds at 28 and 91 days). Targets the bu-level panel on its own terms (~88% of `daily_sales_data.csv` scope — see `DATA.md`; do not mix the two targets).

| Task | Use | Why |
|---|---|---|
| Individual trade-area forecasts | `market_timesfm` (TimesFM 2.5 per market, batched) | Median per-market WAPE 0.163-0.164 vs 0.180-0.193 for per-market seasonal naive; beats naive in 77-78% of markets at both horizons. |
| bu-total or like-for-like aggregate | Direct seasonal naive on the aggregate series | Bottom-up summing adds nothing (0.0608 vs 0.0609 at 28d); per-market model wins do not survive aggregation (TimesFM 0.077, pooled LightGBM 0.086 at the total level). |
| Young/new markets (< 1 year) | `pooled_lightgbm` or `market_timesfm` | Pooled model transfers sqm/format/days-since-open structure across markets; seasonal naive has no year-ago base. |

Notes:

- Per-market growth-adjusted naive collapses at 91 days (mean total WAPE 0.149, +8% bias): noisy per-market YoY ratios — often inflated by openings in the trailing year — extrapolate badly. Do not use it below the chain level.
- Like-for-like (markets with a physical unit open >= 365 days) behaves like the total: direct seasonal naive on the mature-market aggregate is the benchmark.
- Opening lift for scenario work comes from `estimate_market_opening_lift.py` (within-market ecomm-baseline counterfactual), not from these forecast models.

## Decision Rules

### 1. General sales forecasting

Use this when the goal is the most accurate expected sales forecast, not attribution.

Default choices:

| Forecast length | Use |
|---:|---|
| 1-28 days | `seasonal_naive_lag_364`; keep `final_28d_extra_trees_opening_timesfm_pool` as the opening-aware sensitivity comparison, and avoid Extra Trees unhedged on windows spanning major holidays |
| 29-91 days | `seasonal_naive_lag_364`; keep `final_91d_convex_stack_opening_timesfm_pool` as a sensitivity comparison only |
| 92-180 days | `seasonal_naive_lag_364` (won all 5 multi-fold folds); opening-sensitive models for scenario sensitivity when openings matter |
| 181-365 days | `seasonal_naive_lag_364` (won the leak-free fold and the 17-window sweep); opening-sensitive models when future openings materially affect the forecast |

Weekly-grain choices:

| Forecast length | Use |
|---:|---|
| 4 weeks | `daily_reagg_seasonal_naive_lag_365`; keep `seasonal_naive_lag_52` as benchmark |
| 12 weeks | `daily_reagg_seasonal_naive_lag_365`; keep `seasonal_naive_lag_52` as benchmark |
| 26 weeks | `daily_reagg_seasonal_naive_lag_365`; overlapping-window evidence, so report the caveat |
| 52 weeks | `daily_reagg_seasonal_naive_lag_365`; only about 1 independent annual read is feasible with current weekly history |

Implementation (primary, all horizons):

```powershell
python run_production_forecast.py `
  --sales-csv daily_sales_data.csv `
  --output-dir outputs_production_forecast
```

Primary output: `outputs_production_forecast/production_forecast.csv` with rows:

```text
final_<h>d_seasonal_naive_production       (production forecast)
challenger_<h>d_blend_snl75_tfm25          (pre-registered challenger, tracked not shipped)
```

Weekly implementation:

```powershell
python run_weekly_production_forecast.py `
  --sales-csv daily_sales_data.csv `
  --horizons 4,12,26,52 `
  --output-dir outputs_weekly_production_forecast
```

Primary weekly output: `outputs_weekly_production_forecast/weekly_production_forecast.csv` with rows:

```text
final_<h>w_daily_reagg_seasonal_naive_lag_365
benchmark_<h>w_seasonal_naive_lag_52
```

Intervals are residual quantiles from the daily-reaggregation weekly grid. Default interval mode is `global`; `--interval-mode segment` is available for holiday/rest diagnostics but is not the default because the backtest undercovered short weekly holiday windows.

Intervals are residual quantiles from each rule's own recent multifold grid windows; the manifest records the grid and window count per horizon. Requires the matching `outputs_multifold_<h>d_validation/` grid (run `run_multifold_final_validation.py` first for a new horizon or after a data refresh).

The phase-6 package (`run_phase6.py`, see Implementation Commands below) remains available for the legacy stack/ML artifacts; after the 2026-06-10 multi-fold validations its winner-aligned rows are sensitivity comparisons, not production.

### 2. Like-for-like demand forecasting

Use this when the goal is to evaluate underlying demand trends without confusing expansion-driven growth with organic growth.

Recommended setup:

| Target | Opening features | Purpose |
|---|---|---|
| Existing-unit sales only | No | Cleanest benchmark for organic demand. |
| Total sales | Yes | Production forecast when unit openings affect future volume. |

Implementation notes:

- Build or maintain an existing-unit-only target if unit-level sales are available.
- Exclude sales from units opened inside the evaluation window when measuring mature-store demand.
- Keep total-sales forecasts as a separate track because the business still needs a system-wide forecast.

If only aggregate sales are available, the current pipeline cannot perfectly isolate like-for-like demand. In that case, use the opening-aware total-sales model and interpret the opening feature effects as directional, not causal proof.

### 3. Counterfactuals for incremental sales

Use this when the goal is to estimate incremental sales from a promotion, market event, pricing action, operational change, or new-unit opening.

The model should answer:

```text
What sales would we have expected if the event had not happened?
```

Recommended model choice:

| Counterfactual window | Recommended model family | Reason |
|---:|---|---|
| Up to 28 days | Extra Trees opening-aware model for the scenario delta, reported beside the `seasonal_naive_lag_364` benchmark | Extra Trees responds to opening features, which the counterfactual needs; seasonal naive is the more accurate pure forecast (4 of 6 folds, 2026-06-10), so report both. |
| 29-91 days | `seasonal_naive_lag_364` as the baseline forecast, paired with an opening/event-sensitive scenario model when the counterfactual needs to respond to the intervention | The convex stack lost all 5 folds of the multi-fold validation, and like seasonal naive it is not opening-sensitive in its fitted form. |
| 92-180 days | Opening-sensitive gradient boosting plus seasonal and stack benchmarks | Pure seasonal/stack models may forecast well, but cannot attribute opening effects alone. |
| 181-365 days | Opening-sensitive gradient boosting plus seasonal and stack benchmarks | Needed when the counterfactual must reason about openings over a full annual cycle. |

Do not use `seasonal_naive_lag_364` alone for opening-related incrementality. It is a strong benchmark, but it ignores opening features and therefore cannot distinguish organic demand from unit expansion.

Do not use the long-horizon stack alone for opening-related incrementality either. The latest 180/365 stack weights are mostly seasonal/TimesFM/StatsForecast inputs. Pair it with `final_180d_hist_gradient_boosting_opening_sensitive` or `final_365d_hist_gradient_boosting_opening_sensitive` when openings or ramp timing are part of the counterfactual.

Counterfactual workflow:

1. Define the intervention date and the counterfactual window.
2. Freeze all non-intervention inputs as observed or planned.
3. Create two feature sets:
   - Actual scenario: includes the event, opening, promotion, or operational change.
   - Counterfactual scenario: removes or neutralizes that event while keeping calendar, seasonality, and known external context unchanged.
4. Generate forecasts for both scenarios.
5. Calculate incremental sales:

```text
incremental_sales = actual_or_actual_scenario_forecast - counterfactual_forecast
```

6. Report total incrementality, daily incrementality, bias risk, and uncertainty intervals.

For historical events where actual sales are known:

```text
incremental_sales = observed_sales - counterfactual_forecast
```

For future/planned events:

```text
incremental_sales = planned_scenario_forecast - no_event_scenario_forecast
```

### 4. New-unit opening analysis

Use this when the project is specifically about how much sales growth comes from new units entering the system.

Required input:

```text
opening_info.csv
```

Current supported fields:

```text
Name
BU
Opening_date
```

Preferred future structure if unit-level sales become available:

```text
unit_id
open_date
market_area_no
unit_type
```

Opening features currently generated:

```text
new_unit_opened_flag
days_since_latest_opening
num_units_open
num_new_units_last_30_days
num_new_units_last_60_days
num_new_units_last_90_days
num_new_units_last_180_days
grand_opening_0_30_days
ramp_31_90_days
ramp_91_180_days
mature_180_plus_days
post_opening_ramp_flag
grand_opening_window_flag
```

Recommended benchmarks:

| Version | Sales included | Opening features | Purpose |
|---|---|---|---|
| A | Existing-unit only | No | Cleanest like-for-like model test. |
| B | Total sales | No | Diagnostic showing how much openings distort naive time-series models. |
| C | Total sales | Yes | Best production setup for total-sales forecasting. |

Version C should be the default production candidate for total sales. Version A should be the primary demand-model benchmark when unit-level data supports it.

Aggregate identifiability warning (2026-06-10): `estimate_opening_lift_event_study.py` estimated per-opening lift from total sales for all 10 known openings using a pre-opening-YoY-adjusted comparison. Estimates ranged from -15.6% to +6.4% of chain sales with no consistent sign or ramp shape — an order of magnitude larger than any plausible single-unit contribution — because chain-level macro swings dominate. Per-unit opening effects are NOT identifiable from aggregate data. Consequences: (a) do not apply event-study uplifts to production forecasts; (b) treat ML opening-feature effects as directional at best; (c) unit-level daily sales remain the single highest-leverage data improvement for opening analytics.

### 5. Promotion or campaign incrementality

For promotions, the opening features should remain in both the actual and counterfactual scenarios unless the promotion is tied directly to a new-unit opening.

Recommended setup:

| Input | Actual scenario | Counterfactual scenario |
|---|---|---|
| Calendar features | Same as observed | Same as observed |
| Opening features | Same as observed | Same as observed |
| Promotion flag or spend | Observed/planned value | Zero or baseline value |
| Sales target | Observed sales if historical | Forecasted scenario sales if future |

Model choice:

- Use `seasonal_naive_lag_364` as the baseline for short campaign windows, with the 28-day Extra Trees model as the feature-responsive scenario comparison (avoid Extra Trees unhedged across major holidays).
- Use `seasonal_naive_lag_364` as the baseline when the impact window extends past one month; the 91-day convex stack is a sensitivity comparison only after the multi-fold validation.
- For longer windows, use an opening-sensitive model and compare against the seasonal benchmark.

### 6. Model monitoring and refresh

Refresh the validation when any of these occur:

- A new unit opens.
- A new sales file extends the history by at least 4 weeks.
- A major holiday, promotion, pricing, or operational pattern changes.
- The forecast horizon changes materially.
- Counterfactual work requires windows beyond the currently validated 28, 91, 180, or 365 days.

Recommended cadence:

| Task | Cadence |
|---|---|
| Update production forecast | Weekly or after each data refresh. |
| Re-run 28/91 validation | Monthly or after major business changes. |
| Re-run 180/365 validation | Quarterly or when long-range planning is needed. |
| Re-check counterfactual calibration | Before each major incrementality readout. |

## Implementation Commands

### Full opening-aware bakeoff

```powershell
python run_bakeoff.py `
  --sales-csv daily_sales_data.csv `
  --opening-info opening_info.csv `
  --horizons 28,91 `
  --output-dir outputs_opening_phase1
```

### Long-horizon opening-aware validation

```powershell
python run_bakeoff.py `
  --sales-csv daily_sales_data.csv `
  --opening-info opening_info.csv `
  --horizons 180,365 `
  --max-cutoffs 3 `
  --output-dir outputs_opening_long_horizons
```

### TimesFM backtests

```powershell
python run_timesfm_backtests.py `
  --sales-csv daily_sales_data.csv `
  --output-dir outputs_opening_timesfm_backtests
```

### Prophet and TimesFM benchmark

```powershell
python run_prophet_timesfm_benchmark.py `
  --sales-csv daily_sales_data.csv `
  --output-dir outputs_prophet_timesfm
```

### Final production package

```powershell
python run_phase6.py `
  --sales-csv daily_sales_data.csv `
  --opening-info opening_info.csv `
  --phase3-metadata outputs_opening_phase3_statsmodels_timesfm25_venv/phase3_metadata.json `
  --phase4-predictions outputs_opening_phase4_statsmodels_timesfm25_recalibrated_venv/phase4_calibrated_predictions.csv `
  --horizons 28,91 `
  --output-dir outputs_opening_phase6_statsmodels_timesfm25_recalibrated_venv
```

### Long-horizon production package

```powershell
python run_phase6.py `
  --sales-csv daily_sales_data.csv `
  --opening-info opening_info.csv `
  --phase3-metadata outputs_opening_phase3_long_stats_chronos_venv/phase3_metadata.json `
  --phase4-predictions outputs_opening_phase4_long_stats_chronos_venv/phase4_calibrated_predictions.csv `
  --horizons 180,365 `
  --output-dir outputs_opening_phase6_long_stats_chronos_venv
```

This writes both the pure accuracy seasonal benchmarks and the opening-sensitive long-horizon candidates:

```text
final_180d_seasonal_naive_benchmark
final_180d_hist_gradient_boosting_opening_sensitive
final_365d_seasonal_naive_benchmark
final_365d_lightgbm_opening_sensitive
```

Long-horizon bakeoffs now also include `sklearn_hist_gradient_boosting_direct`, a direct multi-step comparator that trains separate horizon-step models instead of recursively feeding predictions back into lag features.

### Opening counterfactual scenario file

To create a no-future-opening scenario:

```powershell
python create_opening_scenario.py `
  --opening-info opening_info.csv `
  --output-csv outputs_counterfactual/opening_info_no_future_after_2026_01_01.csv `
  --mode remove_on_or_after `
  --start-date 2026-01-01
```

Then run phase 6 once with the actual opening file and once with the adjusted opening file. Compare the two forecast outputs with:

```powershell
python run_counterfactual.py `
  --actual-forecast outputs_actual/phase6_final_forecast.csv `
  --counterfactual-forecast outputs_no_event/phase6_final_forecast.csv `
  --actual-model final_365d_lightgbm_opening_sensitive `
  --counterfactual-model final_365d_lightgbm_opening_sensitive `
  --output-dir outputs_counterfactual
```

For historical calibration, run paired backtests with the real opening file and a no-event opening file, then compare them with:

```powershell
python calibrate_counterfactual_from_backtests.py `
  --actual-backtest outputs_actual/model_backtest_predictions.csv `
  --counterfactual-backtest outputs_no_event/model_backtest_predictions.csv `
  --model-name sklearn_hist_gradient_boosting `
  --start-date 2025-07-16 `
  --end-date 2025-10-14 `
  --output-dir outputs_counterfactual_calibration
```

### Unit-level like-for-like targets

When unit-level sales are available, prepare total, existing-unit, and new-unit targets with:

```powershell
python prepare_unit_level_targets.py `
  --unit-sales-csv unit_daily_sales.csv `
  --opening-info opening_info.csv `
  --output-csv outputs_unit_level/unit_level_targets.csv
```

Expected unit-level sales fields:

```text
unit_id
date or transaction_date
sales or net_sales
```

### New-unit ramp curve

When unit-level sales are available, estimate the opening ramp curve with:

```powershell
python fit_opening_ramp_curve.py `
  --unit-sales-csv unit_daily_sales.csv `
  --opening-info opening_info.csv `
  --output-dir outputs_ramp_curve
```

This creates:

```text
daily_opening_ramp_curve.csv
opening_ramp_band_summary.csv
```

## Reporting Templates

### Forecast report

Include:

- Forecast horizon.
- Selected model.
- Forecast date range.
- Forecast total.
- WAPE and bias from the relevant validation.
- Number of validation folds and whether the metric came from rolling cutoffs, phase-3 held-out cutoffs, or one final holdout.
- Interval coverage from final holdout, when available.
- Known caveats, especially whether the model uses opening features.

### Counterfactual report

Include:

- Intervention definition.
- Counterfactual assumption.
- Model used.
- Counterfactual date range.
- Observed or planned scenario sales.
- Counterfactual sales.
- Incremental sales.
- Incremental percentage lift.
- Uncertainty interval.
- Sensitivity check against the seasonal benchmark.

Suggested table:

| Metric | Value |
|---|---:|
| Observed or planned scenario sales | |
| Counterfactual sales | |
| Incremental sales | |
| Incremental lift | |
| Lower interval | |
| Upper interval | |

## Near-Term Improvements

1. Done: extend phase 5 and phase 6 to support configurable horizons and package 180-day and 365-day outputs.
2. Done: add an opening scenario generator and counterfactual comparison utility.
3. Done: add a unit-level target preparation script for existing-unit and new-unit sales, pending unit-level input data.
4. Done: add a new-unit ramp-curve estimator, pending unit-level input data.
5. Done: add model cards for each production candidate, including intended use, validation horizon, and counterfactual suitability.
6. Done (2026-06-10): multi-fold nested final validation at 91 days (`run_multifold_final_validation.py`); seasonal naive won all 5 folds and is now the 91-day production recommendation.
7. In progress (2026-06-10): `explore_naive_beaters.py` swept parameter-free candidates over all grid windows from the four multifold runs. A fixed 0.75 seasonal-naive / 0.25 TimesFM blend beat plain seasonal naive in 25/29 windows at 28 days, 24/27 at 91 days, 19/23 at 180 days, and 16/17 at 365 days (recent-6 windows: 5/6, 6/6, 3/6, 5/6), with worst single-window losses of +0.008/+0.000/+0.004/+0.001 WAPE respectively. The pattern holds at 62/38 and 50/50, so it is not ratio-mined. Pre-registered pick: **75/25**. Track it against seasonal naive on future windows; promote to production after it holds out-of-exploration. Notes: at 180 days the recent edge is roughly neutral, and growth-adjusted seasonal naive (the best non-foundation fallback at short horizons) degrades badly at 365 days because a trailing YoY ratio does not persist a year out.
8. Done (2026-06-10): multi-fold nested validation at 28 days (6 non-overlapping folds, run_id `20260610T044622Z`); seasonal naive won 4 of 6 folds and is now the 28-day production recommendation, with Extra Trees retained as the opening-aware scenario model.
9. Done (2026-06-10): multi-fold nested validation at 180 days (5 folds, run_id `20260610T112337Z`) and 365 days (1 feasible fold, run_id `20260610T113121Z`); seasonal naive won everywhere and the long-horizon stacks were demoted to sensitivity comparators. Their earlier focused-run wins traced to weight-fitting windows overlapping the eval windows by ~5 and ~11 months.
10. Done (2026-06-10): `opening_info.csv` grew from 4 to 10 openings (2 with fully observed ramps, 3 planned after the current data end). The 28/91-day multifold runs were rerun with the new file (run_ids `20260610T124426Z`/`20260610T124857Z`): no verdict changed, and richer opening features slightly worsened the opening-aware ML models. The aggregate event study found per-unit lift unidentifiable (see the warning in section 4). The production forecast manifest now lists planned openings inside each forecast window as an explicit unquantified caveat.
