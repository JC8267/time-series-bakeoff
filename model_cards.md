# Production Model Cards

Last updated: 2026-07-23

Validation fold note: every WAPE/bias claim below should be read with its fold count. The latest 28/91 phase-1 and foundation reruns used 3 rolling cutoffs per horizon, phase 3 trained stack weights on the earlier aligned cutoff and evaluated on 2 held-out cutoffs, and phase 5 used 1 final holdout. The latest 180/365 validation used 3 rolling cutoffs per horizon. Fewer-fold sensitivity tests can make `seasonal_naive_lag_364` win; treat those as sensitivity checks unless the ensemble candidates are rebuilt and scored on the same folds.

2026-06-10 update: multi-fold nested final validations settled the rankings at every horizon (`run_multifold_final_validation.py`; 91 days: 5 folds, run_id `20260610T124857Z`; 28 days: 6 non-overlapping folds, run_id `20260610T124426Z`; 180 days: 5 folds, run_id `20260610T112337Z`; 365 days: 1 fold plus a 17-window parameter-free sweep, run_id `20260610T113121Z`; outputs in the matching `outputs_multifold_*_validation/` directories). The 28/91-day runs were rerun after `opening_info.csv` grew from 4 to 10 openings (initial run_ids `20260610T044622Z`/`20260610T042656Z`, archived); the richer opening features did not change any verdict and slightly worsened the opening-aware ML models, consistent with the aggregate event study (`estimate_opening_lift_event_study.py`) finding per-unit opening lift unidentifiable from total-sales data. The production recommendation at all four horizons is now `seasonal_naive_lag_364`. The learned convex stacks lost every leak-free fold at every horizon; their earlier wins traced to phase-3 weight-fitting windows overlapping the evaluation windows (31 days at 91d, 19 days at 28d, ~5 months at 180d, ~11 months at 365d). A fixed 0.75 seasonal-naive / 0.25 TimesFM blend is pre-registered as the tracked challenger (see `model_usage_plan.md`). As of 2026-07-23, `run_production_forecast.py` accepts 1-365-day horizons, keeps `production_forecast.csv` final-only, and writes the challenger only to an opt-in `diagnostic_forecast.csv` for exact, non-degraded validation grids. See the per-card verdicts below.

2026-06-24/25 weekly-grain update: weekly validation now exists for 4/12/26/52-week horizons using complete `W-WED` weeks (`outputs_weekly_bakeoff_baselines_4_12_26_52`, `outputs_weekly_multifold_*w_baseline_pool`, `outputs_weekly_multifold_*w_timesfm_pool`, `outputs_weekly_multifold_*w_foundation_pool`, `outputs_weekly_multifold_*w_ml_pool`, `outputs_weekly_multifold_*w_daily_reagg_pool`, and `outputs_weekly_multifold_*w_daily_reagg_foundation_pool`). The weekly primary recommendation is now `daily_reagg_seasonal_naive_lag_365`: forecast daily with lag 365, then sum into `W-WED` weeks. Native `seasonal_naive_lag_52` remains the benchmark. Same-fold diagnostic ensembles beat it in the full baseline grid, but those ensembles are selected on the same rows they score. Honest nested baseline, broad foundation, standalone ML tree pools, and a broad foundation pool that included lag-365 did not beat the standalone lag-365 rule. The retired holiday-aligned weekly rule worsened the holiday folds because its anchor can jump a full weekly bucket when the holiday weekday changes. The 12/26/52-week nested reads use 4-week steps and are overlapping windows, not independent folds.

2026-06-25 seasonal trend/averaging update: level-scaled seasonal naive and recency-weighted multi-year seasonal averages were added at both daily and weekly grains (`outputs_multifold_*d_trend_seasonal_pool`, `outputs_weekly_multifold_*w_trend_seasonal_pool`). Standalone scaled/multi-year rules did not beat the existing daily or weekly champions. Overlap-aware significance tests on extended nested windows (`outputs_forecast_significance_tests`) found the daily 180d inverse-WAPE stack directionally better but not statistically confirmed (mean WAPE diff -0.00536, HAC p=0.147, block CI crosses zero). Weekly 26w `pool_median` has a small but statistically cleaner edge over `daily_reagg_seasonal_naive_lag_365` (mean diff -0.00140, HAC p=0.017), so track it as the 26w challenger; keep lag-365 as the general weekly default until the 26w challenger is packaged and holds up after refresh.

2026-06-25 interval coverage update: `run_interval_coverage_backtest.py` measured rolling, prior-cutoff-only residual-quantile coverage. Daily and weekly production scripts now support `--interval-mode global|segment`. Default remains `global`: segment conditioning helped some daily holiday slices but reduced overall coverage and badly undercovered short weekly holiday rows.

2026-07-02 update: holiday-anchor-aware daily seasonal naive was pre-registered and tested (`outputs_multifold_*d_holiday_anchor_pool`, `outputs_weekly_multifold_*w_holiday_anchor_pool`, `outputs_forecast_significance_tests_holiday_anchor`). It is rejected. Daily results: +0.0010 WAPE at 28d, +0.0015 at 91d, -0.0024 at 180d with HAC p=0.409 and block CI crossing zero, and +0.0012 on the single feasible 365d fold. Weekly results versus `daily_reagg_seasonal_naive_lag_365`: +0.0025 at 4w, +0.0025 at 12w, +0.0022 at 26w, and +0.0024 at 52w. Segment summaries confirm the mechanism improved holiday weeks but degraded rest weeks enough to lose overall. Do not tune the windows.

2026-07-02 interval update: `shrunken` interval mode was added to the backtest and production packagers, with shrinkage k tested at 10/30/60 (`outputs_interval_coverage_backtest_shrunken_k*`). It is not the default. None of the shrunken settings met the weekly 4w/12w coverage floor; increasing global calibration windows to 16 or all available also reduced weekly 80% coverage. Keep `global` with 8 windows as the production default pending a better interval design.

2026-07-02 monitoring/covariate update: `run_forecast_monitor.py` now appends relative-only realized WAPE, bias pct, coverage, CUSUM alarm state, and final-vs-challenger verdicts. The weekly production package now emits `challenger_26w_pool_median_trend_seasonal` with intervals. `run_residual_covariate_model.py` scored `residual_ridge_f1` and `residual_ridge_f1f2` at all daily and weekly horizons (`outputs_residual_covariate_model`, `outputs_forecast_significance_tests_residual_covariates`); both lose badly everywhere, so no covariate residual model is promoted.

2026-07-09 update: a pre-registered fixed 50/50 daily lag-364/lag-365 blend was rejected. It won 0/14 extended windows at 28/91/180 days and 0/1 at 365 days, with mean WAPE losses of +0.0720/+0.0685/+0.0620/+0.0719 versus lag 364. Daily lag 365 shifts the weekday pattern; keep it only for daily-to-weekly reaggregation. Multi-fold runs now record the missing-date strategy and filled-day count and write an explicit unweighted-metrics warning plus audit path when gaps are filled.

2026-07-15 offer-level update: at a 123-day daily horizon, the pre-registered fixed 0.75 `seasonal_naive_lag_364` / 0.25 TimesFM blend beat seasonal naive on all 5 rolling folds for the Total series (mean WAPE 0.0700 vs 0.0750) and improved the untouched 2026-05-01 through 2026-07-14 Total holdout (WAPE 0.0771 vs 0.0785). This is offer-specific supporting evidence, not a project-wide promotion: consecutive folds overlap by 95 of 123 days, leaving only about 1.14 effective independent windows. The weekly offer test did not transfer; daily-reaggregated lag-365 remained the coherent weekly winner.

## `weekly_daily_reagg_seasonal_naive_lag_365`

Purpose: weekly-grain total-sales production forecast.

Validated horizons: 4, 12, 26, and 52 weeks.

Use for:

- Weekly-grain planning when forecasts are consumed as `W-WED` weekly totals.
- Forecast windows where holiday week allocation matters.
- Benchmarking future weekly ensembles against a parameter-free calendar-date baseline.

Do not use for:

- Daily forecast delivery; use the day-grain cards instead.
- Opening incrementality attribution by itself.

Notes:

- Forecasts daily sales with a 365-day lag, then sums the daily forecasts into `W-WED` buckets.
- This differs from daily lag-364 reaggregation. Lag 364 is exactly equivalent to native weekly `seasonal_naive_lag_52` under `W-WED` buckets, so it cannot fix the weekly phase issue.
- The main lift is holiday allocation: at 4 weeks the 2025-12-03 fold improved from 0.1048 WAPE for native weekly seasonal naive to 0.0496.
- In the lag-365 foundation pool, no ensemble beat standalone lag-365 at any horizon. The closest rows were `nested_inverse_wape_stack` at 4 weeks, `pool_median` at 12/26/52 weeks; all had higher mean WAPE than lag-365.
- In the seasonal trend/averaging pool, `pool_median` beat lag-365 at 26w and 52w but not at 4w/12w. The 26w edge survived an overlap-aware test; the 52w read remains too small to trust.
- Weekly production output is packaged by `run_weekly_production_forecast.py` in `outputs_weekly_production_forecast/weekly_production_forecast.csv`, with global residual-quantile intervals by default and optional holiday/rest segment or shrunken intervals for diagnostics.
- The package now includes `challenger_26w_pool_median_trend_seasonal` only at 26w so monitor refreshes can adjudicate the statistically cleaner long-weekly challenger out of exploration.
- The 26- and 52-week evidence is still data-limited because the final folds are heavily overlapping.

Latest validation:

- 4 weeks: mean WAPE 0.0386 vs 0.0489 for `seasonal_naive_lag_52`; mean bias 0.0039; 3/5 wins.
- 12 weeks: mean WAPE 0.0410 vs 0.0505; mean bias -0.0032; 4/5 wins.
- 26 weeks: mean WAPE 0.0352 vs 0.0397; mean bias -0.0108; 4/5 wins, heavily overlapping read.
- 52 weeks: mean WAPE 0.0347 vs 0.0389; mean bias 0.0000; 2/2 wins, effectively one annual read.
- Best lag-365-included ensemble WAPE: 0.0492 at 4w, 0.0523 at 12w, 0.0401 at 26w, and 0.0378 at 52w. These do not beat standalone lag-365.

Production read: use `daily_reagg_seasonal_naive_lag_365` as the weekly-grain default for 4/12/26/52 weeks. Track the packaged 26w `pool_median` seasonal-trend challenger, but do not replace the default until it survives a refresh.

## `weekly_seasonal_naive_lag_52`

Purpose: weekly-grain total-sales benchmark.

Validated horizons: 4, 12, 26, and 52 weeks.

Use for:

- Weekly-grain planning when forecasts are consumed as weekly totals.
- Benchmarking weekly ensembles and foundation models.
- Sensitivity comparison beside `daily_reagg_seasonal_naive_lag_365`.

Do not use for:

- Daily forecast delivery; use the day-grain cards instead.
- Opening incrementality attribution by itself.

Notes:

- Weekly rows are complete weeks ending Wednesday (`W-WED`), matching the current data cutoff of 2026-04-01.
- Full baseline grid used 114/27/23/17 cutoffs for 4/12/26/52 weeks.
- Nested baseline, broad foundation, and standalone ML pools did not beat this benchmark robustly, but daily lag-365 reaggregation did.
- Weekly ML pools with `sklearn_hist_gradient_boosting`, direct HistGradientBoosting, random forest, Extra Trees, ridge, and LightGBM did not beat this benchmark. Best standalone ML WAPE was 0.0560 at 4w for ridge, 0.0587 at 12w for LightGBM, 0.0684 at 26w for direct HistGradientBoosting, and 0.0597 at 52w for direct HistGradientBoosting.
- The 12/26/52-week nested reads are overlapping: effective independent windows are about 2.33, 1.62, and 1.08 respectively. The 52-week read is data-limited and has only 2 final folds.
- Weekly `data_gap_filled_flag` is carried through aggregation but is not used for metric weighting. Multi-fold manifests now disclose any fills and warn that affected rows are unweighted. These runs had no filled missing days.

Latest validation:

- 4 weeks: mean WAPE 0.0489, mean bias 0.0045 across 5 non-overlapping nested windows; lost to `daily_reagg_seasonal_naive_lag_365` by 0.0103 mean WAPE.
- 12 weeks: mean WAPE 0.0505, mean bias -0.0031 across 5 overlapping nested windows; lost by 0.0095 mean WAPE.
- 26 weeks: mean WAPE 0.0397, mean bias -0.0117 across 5 heavily overlapping nested windows; lost by 0.0045 mean WAPE.
- 52 weeks: mean WAPE 0.0389, mean bias -0.0001 across 2 heavily overlapping nested windows; lost by 0.0042 mean WAPE.

Production read: keep `seasonal_naive_lag_52` as the benchmark row, not the weekly default.

## `weekly_seasonal_naive_holiday_aligned_lag_52`

Purpose: retired weekly-grain holiday-alignment experiment.

Validated horizons: 4, 12, 26, and 52 weeks.

Use for:

- Research comparisons against plain weekly seasonal naive near Thanksgiving, Christmas, and New Year.

Do not use for:

- Production weekly forecasts.
- Winner claims over `seasonal_naive_lag_52`.

Notes:

- This fixed rule mapped holiday-adjacent target weeks to the prior year's matching holiday week and fell back to `seasonal_naive_lag_52` elsewhere.
- It failed for a concrete reason: when a holiday's weekday changes, the target anchor can jump a full `W-WED` bucket. At the 2025-12-03 4-week fold it worsened WAPE from 0.1048 to 0.1794.
- It has been removed from active `model_specs`; do not patch or reuse it.

Latest validation:

- 4 weeks: mean WAPE 0.0686 vs 0.0489 for `seasonal_naive_lag_52`; 0/5 wins.
- 12 weeks: mean WAPE 0.0697 vs 0.0505 for `seasonal_naive_lag_52`; 0/5 wins.
- 26 weeks: mean WAPE 0.0486 vs 0.0397 for `seasonal_naive_lag_52`; 0/5 wins.
- 52 weeks: mean WAPE 0.0461 vs 0.0389 for `seasonal_naive_lag_52`; 0/2 wins.

## `weekly_pool_median_seasonal_naive_timesfm`

Purpose: weekly-grain TimesFM sensitivity/challenger.

Validated horizons: 4 and 12 weeks as a tracked challenger; checked and rejected at 26 and 52 weeks.

Use for:

- Monitoring whether a simple seasonal-naive/TimesFM blend continues to beat weekly seasonal naive.
- Sensitivity comparison when a foundation-model signal is desired.

Do not use for:

- Default weekly production forecasts until it wins future out-of-exploration folds.
- Opening or promotion incrementality attribution.

Notes:

- This is the row named `pool_median` in the narrow pool containing only `seasonal_naive_lag_52` and `timesfm_2_5`.
- It beat `seasonal_naive_lag_52` on mean WAPE at 4 and 12 weeks, but the evidence is not decisive: 2/5 wins at 4 weeks and 3/5 wins at 12 weeks.
- At 4 weeks, the net mean edge is driven by the 2025-12-03 holiday fold: pool median WAPE 0.0688 vs 0.1048 for `seasonal_naive_lag_52`. Seasonal naive has the better 4-week median WAPE (0.0328 vs 0.0483), so interpret the challenger as tail/holiday cushioning rather than a generally better forecast.
- The advantage did not survive adding Chronos and StatsForecast MSTL models to the broader foundation pool.

Latest validation:

- 4 weeks: mean WAPE 0.0455 vs 0.0489 for `seasonal_naive_lag_52`; mean bias 0.0029; 2/5 wins.
- 12 weeks: mean WAPE 0.0472 vs 0.0505 for `seasonal_naive_lag_52`; mean bias 0.0000; 3/5 wins.
- 26 weeks: mean WAPE 0.0543 vs 0.0397 for `seasonal_naive_lag_52`; rejected.
- 52 weeks: mean WAPE 0.0510 vs 0.0389 for `seasonal_naive_lag_52`; rejected on 2 feasible folds.

## `final_28d_extra_trees_opening_timesfm_pool`

Purpose: short-term total-sales forecasting.

Validated horizon: 28 days.

Use for:

- Weekly and monthly near-term sales planning.
- Short campaign windows up to 28 days.
- Short counterfactuals when opening features should remain controlled.

Do not use for:

- Standalone new-unit incrementality attribution without a matching no-opening scenario.
- Long-range planning beyond the validated horizon.

Notes:

- Best final holdout WAPE in the opening-aware TimesFM-pool rerun.
- Uses an opening-aware feature set through the ML forecast pipeline.
- The Statsmodels-fixed rerun did not change this recommendation.

**2026-06-10 multi-fold verdict: prefer `seasonal_naive_lag_364` at 28 days. Extra Trees won only 1 of 6 folds.**

Provenance: `run_multifold_final_validation.py --horizon 28`, run_id `20260610T124426Z`, `outputs_multifold_28d_validation/` (6 non-overlapping 28-day holdout folds, cutoffs 2025-10-15 to 2026-03-04; rerun with the 10-opening `opening_info.csv`).

- `seasonal_naive_lag_364`: mean WAPE 0.0680. `sklearn_extra_trees`: mean WAPE 0.0892, winning 1 of 6 folds (mean diff +0.021). With the earlier 4-opening file it scored 0.0865 and won 2 of 6 — richer opening features made it slightly worse, consistent with the event-study finding that aggregate data carries no identifiable opening signal.
- The phase-5 single-holdout win below is clean for standalone Extra Trees (no weight fitting) and lands on its best fold (0.0333 vs naive 0.0386) — but it is 1 of 6 folds and not representative.
- Extra Trees is volatile: competitive on most folds but failed badly on the December 2025 window (WAPE 0.177 vs naive 0.125), underforecasting through the holiday season. Do not use it unhedged for windows that span major holidays.
- Production read: use `seasonal_naive_lag_364` as the 28-day production forecast. Keep Extra Trees as the opening-aware scenario/counterfactual model, reported beside the seasonal benchmark.

Latest validation:

- Phase 5 holdout WAPE: 0.0314.
- Phase 5 holdout bias: 0.0020.
- Validation folds: 1 final holdout window; supporting phase-1 rerun used 3 rolling cutoffs.
- Caveat: `seasonal_naive_lag_364` beat Extra Trees on 4 of 6 folds of the 2026-06-10 multi-fold validation (run_id `20260610T044622Z`). Treat `seasonal_naive_lag_364` as the 28-day production model.

## `final_91d_convex_stack_opening_timesfm_pool`

Purpose: medium-term total-sales forecasting.

Validated horizon: 91 days.

Use for:

- Quarterly sales planning.
- Medium-length promotion or operational counterfactuals.
- Production forecasts where TimesFM and classical/ML models should be blended.

Do not use for:

- Attribution where the intervention variable is not represented in the scenario features.
- Windows materially longer than 91 days without a long-horizon sensitivity check.

Notes:

- TimesFM improved the 91-day ensemble in final holdout testing.
- The latest rerun also added StatsForecast and Chronos-Bolt candidates. Chronos-Bolt did not improve the 91-day final holdout, but remains available as a diagnostic foundation-model comparator.
- The Statsmodels-fixed rerun confirmed `statsmodels_ets_weekly` and `statsmodels_sarimax_weekly` now run, but neither improved the 91-day winner.
- The selected stack is not meaningfully opening-sensitive in its current fitted form. Latest convex weights are mostly `seasonal_naive_lag_364` and `timesfm_2_5`, with zero weight on Chronos/MSTL and no opening-feature ML model in the final stack.

**2026-06-10 multi-fold verdict: prefer `seasonal_naive_lag_364` at 91 days. The stack strategy lost all 5 nested validation folds.**

Provenance: `run_multifold_final_validation.py`, run_id `20260610T124857Z`, `outputs_multifold_91d_validation/` (5 rolling 91-day holdout folds, cutoffs 2025-09-10 to 2025-12-31 stepped 28 days; folds overlap, so they are not fully independent; stack weights refit per fold on only the backtest windows ending at or before that fold's cutoff; rerun with the 10-opening file and `sklearn_extra_trees` added to the pool, where it finished near last at 0.1057 mean WAPE).

- Nested convex stack (the honest version of the production strategy): mean WAPE 0.0903, **0/5 wins** vs `seasonal_naive_lag_364` (mean WAPE 0.0757), losing every fold.
- Root cause is weight instability, not blending per se. On the 2024 to mid-2025 fitting windows TimesFM dominated (inner WAPE ~0.08 vs ~0.13-0.15 for seasonal naive), so every nested fit put ~90% weight on TimesFM; on the late-2025 holdout windows the regime flipped and seasonal naive won everywhere. The strategy chases stale relative performance.
- The earlier phase-5 single-holdout "win" (0.0507 vs 0.0550) is contaminated: phase 3 fit the production weights at cutoff 2025-11-01, whose 91-day fitting window (through 2026-01-31) overlaps the phase-5 holdout (2026-01-01 to 2026-04-01) by 31 days.
- The frozen shipped weights (0.62 seasonal naive / 0.38 TimesFM) do beat seasonal naive on all 5 folds by small margins (0.0013-0.0091 WAPE), but those weights postdate 4 of the 5 folds and were fit on data overlapping the 5th, so this is suggestive, not validated. If a blend is wanted, pre-register a fixed ratio and validate it on future folds before shipping it as the winner.
- Production read: use `seasonal_naive_lag_364` as the 91-day production forecast. Keep this stack as a sensitivity/diagnostic model and for cases where TimesFM signal is explicitly wanted alongside the seasonal baseline.
- For opening or ramp-timing counterfactuals at 91 days, do not rely on this stack alone. Pair it with an explicitly opening-sensitive ML scenario model or rerun phase 5 with the opening-sensitive model retained as a named candidate.

**Interval note:** phase 4 was rerun with standalone `timesfm_2_5` included in interval calibration. The corrected phase-5 TimesFM interval metrics are now usable: 91-day standalone TimesFM coverage is 0.857 at the 80% level and 0.945 at the 95% level. The production stack interval metrics remain calibrated from the stack's own phase-3 residuals (`phase3_heldout_cutoffs_convex_stack`).

Latest validation:

- Phase 5 holdout WAPE: 0.0507.
- Phase 5 holdout bias: -0.0116.
- Phase 5 holdout interval coverage: 0.857 at 80%, 0.956 at 95%.
- Validation folds: 1 final holdout window; supporting phase-1/foundation reruns used 3 rolling cutoffs and phase 3 evaluated stacks on 2 held-out cutoffs.
- Caveat: the phase-5 holdout WAPE above is contaminated by the weight-fitting overlap described in the multi-fold verdict, and `seasonal_naive_lag_364` beat the nested stack on all 5 folds of the 2026-06-10 multi-fold validation (run_id `20260610T042656Z`). Treat `seasonal_naive_lag_364` as the 91-day production model.

## `final_180d_seasonal_naive_benchmark`

Purpose: 180-day pure forecast accuracy benchmark.

Validated horizon: 180 days.

Use for:

- Long-horizon benchmark comparisons.
- Sanity checks against opening-sensitive ML forecasts.

Do not use for:

- Opening incrementality or expansion attribution.

Notes:

- Strong 180-day benchmark in the opening-aware long-horizon run.
- It does not use opening features, so it cannot distinguish organic growth from new-unit growth.
- **2026-06-10: promoted to the 180-day production model.** It won all 5 folds of the multi-fold nested validation (mean WAPE 0.0638 vs 0.0879 for the nested convex stack; run_id `20260610T112337Z`, `outputs_multifold_180d_validation/`).

Latest validation:

- Multi-fold mean WAPE: 0.0638, mean bias -0.0096 (5 rolling folds, 2026-06-10).
- Earlier focused validation WAPE: 0.0686, bias -0.0086 (3 rolling cutoffs).

## `final_180d_convex_stack_stats_chronos_pool`

Purpose: 180-day long-horizon forecast candidate.

Validated horizon: 180 days.

Use for:

- Long-horizon forecast sensitivity against the seasonal benchmark.
- Planning forecasts where a small stack is preferred over a single seasonal baseline.

Do not use for:

- Opening incrementality attribution by itself.
- Definitive winner claims without the limited-fold caveat.

Notes:

- Latest focused validation WAPE was 0.0648 for the held-out convex stack and 0.0645 for the inverse-WAPE stack.
- Packaged phase 6 output uses the convex stack for a stable named production-style artifact.

**2026-06-10 multi-fold verdict: demoted to diagnostic. The nested stack lost all 5 leak-free folds.**

- Nested convex stack mean WAPE 0.0879 vs 0.0638 for `seasonal_naive_lag_364` (0/5 wins; run_id `20260610T112337Z`, `outputs_multifold_180d_validation/`).
- The focused-run win above was contaminated: phase 3 fit the 180-day weights at cutoff 2025-08-01, whose fitting window (through 2026-01-28) overlaps the 2025-09-01/2025-10-01 eval windows by roughly 5 months.
- Use `final_180d_seasonal_naive_benchmark` for production; keep this stack only as a sensitivity comparator.

## `final_180d_hist_gradient_boosting_opening_sensitive`

Purpose: 180-day opening-sensitive forecasting and counterfactual support.

Validated horizon: 180 days.

Use for:

- Opening-aware 180-day scenario comparisons.
- Counterfactuals where new-unit timing matters.
- Sensitivity analysis against the seasonal benchmark.

Do not use for:

- Pure accuracy claims without comparing to `final_180d_seasonal_naive_benchmark`.

Notes:

- Opening-sensitive 180-day model that can use opening features.
- It did not beat the seasonal or stack benchmarks on pure WAPE in the latest focused run.
- Use as a counterfactual/scenario sensitivity model when openings or ramp timing matter.
- Validation folds: 3 rolling cutoffs.

## `final_365d_seasonal_naive_benchmark`

Purpose: 365-day pure forecast accuracy benchmark.

Validated horizon: 365 days.

Use for:

- Annual planning benchmark.
- Sanity checks for annual forecasts.

Do not use for:

- Opening, promotion, or expansion attribution.

Notes:

- Strong 365-day benchmark in the opening-aware long-horizon run.
- It is intentionally simple and opening-insensitive.
- **2026-06-10: promoted to the 365-day production model.** It won the single feasible nested fold (WAPE 0.0568 vs 0.0806 for the nested stack; run_id `20260610T113121Z`, `outputs_multifold_365d_validation/`), and beat the nested stack candidates across the 17-window grid. Annual validation remains data-limited; the supporting evidence is the consistency of this result with the 28/91/180-day verdicts.

Latest validation:

- Multi-fold nested fold WAPE: 0.0568, bias 0.0001 (1 fold, 2026-06-10).
- Earlier focused validation WAPE: 0.0591, bias 0.0019 (3 rolling cutoffs).

## `final_365d_convex_stack_stats_chronos_pool`

Purpose: 365-day long-horizon forecast candidate.

Validated horizon: 365 days.

Use for:

- Annual forecast sensitivity against the seasonal benchmark.
- Planning forecasts where a stack of seasonal, StatsForecast, and TimesFM signals is preferred.

Do not use for:

- Opening incrementality attribution by itself.
- Definitive annual model superiority claims without the limited-fold caveat.

Notes:

- Latest focused validation WAPE was 0.0563 for the held-out convex stack.

**2026-06-10 multi-fold verdict: demoted to diagnostic. The nested stack lost the only leak-free fold.**

- Nested convex stack WAPE 0.0806 vs 0.0568 for `seasonal_naive_lag_364` (run_id `20260610T113121Z`, `outputs_multifold_365d_validation/`).
- The focused-run win above was contaminated: phase 3 fit the 365-day weights at cutoff 2025-02-01, whose fitting window (through 2026-02-01) overlaps the 2025-03-01/2025-04-01 eval windows by roughly 11 months — the eval was nearly in-sample.
- Only one leak-free nested fold is feasible with current history, but the verdict matches the 28/91/180-day pattern and the stack also trailed across the 17-window parameter-free sweep.
- Use `final_365d_seasonal_naive_benchmark` for production; keep this stack only as a sensitivity comparator.

## `final_365d_hist_gradient_boosting_opening_sensitive`

Purpose: 365-day opening-sensitive forecasting and counterfactual support.

Validated horizon: 365 days.

Use for:

- Annual opening-aware scenarios.
- Counterfactuals where the no-event case differs by unit openings or ramp timing.
- Long-window incrementality analysis with a seasonal benchmark sensitivity check.

Do not use for:

- Pure annual forecast winner claims without reporting the seasonal naive benchmark.

Notes:

- Opening-sensitive 365-day model that can use opening features.
- It did not beat the seasonal or stack benchmarks on pure WAPE in the latest focused run.
- Use as a counterfactual/scenario sensitivity model when openings or ramp timing matter.
- Validation folds: 3 rolling cutoffs.
