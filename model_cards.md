# Production Model Cards

Last updated: 2026-09-25

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

2026-08-30 total-sales holdout update: the fixed 75/25 challenger improved mean WAPE by about 0.0018 at both 28 and 91 days on newly available forecast origins. It won 3/5 non-overlapping 28-day windows and 3/3 overlapping 91-day windows (about 1.62 effective independent windows). The realized 2026-04-01 production package also favored it at both horizons. This is supportive but the absolute gain is small and 28-day consistency is only 60%, so it remains a diagnostic challenger rather than replacing seasonal naive.

2026-09-17 challenger refresh: `latest_sls_aug.csv` is unchanged at 2026-08-29 (SHA-256 `7639a348c80c4068cbebea18464a040525907121480102be0189bb3edd47399e`), so the daily 28/91 confirmation scorecard is unchanged and no post-registration 180/365-day windows have matured. The weekly 26w trend/seasonal `pool_median` was rerun on the same snapshot with its exact registered six-member pool. On the six newly matured, heavily overlapping windows (origins 2025-10-08 through 2026-02-25; about 1.77 effective independent windows), it lost to `daily_reagg_seasonal_naive_lag_365`: mean WAPE 0.03658 vs 0.03511, 2/6 wins, mean difference +0.00147 and worst loss +0.00478. The weekly challenger therefore failed confirmation; production remains lag-365.

## `challenger_daily_blend_snl75_tfm25`

Purpose: fixed, parameter-free daily challenger to the seasonal-naive production rule.

Validated horizons: exploratory evidence at 28/91/180/365 days; post-registration holdout evidence currently available at 28 and 91 days.

Use for:

- Diagnostic comparison beside the final 28/91-day production forecast.
- Realized monitoring of whether TimesFM adds stable value beyond annual seasonal naive.

Do not use for:

- Replacing `seasonal_naive_lag_364` without another review of the small gain and operational TimesFM cost.
- Opening or promotion incrementality attribution.
- 180/365-day promotion claims before those post-registration windows mature.

Notes:

- Formula is frozen at 0.75 × `seasonal_naive_lag_364` + 0.25 × `timesfm_2_5`; no weights are fitted.
- `run_production_forecast.py --include-challenger` writes it only to `diagnostic_forecast.csv`; production rows remain final-only.
- Refreshed holdout results: 28d WAPE 0.0565 vs 0.0582 (3/5 wins); 91d WAPE 0.0610 vs 0.0627 (3/3 wins, overlapping).
- The 2026-04-01 realized package scored 0.0851 vs 0.0912 at 28d and 0.0611 vs 0.0638 at 91d.

Production read: keep tracking. Do not promote yet.

## `timesfm_3` (exploratory only)

Purpose: one-fold research comparison of TimesFM 3 against the seasonal-naive benchmark and the existing TimesFM 2.5 challenger.

Do not use for:

- Commercial or production forecasting with the default `google/timesfm-3.0-pytorch` weights; their license prohibits that use.
- Winner or promotion claims from these single-fold runs.

August 31 exploratory scorecard (`outputs_multifold_*d_timesfm3_aug2026_test`; one fold per horizon):

| Horizon | Fold origin | Seasonal naive WAPE | TimesFM 3 WAPE | Three-member median WAPE | Verdict |
|---:|---|---:|---:|---:|---|
| 28d | 2026-08-01 | 0.02947 | 0.03730 | 0.03235 | TimesFM 3 and median lose |
| 91d | 2026-05-30 | 0.05312 | 0.05366 | 0.04899 | Median improves one exploratory fold |
| 180d | 2026-03-02 | 0.05585 | 0.06114 | 0.05917 | TimesFM 3 and median lose |
| 365d | 2025-08-29 | 0.06184 | 0.07379 | 0.06748 | TimesFM 3 and median lose |

Notes:

- The median is the pointwise median of `seasonal_naive_lag_364`, `timesfm_2_5`, and `timesfm_3`.
- Standalone TimesFM 3 lost to seasonal naive at every horizon. Only the three-member 91-day median improved, by 0.00413 WAPE on one fold.
- `run_multifold_final_validation.py` blocks the default TimesFM 3 weights unless the explicitly non-commercial override is supplied. Do not bypass that guard for this business shortlist.

Production read: excluded by license and insufficient evidence.

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
- In the original seasonal trend/averaging pool, `pool_median` beat lag-365 at 26w and 52w but not at 4w/12w. The 26w exploratory edge survived an overlap-aware test, but the September refresh below failed confirmation.
- Weekly production output is packaged by `run_weekly_production_forecast.py` in `outputs_weekly_production_forecast/weekly_production_forecast.csv`, with global residual-quantile intervals by default and optional holiday/rest segment or shrunken intervals for diagnostics.
- The package includes `challenger_26w_pool_median_trend_seasonal` only as a 26w diagnostic row. Its September refresh failed confirmation; do not promote it.
- The 26- and 52-week evidence is still data-limited because the final folds are heavily overlapping.

Latest validation:

- 4 weeks: mean WAPE 0.0386 vs 0.0489 for `seasonal_naive_lag_52`; mean bias 0.0039; 3/5 wins.
- 12 weeks: mean WAPE 0.0410 vs 0.0505; mean bias -0.0032; 4/5 wins.
- 26 weeks: mean WAPE 0.0352 vs 0.0397; mean bias -0.0108; 4/5 wins, heavily overlapping read.
- 52 weeks: mean WAPE 0.0347 vs 0.0389; mean bias 0.0000; 2/2 wins, effectively one annual read.
- Best lag-365-included ensemble WAPE: 0.0492 at 4w, 0.0523 at 12w, 0.0401 at 26w, and 0.0378 at 52w. These do not beat standalone lag-365.
- Refreshed 26w confirmation (`outputs_weekly_multifold_26w_trend_seasonal_refresh_sep2026`, run_id `20260917T143838Z`): lag-365 WAPE 0.03511 and bias -0.00672 vs `pool_median` WAPE 0.03658 and bias 0.00195 across 6 windows; challenger won 2/6, mean WAPE difference +0.00147, worst loss +0.00478. Origins were 2025-10-08 through 2026-02-25 with about 1.77 effective independent windows.

Production read: use `daily_reagg_seasonal_naive_lag_365` as the weekly-grain default for 4/12/26/52 weeks. The packaged 26w `pool_median` seasonal-trend row is diagnostic only and failed its first confirmation refresh.

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

## `tirex_2`

Purpose: exploratory zero-shot aggregate daily-sales challenger.

Screened horizons: 28 and 91 days.

Use for:

- Reproducibility checks of the frozen 2026-09-21 screen only.

Do not use for:

- Production forecasts or opening/promotion attribution.
- 180/365-day claims; those horizons were not screened.

Notes:

- Pre-registration: `new_model_screen_registration.json`; checkpoint `NX-AI/TiRex-2@05e5b26db52bfb256f1ae1bdf785589850482de3`, target-only 2,048-day context, median forecast, no covariates.
- Artifact: `outputs_new_model_screen_sep2026` (run_id `20260921T143751Z`). The same registered windows and actuals were used for naive, frozen 75/25 naive/TimesFM and TiRex comparisons.
- Standalone TiRex-2 lost badly: 28d WAPE 0.07693 versus 0.05888 naive; 91d WAPE 0.09202 versus 0.06490 naive.
- The fixed 75/25 naive/TiRex blend slightly beat naive but failed the registered comparator gate: 28d WAPE 0.05852 versus 0.05685 for naive/TimesFM; 91d WAPE 0.06256 versus 0.06084. Verdict: reject at both horizons.
- Local smoke and scoring used an isolated Python 3.12 CPU environment. `tirex-2==0.2.1`, `torch==2.8.0+cpu` and `triton-windows==3.4.0.post21` were required; no production environment changed.

Production read: rejected. Keep `seasonal_naive_lag_364` as the production default and the frozen naive/TimesFM blend as a diagnostic only.

## `tabpfn_ts_3_5`

Purpose: registered exploratory tabular time-series candidate.

Screened horizons: 28 and 91 days.

Do not use for:

- Business production or commercial decision-making without a separate commercial license.
- Cloud inference with the private sales data.

Notes:

- Pre-registration pinned `tabpfn-time-series==1.2.0` and `Prior-Labs/tabpfn_3_5@06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`, with the package's running-index, calendar and auto-seasonal features and no supplied covariates.
- After the research terms were accepted, local smoke passed and the frozen screen completed in `outputs_new_model_screen_tabpfn_sep2026` (run_id `20260921T155428Z`) with `tabpfn-time-series==1.2.0`, `tabpfn==9.0.0` and `torch==2.14.0+cu130` on an NVIDIA RTX A1000. No fallback was substituted.
- Standalone WAPE was 0.05656 at 28d and 0.06842 at 91d. The fixed 75/25 naive/TabPFN blend scored 0.05671 and 0.06146, respectively.
- At 28d the blend was the WAPE winner: 0.05671 versus 0.05685 for naive/TimesFM and 0.05888 for naive. Its frozen rejection came solely from the bias guardrail: 1.022% absolute mean bias versus 0.861% for naive, a 0.161 percentage-point miss. Its worst-window loss was better than the TimesFM blend. Treat this as promising evidence for a pre-registered fresh 28-day confirmation, not as a discarded candidate or a retroactive change to the frozen verdict.
- At 91d the blend was rejected on accuracy: 0.06146 WAPE trailed the 0.06084 naive/TimesFM blend.
- The TabPFN-3.5 weights permit limited evaluation but prohibit commercial/production use and outputs in commercial decision-making absent a separate license. Cloud mode was not used.

Production read: promising 28-day confirmation candidate that failed one frozen guardrail; rejected at 91 days and ineligible for business production without a separate commercial license.

## `google_tabfm_1_0_0`

Purpose: exploratory direct tabular 28/91-day forecasting candidate.

Screened horizons: 28 and 91 days.

Do not use for:

- Production or commercial decision-making; the pretrained weights are non-commercial.
- 180/365-day claims or causal opening/promotion attribution.

Notes:

- Registration: `tabfm_screen_registration.json`. It pinned `tabfm==1.0.1`, source commit `d8678b6895f1428a468d4cc299c1ff4cf704e726`, checkpoint `google/tabfm-1.0.0-pytorch@77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83`, 256 context rows, one estimator, BF16 and CUDA.
- The direct recipe used only cutoff-available horizon, calendar, lag-364 and trailing-history fields. It did not use recursive short lags, future actuals, supplied covariates, tuning or fallback predictions.
- Artifact: `outputs_new_model_screen_tabfm_sep2026` (run_id `20260921T202918Z`). Standalone TabFM mean WAPE was 0.06827 with 2/6 wins versus naive.
- The fixed 75/25 naive/TabFM blend improved mean WAPE versus naive (0.05765 versus 0.05888), had the best median WAPE (0.05405) and the smallest absolute mean bias (0.198%). It nevertheless trailed the TimesFM blend's 0.05685 mean WAPE and had a larger worst-window loss versus naive (0.00508 versus 0.00328).
- The separately registered 91-day extension is in `outputs_new_model_screen_tabfm_91d_sep2026` (run_id `20260921T204255Z`). Standalone TabFM scored 0.09636 WAPE. The blend essentially tied naive (0.064886 versus 0.064897; 6/8 wins) but trailed the TimesFM blend at 0.060840 and had a 0.00723 worst-window loss versus naive.

Production read: rejected from fresh confirmation. Retain only for the bounded error-diversity audit; do not expand the foundation-model search based on this result.

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

## `neuralprophet_trend` and `neuralprophet_ar` (exploratory)

Purpose: bounded aggregate daily-sales NeuralProphet trial at 28 and 91 days. Production remains `seasonal_naive_lag_364`.

Notes:

- Registrations: `neuralprophet_screen_registration.json` and `neuralprophet_ensemble_registration.json`. NeuralProphet 0.8.0 ran in isolated WSL2 Python 3.12.3 with fixed 40 epochs, batch size 64, learning rate 0.01, yearly/weekly seasonality and no supplied covariates. Trend used no autoregression; AR used 91 known lags and direct multi-step forecasts. Both used only training data through each cutoff.
- Artifact: `outputs_neuralprophet_screen_sep2026` (run_id `20260925T140644Z`), six 28-day and eight 91-day historical windows from the frozen August snapshot. These are exploration, not new confirmation windows.
- Standalone 28d mean WAPE: trend 0.07582, AR 0.12534, seasonal naive 0.05888. Standalone 91d: trend 0.10664, AR 0.15595, seasonal naive 0.06490. Neither standalone configuration is competitive.
- Fixed 75/25 naive/trend blend: 28d mean WAPE 0.05618, signed bias -0.00719, 3/6 wins versus naive, worst-window WAPE loss +0.00319. It narrowly beats the frozen naive/TimesFM blend (0.05685, worst loss +0.00328) and clears the registered exploratory gate. At 91d it scores 0.06360 versus 0.06084 for naive/TimesFM and is rejected.
- Fixed 75/25 naive/AR blend: 0.06070 at 28d and 0.07247 at 91d; reject both. The registered three-member 28d naive/TimesFM/trend blend scores 0.05629, a small diagnostic improvement over naive/TimesFM; the two blends involving AR lose. At 91d all registered NeuralProphet ensembles trail naive/TimesFM.
- The 91d November/December slice is difficult for both NeuralProphet versions: trend WAPE 0.21024 and AR 0.20385 versus 0.11733 for naive. The 28d windows contain no November/December dates. Mean run time per fitted cutoff was about 19-27 seconds depending on horizon/configuration.

Production read: preserve the 28d trend blend only as a candidate for frozen fresh confirmation. Its WAPE edge over the TimesFM blend is 0.00067 on previously seen windows, and several formulas were screened. Do not promote or claim 91d benefit.

### 365-day check — 2026-09-25

The unchanged trend and 91-lag AR recipes were tested with direct 365-day output on two approximately non-overlapping historical windows (cutoffs 2024-08-30 and 2025-08-29). Registration: `neuralprophet_365d_registration.json`; artifact: `outputs_neuralprophet_365d_sep2026` (run_id `20260925T141945Z`). Both configurations passed native smoke and completed without fallback.

| Model | 2024-08-30 WAPE | 2025-08-29 WAPE | Mean WAPE | Mean signed bias |
|---|---:|---:|---:|---:|
| Seasonal naive lag 364 | 0.10853 | 0.06184 | 0.08518 | +0.02536 |
| Frozen 75/25 naive/TimesFM | 0.09204 | 0.05898 | 0.07551 | +0.01564 |
| NeuralProphet trend | 0.09000 | 0.15814 | 0.12407 | +0.07048 |
| 75/25 naive/trend | 0.09463 | 0.06848 | 0.08155 | +0.03664 |
| NeuralProphet AR | 0.27758 | 0.19930 | 0.23844 | -0.21492 |
| 75/25 naive/AR | 0.10861 | 0.07957 | 0.09409 | -0.03471 |

The trend model beat naive on the older annual window but failed on the latest; AR failed both. November/December pooled WAPE was 0.16549 for trend and 0.21563 for AR versus 0.15828 for naive and 0.13548 for the TimesFM blend. Both NeuralProphet blends fail the annual comparator gate. This tests an aggregate daily target with no covariates and fixed 40-epoch settings; it does not evaluate a differently configured team tool or its other data. Two annual windows are too few for a general model-family verdict. Production remains seasonal naive.
