# Production Model Cards

Last updated: 2026-06-10

Validation fold note: every WAPE/bias claim below should be read with its fold count. The latest 28/91 phase-1 and foundation reruns used 3 rolling cutoffs per horizon, phase 3 trained stack weights on the earlier aligned cutoff and evaluated on 2 held-out cutoffs, and phase 5 used 1 final holdout. The latest 180/365 validation used 3 rolling cutoffs per horizon. Fewer-fold sensitivity tests can make `seasonal_naive_lag_364` win; treat those as sensitivity checks unless the ensemble candidates are rebuilt and scored on the same folds.

2026-06-10 update: a 5-fold nested final validation at 91 days (`run_multifold_final_validation.py`, run_id `20260610T042656Z`, outputs in `outputs_multifold_91d_validation/`) settled the fragile 91-day ranking. See the 91-day card below; the production recommendation at 91 days is now `seasonal_naive_lag_364`.

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

Latest validation:

- Phase 5 holdout WAPE: 0.0314.
- Phase 5 holdout bias: 0.0020.
- Validation folds: 1 final holdout window; supporting phase-1 rerun used 3 rolling cutoffs.

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

Provenance: `run_multifold_final_validation.py`, run_id `20260610T042656Z`, `outputs_multifold_91d_validation/` (5 rolling 91-day holdout folds, cutoffs 2025-09-10 to 2025-12-31 stepped 28 days; folds overlap, so they are not fully independent; stack weights refit per fold on only the backtest windows ending at or before that fold's cutoff).

- Nested convex stack (the honest version of the production strategy): mean WAPE 0.0918, **0/5 wins** vs `seasonal_naive_lag_364` (mean WAPE 0.0757), losing every fold by +0.014 to +0.022.
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
- The 180-day validation has limited folds with current history; treat this as the current benchmark, not definitive proof that more complex models cannot win.

Latest validation:

- Focused validation WAPE: 0.0686.
- Focused validation bias: -0.0086.
- Validation folds: 3 rolling cutoffs.

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
- The 180-day validation remains underpowered.
- Validation folds: 3 rolling cutoffs. If a fewer-fold run makes seasonal naive win, rerun this ensemble on the same folds before changing the recommendation.

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
- The 365-day validation is underpowered with current history. The practical read is that the seasonal benchmark is hard to beat, not that it is conclusively superior for every annual planning problem.

Latest validation:

- Focused validation WAPE: 0.0591.
- Focused validation bias: 0.0019.
- Validation folds: 3 rolling cutoffs.

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
- The 365-day validation is underpowered with current history, so this should be reported beside the seasonal benchmark.
- Validation folds: 3 rolling cutoffs. If a fewer-fold run makes seasonal naive win, rerun this ensemble on the same folds before changing the recommendation.

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
