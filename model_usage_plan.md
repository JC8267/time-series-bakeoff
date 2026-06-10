# Forecasting and Counterfactual Model Usage Plan

Last updated: 2026-06-10

This document describes which validated models and ensembles to use for different project types, and how to implement each workflow with the current bakeoff pipeline.

## Current Validated Winners

| Use case | Horizon | Recommended model | Why |
|---|---:|---|---|
| Short-term production forecast | 28 days | `seasonal_naive_lag_364` | Won 4 of 6 folds of the 2026-06-10 multi-fold validation (mean WAPE 0.0680 vs 0.0865 for Extra Trees, which failed badly on the December holiday window). `final_28d_extra_trees_opening_timesfm_pool` remains the opening-aware scenario model. |
| Medium-term production forecast | 91 days | `seasonal_naive_lag_364` | Won all 5 folds of the 2026-06-10 nested multi-fold validation (mean WAPE 0.0757 vs 0.0918 for the honestly refit convex stack). The earlier single-holdout stack win was contaminated by weight-fitting overlap. `final_91d_convex_stack_opening_timesfm_pool` is now a sensitivity/diagnostic model. |
| Long-horizon pure forecast benchmark | 180 days | `final_180d_convex_stack_stats_chronos_pool` plus `final_180d_seasonal_naive_benchmark` | The stack had lower WAPE in the focused rerun, but validation remains underpowered. Keep seasonal naive as the benchmark sensitivity. |
| Long-horizon pure forecast benchmark | 365 days | `final_365d_convex_stack_stats_chronos_pool` plus `final_365d_seasonal_naive_benchmark` | The stack had lower WAPE in the focused rerun, but validation remains underpowered. Keep seasonal naive as the benchmark sensitivity. |
| Long-horizon opening-sensitive forecast | 180 days | `final_180d_hist_gradient_boosting_opening_sensitive` | Opening-sensitive scenario model. It did not beat the seasonal/stack benchmarks on WAPE, so use it for counterfactual sensitivity, not pure accuracy claims. |
| Long-horizon opening-sensitive forecast | 365 days | `final_365d_hist_gradient_boosting_opening_sensitive` | Opening-sensitive scenario model. It did not beat the seasonal/stack benchmarks on WAPE, so use it for counterfactual sensitivity, not pure accuracy claims. |

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
| 91-day multi-fold nested validation | `outputs_multifold_91d_validation` (run_id `20260610T042656Z`) |
| 28-day multi-fold nested validation | `outputs_multifold_28d_validation` (run_id `20260610T044622Z`) |

Artifact notes:

- The phase 4/5/6 rows above point to the TimesFM interval-fix reruns. They supersede `outputs_opening_phase4_statsmodels_timesfm25_venv`, `outputs_opening_phase5_statsmodels_timesfm25_venv`, and `outputs_opening_phase6_statsmodels_timesfm25_venv`, where standalone `timesfm_2_5` was missing from phase-4 interval calibration and got zero-width 91-day intervals. Point forecasts/WAPE are identical between the two; only standalone TimesFM interval metrics changed (91-day coverage_80 0.000 → 0.857, coverage_95 0.000 → 0.945).
- Older and scratch output directories were moved to `archive/run_outputs_2026-06-01/` during cleanup. The table above lists the retained root-level artifacts.

Latest key validation metrics:

| Horizon | Candidate | WAPE | Bias | Validation folds |
|---:|---|---:|---:|---:|
| 28 | `seasonal_naive_lag_364` | 0.0680 | 0.0025 | 6 non-overlapping folds (2026-06-10) |
| 28 | `sklearn_extra_trees` | 0.0865 | -0.0313 | 6 non-overlapping folds (2026-06-10) |
| 28 | `sklearn_extra_trees` | 0.0314 | 0.0020 | 1 final holdout (clean but unrepresentative; matches the best of the 6 folds) |
| 28 | `phase5_ridge_stack` | 0.0327 | -0.0164 | 1 final holdout (contaminated; see caveat) |
| 91 | `seasonal_naive_lag_364` | 0.0757 | -0.0024 | 5 nested rolling folds (2026-06-10) |
| 91 | `nested_convex_stack` | 0.0918 | 0.0022 | 5 nested rolling folds (2026-06-10) |
| 91 | `phase5_convex_stack` | 0.0507 | -0.0116 | 1 final holdout (contaminated; see caveat) |
| 91 | `phase5_ridge_stack` | 0.0522 | -0.0216 | 1 final holdout (contaminated; see caveat) |
| 180 | `phase3_heldout_cutoffs_inverse_wape` | 0.0645 | 0.0049 | 3 rolling cutoffs |
| 180 | `seasonal_naive_lag_364` | 0.0686 | -0.0086 | 3 rolling cutoffs |
| 365 | `phase3_heldout_cutoffs_convex_stack` | 0.0563 | -0.0080 | 3 rolling cutoffs |
| 365 | `seasonal_naive_lag_364` | 0.0591 | 0.0019 | 3 rolling cutoffs |

Fold-count caveat:

- The latest 28/91 phase-1 and foundation reruns used 3 rolling cutoffs per horizon. Phase 3 stacking trained on the earlier aligned cutoff and evaluated on 2 held-out cutoffs. Phase 5 is a single final holdout window.
- The phase-5 stack metrics are contaminated at both horizons: phase 3 fit the 91-day weights at cutoff 2025-11-01 (fitting window through 2026-01-31, overlapping the phase-5 holdout by 31 days) and the 28-day weights at cutoff 2026-02-23 (fitting window through 2026-03-23, overlapping the 28-day holdout by 19 days). Standalone models like `sklearn_extra_trees` have no weight fitting and are unaffected.
- The 2026-06-10 multi-fold nested validations (`run_multifold_final_validation.py`, weights refit per fold leak-free) supersede the single holdouts: `seasonal_naive_lag_364` won all 5 folds at 91 days and 4 of 6 folds at 28 days. The 91-day folds overlap (28-day steps, 91-day windows), so they are not fully independent; the 28-day folds do not overlap.
- The latest 180/365 validation used 3 rolling cutoffs per horizon. This is intentionally underpowered.
- Tests with fewer folds can flip the apparent winner back to `seasonal_naive_lag_364`. Do not compare a naive-only, fewer-fold run against an ensemble run unless the ensemble is rebuilt and scored on the same folds.
- When fold count is reduced, report the result as a sensitivity check rather than a replacement recommendation unless phase 3/phase 5 are rerun under the same fold design.

Model additions in the latest rerun:

- StatsForecast `AutoARIMA`, `AutoETS`, `MSTL + AutoARIMA`, and `MSTL + AutoETS`.
- Chronos-Bolt small as a second foundation model beside TimesFM.
- Statsmodels `ExponentialSmoothing` and `SARIMAX` now run without fallback in the latest Statsmodels-fixed rerun. They did not improve the 28-day or 91-day production winners.
- Phase 4 was rerun with standalone `timesfm_2_5` included in interval calibration. Corrected 91-day standalone TimesFM coverage is 0.857 at the 80% level and 0.945 at the 95% level; the 91-day production stack coverage remains 0.857 at 80% and 0.956 at 95%.
- TimesFM loader fallback for the current `huggingface-hub` behavior that forwards `proxies` into the TimesFM constructor.
- Direct dependency pins for `huggingface-hub==0.35.3` and `httpx==0.27.2` to keep TimesFM and Chronos loading in the project venv.

## Decision Rules

### 1. General sales forecasting

Use this when the goal is the most accurate expected sales forecast, not attribution.

Default choices:

| Forecast length | Use |
|---:|---|
| 1-28 days | `seasonal_naive_lag_364`; keep `final_28d_extra_trees_opening_timesfm_pool` as the opening-aware sensitivity comparison, and avoid Extra Trees unhedged on windows spanning major holidays |
| 29-91 days | `seasonal_naive_lag_364`; keep `final_91d_convex_stack_opening_timesfm_pool` as a sensitivity comparison only |
| 92-180 days | Re-run opening-aware validation and compare `seasonal_naive_lag_364` against opening-sensitive ML models. Current benchmark winner is `seasonal_naive_lag_364`. |
| 181-365 days | Use the focused phase-3 stack as the accuracy candidate and `seasonal_naive_lag_364` as the benchmark sensitivity; use opening-sensitive models when future openings materially affect the forecast. |

Implementation:

```powershell
python run_phase6.py `
  --sales-csv daily_sales_data.csv `
  --opening-info opening_info.csv `
  --phase3-metadata outputs_opening_phase3_statsmodels_timesfm25_venv/phase3_metadata.json `
  --phase4-predictions outputs_opening_phase4_statsmodels_timesfm25_recalibrated_venv/phase4_calibrated_predictions.csv `
  --output-dir outputs_opening_phase6_statsmodels_timesfm25_recalibrated_venv
```

Primary output:

```text
outputs_opening_phase6_statsmodels_timesfm25_recalibrated_venv/phase6_final_forecast.csv
```

Use the winner-aligned rows:

```text
final_28d_extra_trees_opening_timesfm_pool
final_91d_convex_stack_opening_timesfm_pool
```

Note: after the 2026-06-10 multi-fold validations, prefer `seasonal_naive_lag_364` for both the 28-day and 91-day production numbers and treat the packaged winner-aligned rows as sensitivity comparisons. Seasonal naive needs no phase-6 packaging; it is reproducible directly from the sales history (year-ago values).

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
7. Open: if a naive/TimesFM blend is still wanted at 91 days, pre-register a fixed ratio now and validate it on future quarters as they arrive (the frozen 62/38 weights won all 5 folds but were fit with hindsight relative to those folds).
8. Done (2026-06-10): multi-fold nested validation at 28 days (6 non-overlapping folds, run_id `20260610T044622Z`); seasonal naive won 4 of 6 folds and is now the 28-day production recommendation, with Extra Trees retained as the opening-aware scenario model.
