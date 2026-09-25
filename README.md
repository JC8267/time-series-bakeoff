# Time Series Bakeoff

Daily sales forecasting bakeoff with opening-aware features, TimesFM/Prophet benchmarks, ensemble selection, production packaging, and counterfactual utilities.

## Start Here

Build final-only production forecasts for any daily horizons from 1 through 365, with leak-free residual-quantile intervals:

```powershell
python run_production_forecast.py --sales-csv daily_sales_data.csv --horizons 28,45,91 --output-dir outputs_production_forecast
```

Add `--include-challenger` only when an exact, non-degraded validation grid exists; the pre-registered 75/25 naive/TimesFM blend is then written separately to `diagnostic_forecast.csv`.

Refresh and score that challenger after new daily sales arrive:

```powershell
python run_multifold_final_validation.py --sales-csv latest_sls_aug.csv --horizon 28 --n-folds 1 --grid-step 28 --max-grid-cutoffs 6 --pool seasonal_naive_lag_364,timesfm_2_5 --missing-date-strategy error --output-dir outputs_multifold_28d_aug2026_validation
python run_multifold_final_validation.py --sales-csv latest_sls_aug.csv --horizon 91 --n-folds 1 --grid-step 28 --max-grid-cutoffs 8 --pool seasonal_naive_lag_364,timesfm_2_5 --missing-date-strategy error --output-dir outputs_multifold_91d_aug2026_validation
python explore_naive_beaters.py --sales-csv latest_sls_aug.csv --holdout-cutoff 2026-04-01 --output-dir outputs_naive_timesfm_challenger_aug2026
```

`run_forecast_monitor.py` automatically uses a packaged sibling `diagnostic_forecast.csv` when given its final-only `production_forecast.csv`, so realized refreshes include champion-vs-challenger adjudication.

### TimesFM 3 test-only bakeoff

TimesFM 3 is available as the explicit `timesfm_3` validation-pool candidate. It is intentionally excluded from default pools and production packaging: Google's checkpoint license permits only non-commercial, non-production testing and forbids using its outputs for commercial decision-making.

```powershell
foreach ($h in 28,91,180,365) {
  python run_multifold_final_validation.py --sales-csv latest_sls_aug.csv --horizon $h --n-folds 1 --grid-step 28 --pool seasonal_naive_lag_364,timesfm_2_5,timesfm_3 --timesfm-context 1024 --missing-date-strategy error --allow-noncommercial-timesfm3 --output-dir "outputs_multifold_${h}d_timesfm3_test"
}
```

Run the opening-aware 28/91 validation:

```powershell
python run_bakeoff.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --horizons 28,91 --output-dir outputs_opening_phase1
```

Re-run the leak-free multi-fold validation for a horizon (required before packaging a new horizon):

```powershell
python run_multifold_final_validation.py --horizon 91 --n-folds 5 --output-dir outputs_multifold_91d_validation
```

Build the legacy stack/ML sensitivity package (demoted models — NOT production; see `model_usage_plan.md`):

```powershell
python run_phase6.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --phase3-metadata outputs_opening_phase3_timesfm/phase3_metadata.json --phase4-predictions outputs_opening_phase4_timesfm/phase4_calibrated_predictions.csv --horizons 28,91 --output-dir outputs_opening_phase6_timesfm_updated
```

## Market-Level Pipeline

```powershell
python prepare_market_level_data.py                  # clean trade-area panel from bu_level_sls.csv
python estimate_market_opening_lift.py               # opening lift vs within-market ecomm baseline
python run_market_bakeoff.py --horizons 28,91        # per-market forecasting bakeoff
```

See `DATA.md` for the trade-area data semantics (ecomm backfill, scope vs the aggregate file).

## Key Docs

- `model_usage_plan.md`: which models to use by project type.
- `model_cards.md`: intended use and limits for each production model.
- `RUN_INDEX.md`: generated index of output directories.

## Tests

```powershell
python -m pytest
```
