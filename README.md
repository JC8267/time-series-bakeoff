# Time Series Bakeoff

Daily sales forecasting bakeoff with opening-aware features, TimesFM/Prophet benchmarks, ensemble selection, production packaging, and counterfactual utilities.

## Start Here

Build the current production forecast package (seasonal naive production rows plus the pre-registered 75/25 naive/TimesFM challenger, all four horizons, leak-free residual-quantile intervals):

```powershell
python run_production_forecast.py --sales-csv daily_sales_data.csv --output-dir outputs_production_forecast
```

Run the opening-aware 28/91 validation:

```powershell
python run_bakeoff.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --horizons 28,91 --output-dir outputs_opening_phase1
```

Re-run the leak-free multi-fold validation for a horizon (required before packaging a new horizon):

```powershell
python run_multifold_final_validation.py --horizon 91 --n-folds 5 --output-dir outputs_multifold_91d_validation
```

Build the current production package:

```powershell
python run_phase6.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --phase3-metadata outputs_opening_phase3_timesfm/phase3_metadata.json --phase4-predictions outputs_opening_phase4_timesfm/phase4_calibrated_predictions.csv --horizons 28,91 --output-dir outputs_opening_phase6_timesfm_updated
```

Build the long-horizon package:

```powershell
python run_phase6.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --phase3-metadata outputs_opening_phase3_timesfm/phase3_metadata.json --phase4-predictions outputs_opening_phase4_timesfm/phase4_calibrated_predictions.csv --horizons 180,365 --output-dir outputs_opening_phase6_long
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

