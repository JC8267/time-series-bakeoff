# Time Series Bakeoff

Daily sales forecasting bakeoff with opening-aware features, TimesFM/Prophet benchmarks, ensemble selection, production packaging, and counterfactual utilities.

## Start Here

Run the opening-aware 28/91 validation:

```powershell
python run_bakeoff.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --horizons 28,91 --output-dir outputs_opening_phase1
```

Build the current production package:

```powershell
python run_phase6.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --phase3-metadata outputs_opening_phase3_timesfm/phase3_metadata.json --phase4-predictions outputs_opening_phase4_timesfm/phase4_calibrated_predictions.csv --horizons 28,91 --output-dir outputs_opening_phase6_timesfm_updated
```

Build the long-horizon package:

```powershell
python run_phase6.py --sales-csv daily_sales_data.csv --opening-info opening_info.csv --phase3-metadata outputs_opening_phase3_timesfm/phase3_metadata.json --phase4-predictions outputs_opening_phase4_timesfm/phase4_calibrated_predictions.csv --horizons 180,365 --output-dir outputs_opening_phase6_long
```

## Key Docs

- `model_usage_plan.md`: which models to use by project type.
- `model_cards.md`: intended use and limits for each production model.
- `RUN_INDEX.md`: generated index of output directories.

## Tests

```powershell
python -m pytest
```

