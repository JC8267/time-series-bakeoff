# Data Semantics

Local-only files (all gitignored). Confirmed with the data owner 2026-06-10.

## daily_sales_data.csv

Total daily sales, `transaction_date, sales`, 2022-01-01 onward. This is the
target of the original bakeoff and the multifold validations.

## bu_level_sls.csv

`sub_market_no, transaction_date, sales`, daily from 2021-09-01, refreshed to
the pull date.

Semantics: each row is a TRADE AREA total — the physical unit's sales (once
open) plus e-commerce attributed to the market area. For some markets, ecomm
is backfilled before the physical unit's opening date, so pre-opening rows are
an ecomm-only baseline, not an error. Markets without ecomm backfill first
appear at (or just before) physical opening.

Known quirks:

- Summed by date, this file covers a stable ~88% of `daily_sales_data.csv`
  across the full overlap. Cause unresolved (possibly returns handling or
  sales not attributed to any sub-market). Do not treat the two files as the
  same measure; reconcile scope before mixing them in one target.
- A handful of far-future-dated rows exist (data errors);
  `prepare_market_level_data.py` drops and audits them.
- Zeros and small negatives are real (closure days, net-returns days).
- Some sub-markets have sales but no row in the opening metadata; these are
  presumed non-store/ecomm-only markets.

## complete_opening_info.csv

NOTE: this is an .xlsx workbook despite the .csv extension — read with
`pd.read_excel`. One row per physical unit: `Bu Number, Total Sqm, Unit Type,
Opening Date, Relocation Date, Unit Age`.

- `Opening Date` is the PHYSICAL unit opening (current format), not the first
  trading day of the trade area — ecomm backfill can long predate it.
- `Relocation Date` only records older relocations; blank does not mean a
  unit never moved.

## opening_info.csv (legacy)

Small subset (`Name, BU, Opening_date`) used by the original aggregate
bakeoff. Superseded by `complete_opening_info.csv` for anything unit-level.

## Derived (gitignored, rebuildable)

- `prepare_market_level_data.py` → `outputs_market_prep/`: clean panel with
  `has_physical`, `days_since_physical_open`, `pre_opening_ecomm` flags,
  market metadata, and a `normalized_openings.csv` compatible with
  `load_openings`.
- `estimate_market_opening_lift.py` → `outputs_market_opening_lift/`:
  within-market opening lift vs the ecomm-baseline counterfactual.
