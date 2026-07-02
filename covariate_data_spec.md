# Covariate Data Spec

Purpose: define committable acquisition asks for known-ahead covariates. This file contains no sales data and no dollar figures.

## Promo/Markdown Calendar

Minimum history: ideally back to 2022-01-01 so weekly validation can cover the full current panel.

Scope options:

- Chain-level flag: enough for first residual-model tests.
- Category-level intensity: preferred if promotions differ materially by category.

Required fields:

| Field | Type | Notes |
|---|---|---|
| `start_date` | date | First active date, inclusive. |
| `end_date` | date | Last active date, inclusive. |
| `promo_type` | string | Controlled label such as markdown, financing, coupon, campaign, clearance. |
| `scope` | string | Chain, category, market, channel, or other stable scope label. |
| `planned_flag` | boolean | `true` when the row was known before the forecast cutoff. |

Leak rule:

- Forecasting can only use planned or known-ahead versions available at the cutoff.
- Realized spend, actual redemption, final margin, or post-event adjustments are attribution inputs only, not forecast covariates.

Preferred long-format handoff for `run_residual_covariate_model.py --covariate-csv`:

| Field | Type | Notes |
|---|---|---|
| `date` | date | Expanded daily date. |
| `name` | string | Stable covariate name, for example `promo_any` or `markdown_category_sofas`. |
| `value` | numeric | Known-ahead value for that date. |

## Unit-Level Daily Sales

This is the unlock for cannibalization, halo, and opening analytics that chain-level totals cannot identify.

Required fields matching `prepare_unit_level_targets.py` expectations:

| Field | Type | Notes |
|---|---|---|
| `unit_id` | string | Stable physical or commerce unit identifier. |
| `date` | date | Daily transaction date. `transaction_date` is also acceptable. |
| `sales` | numeric | Daily net sales for the unit. `net_sales` is also acceptable. |

Preferred additions:

- `channel`: store, ecomm, marketplace, or other stable channel label.
- `market_area_no`: if available, to join with trade-area panels.
- `unit_type`: showroom, outlet, warehouse, ecomm-only, or other operating format.

Leak rule:

- Historical unit sales can train and validate models.
- Future unit-level sales are never forecast inputs.
- Planned openings, closures, relocations, and channel status may be future covariates only when they were known before the forecast cutoff.
