# Units Forecast Bakeoff

Run date: 2026-07-24

## Decision

Use the channel forecasts and sum them for a coherent Total:

| Horizon | Online | Remote | Store | Coherent Total mean WAPE |
|---|---|---|---|---:|
| 28d | TimesFM | TimesFM | TimesFM | 6.69% |
| 91d | TimesFM | TimesFM | TimesFM | 6.58% |
| 180d | TimesFM | TimesFM | TimesFM | 8.17% |
| 365d | TimesFM | TimesFM | Damped annual seasonal | 7.20% |

At 180 days, the nested convex ensemble is the numerical channel winner for Online and Remote, but its margins over TimesFM are only 0.15 and 0.18 WAPE points on heavily overlapping folds. It also produces a slightly worse bottom-up Total than using TimesFM for all three channels (8.22% versus 8.17%). Keep it as a challenger.

At 365 days, the mixed channel stack beats the best direct-Total model in all five nominal folds (7.20% versus 7.87% WAPE). This result is provisional because those folds represent only about 1.31 independent holdout windows.

## Data

- Source: `units_raw.csv`
- Complete training range: 2022-01-01 through 2026-07-23 (1,665 daily rows per series)
- Series: Online, Remote, Store, and direct Total
- Missing channel-dates: 0
- Excluded: the partial 2026-07-24 rows and ten later zero placeholders
- Preserved as observed net units: 38 negative Remote days and 8 negative Store days
- Reconciliation check: channel actuals sum exactly to Total

## Validation

| Horizon | History-grid cutoffs | Final cutoff range | Nominal folds | Adjacent overlap | Approx. independent windows |
|---:|---:|---|---:|---:|---:|
| 28d | 33 | 2026-03-05 to 2026-06-25 | 5 | 0.0% | 5.00 |
| 91d | 31 | 2026-01-01 to 2026-04-23 | 5 | 69.2% | 2.23 |
| 180d | 27 | 2025-10-04 to 2026-01-24 | 5 | 84.4% | 1.62 |
| 365d | 21 | 2025-04-02 to 2025-07-23 | 5 | 92.3% | 1.31 |

- Nested ensemble weights use only windows ending before each evaluation cutoff.
- The 180- and 365-day runs used the same predeclared eight-model pool for every series.
- All manifests report zero filled dates and no degraded model fallbacks.

## Numerical Winners

This table ranks standalone models and ensembles together. "Fold wins" is the winner's head-to-head count against the listed runner-up.

| Horizon | Series | Numerical winner | Mean WAPE | Mean bias | Runner-up | Runner-up WAPE | Fold wins |
|---|---:|---|---:|---:|---|---:|---:|
| 28d | Online | TimesFM | 5.60% | +2.91% | Pool median | 5.63% | 3/5 |
| 28d | Remote | TimesFM | 20.89% | -6.13% | Nested convex | 21.01% | 3/5 |
| 28d | Store | TimesFM | 8.03% | -0.51% | Extra Trees | 8.23% | 2/5 |
| 28d | Total | Extra Trees | 6.59% | +1.95% | TimesFM | 6.70% | 4/5 |
| 91d | Online | Chronos Bolt | 7.71% | +5.15% | TimesFM | 7.72% | 3/5 |
| 91d | Remote | TimesFM | 23.03% | -10.63% | Nested convex | 23.47% | 5/5 |
| 91d | Store | TimesFM | 7.91% | -1.15% | Nested convex | 8.43% | 5/5 |
| 91d | Total | TimesFM | 6.58% | +0.21% | Extra Trees | 7.56% | 5/5 |
| 180d | Online | Nested convex | 8.61% | +4.39% | TimesFM | 8.76% | 4/5 |
| 180d | Remote | Nested convex | 25.43% | -7.23% | TimesFM | 25.61% | 3/5 |
| 180d | Store | TimesFM | 9.71% | +0.15% | Nested convex | 10.21% | 5/5 |
| 180d | Total | TimesFM | 8.24% | +0.83% | Damped annual seasonal | 9.00% | 5/5 |
| 365d | Online | TimesFM | 9.36% | +1.09% | Fixed 75/25 blend | 9.40% | 3/5 |
| 365d | Remote | TimesFM | 26.55% | -5.81% | Nested convex | 26.63% | 3/5 |
| 365d | Store | Damped annual seasonal | 8.19% | +4.19% | Fixed 75/25 blend | 8.99% | 5/5 |
| 365d | Total | Damped annual seasonal | 7.87% | +4.55% | Fixed 75/25 blend | 8.09% | 3/5 |

Model labels: "Damped annual seasonal" is `seasonal_naive_scaled_lag_364_w56_damped`; "Fixed 75/25 blend" is 75% `seasonal_naive_lag_364` plus 25% TimesFM.

## Ensemble Findings

- Honest nested convex, nested inverse-WAPE, and pool-median ensembles were evaluated at every horizon.
- The nested convex stack is the only automatic ensemble to win a series/horizon cell: Online and Remote at 180 days, 2 of 16 cells total.
- The fixed 75/25 annual-naive/TimesFM blend wins 0 of 16 cells. Its closest result is Online at 365 days: 9.40% versus TimesFM's 9.36%.
- No ensemble beats the standalone winner for Store or direct Total at any horizon.

## Coherent Total

| Horizon | Bottom-up channel stack | Bottom-up WAPE | Bottom-up bias | Best direct Total | Direct WAPE | Direct bias | Bottom-up fold wins |
|---|---|---:|---:|---|---:|---:|---:|
| 28d | TimesFM / TimesFM / TimesFM | 6.69% | +0.03% | Extra Trees | 6.59% | +1.95% | 2/5 |
| 91d | TimesFM / TimesFM / TimesFM | 6.58% | -0.04% | TimesFM | 6.58% | +0.21% | 2/5 |
| 180d | TimesFM / TimesFM / TimesFM | 8.17% | +0.55% | TimesFM | 8.24% | +0.83% | 3/5 |
| 365d | TimesFM / TimesFM / Damped annual seasonal | 7.20% | +3.45% | Damped annual seasonal | 7.87% | +4.55% | 5/5 |

The 28-day bottom-up accuracy cost is only 0.10 WAPE points and buys exact channel-to-total reconciliation plus near-zero bias. Keep direct Total Extra Trees as a 28-day challenger.

For Online at 91 days, Chronos beats TimesFM by only 0.0135 WAPE points and 3/5 folds. Prefer TimesFM because Chronos warns that prediction lengths above 64 are outside its optimized range.

Remote is the weak series: its best WAPE ranges from 20.89% to 26.55%, with material negative bias. Treat its forecasts and intervals cautiously.

The 91-, 180-, and especially 365-day rankings are less conclusive than five nominal folds suggest because their holdouts overlap.

## Artifacts

- Broad screens: `outputs_phase1_28_91/<Series>/`
- Nested 28-day runs: `outputs_multifold_28d_validation/<Series>/`
- Nested 91-day runs: `outputs_multifold_91d_validation/<Series>/`
- Nested 180-day runs: `outputs_multifold_180d_validation/<Series>/`
- Nested 365-day runs: `outputs_multifold_365d_validation/<Series>/`
- Prepared inputs and exclusion audit: `outputs_prepared/`
- Run inventory: `RUN_INDEX.md`
