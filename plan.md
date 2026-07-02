# Improvement Plan — Time Series Bakeoff (2026-07-02)

Implementation plan for the next round of forecast improvements. Written for an
implementing agent with no prior context. Read this whole file before starting.

## Required reading (in order)

1. `model_usage_plan.md` — current validated winners, decision rules, caveats.
2. `model_cards.md` — per-model verdicts, especially the weekly-grain cards.
3. `DATA.md` — data semantics. Do not mix `daily_sales_data.csv` and `bu_level_sls.csv` targets.
4. `C:\Users\jacol9\Vault\AI\30_Projects\Time Series Bakeoff.md` — the June 2026
   critique. Its "Standing Methodology Lessons" section is binding on this plan.

## Ground rules (non-negotiable)

- **Data safety:** the repo is private (`JC8267/time-series-bakeoff`) but sales CSVs,
  `outputs_*` directories, and any file containing absolute dollar figures must NEVER
  be committed or pushed. Run summaries and docs report relative metrics only
  (WAPE, bias pct, coverage). Absolute values stay in gitignored output CSVs.
  Check `.gitignore` covers any new output directory pattern you introduce.
- **Environment:** Windows, PowerShell syntax for commands. Python runs from the
  project `.venv`. Package installs use `uv`, never bare `pip`. Dependency pins
  matter (`huggingface-hub==0.35.3`, `httpx==0.27.2` keep TimesFM/Chronos loading).
- **Reproducibility conventions:** every run script calls `set_seeds()`, writes a
  `manifest.json`, and registers itself via `experiment_registry.append_experiment`.
  New scripts must do the same. Rebuild `RUN_INDEX.md` with `build_run_index.py`
  after producing new output directories.
- **Promotion discipline:** no candidate replaces a production default without
  (a) winning in the leak-free nested multifold harness
  (`run_multifold_final_validation.py`), and (b) passing an overlap-aware
  significance test (`run_forecast_significance_tests.py`: HAC p < 0.05 AND
  block-bootstrap 95% CI excluding zero). "4/5 wins by eyeball" is not sufficient.
  Pre-register candidates in this file's log section before scoring them.
- **Tests:** `python -m pytest tests/` must pass before every commit (15/15 at last
  review, more since). Add tests for any new pure function.

## Current state (why these workstreams)

- Production champions: daily `seasonal_naive_lag_364` at 28/91/180/365d; weekly
  `daily_reagg_seasonal_naive_lag_365` at 4/12/26/52w (forecast daily with lag 365,
  sum into W-WED weeks).
- Six-plus model families (ML trees, TimesFM, Chronos-Bolt, Chronos-2, learned
  stacks, scaled/averaged seasonal variants) have failed to beat these champions
  outside noise. Conclusion from the critique: the aggregate series is at its
  point-accuracy ceiling for history-only models. Remaining levers: smarter
  calendar alignment, covariates, interval calibration, and monitoring.
- Known residual concentration: November/December. Holiday-week residuals run
  2–3x rest-of-year. Weekly global 80% intervals undercover (0.680 at 4w, 0.717
  at 12w); segment-conditioned intervals are worse at short horizons because the
  holiday calibration sample is tiny (4w holiday 95% coverage collapsed to 0.471).
- One statistically clean challenger exists: weekly 26w `pool_median` from the
  trend/seasonal pool (12/14 wins vs champion, mean WAPE diff -0.0014, HAC
  p=0.017). Tracked, not promoted.
- All work since 6/20 is uncommitted (last commit `e922c66`, 2026-06-10).

---

## WS0 — Commit the outstanding 6/24–6/25 work (do this first)

Modified: `model_cards.md`, `model_usage_plan.md`, `run_bakeoff.py`,
`run_multifold_final_validation.py`, `run_production_forecast.py`,
`run_prophet_timesfm_benchmark.py`, `tests/test_baselines.py`,
`tests/test_intervals.py`, `tests/test_load_sales.py`.
Untracked: `run_forecast_significance_tests.py`, `run_interval_coverage_backtest.py`,
`run_weekly_production_forecast.py`, `tests/test_foundation_models.py`.

Steps:

1. Run the test suite; fix nothing silently — if something fails, report it.
2. Verify no file being committed contains absolute dollar figures or data rows
   (the md docs use relative metrics only — spot-check the diffs).
3. Commit in three logical commits:
   - weekly-grain harness + trend/seasonal + Chronos-2 candidates
     (`run_bakeoff.py`, `run_multifold_final_validation.py`,
     `run_prophet_timesfm_benchmark.py`, baseline/load tests, foundation tests)
   - significance + interval tooling + production packaging
     (`run_forecast_significance_tests.py`, `run_interval_coverage_backtest.py`,
     `run_weekly_production_forecast.py`, `run_production_forecast.py`,
     interval tests)
   - docs (`model_cards.md`, `model_usage_plan.md`, `RUN_INDEX.md` if regenerated)
4. Push to `main` (repo is private; code/docs only).

Acceptance: clean `git status` (aside from gitignored outputs), tests green, pushed.

---

## WS1 — Holiday-anchor-aware seasonal naive (highest-leverage accuracy candidate)

### Rationale

The two champions embody a tension:

- Daily champion uses **lag 364** → preserves weekday. Thanksgiving/Black Friday
  (weekday-anchored) align; Christmas/New Year (date-anchored) land 1 day off,
  drifting +1 day per year.
- Weekly champion re-aggregates daily **lag 365** → preserves calendar date.
  Christmas aligns; weekday-anchored events (Black Friday, Easter, Memorial/Labor
  Day) misalign by up to 2 days, and weekday-of-week structure shifts.

A per-day hybrid should dominate both: for each target date, choose the source
date by the nearest holiday's anchor type. This is parameter-free (no fitting),
so it can be validated on the full grid without leakage concerns.

**This is NOT the retired `seasonal_naive_holiday_aligned_lag_52`.** That weekly
rule failed because a holiday anchor can jump a full W-WED bucket when the
holiday weekday changes (documented in `model_cards.md`). At daily grain there is
no bucket to jump; the mapping moves by at most ±3 days of source data.

### Implementation

In `run_bakeoff.py` (near `seasonal_naive` at ~line 442), add:

```python
def holiday_anchor_aware_source_dates(target_dates: pd.Series) -> pd.Series:
    """Map each target date to last year's source date by holiday anchor type."""
```

Rules (pre-registered — do not tune after seeing results):

- **Date-anchored windows** (use lag 365, same calendar date last year):
  Dec 18–Jan 3 (Christmas/New Year), Jul 1–7 (July 4th).
- **Weekday-anchored windows** (use lag 364, same weekday last year):
  the 14 days centered on Thanksgiving, Easter (±7 days; compute Easter with a
  standard algorithm — implement Anonymous Gregorian algorithm inline, no new
  dependency), Memorial Day (last Mon of May ±3), Labor Day (first Mon of Sep ±3).
- **Everywhere else:** grain default — lag 364 for the daily candidate,
  lag 365 for the weekly daily-reagg candidate. This keeps each hybrid a strict
  refinement of its own champion, so any WAPE change is attributable to the
  holiday windows.
- If a target date falls in two windows (can't happen with the windows above,
  but guard it), date-anchored wins.
- Weekday-anchored source for a date-anchored default (and vice versa) can
  produce duplicate/skipped source days at window boundaries; that is expected
  and fine — each target day independently picks its source day.

Two new candidate names in `make_predictors` in `run_multifold_final_validation.py`
(follow the existing naming/dispatch pattern):

- `seasonal_naive_holiday_anchor_aware` (daily grain; default lag 364)
- `daily_reagg_seasonal_naive_holiday_anchor_aware` (weekly grain; default lag 365,
  built on `daily_reagg_seasonal_naive_from_full_history`-style plumbing)

Unit tests (`tests/test_baselines.py`): Easter date computation for known years
(2024-03-31, 2025-04-20, 2026-04-05); Christmas maps to prior Christmas date;
Thanksgiving 2025-11-27 maps to 2024-11-28 (lag 364, same weekday, which was
Thanksgiving 2024); non-holiday March date uses grain default; boundary dates.

### Validation runs

Daily (4 runs). Use extended grids like the 6/25 significance runs so ~14 windows
are available for the significance test at 180d (`--n-folds` up to the grid size;
mirror how `outputs_multifold_180d_trend_seasonal_extended_pool` was produced):

```powershell
python run_multifold_final_validation.py --horizon 28  --grain daily `
  --pool seasonal_naive_lag_364,seasonal_naive_holiday_anchor_aware `
  --output-dir outputs_multifold_28d_holiday_anchor_pool
# repeat for 91, 180, 365
```

Weekly (4 runs):

```powershell
python run_multifold_final_validation.py --horizon 4 --grain weekly `
  --pool seasonal_naive_lag_52,daily_reagg_seasonal_naive_lag_365,daily_reagg_seasonal_naive_holiday_anchor_aware `
  --output-dir outputs_weekly_multifold_4w_holiday_anchor_pool
# repeat for 12, 26, 52
```

Then significance vs the incumbent champion at each horizon:

```powershell
python run_forecast_significance_tests.py `
  --comparison "daily28_holiday_anchor|outputs_multifold_28d_holiday_anchor_pool|seasonal_naive_holiday_anchor_aware|seasonal_naive_lag_364" `
  ... (one per horizon/grain)
```

Also inspect the weekly holiday-segment summary (`multifold_segment_summary.csv`)
— the mechanism predicts the gain concentrates in `holiday_weeks`; if the gain
comes from `rest_weeks` instead, treat the result as suspect and investigate.

### Promotion gate

- Beats the champion at ≥1 horizon with HAC p < 0.05 and block CI excluding zero,
  AND does not lose (mean WAPE diff > +0.002) at any other horizon.
- If it passes: update `run_production_forecast.py` / `run_weekly_production_forecast.py`
  to emit it as the `final_*` row (champion becomes `benchmark_*`), update both md
  docs and the vault note.
- If it fails: document the negative result in `model_cards.md` (like the retired
  holiday-aligned rule) and stop — do not iterate on window widths (that would be
  window-mining; one pre-registered refinement round max, and only if the failure
  mode is diagnosable from segment metrics).

Estimated effort: ~1 day including runs (seasonal-naive-only pools run in minutes).

---

## WS2 — Fix weekly interval undercoverage (shrinkage intervals)

### Rationale

Weekly global 80% coverage: 0.680 (4w), 0.717 (12w) — target is 0.80. Pure
segment conditioning fails from small holiday samples. Fix: shrink segment
quantiles toward global quantiles by sample size, and/or widen with a holiday
scale factor. Also, weekly intervals currently calibrate on only the last 8 grid
windows (`--interval-windows 8` default in `run_weekly_production_forecast.py`)
— part of the undercoverage may simply be too little calibration data.

### Implementation

1. In `run_interval_coverage_backtest.py`, add mode `shrunken` beside
   `global|segment` in `quantiles_for_group` / `add_rolling_intervals`:
   - Segment residual quantile `q_seg` from `n_seg` rows; global `q_glob`.
   - `q = w * q_seg + (1 - w) * q_glob`, `w = n_seg / (n_seg + k)`, k = 30 rows
     (pre-registered default; sweep k in {10, 30, 60} in the backtest ONLY,
     pick by 4w+12w coverage, and record the choice — this is calibration
     tuning, not model selection, so a small sweep is acceptable).
2. Also test simply raising `--interval-windows` (8 → 16 and all-available) in
   global mode as the null hypothesis fix. If that alone reaches coverage
   targets, prefer it — simpler beats cleverer.
3. Re-run the coverage backtest for all 16 run/mode combos from
   `outputs_interval_coverage_backtest/manifest.json` plus the new modes.
4. Adopt the winning mode as the default in `run_weekly_production_forecast.py`
   and `run_production_forecast.py` (`--interval-mode` choices + default), and
   add `shrunken` to their row-building logic.
5. Extend `tests/test_intervals.py`: shrinkage weight math, degenerate segment
   (0 rows → pure global), k boundary behavior.

### Acceptance

- Backtest 80% coverage ≥ 0.75 and 95% coverage ≥ 0.90 at every weekly horizon.
- No segment (holiday/rest) below 0.65/0.85 at 80%/95% anywhere.
- Average 95% width increase ≤ 25% vs current global mode at the same horizon.
- Daily horizons must not regress below current global coverage.

Estimated effort: ~half a day; backtests are fast (they reuse existing grid CSVs).

---

## WS3 — Forecast monitoring + challenger adjudication (`run_forecast_monitor.py`)

### Rationale

The docs say "track the 75/25 blend" and "track the 26w pool_median" but nothing
operationalizes tracking. Monitoring is also the declared regime-change trigger:
a sustained one-sided bias on the shipped naive is what would make a
trend-scaled model correct (see vault note). Without an alarm, that trigger
never fires.

### Implementation

New script `run_forecast_monitor.py`:

- Inputs: `--forecast-csv` (a prior `production_forecast.csv` or
  `weekly_production_forecast.csv`), `--sales-csv` (refreshed actuals),
  `--log-csv` (default `outputs_forecast_monitor/monitoring_log.csv`, gitignored).
- For every forecast row whose `ds` is now realized in the actuals: compute
  per-model WAPE, bias pct, and interval hit/miss (was the actual inside p10–p90
  / p025–p975), grouped by model_name and by elapsed-horizon bucket.
- Append one row per (run_id of the source forecast manifest, model_name,
  evaluation window) to the log. Idempotent: skip (run_id, model, window) keys
  already logged.
- **Bias alarm:** one-sided CUSUM on daily bias pct of the shipped `final_*`
  rows: `S_t = max(0, S_{t-1} + (bias_t - slack))` with slack = 0.5 × the
  mean |bias| observed in the multifold backtests (read from the relevant
  `multifold_summary.csv`), alarm when S_t exceeds 4 × slack. Print a loud
  warning and write `alarm_state.json`. These constants are starting points —
  record them in the manifest so they can be revisited; do not silently retune.
- **Challenger adjudication:** for each realized window, also score the tracked
  challenger rows already present in the forecast CSVs (`challenger_*_blend_snl75_tfm25`)
  against the `final_*` rows, and append a champion-vs-challenger verdict line.
  This turns each data refresh into an out-of-exploration test automatically.
- Manifest + `append_experiment` like every other script. Tests for the CUSUM
  math and idempotent appending.

Additionally, emit the 26w challenger in the weekly package: in
`run_weekly_production_forecast.py`, add row
`challenger_26w_pool_median_trend_seasonal` — the median of the parameter-free
members of the 26w trend/seasonal pool (`daily_reagg_seasonal_naive_lag_365`,
`daily_reagg_scaled_seasonal_naive_lag_365_w28`,
`daily_reagg_scaled_seasonal_naive_lag_365_w56_damped`,
`daily_reagg_seasonal_average_lag_365_y2`, `daily_reagg_seasonal_average_lag_365_y3`,
`seasonal_naive_lag_52` — confirm the exact member list against
`outputs_weekly_multifold_26w_trend_seasonal_pool/manifest.json` `pool` field;
exclude the fitted stack rows, which are not reproducible without inner windows).
Intervals: its own residual quantiles from that pool's grid CSV, same mechanism
as existing rows. 26w only — it has no evidence at other horizons.

### Acceptance

- Running the monitor against the 2026-04-01-cutoff forecasts with the current
  (already partially realized) sales file produces a populated log and correct
  interval hit rates on a hand-checked sample.
- Weekly package emits the challenger row; docs updated to say the challenger is
  now packaged and auto-adjudicated.

Estimated effort: ~1 day.

---

## WS4 — Covariate residual-model track (the strategic lever; start after WS1–WS3)

### Rationale

From the critique: the only path to genuinely tighter point forecasts is new
information, and the architecture is fixed in advance — do NOT replace seasonal
naive; model its **residual** with a small regularized learner on covariates
that are known or forecastable over the horizon. Prove each covariate's
incremental OOS WAPE one at a time.

### Implementation

New script `run_residual_covariate_model.py`:

- Base: champion forecasts on the existing multifold grids (reuse
  `multifold_grid_predictions.csv` from the champion pools — residual = actual −
  champion forecast per (cutoff, ds)).
- Model: ridge regression (sklearn, standardized features, alpha from inner-CV
  on windows ending ≤ the fold cutoff — same nesting discipline as
  `run_folds`). Final forecast = champion + predicted residual, floored at 0.
- Feature sets, evaluated cumulatively one at a time (pre-registered order):
  1. `F1` moving-holiday distances: signed days to Easter, days to Thanksgiving,
     days to Black Friday, clipped to ±21 and one-hot-banded.
     (Skip if WS1 passed and already captured this — check whether F1 still adds
     signal on top of the WS1 champion; the residual model must always target
     whatever the current champion is.)
  2. `F2` opening structure from `opening_info.csv`: `num_units_open`,
     openings-in-window counts (features already exist in
     `add_opening_features`; reuse, don't reimplement). As-of joins only —
     planned openings are known ahead, so they are legitimate future features,
     but the event study showed aggregate opening signal is weak; expect F2 to
     fail and document it either way.
  3. `F3` promo/markdown calendar — blocked on data acquisition (WS5). Wire the
     join interface now (a generic `--covariate-csv date,name,value` long-format
     input) so F3 is a data drop, not a code change.
- Evaluation: same fold structure as the champion pools at daily 28/91/180/365
  and weekly 4/12/26/52 (weekly via daily residual model, re-aggregated).
  Promotion gate identical to WS1 (HAC + block CI vs current champion).
- Leakage checklist (each item must be explicitly verified and noted in the run
  summary): features computable at forecast time from information available at
  the cutoff; ridge alpha and scaler fit inside the fold; no covariate derived
  from the target series.

### Acceptance

- A `residual_ridge_f1` (and `_f2`, cumulative) candidate scored in the nested
  harness at all horizons with significance tests vs champion.
- Verdicts (positive or negative) written to `model_cards.md` and
  `model_usage_plan.md`. A negative result here is a valid, expected outcome —
  the deliverable is the harness plus documented evidence, not a win.

Estimated effort: ~2 days.

---

## WS5 — Covariate data spec (parallel, no code)

Write `covariate_data_spec.md` (committable — no data, no dollar figures)
defining the two acquisition asks from the critique:

1. **Promo/markdown calendar:** date range, scope (chain-level flag vs
   category-level intensity), fields (`start_date, end_date, promo_type,
   scope, planned_flag`), minimum history (ideally back to 2022-01 to cover the
   weekly panel), and the leak rule (only planned/known-ahead versions usable
   for forecasting; realized-spend versions only for attribution).
2. **Unit-level daily sales:** fields per `prepare_unit_level_targets.py`
   expectations (`unit_id, date, sales`), plus store-vs-ecomm channel split —
   this is the declared unlock for cannibalization/halo and opening analytics.

Estimated effort: ~1 hour.

---

## Explicitly out of scope (per the standing critique — do not do these)

- New model families on the aggregate series (no N-BEATS, no more foundation
  models, no deep nets). Six-plus families have already failed.
- Trend/level scaling revisits (fails because the decline is decelerating;
  documented). Revisit only if the WS3 bias alarm fires.
- Chronos-2 promotion (tested 6/25 across all daily and weekly horizons; did
  not beat champions).
- Re-tuning the retired `seasonal_naive_holiday_aligned_lag_52` weekly rule.
- Any comparison of a fewer-fold rerun against numbers from a different fold
  design (documented trap in `model_usage_plan.md`).

## Suggested execution order

| Order | Workstream | Depends on | Rough effort |
|---|---|---|---|
| 1 | WS0 commits | — | 1–2 h |
| 2 | WS1 holiday-anchor naive | WS0 | 1 day |
| 3 | WS2 intervals | WS0 (parallel with WS1) | 0.5 day |
| 4 | WS3 monitor + challenger row | WS0 | 1 day |
| 5 | WS4 residual covariates | WS1 verdict (champion may change) | 2 days |
| 6 | WS5 data spec | — (anytime) | 1 h |

After each workstream: update `model_cards.md` / `model_usage_plan.md`, rebuild
`RUN_INDEX.md`, commit (code + docs only), and record the verdict below.

## Pre-registration log

Record candidates and their exact definitions BEFORE scoring; append verdicts after.

- 2026-07-02: `seasonal_naive_holiday_anchor_aware` and
  `daily_reagg_seasonal_naive_holiday_anchor_aware` as defined in WS1
  (fixed windows, no tuning). Verdict: rejected. Daily 180d improved mean WAPE
  by -0.0024 but failed significance (HAC p=0.409; block CI crossed zero);
  28d/91d/365d lost. Weekly lost to `daily_reagg_seasonal_naive_lag_365` at
  4/12/26/52w by about +0.0022 to +0.0025 mean WAPE. Holiday weeks improved,
  but rest weeks degraded enough to lose overall. No window tuning.
- 2026-07-02: interval shrinkage k in {10, 30, 60} (calibration sweep, backtest
  only). Verdict: implemented but not adopted. k=10/30/60 did not meet weekly
  coverage gates; 16-window and all-available global intervals also reduced
  weekly 80% coverage. Keep production default `global` with 8 windows.
- 2026-07-02: `residual_ridge_f1` (moving-holiday distances),
  `residual_ridge_f1f2` (+ opening structure) as defined in WS4. Verdict:
  rejected. Every daily and weekly horizon had positive WAPE deltas versus the
  current champion; the harness remains for future known-ahead promo covariates.
