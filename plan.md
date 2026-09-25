# Improvement Plan — Time Series Bakeoff

## Active handoff — 2026-09-08

Status: in progress; items 1 and 2 completed 2026-09-17 and item 3 completed
2026-09-21. The bounded TabFM 28-day follow-on and requested 91-day extension
also completed 2026-09-21. Production models remain unchanged. Resume at next
action 2 below; do not rerun completed work. This section supersedes conflicting
July instructions and older vault conclusions, including the blanket exclusion
of new model families. The July workstreams and registration log below remain
historical context, not an instruction to repeat completed work or commit/push.

### Starting point and scope

- Read `model_usage_plan.md`, `model_cards.md`, `DATA.md`, then the manifests for
  the runs being compared. Inspect the current working tree before editing;
  preserve unrelated changes. Use the project `.venv` and `uv` for dependencies.
- Keep daily `seasonal_naive_lag_364` and weekly
  `daily_reagg_seasonal_naive_lag_365` as production defaults during experiments.
- Current evidence supports "tested candidates have not reliably beaten the
  benchmark", not proof that history-only forecasting has reached a ceiling.
- Reuse the nested validation, significance, monitoring and experiment-registry
  tooling. Do not rebuild the pipeline or add a broad hyperparameter search.
- Keep sales data and absolute sales figures local and gitignored. Report WAPE,
  bias percentages, coverage and relative improvements in committed documents.

### 1. Repair cumulative-interval validation first — completed 2026-09-17

Completion record: `rolling_intervals` now requires each same-step calibration
residual's `ds <=` the evaluated cutoff before window selection and minimum-count
checks. Regression tests pass. Replacement artifact:
`outputs_cumulative_error_calibration_corrected_20260917` (run_id
`20260917T143836Z`). At 180d, 13 terminal origins were eligible and observed
80%/95% coverage was 0.000; at 365d, four observable terminal residuals do not
exist, so results stop at step 364. The original output is provenance only.

Review `calibrate_cumulative_error_paths.py::rolling_intervals` and its callers.
The reviewed implementation selects earlier forecast origins but does not require
their cumulative residuals to be observable at the evaluated origin. Overlapping
180/365-day windows therefore admit future actuals into calibration.

- At each horizon step, require the calibration residual's `ds <= cutoff` of
  the evaluated forecast. Apply `recent8`/all-prior selection to eligible
  residuals, and enforce the minimum calibration count after this filter.
- Add a regression test with overlapping origins: changing actuals after the
  evaluated cutoff must not change its calibrated bands. Check the boundary
  where a residual becomes available and the insufficient-history case.
- Rerun the existing cumulative calibration design into a new output directory.
  Mark `outputs_cumulative_error_calibration` as superseded for coverage claims;
  retain it for provenance. Report eligible windows, coverage and width by step
  and at the terminal horizon. Do not sum daily interval endpoints.
- Acceptance: no future residual enters calibration; relevant tests pass;
  corrected results and limitations are recorded. Sparse long-horizon evidence
  is a valid limitation, not a reason to relax the availability rule.

### 2. Refresh the existing challengers and reconcile the written plan — completed 2026-09-17

Completion record: `latest_sls_aug.csv` is unchanged through 2026-08-29, so the
reproduced daily scorecard adds no new 28/91 windows and no post-registration
180/365 windows are mature. The exact registered weekly 26w pool was refreshed
on six newly matured origins; its `pool_median` lost to lag-365 (WAPE 0.03658 vs
0.03511, 2/6 wins, about 1.77 effective windows). Production models are unchanged.
Artifacts: `outputs_naive_timesfm_challenger_sep2026` and
`outputs_weekly_multifold_26w_trend_seasonal_refresh_sep2026`. Model cards now
record the one-fold TimesFM 3 results and non-commercial license restriction;
the July workstreams below are archived.

- Freeze the existing daily blend at 0.75 seasonal naive + 0.25 TimesFM 2.5.
  Reuse `explore_naive_beaters.py`, `run_forecast_monitor.py`, and exact validation
  grids. Score newly available completed windows at 28/91 days; add 180/365-day
  confirmation only as post-registration windows mature.
- Refresh the existing weekly 26w trend/seasonal `pool_median` challenger using
  its exact registered membership. Do not transfer its verdict to other horizons.
- Reference evidence: `outputs_naive_timesfm_challenger_aug2026` reports daily
  28d WAPE 0.0582 -> 0.0565 (3/5 wins) and 91d 0.0627 -> 0.0610 (3/3 wins,
  overlapping). These are tracking results, not a production promotion.
- Add the August 31 TimesFM 3 results to the model cards: all four summaries in
  `outputs_multifold_*d_timesfm3_aug2026_test` evaluate just one fold. Standalone
  TimesFM 3 loses to naive at each horizon; the three-member median improves
  91d WAPE 0.0531 -> 0.0490. Label this exploratory. Default TimesFM 3 weights
  currently prohibit commercial/production use; exclude them from this business
  production shortlist, and do not bypass the existing license guard.
- Update stale guidance in `time_series_forecasting_bakeoff_plan.md`: inverse-
  squared WAPE amplifies weight differences; it is not inherently more stable
  than inverse WAPE. Prefer fixed weights; any learned weights need nested
  fitting and shrinkage toward the benchmark. Archive completed July tasks.
- Acceptance: a refreshed, provenance-backed scorecard distinguishes exploration,
  confirmation and production, with dates, folds, overlap and explicit verdicts.

### 3. Run a small new-model screen at 28/91 days — completed 2026-09-21

Completion record: `new_model_screen_registration.json` froze the data hash,
28/91 cutoffs, pinned packages/checkpoints, recipes, context limits, fixed blends,
and keep/reject rule before scoring. The exact TiRex-2 screen is in
`outputs_new_model_screen_sep2026` (run_id `20260921T143751Z`). Standalone
TiRex-2 and its fixed 75/25 naive blend were rejected at both horizons: the
blend scored 0.05852 WAPE at 28d versus 0.05888 naive and 0.05685 for the frozen
naive/TimesFM blend, and 0.06256 at 91d versus 0.06490 and 0.06084. TiRex-2
ran locally in an isolated CPU environment; Torch 2.8 required the documented
`triton-windows==3.4.0.post21` compatibility pin. After the research terms were
accepted, TabPFN-TS also completed the frozen screen in
`outputs_new_model_screen_tabpfn_sep2026` (run_id `20260921T155428Z`) using
local weights and CUDA. Its blend was rejected at both horizons: 0.05671 WAPE
at 28d beat the 0.05685 TimesFM blend but worsened absolute bias versus naive,
while 0.06146 at 91d trailed the 0.06084 TimesFM blend. The weights remain
ineligible for business production without a separate commercial license.
No degraded or fallback prediction was counted and production defaults are unchanged.

Pre-register exact checkpoints, feature recipes, context limits, data snapshot,
cutoffs and candidate formulas before scoring. Freeze this candidate set:

| Candidate | Evaluation | Hypothesis |
|---|---|---|
| TiRex-2 | Standalone and 0.75 naive + 0.25 TiRex-2 | A recurrent foundation model contributes useful errors different from TimesFM. |
| TabPFN-TS | Standalone and 0.75 naive + 0.25 TabPFN-TS | Tabular forecasting with a fixed calendar/seasonal feature recipe adds complementary signal. |

- Compare on identical windows against naive and the frozen naive/TimesFM 2.5
  blend. Measure incremental ensemble value even when a standalone model loses.
  Do not tune blend weights or add a pooled mega-ensemble after seeing results.
- Start with one integration smoke run per new model. Verify package/checkpoint
  terms and local runtime before the full screen; pin working versions. TiRex-2
  documents Windows Docker support but native installation is tested mainly on
  Linux/macOS. Keep dependency experiments isolated from the production venv.
- TabPFN-TS's documented wrapper uses known-future covariates and drops past-only
  and static covariates. Confirm the pinned version's behavior; never silently
  claim ignored inputs were evaluated.
- Existing historical windows are exploration for these new recipes. Freeze
  survivors before evaluating fresh confirmation windows. Expand to 180/365 days
  only after a useful short/medium-horizon signal; do not claim transfer.
- Acceptance: reproducible manifests, no degraded/fallback predictions counted
  as model successes, same-window scorecards, runtime and explicit keep/reject
  decisions. A negative result completes the experiment.

### Next actions — ordered by expected information value

1. **Done 2026-09-21: bounded Google TabFM 28-day screen.** A new registration
   was created without amending the scored TiRex/TabPFN registration. It pinned
   the source revision, `google/tabfm-1.0.0-pytorch` checkpoint, non-commercial
   license, seed, 256-row context, one estimator, BF16/CUDA runtime and a direct
   leak-free calendar/horizon/lag-364/trailing-history recipe. The same six
   28-day cutoffs and comparators were used. Smoke and full scoring completed in
   `outputs_new_model_screen_tabfm_sep2026` (run_id
   `20260921T202918Z`) with no fallback. Standalone TabFM scored 0.06827 mean
   WAPE and won 2/6 windows versus naive. Its fixed blend improved naive
   (0.05765 versus 0.05888), had the best median WAPE (0.05405), and much lower
   absolute mean bias, but lost to the TimesFM blend (0.05685) and had a larger
   worst-window loss (0.00508 versus 0.00328). It is rejected from fresh
   confirmation; retain it only for the next error-diversity audit. A separately
   registered 91-day extension (`outputs_new_model_screen_tabfm_91d_sep2026`,
   run_id `20260921T204255Z`) reinforced the rejection: standalone WAPE was
   0.09636, while the blend essentially tied naive (0.064886 versus 0.064897)
   but trailed TimesFM (0.06084) and had a 0.00723 worst-window loss versus naive.
2. **Use the existing 28-day predictions to design, not score, confirmation.**
   Audit paired daily/fold errors for naive, TimesFM and TabPFN, with TabFM kept
   only as a diagnostic, by horizon step and holiday/rest slice. Freeze the symmetric
   `0.75 naive + 0.125 TimesFM + 0.125 TabPFN` blend only for future confirmation
   if the errors are complementary; do not claim a historical win or tune its
   weights on the six seen windows. Preserve TabPFN's frozen rejection while
   recording that it was the 28-day WAPE winner and missed only the bias guardrail.
3. **Run fresh 28-day confirmation when new actuals mature.** Do not rerun the
   same August snapshot and call it confirmation. Compare the frozen TabPFN and,
   if it survives exploration, TabFM recipes on new complete non-overlapping
   windows. Before scoring, obtain a business bias tolerance; absent one, report
   WAPE leadership and bias separately instead of forcing another binary verdict.
4. **Then prioritize new information over more model families.** Execute item 4
   only with versioned, as-of-available market, promotion, closure and opening-
   maturity inputs. This is more likely to create durable lift than continuing
   a broad zero-shot model search. If those inputs are unavailable, document the
   blocker and proceed to the two narrow structural ensembles in item 5.
5. **Bounded NeuralProphet trial — completed 2026-09-25.**
   Start with aggregate daily sales at 28 days using the existing rolling
   cutoffs. Pre-register two configurations: trend/seasonality without
   autoregression, and the same recipe with autoregression. Freeze package
   version, seed, lags, training settings, data snapshot and evaluation gates
   before scoring; keep any tuning inside training folds, with no broad search.
   Run one local smoke test in an isolated environment (upstream recommends
   WSL2 on Windows), then reuse existing scoring and registry tooling.
   Compare each configuration standalone and as a fixed 0.75 seasonal naive +
   0.25 NeuralProphet blend against seasonal naive and the frozen naive/TimesFM
   blend on identical windows. Report WAPE, signed bias, worst-window loss,
   holiday/rest performance and runtime, with no future actuals used as inputs.
   Treat these historical windows as exploration. Advance to 91 days or shared
   market modeling only if the screen shows useful incremental value under
   the evaluation contract below; freeze survivors for fresh confirmation.
   Known-ahead promotion, closure or opening features require versioned as-of
   inputs and a separate added-input comparison. Component effects are not
   causal lift estimates. Record a reproducible verdict even if negative;
   production defaults remain unchanged.
   Completion: `neuralprophet_screen_registration.json` froze both configurations
   and `neuralprophet_ensemble_registration.json` froze three additional blends
   before the scorecard was read. `outputs_neuralprophet_screen_sep2026` records
   six 28-day and eight 91-day historical windows, run_id `20260925T140644Z`.
   Standalone trend and AR models lost at both horizons. The 28-day fixed
   75/25 naive/trend blend narrowly beat naive/TimesFM (0.05618 vs 0.05685
   WAPE) and cleared the exploratory gate; retain only for fresh confirmation.
   The 91-day trend blend and all AR blends failed. A frozen three-member
   naive/TimesFM/trend blend scored 0.05629 at 28 days but remains diagnostic.
   Production recommendations did not change.
   A requested 365-day extension completed on 2026-09-25 using two nearly
   non-overlapping annual windows and the unchanged recipes. Its artifact is
   `outputs_neuralprophet_365d_sep2026` (run_id `20260925T141945Z`). Standalone
   trend and AR scored mean WAPE 0.12407 and 0.23844 versus 0.08518 for naive;
   both 75/25 naive/NeuralProphet blends lost to naive/TimesFM at 0.07551.
   This does not evaluate a differently configured team forecasting tool.

TabFM did not clear the registered mean-WAPE/worst-loss gate, but its strong
median and two standalone fold wins justify only the already bounded error-
diversity audit in action 2. Do not add another foundation model until fresh
confirmation or new covariates justify the cost.

### 4. Add information through markets and covariates

Proceed when suitable input data are available; document missing inputs instead
of substituting realized future information.

- Test Chronos-2 with related market series and/or versioned known-ahead promo
  plans. This changes the information set from the existing single-series
  wrapper. Compare target-only and added-input versions on identical folds.
- Test seasonal naive plus a small pooled CatBoost or LightGBM residual model
  across markets, using promotions, closures and opening maturity. Reuse existing
  features and compare with market TimesFM, market naive and pooled LightGBM.
  Train residual targets from out-of-sample forecasts available at each cutoff;
  fit preprocessing and any tuning inside the training fold.
- Treat future opening/promotion schedules as known only if historical as-of
  versions substantiate that claim. Forecast skill does not establish causal
  opening or promotion lift; keep attribution work separate.
- Acceptance: incremental value of the added information is isolated, data
  availability is documented, and both per-market and aggregate metrics appear.

### 5. Test two narrowly defined structural ensembles

These are follow-on experiments, not prerequisites for the initial screen.

- Weekly-to-daily allocation: use the validated weekly lag-365 reaggregation
  forecast as the total and allocate with weekday shares estimated only from
  prior data. Freeze the share-estimation window before scoring. Verify exact
  weekly coherence and score daily error, holiday slices and weekly totals.
  This is distinct from the rejected direct daily lag-364/365 blend.
- Aggregate-anchored market reconciliation: reconcile market TimesFM forecasts
  to a direct forecast of the same aggregate scope. Start with simple historical
  shares; defer covariance-based reconciliation until that baseline warrants it.
  Resolve the documented ~88% market-panel coverage of total sales first, or
  restrict evaluation to the panel's own aggregate. Do not silently equate targets.
- Acceptance: exact aggregation identities, no future-derived shares, and gains
  assessed at both levels without hiding deterioration in component forecasts.

### Evaluation and promotion contract

- Freeze a common data snapshot and explicit origin/date grid. Use only completed
  inner windows for fitting and calibration; audit missing/synthetic dates.
- Record mean and median WAPE, signed bias, paired changes versus the incumbent,
  worst-window loss, holiday/rest performance, interval coverage/width and runtime.
  Report WAPE percentage-point changes separately from relative improvement.
- Before confirmation, record horizon-specific minimum useful WAPE improvement,
  maximum acceptable bias/worst-window regression, interval and runtime limits.
  These business thresholds remain to be set; absent them, report evidence and
  retain the incumbent rather than inventing promotion criteria after scoring.
- Retain paired HAC and moving-block-bootstrap evidence, disclose overlap and
  the approximate effective window count, and treat tiny long-horizon samples
  as inconclusive. Account for multiple candidate/horizon comparisons with a
  declared correction or a separately frozen fresh-confirmation stage.
- Promote per horizon only when statistical evidence, practical thresholds and
  operational requirements pass. Do not require a candidate to win every horizon.
- After code changes, run relevant regression tests and the full project suite
  before handoff. Update `model_cards.md` and `model_usage_plan.md`; rebuild
  `RUN_INDEX.md` after new runs. Record each recipe and verdict in the registration
  log. Do not commit/push unrelated working-tree changes.

### Handoff deliverables

1. Corrected cumulative-interval code, tests and replacement coverage artifacts.
2. Refreshed daily/weekly challenger scorecards and current model documentation.
3. Registered TiRex-2/TabPFN-TS and Google TabFM screens with keep/reject decisions.
4. Market/covariate and structural-ensemble results, or explicit data blockers;
   bounded NeuralProphet screen registration, scorecard and verdict.
5. A final per-horizon recommendation listing evidence, remaining uncertainty,
   practical thresholds and whether any production change is justified.

References checked during the review (recheck versions/terms before execution):
[TimesFM](https://github.com/google-research/timesfm),
[TiRex-2](https://github.com/NX-AI/tirex-2),
[TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series),
[TabFM](https://github.com/google-research/tabfm),
[NeuralProphet](https://github.com/ourownstory/neural_prophet),
[Chronos-2](https://github.com/amazon-science/chronos-forecasting).

---

## Archived July 2026 plan

Status: completed and archived. The workstreams below are retained only for
recipe and decision provenance; do not execute their imperative steps. Use the
active September handoff above for current work.

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
- 2026-07-09: `pool_median` from the exact two-member daily pool
  `seasonal_naive_lag_364,seasonal_naive_lag_365`. With two members this is a
  fixed 50/50 arithmetic blend, pre-registered to hedge weekday alignment
  (lag 364) against calendar-date alignment (lag 365) without fitted weights.
  Score 28/91/180 days on the same 14-window extended design used for the
  holiday-anchor experiment and 365 days on its single feasible nested fold.
  Promotion gate: HAC p < 0.05 and block-bootstrap 95% CI below zero at at least
  one horizon, with no mean WAPE loss greater than +0.002 at another horizon.
  Verdict: rejected. It won 0/14 windows at 28/91/180 days and 0/1 at 365 days;
  mean WAPE losses were +0.0720/+0.0685/+0.0620/+0.0719 respectively. Daily
  lag 365 shifts the weekday pattern, and equal blending preserves too much of
  that error. Keep lag 365 only for the validated daily-to-weekly reaggregation
  workflow; do not add this blend to production.
- 2026-09-17: refreshed the pre-registered weekly 26w `pool_median` using the
  exact six-member pool from
  `outputs_weekly_multifold_26w_trend_seasonal_pool/manifest.json`. On six newly
  matured, heavily overlapping origins (2025-10-08 through 2026-02-25), it lost
  to `daily_reagg_seasonal_naive_lag_365`: WAPE 0.03658 vs 0.03511, 2/6 wins,
  mean difference +0.00147, worst loss +0.00478. Verdict: confirmation failed;
  retain lag-365 in production and keep the packaged pool median diagnostic only.
