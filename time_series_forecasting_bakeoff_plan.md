# Time Series Forecasting Model Bakeoff Plan

## Objective

Build a structured forecasting benchmark for dense daily retail sales data using multiple modern forecasting libraries and model families, then evaluate whether an ensemble of the strongest performers improves accuracy and stability.

The input data will contain:

```text
date
sales
```

The goal is to produce accurate daily forecasts with prediction intervals where possible, evaluate models through rolling backtests, and identify the best model or ensemble for production use.

The key constraint is important: **data is only `date` and `sales`**. That means this test is focused on pure time-series signal: trend, seasonality, autocorrelation, holiday-like calendar effects if derived from date, and model generalization. No price, promo, stockout, channel, or product attributes yet.

---

## 1. Data Assumptions

### Required Input Schema

```text
date: daily date
sales: numeric sales value
```

Optional if testing multiple series later:

```text
unique_id: SKU, store, product area, market, or other series identifier
```

If the dataset is currently only one aggregate sales series, create:

```text
unique_id = "total_sales"
```

Final standardized format:

```text
unique_id | ds | y
```

Where:

```text
unique_id = series identifier
ds = date
y = sales
```

---

## 2. Data Preparation

### 2.1 Validate Date Continuity

Ensure every date exists in the series.

For missing dates:

```text
If business was open and no sales occurred: fill sales as 0
If data is missing due to system issue: impute or flag, but do not blindly fill with 0
```

For dense retail sales, missing dates are usually more likely to be data gaps than true zero-sales days unless this is SKU-level data.

### 2.2 Handle Negative Sales

Retail sales data may contain negative days due to returns, adjustments, or cancellations.

Recommended approach:

```text
Keep negative sales in the base benchmark.
Create a second sensitivity version where sales are winsorized or adjusted.
Compare results.
```

Do not remove negative values automatically because returns are part of the actual observed net-sales process.

### 2.3 Target Transformation

Negative sales rule out a plain `log(y)` transform. Pick one transformation strategy upfront and apply it uniformly across all models:

```text
Option A: no transform (simplest, fine for stable variance)
Option B: log1p, only after dropping or flooring negatives
Option C: arcsinh (sign-preserving, handles negatives natively)
Option D: Box-Cox via StatsForecast / MLForecast target_transforms
```

Recommended default:

```text
arcsinh transform during fit, inverse-transform at predict time
```

This stabilizes variance, keeps the sign of returns, and improves interval calibration for models that assume Gaussian residuals.

### 2.4 COVID and Structural Break Handling

If the training window includes 2020–2021, every model fits a shock that will not repeat. Pick one strategy and apply it consistently:

```text
Option A: exclude 2020-03 through 2021-06 entirely
Option B: include with a COVID indicator feature (derived from date)
Option C: train two variants, with and without COVID, compare
```

Recommended default for production-style models:

```text
Include the period but add a binary COVID-window flag for ML models.
For stats models that cannot accept flags, exclude the window.
```

This must be decided before phase 1, not retrofitted after seeing results.

### 2.5 Outlier Treatment

Create two benchmark datasets:

```text
Dataset A: raw sales
Dataset B: outlier-adjusted sales
```

Outlier adjustment options:

```text
Cap extreme values at p1 / p99
Cap extreme residuals after seasonal decomposition
Flag extreme days but keep them in model training
```

Since only `date` and `sales` are available, avoid aggressive outlier removal unless there are obvious data errors.

### 2.6 Calendar Features

Even with only `date` and `sales`, feature-based models can use derived calendar features:

```text
day_of_week
week_of_year
month
quarter
day_of_month
day_of_year
is_weekend
is_month_start
is_month_end
is_quarter_start
is_quarter_end
fiscal_year
fiscal_week
fiscal_month
```

For IKEA-style fiscal years, use:

```text
Fiscal year starts September 1
```

Holiday flags can technically be added from `date`, but since the data is simply date and sales, run two versions:

```text
Version 1: date-derived calendar features only
Version 2: calendar features + retail-relevant holiday flags
```

"US holidays" is too vague for retail. Use the `holidays` Python package as a base, then overlay retail-specific events that federal calendars miss:

```text
fixed-date: Memorial Day, July 4, Labor Day, Halloween, Thanksgiving, Christmas Eve, Christmas, New Year's Eve
moving-date: Easter, Mother's Day, Father's Day
retail-specific: Black Friday, Cyber Monday, back-to-school window, Prime Day
day-before / day-after flags for major closures
```

For each holiday, prefer separate features for the day, day-before, and day-after rather than one binary flag.

---

## 3. Forecast Horizons

Test multiple horizons because model performance can vary heavily by use case.

Recommended horizons:

```text
7 days
14 days
28 days
56 days
91 days
365 days
```

Suggested mapping:

| Horizon | Business Use |
|---:|---|
| 7 days | Short-term operations |
| 14 days | Tactical planning |
| 28 days | Monthly planning |
| 56 days | Promo / campaign planning |
| 91 days | Quarterly outlook |
| 365 days | Annual planning |

For dense daily retail sales, make **28 days and 91 days** the primary evaluation horizons.

**Caveat on the 365-day horizon.** With only a few years of history and monthly cutoffs, the available test windows for a 365-day horizon are few in number and heavily overlapping, all dominated by one or two annual cycles. Rankings at this horizon will be noisy and should be treated as directional only — not used to select the production model. If annual forecasts are a real business need, split that off as a separate exercise rather than letting it influence the bakeoff ranking.

---

## 4. Backtesting Design

Use rolling-origin cross-validation.

### 4.1 Minimum Training Window

Since there are years of daily data, start with:

```text
Minimum training window = 2 years
```

This gives the models enough history to learn:

```text
weekly seasonality
annual seasonality
trend
holiday-like patterns
```

### 4.2 Rolling Cutoffs

Use multiple forecast origins.

Example:

```text
Train through 2023-08-31 → forecast next 91 days
Train through 2023-09-30 → forecast next 91 days
Train through 2023-10-31 → forecast next 91 days
...
```

Recommended cutoff frequency:

```text
Monthly cutoffs for 91-day and 365-day horizons
Weekly cutoffs for 7-day, 14-day, and 28-day horizons
```

Note that consecutive monthly cutoffs at a 91-day horizon produce heavily overlapping test windows. Track the **effective sample size** (count of non-overlapping windows) separately from the raw cutoff count when reasoning about whether differences are real.

For each cutoff, enforce a strict train/test gap equal to the forecast horizon — no part of the test window may overlap with training data, and any features computed at the cutoff must use only data up to and including that cutoff. MLForecast and StatsForecast handle this correctly when called with `cross_validation`; verify it explicitly for any custom code.

### 4.3 Retraining Cadence

Decide upfront whether models refit at every cutoff or fit once and re-forecast. This changes both runtime and the realism of the evaluation:

```text
StatsForecast models: refit at every cutoff (fast, cheap)
MLForecast tree models: refit at every cutoff (minutes)
NeuralForecast NHITS/NBEATS: refit every N cutoffs, N = 3 to 6 (hours otherwise)
AutoGluon-TimeSeries: refit every N cutoffs, N = 3 to 6
Chronos / TimesFM zero-shot: no refit, only re-forecast
```

Document the refit cadence per model in the results table. A model that "wins" by being refit twice as often as competitors did not really win.

### 4.3 Holdout Period

Keep a final untouched test window.

Recommended:

```text
Final holdout = most recent 6 to 12 months
```

Use the earlier history for model selection and the final holdout for final reporting.

---

## 5. Models to Test

### 5.1 Naive Baselines

These are mandatory. They establish whether advanced models are actually adding value.

Test:

```text
Naive
Seasonal naive, lag 7
Seasonal naive, lag 364
Moving average 7
Moving average 28
Last year same day
```

For retail daily data, strong baselines are:

```text
same day last week
same day last year
average of same weekday over recent weeks
```

These should be hard to beat.

---

### 5.2 StatsForecast Models

Library:

```text
statsforecast
```

Models to test:

```text
AutoETS
AutoARIMA
MSTL
SeasonalNaive
DynamicOptimizedTheta
HistoricAverage
WindowAverage
Croston-style models if SKU-level intermittent data is later added
```

Use this group for fast, scalable statistical benchmarks.

Expected strengths:

```text
fast
stable
good with seasonality
good baselines for many series
native prediction intervals for many models
```

Expected weaknesses:

```text
limited ability to learn complex nonlinear patterns
less useful without exogenous variables for promo/price effects
```

---

### 5.3 MLForecast Models

Library:

```text
mlforecast
```

Models:

```text
LightGBM
CatBoost
XGBoost
RandomForest or ExtraTrees as a secondary baseline
ElasticNet / Ridge as a simple ML baseline
```

Feature set:

```text
lags: 1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 364
rolling means: 7, 14, 28, 56, 91
rolling std: 7, 14, 28
rolling min/max: 7, 28
expanding mean
day_of_week
month
week_of_year
day_of_year
is_weekend
fiscal_month
fiscal_week
```

Primary candidates:

```text
MLForecast + LightGBM
MLForecast + CatBoost
```

Forecasting strategy — must be decided per horizon:

```text
Recursive (one model, feeds its own predictions back as lag features):
    - cheap, single model per series
    - error compounds at long horizons
    - use for 7-day and 14-day horizons

Direct (one model per horizon step):
    - no compounding, but H models to train
    - better calibrated long-horizon forecasts
    - use for 28-day, 91-day, 365-day horizons

Hybrid (DirRec / blocked direct):
    - one model per block of horizon steps
    - balances training cost and accuracy
    - useful when H is large
```

Recommended:

```text
Recursive for 7 and 14 day horizons.
Direct (or blocked direct with 7-day blocks) for 28, 91, 365 day horizons.
```

Document which strategy was used per row in the results table — recursive and direct LightGBM are effectively different models.

Expected strengths:

```text
usually very strong for retail
fast enough for production
easy to add future covariates later
works well with calendar/lags
can support conformal prediction intervals
```

Expected weaknesses:

```text
requires feature engineering
long-horizon recursive forecasts can accumulate error
without price/promo/stockout variables, some shocks are unknowable
```

---

### 5.4 NeuralForecast Models

Library:

```text
neuralforecast
```

Models to test:

```text
NHITS
NBEATS
TFT
PatchTST or transformer-style model if available in your environment
```

For only `date` and `sales`, start with:

```text
NHITS
NBEATS
```

Then add more complex models only if they beat the simpler options.

NBEATSx is intentionally excluded — it expects exogenous variables, and on a univariate calendar-feature-only setup it adds little beyond NBEATS while costing more tuning. Reintroduce it once price/promo/stockout features are available.

Expected strengths:

```text
can learn nonlinear seasonal patterns
can benefit from many related series
can produce quantile forecasts
good challenger for dense data
```

Expected weaknesses:

```text
more tuning required
may underperform simple models on single series
GPU helpful for larger tests
less interpretable
```

Important note:

```text
Neural models become much more compelling when there are many related series.
For one aggregate sales series, they may not beat StatsForecast or MLForecast.
```

---

### 5.5 AutoGluon-TimeSeries

Library:

```text
autogluon.timeseries
```

Test presets:

```text
medium_quality
high_quality
best_quality, if runtime allows
```

Models may include:

```text
statistical models
tabular ML models
deep learning models
Chronos / foundation-model variants
ensembles
```

Expected strengths:

```text
excellent automated benchmark
built-in quantile forecasts
can create ensembles automatically
good for comparing many model families quickly
```

Expected weaknesses:

```text
less transparent
can be heavier computationally
production integration may be more complex than a focused Nixtla pipeline
```

---

### 5.6 Chronos / Chronos-2

Use through:

```text
AutoGluon-TimeSeries
or standalone Chronos package if preferred
```

Expected strengths:

```text
strong zero-shot or low-tuning baseline
useful for cold-start or limited-history cases
probabilistic forecasts
good challenger model
```

Expected weaknesses:

```text
must validate heavily on retail data
may be less controllable for business scenarios
without exogenous variables, cannot understand promos or price changes directly
```

Context length note:

```text
Chronos-bolt context window is roughly 2048 tokens.
For daily data that is about 5-6 years.
Earlier variants are shorter; verify the specific checkpoint used.
If history exceeds the window, the model sees only the most recent slice.
```

If your daily history is longer than the model's context window, either truncate to the most recent N days deliberately or pre-aggregate to weekly. Do not assume the model sees your full multi-year history just because you passed it in.

---

### 5.7 TimesFM

Use either:

```text
BigQuery TimesFM
open-source TimesFM
Hugging Face implementation
```

Expected strengths:

```text
fast foundation-model benchmark
potentially easy to run close to BigQuery data
good zero-shot baseline
useful for aggregate forecasts
```

Expected weaknesses:

```text
limited control
may not handle business-specific drivers without extra modeling
quantile support depends on implementation
needs calibration checks
```

Context length note:

```text
TimesFM base context length is around 512 tokens.
For daily data that is roughly 18 months.
Newer 2.0 checkpoints extend this; verify the version used.
```

Same rule as Chronos — explicitly choose how much history the model sees. For multi-year history on daily data, TimesFM will only consume the most recent slice that fits its window.

---

### 5.8 Prophet, Cut from the Bakeoff

Prophet is not part of the bakeoff. It is rarely SOTA for dense retail data, intervals are commonly miscalibrated, and at scale it is slow. Keep it available as a one-off if a stakeholder asks for component decomposition charts (trend / weekly / yearly), but do not benchmark it against the production candidates.

If interpretable decomposition is genuinely needed, prefer StatsForecast `MSTL` — it produces the same kind of components and is already in the bakeoff.

---

## 6. Evaluation Metrics

Use multiple metrics because each answers a different business question.

### 6.1 Point Forecast Metrics

Primary:

```text
WAPE
bias
MASE
MAE
weighted RMSE
```

Recommended primary metric:

```text
WAPE
```

Formula:

```text
WAPE = sum(abs(actual - forecast)) / sum(abs(actual))
```

Use WAPE because it behaves well for retail sales and is easy to explain.

sMAPE is intentionally excluded. It has asymmetric behavior near zero and can rank models perversely when actuals are small or negative. WAPE plus bias plus MASE cover the same ground without the foot-guns.

Recommended secondary metric:

```text
MASE — scaled against an in-sample seasonal-naive baseline
```

MASE auto-normalizes against a seasonal-naive benchmark (seasonality = 7 for daily data), which makes "are advanced models earning their keep?" trivially answerable: MASE < 1 means the model beats seasonal naive.

Caveat for windows containing many returns:

```text
WAPE denominator = sum(abs(actual)) — uses absolute values, so a window
heavy with returns does NOT collapse the denominator.
Bias denominator = sum(actual) — CAN collapse if returns roughly offset
sales in a window, producing absurd bias percentages.
```

Mitigation:

```text
Aggregate bias over the FULL evaluation window, not per-cutoff,
to keep the denominator stable.
Report bias both as a percentage and as an absolute total
(sum(forecast - actual)) so a near-zero denominator is detectable.
```

### 6.2 Bias

Bias is critical.

```text
Bias_pct = sum(forecast - actual) / sum(actual)
Bias_abs = sum(forecast - actual)
```

Interpretation:

```text
Positive bias = over-forecasting
Negative bias = under-forecasting
```

A model with slightly worse WAPE but materially lower bias may be better for planning. Always report both forms — the percentage is interpretable, the absolute total is robust when the denominator is near zero.

### 6.3 Interval Metrics

For models that produce intervals:

```text
coverage
average interval width
pinball loss
weighted interval score
```

Example:

```text
If a model provides an 80% prediction interval, actuals should land inside that range approximately 80% of the time.
```

Track calibration by horizon:

```text
7-day coverage
28-day coverage
91-day coverage
365-day coverage
```

### 6.4 Statistical Significance of Rankings

A 0.5 to 1.5 percent WAPE gap between two models is often noise. Before declaring a winner, test whether the gap is real:

```text
Pairwise: Diebold-Mariano test on per-cutoff loss differences
Multi-model: Model Confidence Set (MCS) or MCB ranking test
Cheap fallback: block bootstrap on the cutoff dimension to get
                  a 95% CI on each model's WAPE
```

Use the bootstrap CIs to set the threshold in the decision rule (section 13). If the typical CI width on WAPE is +/- 1.5 percent, then any "improvement" smaller than 1.5 percent is not actionable.

When using Diebold-Mariano, use the small-sample-corrected variant (Harvey-Leybourne-Newbold) since the number of cutoffs is modest.

---

## 7. Evaluation Cuts

Evaluate overall and by time period.

Since the data only contains date and sales, cut by calendar/time dimensions:

```text
overall
by horizon
by month
by day_of_week
by fiscal month
by fiscal quarter
holiday windows, if holiday features are added
high-volume days
low-volume days
recent periods
COVID/post-COVID periods if relevant
```

If multiple series are added later:

```text
by SKU
by PA
by PRA
by HFB
by store/unit
by market
by sales channel
by volume tier
by intermittency tier
```

---

## 8. Model Selection Framework

Rank models using a weighted score.

Example scoring:

```text
60% WAPE
20% absolute bias
10% interval calibration
10% operational simplicity
```

For pure model evaluation, use:

```text
70% WAPE
20% bias
10% interval quality
```

For production readiness, include:

```text
runtime
memory use
ease of retraining
ease of deployment
explainability
support for future exogenous features
```

### 8.1 Hyperparameter Tuning Protocol

A bakeoff that compares default LightGBM to default AutoETS to default NHITS is comparing defaults, not models. Pick one regime and apply it to every model:

```text
Regime A (defaults bakeoff):
    Use library defaults for every model.
    Cheapest, fastest, and a perfectly defensible first cut.
    State explicitly in the writeup that this is a defaults comparison.

Regime B (fixed-budget tuning):
    Each model gets the same wall-clock budget (e.g., 60 minutes)
    for hyperparameter search via Optuna or library-native tuners.
    Tune on a dedicated tuning window that is held out from
    both backtest cutoffs and the final holdout.
```

Recommended:

```text
Phase 1: regime A (defaults) to get a fast read.
Phase 2: regime B applied only to the top 3 to 5 models from phase 1.
```

Never tune on the same cutoffs used for evaluation. Reserve a contiguous tuning window (e.g., 6 months immediately before the rolling-cutoff range) for hyperparameter search.

### 8.2 Compute Budget

Cap runtime per model upfront. Without a cap, AutoGluon `best_quality` will eat your evening to gain 0.5 percent WAPE over a 30-second baseline.

Suggested caps per cutoff fit:

```text
StatsForecast models: 5 minutes
MLForecast tree models: 15 minutes (including tuning if regime B)
NeuralForecast: 60 minutes
AutoGluon-TimeSeries medium_quality: 30 minutes
AutoGluon-TimeSeries high_quality: 90 minutes
Chronos / TimesFM zero-shot: 5 minutes
```

If a model exceeds its cap, record DNF (did not finish) for that cutoff and move on. A model that needs 4 hours to win by 1 percent is not a production candidate.

---

## 9. Ensemble Strategy

Once backtests are complete, test ensembles using the strongest performers.

### 9.1 Candidate Models for Ensemble

Likely candidates:

```text
StatsForecast MSTL or AutoETS
MLForecast LightGBM
MLForecast CatBoost
NeuralForecast NHITS
AutoGluon / Chronos
TimesFM
```

Do not ensemble every model. Use only models that are:

```text
accurate
low-bias
diverse from each other
stable across cutoffs
```

A good ensemble needs model diversity. For example:

```text
LightGBM + AutoETS + NHITS + TimesFM
```

is likely more useful than:

```text
LightGBM + XGBoost + CatBoost
```

because the latter are too similar.

---

### 9.2 Ensemble Methods to Test

#### Simple Average

```text
ensemble = mean(model forecasts)
```

Good first baseline.

#### Median Ensemble

```text
ensemble = median(model forecasts)
```

Often robust to one model going off the rails.

#### Performance-Weighted Ensemble

Weight models by historical backtest performance.

Plain inverse-WAPE weighting is unstable — when several models cluster around the same WAPE they get near-equal weights and outliers get crushed. Use one of these stabler alternatives:

```text
Option A: inverse-squared WAPE
    weight_i = 1 / WAPE_i^2

Option B: softmax over negative WAPE with temperature tau
    weight_i = exp(-WAPE_i / tau) / sum(exp(-WAPE_j / tau))
    tau controls how peaked the weighting is

Option C: constrained convex stacking (recommended)
    Solve weights >= 0, sum(weights) = 1 minimizing
    WAPE of the weighted combination on a held-out window.
    Use scipy.optimize.minimize with SLSQP.
```

Then:

```text
ensemble_forecast = sum(weight_i * forecast_i)
```

Critical: fit the weights on a window that was NOT used to select the candidate models. Otherwise the weights overfit to the same data that picked the candidates and the ensemble will look better than it actually is.

#### Horizon-Specific Weighted Ensemble

Different models may be better at different horizons.

Example:

```text
7-day horizon: more weight to LightGBM
91-day horizon: more weight to AutoETS / TimesFM
365-day horizon: more weight to MSTL / seasonal models
```

#### Cutoff-Recency Weighted Ensemble

Use more recent backtests to assign weights.

Example:

```text
Recent cutoffs receive higher weight than older cutoffs.
```

This is useful if the sales pattern is changing over time.

#### Stacked Ensemble

Train a meta-model that combines the candidate forecasts.

Inputs:

```text
forecast_model_1
forecast_model_2
forecast_model_3
day_of_week
month
horizon_step
```

Output:

```text
actual sales
```

Recommended meta-model:

```text
Ridge regression
ElasticNet
LightGBM, only if enough backtest rows exist
```

Start with Ridge or ElasticNet to avoid overfitting.

---

## 10. Prediction Interval Strategy for Ensemble

For individual models:

```text
Use native prediction intervals where available.
Use quantile forecasts where available.
Use conformal prediction for common calibration.
```

For the ensemble, use a common conformal layer.

**Critical data-split rule.** The residuals used to calibrate intervals must come from a window that was NOT used to (a) select models, (b) fit ensemble weights, or (c) tune hyperparameters. If the same cutoffs feed all three, the resulting intervals will be optimistically narrow.

Recommended split of the available history:

```text
Training cutoffs window:    earliest history through T-12 months
                            used for rolling-origin backtests, model
                            selection, and hyperparameter tuning
Calibration window:         T-12 months through T-6 months
                            used ONLY to compute conformal residuals
                            and fit ensemble stacking weights
Final holdout window:       most recent 6 months
                            touched only once at the very end
```

Recommended approach:

```text
1. Generate ensemble point forecast on the calibration window.
2. Calculate residuals (actual - forecast) per horizon step.
3. Estimate residual quantiles by horizon step.
4. Add residual quantiles around future forecasts.
```

For stronger calibration consider Conformalized Quantile Regression (CQR) — use the model's quantile forecasts directly and conformalize the gap, rather than building intervals from point-forecast residuals.

For non-stationary periods (post-COVID retail shifts), consider Adaptive Conformal Inference (ACI), which adjusts coverage online as new data arrives.

Example:

```text
p50 = ensemble forecast
p10 = forecast + 10th percentile historical residual
p90 = forecast + 90th percentile historical residual
```

Better version:

```text
Calculate residual quantiles separately by horizon step.
```

For example:

```text
Day 1 ahead has narrower error.
Day 91 ahead has wider error.
```

This creates more realistic widening intervals.

---

## 11. Recommended Bakeoff Phases

### Phase 1: Fast Baseline Benchmark

Models:

```text
Seasonal naive
Window average
StatsForecast AutoETS
StatsForecast MSTL
StatsForecast AutoARIMA
MLForecast LightGBM
MLForecast CatBoost
```

Goal:

```text
Identify the strongest classical and ML baselines quickly.
```

### Phase 2: Add Advanced Challengers

Models:

```text
NeuralForecast NHITS
NeuralForecast NBEATS
AutoGluon-TimeSeries
Chronos / Chronos-2
TimesFM
Prophet, optional
```

Goal:

```text
Determine whether neural or foundation models beat the strong ML/statistical baseline.
```

### Phase 3: Ensemble Test

Candidate ensemble members:

```text
Top 3 to 5 models by WAPE
Models with low bias
Models with diverse forecast behavior
```

Test:

```text
simple average
median ensemble
inverse-WAPE weighted ensemble
horizon-specific weighted ensemble
stacked ensemble
```

### Phase 4: Interval Calibration

For the final selected model or ensemble:

```text
Generate p10/p50/p90
Generate p025/p50/p975
Check actual coverage
Recalibrate using conformal residuals
```

### Phase 5: Final Holdout Validation

Run the selected model and ensemble on the untouched final holdout period.

Report:

```text
WAPE
bias
MAE
RMSE
coverage
interval width
runtime
model complexity
recommendation
```

---

## 11.6 Reproducibility Requirements

Every phase output must be reproducible from a clean clone of the repo. Without this, an end-of-bakeoff result that someone wants to revisit in three months becomes a guessing game.

Minimum requirements:

```text
Environment pinning:
    requirements.txt or pyproject.toml with exact versions for
    statsforecast, mlforecast, neuralforecast, autogluon.timeseries,
    chronos-forecasting, timesfm, lightgbm, xgboost, catboost, torch.
    Pin Python version (3.10 or 3.11 recommended for compatibility).

Seed control:
    Single SEED constant set at the top of each script.
    Pass to numpy, random, torch, lightgbm, xgboost, catboost.
    For neural models, also set deterministic=True where supported.

Run tracking:
    Every backtest run writes to a parquet artifact with columns:
        run_id, model_name, hyperparams_json, refit_cadence,
        train_start, train_end, cutoff, ds, horizon_step, forecast,
        p10, p50, p90, runtime_seconds, library_version, git_sha
    Use MLflow if available, otherwise a flat parquet directory
    with a manifest.json index.

Determinism caveats:
    LightGBM and CatBoost are deterministic with fixed seed AND
    fixed thread count. Document the thread count used.
    Neural models on GPU are not bit-deterministic — record the
    median of 3 seeds for any neural model that goes into the
    final decision.
```

A reviewer should be able to run `python run_bakeoff.py --phase 1` and reproduce phase 1 results within bootstrap-CI noise.

---

## 12. Final Deliverables

### 12.1 Data Outputs

Create CSV or Parquet outputs:

```text
model_backtest_predictions.parquet
model_metrics_by_cutoff.csv
model_metrics_by_horizon.csv
model_metrics_by_month.csv
ensemble_results.csv
final_forecast.parquet
```

Suggested prediction output schema:

```text
unique_id
cutoff
ds
horizon_step
actual
model_name
forecast
p10
p50
p90
p025
p975
```

For future forecasts:

```text
unique_id
ds
model_name
forecast
p10
p50
p90
p025
p975
created_at
training_cutoff
```

---

### 12.2 Charts

Create these charts:

```text
Actual vs forecast by model
Actual vs final ensemble forecast
Forecast error by horizon
WAPE by model
Bias by model
Prediction interval coverage by model
Interval width by model
Forecast distribution for final horizon
Residuals over time
```

Most useful executive chart:

```text
Actual sales vs forecast with 80% and 95% prediction bands
```

Most useful model-selection chart:

```text
WAPE vs bias scatterplot by model
```

A good model is:

```text
low WAPE
near-zero bias
stable intervals
reasonable runtime
```

---

## 13. Recommended Final Decision Rule

Before applying the rule below, compute the WAPE noise floor:

```text
noise_floor = half-width of the 95% bootstrap CI on WAPE for the
              best single model, computed across cutoffs
```

The "improvement threshold" used below should be at least `noise_floor`. If the typical bootstrap CI is +/- 1.5 percent, then a 1 percent improvement is not real and should not change the decision.

Select the final model using this logic:

```text
If one model clearly wins on WAPE (gap > noise_floor) and bias is acceptable:
    choose that model

If several models are within noise_floor of each other:
    choose the simplest stable model, or an equal-weight ensemble of them

If the ensemble improves WAPE by more than max(noise_floor, 2%) AND reduces bias:
    choose the ensemble

If the ensemble only marginally improves WAPE but adds complexity:
    keep the single best model

If foundation models perform well but intervals are poorly calibrated:
    use them as ensemble members but calibrate with conformal residuals

If the Diebold-Mariano test cannot reject equality between top candidates:
    treat them as tied and break the tie on operational simplicity
```

---

## 14. Expected Outcome

Expected strongest single models for dense daily retail sales using only `date` and `sales`:

```text
MLForecast + LightGBM
StatsForecast MSTL
StatsForecast AutoETS
AutoGluon ensemble
NeuralForecast NHITS
```

Expected strongest ensemble:

```text
Weighted ensemble of:
- MLForecast LightGBM
- StatsForecast MSTL or AutoETS
- NeuralForecast NHITS
- TimesFM or Chronos challenger
```

Likely production winner:

```text
MLForecast + LightGBM with conformal intervals
```

or:

```text
Horizon-specific weighted ensemble with conformal intervals
```

---

## 15. Suggested First Version

For a first runnable version, keep it tight:

```text
Models:
1. Seasonal naive lag 7
2. Seasonal naive lag 364
3. StatsForecast AutoETS
4. StatsForecast MSTL
5. StatsForecast AutoARIMA
6. MLForecast LightGBM
7. MLForecast CatBoost
8. NeuralForecast NHITS
9. AutoGluon-TimeSeries
10. TimesFM or Chronos
```

Primary horizon:

```text
91 days
```

Secondary horizon:

```text
28 days
```

Primary metric:

```text
WAPE
```

Secondary metrics:

```text
bias
MAE
RMSE
coverage
interval width
```

Final candidate comparison:

```text
Best single model vs best 3-model ensemble vs best 5-model ensemble
```

That gives a strong, defensible benchmark without over-engineering the first pass.
