# Forecasting 101 — Presentation Framework

Audience: analysts and planners across the org who already produce something they call a
forecast, but have not been trained in forecasting. Assume spreadsheet fluency, no stats
background, no ML background.

Goal of the session: change four behaviors.

1. Always ship a **baseline** beside your forecast.
2. Always ship an **interval**, not just one number.
3. Always report **bias and error separately**, at the grain the decision is made.
4. Always score on a **held-out backtest**, never on the data you built the forecast from.

Everything below is structured so you can cut it down without losing the spine.

---

## How to use this document

Each slide entry has four parts:

- **Message** — the one sentence the audience should leave with. If a slide has two, split it.
- **Show** — what is on the slide. Keep it to one chart or one table.
- **Say** — the talking points, in order.
- **Watch for** — the objection or misunderstanding that shows up here.

Numbered slides are the full 50-minute version. Slides marked `[CORE]` are the 25-minute cut.

---

## The spine

Seven beats. If you can only remember the spine, you can rebuild the deck.

1. A forecast is a **falsifiable statement about the future** — and most things called forecasts in the org are targets, not forecasts.
2. One number is not a forecast. Demand is a **distribution**; a point forecast throws away the part the business actually needs.
3. Error metrics exist to make "is this good?" answerable. **Bias and accuracy are different questions.**
4. "Good" only means anything **relative to a baseline**. The baseline is not optional.
5. The baseline is **embarrassingly strong**: last year, same day. Beating it is the bar, not a formality.
6. You only know any of this from a **backtest** — and backtests are easy to fake yourself out with.
7. Here is the **minimum standard** for a forecast readout in this org, on one page.

---

## Timing cuts

| Version | Slides | Use |
|---|---|---|
| 25 min `[CORE]` | 1, 3, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20, 23, 27 | Staff meeting / lunch-and-learn |
| 50 min (full) | All 29 | Working session with the teams that own forecasts |
| Follow-up workshop | Section 7 + demo | Hands-on, bring their own forecast file |

---

# Section 1 — What a forecast actually is

### Slide 1 `[CORE]` — Three things people call a forecast

**Message:** A forecast, a target, and a plan are three different objects. Mixing them is the
single most common failure in the room.

**Show:** Three columns.

| | Forecast | Target | Plan |
|---|---|---|---|
| Question it answers | What do we *expect* to happen? | What do we *want* to happen? | What will we *do*? |
| Owner | Analyst | Leadership | Operator |
| Can it be wrong? | Yes, measurably | No, it's a choice | No, it's a commitment |
| Should it be unbiased? | **Yes** | Not applicable | Not applicable |

**Say:**
- Only the first column is scored for accuracy. The other two are decisions.
- If the same number does two of these jobs, it will be pulled toward the incentive and away
  from the truth — and it will be biased in a *consistent direction*.
- Ask the room: "when you produce your number, which column is it in?" Most will say all three.

**Watch for:** "But leadership wants one number." Answer: they can have one number — a chosen
quantile of the forecast, chosen deliberately (see Slide 6). That's different from bending the
expectation to match the ask.

---

### Slide 2 — The three properties every forecast must state

**Message:** A number without horizon, grain, and origin cannot be scored, so it cannot be improved.

**Show:**

- **Horizon** — how far ahead. 7 days out and 180 days out are different problems with different
  achievable accuracy. A single "accuracy" number across mixed horizons is meaningless.
- **Grain** — the time bucket *and* the entity level. Daily by item is a different problem than
  monthly by channel. Error at one grain tells you almost nothing about error at another.
- **Origin** — the as-of date. What did you know when you made it? Everything after the origin
  must be hidden from you.

**Say:**
- Origin is the one people skip, and it's the one that invalidates the most work.
- If you built a "forecast" for March using data through April, you built a fit, not a forecast.

**Watch for:** "We refresh it every week, so the origin moves." Fine — then you have a *series*
of forecasts, each with its own origin, and you score each at a fixed horizon from its own origin.

---

### Slide 3 `[CORE]` — Why forecasting is hard in one picture

**Message:** A time series is signal (level, trend, seasonality, known events) plus noise you
cannot predict — and the goal is to capture all the signal and none of the noise.

**Show:** One real series, decomposed: level / weekly seasonality / annual seasonality / residual.

**Say:**
- Level: where the series sits. Trend: which way it's drifting.
- Seasonality is usually *two* cycles at once: day-of-week and time-of-year. Both matter.
- Known events (holidays, promos, openings) are signal you can know in advance.
- Everything left is noise. A model that "fits" the noise looks brilliant on history and fails
  in the future. That is the definition of overfitting, and it's the trap the rest of this talk
  is about avoiding.

**Watch for:** "Our business is unpredictable." Response: partly true, and that's the point of
Section 2 — quantifying how unpredictable is the job, not an excuse to skip it.

---

# Section 2 — Why point forecasts alone are bad

### Slide 4 `[CORE]` — The number you deliver is a summary of something you didn't show

**Message:** Reality is a distribution of possible outcomes. A point forecast is one summary
statistic of that distribution — and you almost certainly never chose which one.

**Show:** A distribution curve for a single future day, with the point forecast as a single tick
mark on the x-axis, and the rest of the curve shaded.

**Say:**
- When you say "we'll sell 1,000," you mean something like "the middle of a range that runs from
  700 to 1,400." The range is real information and you deleted it.
- The receiver assumes 1,000 is what will happen. It won't. It never does. The probability of
  hitting any exact point is essentially zero.
- The honest object is: p50 = 1,000, 80% of outcomes fall in 800–1,250.

---

### Slide 5 — Which point is the point? (mean vs median)

**Message:** "The forecast" is ambiguous even as a single number — mean and median differ, and
your error metric silently picks one for you.

**Show:** A right-skewed demand distribution with mean and median marked apart.

**Say:**
- Minimizing squared error targets the **mean**. Minimizing absolute error targets the **median**.
- Demand is usually right-skewed (a floor at zero, occasional big spikes), so mean > median.
- So two teams can both be "right" and disagree by a meaningful margin — because they optimized
  different things without knowing it.

**Watch for:** This is the first slide where people realize their metric choice is a modeling
choice. Let that land; it sets up Section 3.

---

### Slide 6 `[CORE]` — No decision actually wants the middle

**Message:** Every real decision downstream of a forecast wants a *quantile*, not the average —
and usually not the middle one.

**Show:** Table of decisions and the quantile they actually need.

| Decision | Cost of under | Cost of over | What it needs |
|---|---|---|---|
| Inventory / safety stock | Lost sale, lost customer | Carrying cost, markdown | High quantile (p90+) |
| Staffing a shift | Service collapse | Idle labor hours | High-ish quantile |
| Cash / commitment | Breach, emergency spend | Idle capital | Depends, but stated |
| Capacity investment | Can't serve demand | Stranded asset | Full distribution |

**Say:**
- If running out costs 4x what holding extra costs, you should deliberately plan **above** your
  expected value. That's not optimism, that's arithmetic.
- You cannot do that arithmetic with a point forecast. The information required — how wide the
  uncertainty is — was thrown away before the decision-maker got it.
- This is the strongest argument in the deck. Point forecasts don't just lose precision, they
  make the correct decision *unavailable*.

---

### Slide 7 — You can't tell a bad forecast from an unlucky one

**Message:** Without an interval, "we missed by 12%" has no interpretation.

**Say:**
- Point forecast said 1,000, actual was 1,200. Is that a broken model or a normal Tuesday?
- With an interval, the question is answerable: if your 80% band was 950–1,050 and you missed by
  200, the model is broken. If it was 800–1,250, that was a normal outcome and nothing needs
  fixing.
- Without the band, every miss triggers the same reaction — a meeting, a manual override, a
  "the model doesn't understand our business." Most of those meetings are about noise.

**Watch for:** This slide converts skeptics who don't care about statistics but hate wasted
meetings.

---

### Slide 8 `[CORE]` — Manual overrides without a band are unmeasurable

**Message:** If you adjust a forecast, you need to know whether your adjustment was bigger than
the noise. Otherwise you are adding variance and calling it judgment.

**Say:**
- Overrides are often right — a planner knows about a promo the model doesn't.
- But an override of +3% on a series whose 80% band is ±20% is indistinguishable from doing
  nothing, and it costs you the audit trail.
- Minimum standard: log the pre-override and post-override forecast, and score both. Over a year
  you will learn exactly which overrides earn their keep.

---

### Slide 9 — Intervals are cheap: you already have what you need

**Message:** You do not need a fancy model to produce an interval. You need your own past errors.

**Show:** Three steps.

1. Backtest your existing forecast over the last N origins (Section 6). You now have a pile of
   errors: `error = forecast − actual`, one per row.
2. Take the 10th and 90th percentile of those errors, per horizon step.
3. Your 80% band is `forecast + p10(error)` to `forecast + p90(error)`.

**Say:**
- This is empirical residual quantiles. It's ten lines of code and it works on *any* forecast,
  including a spreadsheet one.
- Then measure **coverage**: over your backtest, what fraction of actuals landed inside the 80%
  band? It should be about 80%. If it's 60%, your bands are too narrow and you're overconfident.
- In our own work, weekly 80% bands measured 0.68–0.77 coverage — honestly undercovering, and
  we report it that way rather than pretending otherwise. Measuring it is the win; a wrong band
  you know about beats a point forecast you don't.

**Watch for:** "Our errors aren't normally distributed." Correct — that's *why* we use empirical
quantiles of actual observed errors rather than a formula with a standard deviation in it.

---

### Slide 10 `[CORE]` — Two rules about intervals people get wrong

**Message:** Bands don't add, and errors are bigger where it matters most.

**Say:**
- **Do not sum daily bands into a monthly band.** Summing lower bounds gives you an
  absurdly pessimistic month that essentially cannot happen (every single day would have to hit
  its floor simultaneously). If you need a monthly band, build the band at monthly grain.
- **Uncertainty is not uniform across the year.** In our series, November–December residuals run
  2–3x the rest of the year. A single band applied to all weeks is too wide in June and too
  narrow in December — exactly backwards from what planning needs.
- So: report bands by horizon step, and know your high-variance windows.

---

# Section 3 — Error metrics, from scratch

### Slide 11 — The one definition everything else is built on

**Message:** `error = forecast − actual`. State your sign convention out loud, every time.

**Say:**
- Positive error = you forecast too high (over). Negative = too low (under).
- Half the arguments about "is our bias positive or negative" are people using opposite
  conventions. Pick one, write it on the report, never flip it.
- Live example worth naming: our reporting calls this bias **negative**, because it computes
  `actual − forecast`. The underlying fact is that forecasts sit **above** actuals. Anyone who
  hears "negative bias" without the convention goes hunting for the opposite problem.
- The rest of this section is just different ways of adding up these errors.

---

### Slide 12 `[CORE]` — Bias: are you consistently wrong in one direction?

**Message:** Bias measures direction, not size. Persistent one-sided bias is a **process defect**,
not bad luck, and it is the cheapest thing to fix.

**Show:**

```
bias%  =  sum(forecast − actual) / sum(actual)
```

**Say:**
- Errors should cancel out over a full seasonal cycle. Bias near zero means over- and
  under-forecasts balance.
- A well-centered forecast lands above the actual about **half** the time. Ours landed above
  **92%** of the time over the last year. That is not a run of bad luck — coin flips don't do
  that. Something in the process pushes every number up.
- Bias is *not* accuracy. You can be 20% off every day and have zero bias (half high, half low).
  You can be 3% off every day and have −3% bias (always low). These need different fixes.
- **Report both. Always. Every forecast readout in this org should show bias and error side by side.**

---

### Slide 13 `[CORE]` — Common causes of persistent one-sided bias

**Message:** If your bias is always the same sign, the cause is almost always on this list.

**Show:** Diagnosis table. Read it against the *fact*, not the sign: our forecasts land **above**
actuals. Start at the top block.

| Symptom | Likely cause | Fix |
|---|---|---|
| **Over** | The forecast *is* the plan or budget, and the plan assumes growth | Separate forecast from target; score them separately |
| **Over** | A growth factor applied to a level that is flat or declining | Drop the uplift; test replay-last-year-unscaled as the baseline |
| **Over** | Planned initiatives added at full projected value, none discounted for the ones that won't land | Apply a realization rate learned from past initiatives |
| **Over** | Anchored on an inflated base year (a promo year, a one-off spike) | Re-base on a normal year, or model the spike as an event |
| **Over** | Bottom-up sum of individually optimistic regional or category plans | Reconcile to a top-down total; check bias at the total |
| **Over** | Extrapolating a trend that has already flattened or turned | Damp the trend, or drop it at long horizons |
| Under | Optimizing MAPE (see Slide 16) — it structurally rewards forecasting low | Report WAPE alongside it |
| Under | "Outlier" cleaning removes spikes but not dips | Symmetric rules only, or don't clean at all |
| Under | Sales used as demand during stockouts (censored demand) | Flag stockout periods; don't train on them |
| Drifting | A level shift the process hasn't absorbed | Monitor bias over time; alarm on sustained drift |

**Say:**
- Nearly every over-forecasting cause is a process problem wearing a math costume. None of them
  are fixed by a better algorithm.
- The first two rows are the ones to press on. If the forecast doubles as the plan, it inherits
  the plan's growth assumption — and if the underlying level is flat or drifting down, that
  assumption is wrong in the same direction every single period.
- Fixing bias is nearly free and usually worth more than a better model, because it's systematic —
  it compounds across every row.

**Watch for:** This is the slide the audience will recognize themselves in, and the room may get
defensive. Keep it structural: nobody chose to over-forecast, the process chose for them. Confirm
the 92% figure yourself before presenting it.

---

### Slide 14 `[CORE]` — The 92% problem: subtract the offset before you buy a model

**Message:** When 92% of your forecasts land above actuals, most of your error isn't noise — it's
a fixed offset. Subtracting it is arithmetic, and it is free.

**Show:**

```text
Share of periods where forecast > actual, over one year

    a well-centered forecast     ~50%
    ours                          92%

The fix, in one line:

    debiased = forecast − (median error over a trailing window)
```

**Say:**
- Split your error into two parts: a **systematic** part that's there every period, and a
  **random** part that isn't. Only the random part is genuinely hard.
- 92% one-sided tells you the split. Under a normal-error assumption it implies the offset is about
  **1.4x** the size of the random noise — the systematic part dominates. Removing it alone would
  cut average absolute error by roughly **40–45%**.
- Treat that as an order-of-magnitude estimate, not a promise: it assumes the offset is stable and
  the errors are roughly symmetric around it. **Measure the real figure on your own backtest** —
  it takes an afternoon, and it's the highest-return afternoon available to this team.
- This is also why a one-line seasonal naive is beating us by 20–30%. It isn't smarter. It simply
  has no growth assumption baked in, so it doesn't carry the offset.
- Do it honestly: estimate the offset on a trailing window of *past* origins only, never on the
  window you're scoring, and re-check quarterly. A bias correction is itself a forecast, and it
  goes stale.
- Correct only what's stable. If the offset drifts period to period, subtracting a constant will
  overshoot — go fix the cause on Slide 13 instead.

**Watch for:** "That's just fudging the number." The opposite: a persistent one-sided error is free
information you are currently throwing away every period. Ignoring it is the fudge. The real
objection to surface is whether the offset exists because the forecast is doing a second job as the
plan — if so, this slide is a patch and Slide 13 is the cure.

---

### Slide 15 — Accuracy: MAE and RMSE

**Message:** MAE is the typical miss in units. RMSE punishes big misses much harder. Neither
compares across series.

**Show:**

```
MAE   =  mean( |forecast − actual| )          → "average miss, in units"
RMSE  =  sqrt( mean( (forecast − actual)^2 ) ) → "average miss, big misses weighted heavier"
```

**Say:**
- Use MAE when a 10-unit miss is exactly twice as bad as a 5-unit miss.
- Use RMSE when one enormous miss is disproportionately damaging (capacity, service failure).
- Both are in units, so they're readable — but they can't be compared across items of different
  size. An MAE of 500 is superb for one series and catastrophic for another.

---

### Slide 16 `[CORE]` — WAPE, and why not MAPE

**Message:** WAPE is the default percentage-error metric. MAPE is the trap.

**Show:**

```
WAPE  =  sum( |forecast − actual| )  /  sum( actual )        ← use this
MAPE  =  mean( |forecast − actual| / actual )                ← avoid
```

**Say:**
- WAPE is total absolute error as a share of total volume. It's volume-weighted, so the big days
  count more — which is what the business cares about.
- MAPE fails in three ways:
  1. **Undefined at zero** and explodes for small actuals. One near-zero day can dominate.
  2. **Asymmetric.** Forecasting 50 when actual is 100 gives 50% error. Forecasting 150 when
     actual is 100 gives 50% error too — but forecasting 200 gives 100%, while the worst you can
     score by under-forecasting is 100%. Over-forecasting is punished without limit;
     under-forecasting is capped. **Optimizing MAPE trains you to forecast low.**
  3. **Unweighted.** A 100% error on a 3-unit day counts the same as a 5% error on a 10,000-unit day.
- Note the direction carefully here, because it's diagnostic. MAPE's asymmetry pushes forecasts
  *down*. Ours run *high* anyway. So whatever is lifting our numbers is strong enough to overpower
  the metric pushing back — which points squarely at the organizational causes on Slide 13, not at
  a technique problem.

**Watch for:** MAPE is entrenched. Don't fight it head-on; just add WAPE next to it in every
report and let people watch the two disagree.

---

### Slide 17 — Metric hygiene

**Message:** Most metric disputes are about *which rows were scored*, not which formula.

**Say:**
- Never average percentages across items. Aggregate the numerator and denominator, then divide.
- Score at the grain where the decision is made. A forecast can be excellent monthly and useless
  daily (or vice versa) — errors cancel when you aggregate up.
- Compare models only on **identical rows**: same dates, same items, same horizons.
- Exclude nothing after seeing results. If you drop a bad window, disclose it and say why.
- If you filled missing dates, say how many and treat the metric as approximate.

---

### Slide 18 `[CORE]` — For probabilistic forecasts: coverage and width

**Message:** Once you ship intervals, you need two more numbers to know whether they're honest.

**Say:**
- **Coverage**: what % of actuals fell inside the band. An 80% band should catch ~80%.
  Below → overconfident. Well above → uselessly wide.
- **Average width**: how wide the band is. Coverage alone is gameable — a band from zero to
  infinity has perfect coverage and zero value. Always report width beside it.
- Optional, for the analytics-minded: **pinball loss** scores the whole distribution in one
  number. Mention it, don't teach it. Coverage + width covers 90% of the need.

---

# Section 4 — The baseline rule

### Slide 19 — "Is 8% WAPE good?"

**Message:** The question is unanswerable in isolation. Accuracy is only meaningful relative to a
baseline on the same rows.

**Say:**
- 8% WAPE forecasting a stable aggregate might be terrible. 20% WAPE on a volatile sparse item
  might be excellent.
- The only sound question: **did it beat a dumb rule?**

---

### Slide 20 `[CORE]` — The baseline: seasonal naive

**Message:** "Same period last year" is not a strawman — it is a genuinely strong forecast, and
it is the bar.

**Show:**

```
seasonal naive, lag 364:  forecast(today) = actual(364 days ago)   → keeps the SAME WEEKDAY
seasonal naive, lag 365:  forecast(today) = actual(365 days ago)   → keeps the SAME CALENDAR DATE
```

**Say:**
- 364 = exactly 52 weeks, so Saturday maps to Saturday. Use it when day-of-week drives the series.
- 365 maps calendar date to calendar date. Use it when fixed-date events (Christmas, July 4)
  dominate.
- You have to choose, and the choice is worth real accuracy. In our work, the daily champion uses
  lag 364; the weekly champion forecasts *daily* at lag 365 and sums into weeks — because weekly
  buckets misallocate the pre-Christmas surge when the holiday weekday shifts year to year. That
  one change roughly halved the error on the December fold (10.5% → 5.0% WAPE).
- Why it beats us specifically: replaying last year unscaled carries *no growth assumption*. When
  the underlying level is flat or drifting down, that makes it close to bias-centered by
  construction — while any process that starts from a plan number inherits the plan's optimism
  every period.
- It costs one line of SQL. There is no excuse for not having it beside every forecast.

---

### Slide 21 — Skill: the only number that says "worth it"

**Message:** Report improvement over baseline, as a percentage, on identical rows.

**Show:**

```
skill  =  1  −  ( WAPE_yours / WAPE_baseline )
```

**Say:**
- +20% skill: your forecast reduces error by a fifth versus doing nothing clever. Real value.
- 0%: you spent a quarter to match a one-line rule.
- Negative: **the baseline is better and should replace your forecast today.**
- Make this a required field on every forecast readout. It ends the "our model is sophisticated"
  conversation immediately and replaces it with a number.

---

### Slide 22 — What we learned running this bakeoff properly

**Message:** We tested six-plus model families against seasonal naive on honest backtests. The
naive baseline won essentially everywhere.

**Show:** The list of what lost.

- Gradient boosting trees (HistGradientBoosting, Extra Trees, Random Forest, LightGBM)
- Ridge regression on lag features
- Classical statistical models (ARIMA, ETS, SARIMAX, MSTL)
- Foundation models (TimesFM, Chronos-Bolt, Chronos-2)
- Learned ensembles / stacks (convex weights, inverse-error weights)
- Hand-designed seasonal variants (level-scaled, multi-year averaged, holiday-aligned)

**Say:**
- This is not "models are useless." It's that all of these re-model the **same sales history**.
  Once seasonality and level are captured, there's nothing left to squeeze without **new
  information** — covariates (promo calendar, planned openings, price) or finer structure
  (per-item, per-market).
- At *sub-aggregate* levels the picture changes: per-market and per-channel models did beat naive.
  Complexity pays where the series is noisier and the baseline is weaker.
- Practical takeaway for the room: **start with the baseline, then justify every step away from it.**

**Watch for:** "So we shouldn't use models?" No — you should *earn* them. The bakeoff is how you
find out, and it takes hours, not months.

---

# Section 5 — Backtesting: how you actually know

### Slide 23 `[CORE]` — The only honest test

**Message:** Pick a cutoff. Hide everything after it. Forecast. Compare. Roll forward. Repeat.

**Show:** Timeline diagram — training window, cutoff line, forecast horizon, held-out actuals.
Then the same picture stepped forward three times.

**Say:**
- One cutoff is an anecdote. You need several, at the horizon you actually use.
- Score at the same horizon each time, so you're measuring one thing.
- This is the whole method. Everything after this slide is about ways it goes wrong.

---

### Slide 24 — Leakage: the mistake that produced our own fake winner

**Message:** If any information from after the cutoff touched the forecast, the score is fiction
— and it will be *flatteringly* wrong.

**Say:**
- Real example from this project: we built an ensemble whose blend weights were fitted on a
  window that **overlapped the evaluation window** — by 31 days at the 91-day horizon and by
  ~11 months at the 365-day horizon.
- Result: the ensemble looked like a clear winner and was on track to ship.
- When we refit the weights properly inside each fold — using only data ending at or before that
  fold's cutoff — the ensemble **lost every single fold at every horizon** to plain seasonal naive.
- The root cause was mundane: the weights were chasing which model *used to* be better, on a
  window it had already seen.
- Common leaks to name for the room: fitting anything (weights, parameters, scalers, feature
  selection, "which model to use") on data after the cutoff; using a data field that isn't
  actually knowable at forecast time; cleaning outliers using the full history.

**Watch for:** This is the credibility slide. Telling a story where *you* were wrong buys you the
authority to critique their work in Section 7.

---

### Slide 25 — Overlapping folds: "5 tests" that are really 1

**Message:** If your test windows overlap, you have fewer independent reads than you think.

**Say:**
- 5 folds of a 52-week forecast stepped 4 weeks apart share ~48 of 52 weeks with their neighbor.
  That's roughly **1 effective independent window** — one opinion, quoted five times.
- We compute and print "effective independent windows" on every run for exactly this reason. At
  26/52-week horizons it comes out around 1.1–1.6.
- Practical rule: at long horizons you simply do not have much evidence, and you should say so
  out loud rather than reporting "won 4 of 5 folds" as if that were strong.

---

### Slide 26 — Don't fool yourself: pre-registration and tiny deltas

**Message:** If you try forty things, one wins by chance. Decide what counts as a win *before*
you look.

**Say:**
- Write down: the candidate, the baseline, the horizons, the folds, and the threshold that would
  make you switch — **before** scoring. We keep a pre-registration log in the project plan, with
  the verdict appended after. Most entries read "rejected," and that is a healthy log.
- A challenger that beats the champion by <2% relative on overlapping folds is a **tie**, not a
  win. Eyeballing "4 of 5 folds" is not evidence.
- If you want to promote on a small margin, test the per-window differences properly (paired
  test with overlap-aware error bars). Otherwise, keep the incumbent.
- Corollary: **never compare a model scored on one fold design against a model scored on another.**
  Rebuild both on the same folds or don't compare them.

---

# Section 6 — The minimum standard

### Slide 27 `[CORE]` — The one-page forecast readout

**Message:** From now on, a forecast isn't done until it has this page attached.

**Show:** The template — hand this out.

| Field | Value |
|---|---|
| Horizon | e.g. 28 days |
| Grain | e.g. daily, by channel |
| Forecast origin (as-of date) | |
| Model / method | |
| **Baseline compared against** | e.g. seasonal naive lag 364 |
| WAPE (yours) | |
| WAPE (baseline) | |
| **Skill vs baseline** | `1 − WAPE_yours / WAPE_baseline` |
| Bias % (state the sign convention) | |
| 80% interval coverage / avg width | |
| Backtest design: # folds, cutoff dates, overlap | |
| Known caveats | filled dates, excluded rows, planned events not modeled |

**Say:**
- Every field is cheap. None of it requires a new tool.
- If a field is blank, that's the conversation — not a failure, just the next thing to build.

---

### Slide 28 — Maturity ladder: where is your forecast today?

**Message:** Nobody is expected to be at level 3 tomorrow. But everyone should know their level
and their next step.

| Level | What you have | Next step |
|---|---|---|
| 0 | A number, built by judgment or spreadsheet trend. No score. | Backtest it once. Get a WAPE. |
| 1 | Backtested point forecast with WAPE and bias. | Add a seasonal naive baseline and report skill. |
| 2 | Beats baseline, reports bias, ships p10/p50/p90. | Measure coverage. Fix if it's not ~80%. |
| 3 | Calibrated intervals + monitoring + a tracked challenger. | Add new information (covariates), not new models. |

**Say:**
- Level 0 → 1 is the biggest single jump in value in this entire deck, and it takes an afternoon.
- Note the level 3 → next step: once you're past the baseline, the lever is **new data**, not
  fancier algorithms. That's the durable lesson from our bakeoff.

---

### Slide 29 — What to do Monday

**Message:** Three concrete actions, in order.

**Say:**
1. **Compute your baseline.** Pull last year's same-weekday actuals for the period you forecast.
   Score both your forecast and the baseline on the same rows. Report skill. (One afternoon.)
2. **Count your one-sided rate.** Over the last 12 months, in what share of periods was your
   forecast above the actual? If it isn't near 50%, subtract the trailing median error and
   re-score — then walk the Slide 13 table to find out *why* the offset is there. (One hour for
   the count, an afternoon for the re-score.)
3. **Build a band** from your own backtest errors — p10/p90 of past errors by horizon step — and
   start reporting three numbers instead of one. Measure coverage next quarter. (One day.)

Offer: bring your forecast file and we'll do step 1 together in the follow-up session.

---

# Section 7 — Follow-up workshop material (not slides)

## Live demo script (10 minutes, do this live if you can)

Nothing lands like watching a one-line rule beat a forecast someone defended in a meeting.

1. Take a real forecast the audience produced, with its origin date and horizon.
2. Pull actuals for that window, and pull actuals from 364 days before each date.
3. Compute on the **same rows**: WAPE and bias for their forecast, WAPE and bias for the naive.
4. Show skill: `1 − WAPE_theirs / WAPE_naive`.
5. Show the **third** line: their forecast minus its own trailing median error. This is usually the
   moment the room goes quiet — a forecast nobody re-modelled, just re-centred, that beats the one
   they defended last quarter.
6. Then compute p10/p90 of their errors and show what their 80% band would have been.

Handle the result carefully either way:
- **If naive wins:** frame it as "this is the bar, and now you have a free 20-30% improvement
  available today" — not as an indictment. Some of the audience built that forecast.
- **If their forecast wins:** excellent — now they have a *defensible* claim they never had before,
  and the readout template lets them keep proving it.

## Objection handling

| Objection | Response |
|---|---|
| "Our business is different / too volatile to forecast." | Then quantify the volatility and ship a wide interval. That *is* the forecast. Volatility is a finding, not a blocker. |
| "Naive can't know about our strategy/promos." | Correct — and it still beats us, which means our strategy knowledge isn't making it into the number. That's a fixable gap, and it's the highest-value one. |
| "Leadership only wants one number." | They get one number: a deliberately chosen quantile. The band goes in the appendix and drives the safety-stock decision. |
| "We don't have time to backtest." | The baseline + backtest is an afternoon. It's cheaper than one meeting arguing about whether the forecast is good. |
| "Our accuracy is 92%." | 92% by which metric, on which rows, at which horizon, versus which baseline? Usually this is 100 − MAPE at a grain where errors cancel. |
| "We already adjust for that manually." | Great — log the pre- and post-override numbers and let's score the overrides. If they add skill, we'll build them in. |

## Glossary — define once, use consistently

- **Origin / cutoff** — the as-of date. Nothing after it may be used.
- **Horizon** — how far past the origin you forecast.
- **Grain** — time bucket + entity level.
- **Baseline** — the dumb rule you must beat. Default: seasonal naive.
- **Skill** — % error reduction vs baseline.
- **Bias** — signed average error; direction of systematic wrongness.
- **WAPE** — total absolute error ÷ total actual.
- **Coverage** — % of actuals landing inside your stated interval.
- **Backtest / fold** — a single hidden-future test at one origin.
- **Leakage** — any post-cutoff information influencing the forecast.
- **Overfitting** — fitting noise; looks great on history, fails forward.

## Facilitator prep checklist

- [ ] Pick one real series the audience knows and use it for **every** chart. Don't switch series.
- [ ] Verify the 92% one-sided figure and the room's WAPE yourself before the session, and put both
      on Slides 12–14. Confirm which sign convention their reporting uses, and say it out loud.
- [ ] Run the demo end-to-end beforehand — including the case where their forecast wins.
- [ ] Print the Slide 27 readout template as a physical handout.
- [ ] Have the seasonal-naive SQL/Python snippet ready to paste into chat.
- [ ] Do not name individuals or teams next to bad numbers. Frame every symptom as a process defect.
- [ ] Close with the offer of a hands-on follow-up. The behavior change happens there, not here.

---

## Receipts (for the presenter, not the deck)

Claims in this framework are grounded in this project's artifacts. If challenged, cite:

| Claim | Source |
|---|---|
| Six-plus model families lost to seasonal naive | `model_cards.md`; `model_usage_plan.md` "Current Validated Winners" |
| Ensemble weight leakage flipped the verdict at every horizon | `model_cards.md` (91d/180d/365d multi-fold verdicts); `model_usage_plan.md` fold-count caveat |
| Weekly December fold improved 10.5% → 5.0% WAPE via daily lag-365 reaggregation | `model_cards.md`, `weekly_daily_reagg_seasonal_naive_lag_365` card |
| Weekly 80% interval coverage 0.68–0.77 | `model_usage_plan.md` interval coverage backtest section |
| Nov/Dec residuals 2–3x rest of year | `plan.md` "Current state" |
| Effective independent windows ~1.1–1.6 at 26/52 weeks | `model_usage_plan.md` weekly validation section |
| Don't sum daily bands into a cumulative band | Vault note, July 2026 productionization |
| Sub-aggregate models *do* beat naive (per-market, per-channel) | `model_usage_plan.md` market-level section; `units_forecast/BAKEOFF_SUMMARY.md` |
| Pre-registration log with mostly-rejected verdicts | `plan.md` pre-registration log |

Metric definitions match `metrics.py` (`bias_pct = sum(forecast − actual) / sum(actual)`;
`wape = sum|forecast − actual| / sum|actual|`), so anything computed with this repo's tooling
lines up with the deck.
