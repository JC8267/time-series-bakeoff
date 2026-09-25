"""Frozen candidate screen; September actuals are used only after selection."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from metrics import score_group
from run_bakeoff import labor_day, memorial_day, thanksgiving_date, gregorian_easter
from run_sls_units_labor_day import CUTOFF, BLEND, holiday_factors, multipliers
from run_prophet_timesfm_benchmark import load_timesfm3, forecast_timesfm3


OUT = Path('outputs_sls_units_challengers_20260831')
SOURCE = Path('outputs_sls_units_labor_day_20260831_timesfm3')
RECENT = [CUTOFF - pd.Timedelta(days=28*k) for k in range(32, 0, -1)]
HOLIDAY = [pd.Timestamp(f'{y}-08-31') for y in (2023, 2024, 2025)]
MODELS = ['original_blend', 'seasonal_naive', 'timesfm3', 'event_replay',
          'event_replay_scaled', 'timesfm3_event_replay', 'event_blend_50_50',
          'recent_weekday_event', 'ridge_calendar', 'lightgbm_calendar', 'structural_median',
          'event_replay_4week', 'event_replay_4week_scaled', 'event_4week_blend']


def event_replay(history, dates, window=(-3, 3)):
    """Align the event itself; replace displaced source holiday days with normal weekdays."""
    result = []
    for date in dates:
        offset = (date - labor_day(date.year)).days
        source = date - pd.Timedelta(days=364)
        if window[0] <= offset <= window[1]:
            source = labor_day(date.year-1) + pd.Timedelta(days=offset)
        elif window[0] <= (source - labor_day(source.year)).days <= window[1]:
            distances = (-42, -35, 35, 42) if window == (-7, 20) else (-21, -14, 14, 21)
            neighbors = [source + pd.Timedelta(days=k) for k in distances]
            result.append(float(history.loc[neighbors].median()))
            continue
        result.append(float(history.loc[source]))
    return np.asarray(result)


def event_study(history):
    rows = []
    for year in range(history.index.min().year, history.index.max().year+1):
        for offset in range(-7, 21):
            day = labor_day(year)+pd.Timedelta(days=offset)
            neighbors = [day+pd.Timedelta(days=k) for k in (-42,-35,35,42)]
            if day in history.index and all(d in history.index for d in neighbors):
                baseline = float(history.loc[neighbors].median())
                rows.append({'year':year,'offset':offset,'ds':day,'actual':history.loc[day],
                             'weekday_baseline':baseline,'relative_to_baseline':history.loc[day]/baseline-1})
    return pd.DataFrame(rows)


def calendar(dates):
    dates = pd.DatetimeIndex(dates)
    t = (dates - pd.Timestamp('2022-01-01')).days.to_numpy()/365.25
    features = {'trend': t}
    for day in range(7):
        features[f'dow_{day}'] = (dates.dayofweek == day).astype(int)
    for k in range(1, 7):
        features[f'sin_{k}'] = np.sin(2*np.pi*k*t)
        features[f'cos_{k}'] = np.cos(2*np.pi*k*t)
    for name, fn in [('labor', labor_day), ('memorial', memorial_day),
                     ('thanksgiving', thanksgiving_date), ('easter', gregorian_easter),
                     ('christmas', lambda y: pd.Timestamp(y,12,25)),
                     ('newyear', lambda y: pd.Timestamp(y,1,1)),
                     ('july4', lambda y: pd.Timestamp(y,7,4))]:
        offset = np.array([(d-fn(d.year)).days for d in dates])
        for k in range(-3, 4):
            features[f'{name}_{k}'] = (offset == k).astype(int)
    return pd.DataFrame(features, index=dates)


def statistical_models(history, dates):
    scale = float(history.abs().median())
    train = history.iloc[-730:]
    ridge = Ridge(alpha=1.0)
    weights = np.exp2(-(train.index.max()-train.index).days.to_numpy()/365)
    ridge.fit(calendar(train.index), np.arcsinh(train/scale), sample_weight=weights)
    ridge_pred = np.sinh(ridge.predict(calendar(dates)))*scale
    all_dates = history.index.append(dates)
    features = calendar(all_dates)
    y = history.reindex(all_dates)/scale
    for lag in (28,35,56,364,371):
        features[f'lag_{lag}'] = y.shift(lag)
    features['mean_28_at_lag28'] = y.shift(28).rolling(28).mean()
    features['mean_56_at_lag28'] = y.shift(28).rolling(56).mean()
    # All predictors are known at the origin, including on horizon day 28.
    train_features = features.loc[history.index].dropna().iloc[-730:]
    tree = LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=15,
                         learning_rate=.04, verbosity=-1, n_jobs=1, random_state=42,
                         deterministic=True, force_col_wise=True)
    tree.fit(train_features, np.arcsinh(history.loc[train_features.index]/scale))
    tree_pred = np.sinh(tree.predict(features.loc[dates]))*scale
    return ridge_pred, tree_pred


def candidates(history, dates, tfm):
    factors = holiday_factors(history)
    clean = history/multipliers(history.index, factors)
    adjusted = clean.loc[dates-pd.Timedelta(days=364)].to_numpy()*multipliers(dates, factors)
    replay = event_replay(history, dates)
    recent_dates = history.index[-56:]
    # Exclude either year's Labor Day neighborhood from level estimation.
    source_dates = recent_dates-pd.Timedelta(days=364)
    mask = np.array([abs((d-labor_day(d.year)).days)>7 and
                     abs((s-labor_day(s.year)).days)>7 for d,s in zip(recent_dates,source_dates)])
    ratio = history.loc[recent_dates[mask]].sum()/history.loc[source_dates[mask]].sum()
    scaled = replay*ratio
    patched_tfm = tfm.copy()
    event = np.array([abs((d-labor_day(d.year)).days)<=3 for d in dates])
    patched_tfm[event] = scaled[event]
    weekday = clean.iloc[-56:].groupby(clean.index[-56:].dayofweek).median()
    local = weekday.reindex(dates.dayofweek).to_numpy()*multipliers(dates,factors)
    ridge, tree = statistical_models(history, dates)
    extended = event_replay(history, dates, window=(-7,20))
    broad_mask = np.array([not -7 <= (d-labor_day(d.year)).days <= 20 and
                           not -7 <= (s-labor_day(s.year)).days <= 20 for d,s in zip(recent_dates,source_dates)])
    broad_ratio = history.loc[recent_dates[broad_mask]].sum()/history.loc[source_dates[broad_mask]].sum()
    extended_scaled = extended*broad_ratio
    event_blend = .5*scaled+.5*patched_tfm
    return dict(zip(MODELS, [.75*adjusted+.25*tfm,
                            history.loc[dates-pd.Timedelta(days=364)].to_numpy(), tfm,
                            replay, scaled, patched_tfm, event_blend,
                            local, ridge, tree, np.median([scaled,patched_tfm,tree],axis=0),
                            extended, extended_scaled, .5*extended_scaled+.5*event_blend]))


def calibrate(predictions):
    result = predictions.copy()
    result[['p10','p50','p90']] = np.nan
    result['calibration_folds'] = 0
    for (target, model, origin), g in result.groupby(['target','model','origin']):
        prior = predictions[(predictions.target==target)&(predictions.model==model)&
                            (predictions.origin.isin(RECENT))&
                            (predictions.origin+pd.Timedelta(days=28)<=origin)]
        origins = sorted(prior.origin.unique())[-8:]
        prior = prior[prior.origin.isin(origins)]
        if len(origins)>=4:
            residual = (prior.actual-prior.forecast)/prior.forecast.abs().clip(lower=1)
            q = residual.quantile([.1,.5,.9]).to_numpy()
            result.loc[g.index,['p10','p50','p90']] = g.forecast.to_numpy()[:,None]+np.abs(g.forecast.to_numpy()[:,None])*q
            result.loc[g.index,'calibration_folds'] = len(origins)
    return result


def choose_model(calibrated, target, origin):
    """Rank on completed calibrated forecasts only; no future-window residuals."""
    prior = calibrated[(calibrated.target==target)&(calibrated.origin+pd.Timedelta(days=28)<=origin)].dropna(subset=['actual','p50'])
    recent_origins = sorted(prior.loc[prior.origin.isin(RECENT),'origin'].unique())[-8:]
    if len(recent_origins)<4:
        return None, pd.DataFrame()
    ranks = []
    for model, g in prior.groupby('model'):
        recent = g[g.origin.isin(recent_origins)]
        event = g[g.origin.isin(HOLIDAY)]
        r = (recent.p50-recent.actual).abs().sum()/recent.actual.abs().sum()
        e = (event.p50-event.actual).abs().sum()/event.actual.abs().sum() if len(event) else r
        ranks.append({'target':target,'model':model,'recent_wape':r,'holiday_wape':e,
                      'score':.75*r+.25*e,'recent_folds':len(recent_origins),'holiday_folds':event.origin.nunique()})
    ranking = pd.DataFrame(ranks).sort_values(['score','model'])
    return ranking.iloc[0].model, ranking


def write_research_summary():
    """Keep retrospective best-fit outputs separate from cutoff-only selections."""
    future = pd.read_csv(OUT/'all_model_forecasts.csv',parse_dates=['ds'])
    scores = pd.read_csv(OUT/'actuals_error_summary.csv')
    nested = pd.read_csv(OUT/'nested_backtest_summary.csv')
    overall = scores[(scores.segment=='all_available')&(scores.model!='selected_policy')]
    best = overall.sort_values(['wape','model']).groupby('target').head(1)
    chosen = future.merge(best[['target','model']],on=['target','model'],validate='many_to_one')
    fields = ['p10','p50','p90','actual','error','absolute_error','absolute_percentage_error','covered_80','model']
    wide = chosen.pivot(index='ds',columns='target',values=fields)
    wide.columns = [f'{t}_{q}' for q,t in wide.columns]
    wide[[f'{t}_{q}' for t in ('sales','units') for q in fields]].to_csv(OUT/'retrospective_challenger_forecast.csv')
    lines = ['# Labor Day forecast challenger evaluation', '',
             'Forecast: September 1–28, 2026. Training cutoff: August 31. Actual comparison: September 1–16 (16 days). Non-commercial TimesFM 3 evaluation.', '',
             'Fourteen fixed candidates were screened on 32 non-overlapping 28-day recent windows and three historical September windows. Eight initial recent windows warm up calibration and selection, leaving 24 comparable rolling policy-evaluation windows (672 days per target).', '',
             'The added four-week alignment covers day -7 through +20 relative to Labor Day, preserving the lead-up and post-event pattern. It transfers observed timing from the previous year, removes holiday values displaced onto ordinary dates, and optionally scales levels using the preceding 56 days excluding event neighborhoods. It does not assume that all holiday lift is incremental, or force net demand to be zero.', '',
             '## Outputs and selection', '',
             '- `daily_forecast.csv`: model selections using only information available by August 31 (sales: four-week event blend; units: original ensemble).',
             '- `retrospective_challenger_forecast.csv`: lowest September 1–16 WAPE among the tested candidates for each target. This is a retrospective best-fit shortlist, not an independent validation or production promotion. September 17–28 values are extrapolations of those candidates.',
             '- `all_model_forecasts.csv`: every candidate, actuals, p10/p50/p90, errors, and coverage. Future actuals/errors remain blank.',
             '- `actuals_error_summary.csv`: overall, Labor Day weekend, and other-day results. `nested_backtest_summary.csv` and `selection_history.csv` provide earlier evidence.', '',
             '## Retrospective September comparison', '',
             '| Target | Candidate | WAPE | Bias | 80% coverage |', '|---|---|---:|---:|---:|']
    for target in ('sales','units'):
        winner = best[best.target==target].iloc[0].model
        for name in ['original_blend',winner]:
            row = overall[(overall.target==target)&(overall.model==name)].iloc[0]
            lines.append(f'| {target} | {name} | {row.wape:.2%} | {row.bias_pct:+.2%} | {row.coverage_80:.2%} |')
    lines += ['', 'Sales shortlist: `event_blend_50_50` is 50% level-scaled event-aligned naive and 50% TimesFM 3 with the seven-day event path replaced by that aligned path. Units shortlist: `event_4week_blend` is 50% four-week aligned/scaled path and 50% that shorter event blend.', '',
              '## Earlier backtest comparison', '',
              '| Target | Candidate | WAPE | Bias | 80% coverage |', '|---|---|---:|---:|---:|']
    for target in ('sales','units'):
        winner = best[best.target==target].iloc[0].model
        for name in ['original_blend',winner,'selected_policy']:
            row = nested[(nested.target==target)&(nested.model==name)].iloc[0]
            lines.append(f'| {target} | {name} | {row.wape:.2%} | {row.bias_pct:+.2%} | {row.coverage_80:.2%} |')
    lines += ['', 'The broader units profile improves this September but loses on the earlier common backtest windows. The cutoff-only selector therefore retains the original units blend. Calendar Ridge and LightGBM failed to improve this September; adding complexity alone did not resolve event timing.', '',
              '## Historical lead-up and aftermath', '',
              'Historical 2022–2025 values are compared with medians of same weekdays 35 and 42 days before and after each event date. These are descriptive seasonal associations, not a causal estimate of pulled-forward demand.', '',
              '| Target | Event phase | Difference versus matching-weekday baseline |', '|---|---|---:|']
    profile = pd.read_csv(OUT/'historical_labor_day_profile.csv')
    profile['phase'] = profile.offset.map(lambda k: 'Week before' if k<0 else 'Labor Day' if k==0 else 'Days 1–20 after')
    for (target,phase), g in profile.groupby(['target','phase']):
        lines.append(f'| {target} | {phase} | {g.actual.sum()/g.weekday_baseline.sum()-1:+.1%} |')
    lines += ['', '## Limits and verification', '',
              'September actuals were already viewed before these experiments, so even cutoff-only calculations are retrospective research. No September values entered model fitting, residual calibration, or the cutoff-only ranking. The separate retrospective shortlist explicitly uses September error to choose candidates.', '',
              'Point forecasts are residual-median corrected. Intervals use each model’s signed relative residual quantiles from its latest eight fully completed recent windows (minimum four). Coverage is evaluated with prior-only calibration. Narrower intervals are not justified: the shortlist still covers only 37.5% of sales days and 75% of units days in September, versus the nominal 80%. The bands are daily marginal prediction intervals, not bounds for the total-period sum.', '',
              'Only three historical September evaluation windows are available; two have enough prior calibration data. None reproduces the 371-day Labor Day gap of 2026. First-stage registration is retained; the fixed four-week extension followed the user’s pull-forward observation. No window-width or ensemble-weight search was performed.', '',
              'Reproduce: `.venv/Scripts/python.exe run_sls_units_challengers.py` (qualifying non-commercial evaluation only, as authorized in this task).',
              'Checks: `.venv/Scripts/python.exe -m pytest tests/test_sls_units_challengers.py tests/test_sls_units_labor_day.py -q`.']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv('sls_units.csv',parse_dates=['transaction_date'])
    if raw.isna().any().any() or raw.transaction_date.duplicated().any() or not np.isfinite(raw[['sales','units']]).all().all():
        raise ValueError('Invalid input values or duplicate dates')
    history = raw.set_index('transaction_date').sort_index().loc[:CUTOFF]
    assert history.index.equals(pd.date_range(history.index.min(),CUTOFF))
    digest = hashlib.sha256(Path('sls_units.csv').read_bytes()).hexdigest()
    assert digest == json.loads((SOURCE/'manifest.json').read_text())['sha256'], 'Cached predictions belong to another input snapshot'
    registration = {'created_at':datetime.now(timezone.utc).isoformat(),'training_cutoff':str(CUTOFF.date()),
                    'input_sha256':digest,'models':MODELS,'horizons':[28],
                    'selection':'0.75 recent WAPE + 0.25 prior Labor Day-window WAPE on calibrated p50; last 8 available recent folds; at least 4 required.',
                    'calibration':'Own signed relative residual quantiles, latest 8 fully completed recent folds, minimum 4.',
                    'scope':'Exploratory retrospective screen; September 2026 already inspected, excluded from fitting/selection/calibration.',
                    'use':'User-authorized non-commercial TimesFM 3 evaluation',
                    'candidates':'Fixed settings; no numeric tuning against September actuals. Initial event replay radius +/-3 days; user-requested pull-forward extension uses a fixed four-week window from day -7 through +20. Scale from prior 56 days excluding corresponding holiday neighborhoods.',
                    'stages':'The four-week extension was added after stage-one results and user input; both stages are retrospective exploration.'}
    (OUT/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    cached = pd.concat([pd.read_csv(SOURCE/'backtest_predictions.csv',parse_dates=['ds','origin']),
                        pd.read_csv(SOURCE/'all_model_forecasts.csv',parse_dates=['ds','origin'])])
    cached = cached[cached.model=='labor_day_timesfm_3.0']
    previous = OUT/'backtest_and_future_predictions.csv'
    if previous.exists():
        old = pd.read_csv(previous,parse_dates=['ds','origin'])
        cached = pd.concat([cached,old[old.model=='timesfm3']]).drop_duplicates(['target','origin','ds'])
    tfm_model = None
    rows = []
    with threadpool_limits(limits=1):
        for target in ('sales','units'):
            for origin in sorted(set(RECENT+HOLIDAY+[CUTOFF])):
                print(target,origin.date(),flush=True)
                train = history[target].loc[:origin]
                dates = pd.date_range(origin+pd.Timedelta(days=1),periods=28)
                cache = cached[(cached.target==target)&(cached.origin==origin)].sort_values('ds')
                if len(cache)==28:
                    assert pd.DatetimeIndex(cache.ds).equals(dates)
                    tfm = cache.forecast.to_numpy()
                else:
                    if tfm_model is None:
                        tfm_model = load_timesfm3()
                    factors = holiday_factors(train)
                    clean = train/multipliers(train.index,factors)
                    frame = pd.DataFrame({'ds':clean.index,'y':clean.to_numpy()})
                    tfm = forecast_timesfm3(tfm_model,frame,28,1024).p50.to_numpy()*multipliers(dates,factors)
                for model, values in candidates(train,dates,tfm).items():
                    rows.append(pd.DataFrame({'target':target,'model':model,'origin':origin,'ds':dates,
                                              'forecast':values,'actual':history[target].reindex(dates).to_numpy()}))
    calibrated = calibrate(pd.concat(rows,ignore_index=True))
    calibrated.to_csv(OUT/'backtest_and_future_predictions.csv',index=False)
    selections, selected, ranks = [], [], []
    for target in ('sales','units'):
        for origin in sorted(set(RECENT+[CUTOFF])):
            model, ranking = choose_model(calibrated,target,origin)
            if model is None:
                continue
            selections.append({'target':target,'origin':origin,'selected_model':model})
            g = calibrated[(calibrated.target==target)&(calibrated.origin==origin)&(calibrated.model==model)].copy()
            g['selected_model'] = model
            g['model'] = 'selected_policy'
            selected.append(g)
            if origin==CUTOFF:
                ranks.append(ranking)
    selected = pd.concat(selected,ignore_index=True)
    pd.DataFrame(selections).to_csv(OUT/'selection_history.csv',index=False)
    pd.concat(ranks).to_csv(OUT/'selection_scores.csv',index=False)
    # Score September actuals only after the model choices and forecasts are fixed.
    future = pd.concat([calibrated[calibrated.origin==CUTOFF],selected[selected.origin==CUTOFF]],ignore_index=True)
    for target in ('sales','units'):
        mask = future.target==target
        future.loc[mask,'actual'] = raw.set_index('transaction_date')[target].reindex(future.loc[mask,'ds']).to_numpy()
    future['error'] = future.p50-future.actual
    future['absolute_error'] = future.error.abs()
    future['absolute_percentage_error'] = future.absolute_error/future.actual.abs().replace(0,np.nan)
    future['covered_80'] = ((future.actual>=future.p10)&(future.actual<=future.p90)).astype('boolean').where(future.actual.notna())
    assert future[['p10','p50','p90']].notna().all().all()
    assert ((future.p10<=future.p50)&(future.p50<=future.p90)).all()
    future.to_csv(OUT/'all_model_forecasts.csv',index=False)
    chosen = future[future.model=='selected_policy']
    fields = ['p10','p50','p90','actual','error','absolute_error','absolute_percentage_error','covered_80','selected_model']
    wide = chosen.pivot(index='ds',columns='target',values=fields)
    wide.columns = [f'{t}_{q}' for q,t in wide.columns]
    wide[[f'{t}_{q}' for t in ('sales','units') for q in fields]].to_csv(OUT/'daily_forecast.csv')
    scores = []
    for (target,model), g in future.dropna(subset=['actual']).groupby(['target','model']):
        for segment, mask in [('all_available',np.ones(len(g),dtype=bool)),
                              ('labor_weekend',g.ds.between('2026-09-04','2026-09-07')),
                              ('other_days',~g.ds.between('2026-09-04','2026-09-07'))]:
            part = g[mask]
            scores.append({'target':target,'model':model,'segment':segment,**score_group(part.assign(forecast=part.p50)).to_dict()})
    pd.DataFrame(scores).to_csv(OUT/'actuals_error_summary.csv',index=False)
    summaries = []
    for target in ('sales','units'):
        policy = selected[(selected.target==target)&(selected.origin<CUTOFF)]
        compare = calibrated[(calibrated.target==target)&calibrated.origin.isin(policy.origin.unique())]
        for model,g in pd.concat([compare,policy]).groupby('model'):
            summaries.append({'target':target,'model':model,'folds':g.origin.nunique(),
                              **score_group(g.assign(forecast=g.p50)).to_dict()})
    pd.DataFrame(summaries).to_csv(OUT/'nested_backtest_summary.csv',index=False)
    studies = []
    for target in ('sales','units'):
        study = event_study(history[target])
        study['target'] = target
        studies.append(study)
    pd.concat(studies,ignore_index=True).to_csv(OUT/'historical_labor_day_profile.csv',index=False)
    registration['selected_models'] = chosen.groupby('target').selected_model.first().to_dict()
    registration['actuals_available_through'] = str(raw.transaction_date.max().date())
    registration['limitations'] = ['September comparison is retrospective, not an independent holdout.',
                                   'Candidate definitions are exploratory; nested policy avoids per-origin lookahead but is not external confirmation.',
                                   'Only 3 historical Labor Days; none reproduces the 2026 371-day event gap.',
                                   'Intervals are marginal daily estimates, not simultaneous or period-total intervals.']
    (OUT/'manifest.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    write_research_summary()
    print('SELECTED',registration['selected_models'],flush=True)
    print(pd.DataFrame(scores).query("segment == 'all_available'")[['target','model','wape','bias_pct','coverage_80']].to_string(index=False))


if __name__ == '__main__':
    main()
