"""Cutoff-safe sales/units forecast with an explicit moving Labor Day effect."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import labor_day
from metrics import score_group
from run_prophet_timesfm_benchmark import load_timesfm, forecast_timesfm, load_timesfm3, forecast_timesfm3


CUTOFF = pd.Timestamp('2026-08-31')
OUT = Path('outputs_sls_units_labor_day_20260831')
BLEND = 'labor_day_naive75_timesfm25'


def holiday_factors(history):
    observations = {offset: [] for offset in range(-3, 4)}
    for year in range(history.index.min().year, history.index.max().year + 1):
        for offset in observations:
            day = labor_day(year) + pd.Timedelta(days=offset)
            neighbors = [day + pd.Timedelta(days=k) for k in (-21, -14, 14, 21)]
            if day in history.index and all(d in history.index for d in neighbors):
                baseline = float(history.loc[neighbors].median())
                if baseline > 0 and history.loc[day] > 0:
                    observations[offset].append(float(history.loc[day] / baseline))
    return {k: float(np.median(v)) if v else 1.0 for k, v in observations.items()}


def multipliers(dates, factors):
    return np.array([factors.get((d - labor_day(d.year)).days, 1.0) for d in dates])


def predict(history, model, version='2.5'):
    dates = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=28)
    factors = holiday_factors(history)
    clean = history / multipliers(history.index, factors)
    future_effect = multipliers(dates, factors)
    source = dates - pd.Timedelta(days=364)
    naive = history.loc[source].to_numpy()
    adjusted = clean.loc[source].to_numpy() * future_effect
    frame = pd.DataFrame({'ds': clean.index, 'y': clean.to_numpy()})
    runner = forecast_timesfm3 if version == '3.0' else forecast_timesfm
    tfm = runner(model, frame, 28, 1024)['p50'].to_numpy() * future_effect
    return dates, {
        'seasonal_naive_lag364': naive,
        'labor_day_seasonal_naive': adjusted,
        f'labor_day_timesfm_{version}': tfm,
        BLEND: .75 * adjusted + .25 * tfm,
    }, factors


def write_forecast_actuals(fc, raw, output_dir):
    actuals = raw.melt(id_vars='transaction_date', value_vars=['sales', 'units'],
                       var_name='target', value_name='actual').rename(columns={'transaction_date': 'ds'})
    columns = ['actual', 'error', 'absolute_error', 'absolute_percentage_error', 'covered_80']
    fc = fc.drop(columns=columns, errors='ignore').merge(actuals, on=['ds', 'target'], how='left', validate='many_to_one')
    fc['error'] = fc.p50 - fc.actual
    fc['absolute_error'] = fc.error.abs()
    fc['absolute_percentage_error'] = fc.absolute_error / fc.actual.abs().replace(0, np.nan)
    fc['covered_80'] = ((fc.actual >= fc.p10) & (fc.actual <= fc.p90)).astype('boolean').where(fc.actual.notna())
    fc.to_csv(output_dir / 'all_model_forecasts.csv', index=False)
    primary = fc[fc.model == BLEND].pivot(index='ds', columns='target', values=['p10', 'p50', 'p90', *columns])
    primary.columns = [f'{target}_{q}' for q, target in primary.columns]
    primary = primary[[f'{target}_{q}' for target in ('sales', 'units') for q in ('p10', 'p50', 'p90', *columns)]]
    primary.to_csv(output_dir / 'daily_forecast.csv')
    summaries = []
    for (target, model), g in fc.groupby(['target', 'model']):
        observed = g.dropna(subset=['actual'])
        scores = score_group(observed.assign(forecast=observed.p50)).to_dict() if len(observed) else {}
        summaries.append({'target': target, 'model': model, 'observed_days': len(observed),
                          'actual_start': observed.ds.min(), 'actual_end': observed.ds.max(), **scores})
    pd.DataFrame(summaries).to_csv(output_dir / 'actuals_error_summary.csv', index=False)
    return primary


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesfm-version', choices=['2.5', '3.0'], default='2.5')
    parser.add_argument('--allow-noncommercial-timesfm3', action='store_true')
    parser.add_argument('--refresh-actuals-only', action='store_true', help='Score saved forecasts without refitting')
    args = parser.parse_args()
    if args.timesfm_version == '3.0' and not args.allow_noncommercial_timesfm3:
        raise ValueError('TimesFM 3 requires explicit non-commercial evaluation acknowledgement')
    if args.timesfm_version == '3.0':
        OUT = Path(str(OUT) + '_timesfm3')
    raw = pd.read_csv('sls_units.csv', parse_dates=['transaction_date'])
    if raw.isna().any().any() or raw.transaction_date.duplicated().any():
        raise ValueError('Missing values or duplicate dates in input')
    history = raw.set_index('transaction_date').sort_index().loc[:CUTOFF]
    if not history.index.equals(pd.date_range(history.index.min(), CUTOFF)):
        raise ValueError('Missing daily history through cutoff')
    if not np.isfinite(history[['sales', 'units']].to_numpy()).all():
        raise ValueError('Non-finite target')
    if not np.isfinite(raw[['sales', 'units']].to_numpy()).all():
        raise ValueError('Non-finite actual')
    if args.refresh_actuals_only:
        fc = pd.read_csv(OUT / 'all_model_forecasts.csv', parse_dates=['ds'])
        write_forecast_actuals(fc, raw, OUT)
        return
    OUT.mkdir(exist_ok=True)
    raw[raw.transaction_date > CUTOFF].to_csv(OUT / 'excluded_post_cutoff.csv', index=False)
    model = load_timesfm3() if args.timesfm_version == '3.0' else load_timesfm(1024, 32)
    recent = [CUTOFF - pd.Timedelta(days=28*k) for k in range(16, 0, -1)]
    holiday = [pd.Timestamp(f'{year}-08-31') for year in (2023, 2024, 2025)]
    origins = sorted(set(recent + holiday))
    rows, future, factor_rows = [], [], []
    for target in ('sales', 'units'):
        for origin in [*origins, CUTOFF]:
            print(f'{target} {origin.date()}', flush=True)
            dates, forecasts, factors = predict(history[target].loc[:origin], model, args.timesfm_version)
            for name, values in forecasts.items():
                frame = pd.DataFrame({'target': target, 'model': name, 'origin': origin,
                                      'ds': dates, 'forecast': values})
                if origin == CUTOFF:
                    future.append(frame)
                else:
                    frame['actual'] = history[target].reindex(dates).to_numpy()
                    frame['window'] = 'labor_day' if origin in holiday else 'recent'
                    rows.append(frame)
            if origin == CUTOFF:
                factor_rows.extend({'target': target, 'offset': k, 'multiplier': v} for k,v in factors.items())
    bt = pd.concat(rows, ignore_index=True)
    bt['relative_error'] = (bt.actual - bt.forecast) / bt.forecast.abs().clip(lower=1)
    bt['p10'], bt['p50'], bt['p90'] = np.nan, np.nan, np.nan
    # Evaluate calibration using only completed earlier forecast windows.
    for (target, name, origin), g in bt.groupby(['target', 'model', 'origin']):
        eligible = bt[(bt.target == target) & (bt.model == name) &
                      (bt.origin + pd.Timedelta(days=28) <= origin) & (bt.window == 'recent')]
        chosen = sorted(eligible.origin.unique())[-8:]
        eligible = eligible[eligible.origin.isin(chosen)]
        if len(chosen) >= 4:
            q = eligible.relative_error.quantile([.1,.5,.9]).to_numpy()
            bt.loc[g.index, ['p10','p50','p90']] = g.forecast.to_numpy()[:,None] + np.abs(g.forecast.to_numpy()[:,None]) * q
    bt.to_csv(OUT / 'backtest_predictions.csv', index=False)
    metrics = []
    for (target, name, window), g in bt.groupby(['target','model','window']):
        valid = g.dropna(subset=['p10'])
        metrics.append({'target':target,'model':name,'window':window,'folds':g.origin.nunique(),
                        'wape':(g.actual-g.forecast).abs().sum()/g.actual.abs().sum(),
                        'bias_pct':(g.forecast-g.actual).sum()/g.actual.sum(),
                        'coverage80_prior_only': ((valid.actual >= valid.p10)&(valid.actual<=valid.p90)).mean(),
                        'coverage_days':len(valid)})
    pd.DataFrame(metrics).to_csv(OUT / 'backtest_summary.csv', index=False)
    fc = pd.concat(future, ignore_index=True)
    for (target,name), g in fc.groupby(['target','model']):
        cal = bt[(bt.target==target)&(bt.model==name)&bt.origin.isin(recent[-8:])]
        q = cal.relative_error.quantile([.1,.5,.9]).to_numpy()
        fc.loc[g.index,['p10','p50','p90']] = g.forecast.to_numpy()[:,None] + np.abs(g.forecast.to_numpy()[:,None])*q
    assert len(fc) == 28*2*4 and fc[['p10','p50','p90']].notna().all().all()
    assert ((fc.p10<=fc.p50)&(fc.p50<=fc.p90)).all()
    fc['is_labor_day'] = fc.ds.eq(labor_day(2026))
    primary = write_forecast_actuals(fc, raw, OUT)
    pd.DataFrame(factor_rows).to_csv(OUT / 'labor_day_factors.csv', index=False)
    manifest = {'input':'sls_units.csv','sha256':hashlib.sha256(Path('sls_units.csv').read_bytes()).hexdigest(),
                'training_cutoff':str(CUTOFF.date()),'horizon':28,'forecast_start':'2026-09-01','forecast_end':'2026-09-28',
                'excluded_rows':int((raw.transaction_date>CUTOFF).sum()),'missing_dates':0,
                'negative_training_values':(history<0).sum().to_dict(), 'model':BLEND,
                'checkpoint': 'google/timesfm-3.0-pytorch' if args.timesfm_version == '3.0' else 'google/timesfm-2.5-200m-pytorch',
                'use':'Non-commercial evaluation only','weights':{'naive':.75,'timesfm':.25},
                'labor_day':'2026-09-07','holiday_method':'Median historical event / nearby same-weekday ratios, offsets -3 through +3; remove from history and restore on future calendar.',
                'interval_method':'Own-model signed relative residual quantiles, latest 8 non-overlapping completed 28-day windows; p50 includes median residual correction.',
                'caveats':['80% prediction intervals are marginal daily intervals, not simultaneous or total-period intervals.',
                           'Recent interval calibration does not establish Labor Day conditional coverage.',
                           'Only three historical Labor Day evaluation windows; new adjustment is exploratory.',
                           'No post-cutoff observations used for model selection, factors, or calibration.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(primary[[f'{target}_p50' for target in ('sales', 'units')]].sum().to_string())


if __name__ == '__main__':
    main()
