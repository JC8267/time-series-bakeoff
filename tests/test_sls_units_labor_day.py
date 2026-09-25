import numpy as np
import pandas as pd

from run_sls_units_labor_day import BLEND, holiday_factors, multipliers, write_forecast_actuals
from run_bakeoff import labor_day


def test_moving_labor_day_effect_without_future_leakage():
    dates = pd.date_range('2022-01-02', '2026-08-31')
    history = pd.Series(100.0, index=dates)
    for year in range(2022, 2026):
        history.loc[labor_day(year)] = 200.0
    factors = holiday_factors(history)
    assert factors[0] == 2.0
    target = pd.DatetimeIndex(['2026-09-01', '2026-09-07'])
    np.testing.assert_allclose(multipliers(target, factors), [1, 2])
    clean = history / multipliers(history.index, factors)
    assert clean.loc['2025-09-01'] == 100
    assert labor_day(2026) == pd.Timestamp('2026-09-07')
    # Incomplete future-side neighbors cannot leak into historical factor fitting.
    cutoff = pd.Timestamp('2023-09-05')
    changed = history.copy()
    changed.loc[changed.index > cutoff] *= 100
    assert holiday_factors(changed.loc[:cutoff]) == holiday_factors(history.loc[:cutoff])


def test_actuals_score_medians_and_leave_unobserved_blank(tmp_path):
    dates = pd.date_range('2026-09-01', periods=3)
    fc = pd.DataFrame([{'ds': day, 'target': target, 'model': BLEND,
                        'forecast': 999, 'p10': 0, 'p50': 12, 'p90': 15}
                       for target in ('sales', 'units') for day in dates])
    raw = pd.DataFrame({'transaction_date': dates[:2], 'sales': [10, 0], 'units': [10, 0]})
    primary = write_forecast_actuals(fc, raw, tmp_path)
    assert primary.loc[dates[0], 'sales_error'] == 2
    assert primary.loc[dates[0], 'sales_absolute_percentage_error'] == .2
    assert pd.isna(primary.loc[dates[1], 'sales_absolute_percentage_error'])
    assert primary.loc[dates[2], ['sales_actual', 'sales_error', 'sales_covered_80']].isna().all()
    summary = pd.read_csv(tmp_path / 'actuals_error_summary.csv')
    np.testing.assert_allclose(summary.wape, 1.4)
    assert (summary.observed_days == 2).all()
    assert (summary.coverage_80 == 1).all()
    saved = pd.read_csv(tmp_path / 'all_model_forecasts.csv', parse_dates=['ds'])
    refreshed = write_forecast_actuals(saved, raw, tmp_path)
    pd.testing.assert_frame_equal(primary, refreshed)
