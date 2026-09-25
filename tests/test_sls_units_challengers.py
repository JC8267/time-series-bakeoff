import numpy as np
import pandas as pd

from run_bakeoff import labor_day
from run_sls_units_challengers import RECENT, calibrate, choose_model, event_replay, statistical_models


def test_event_replay_moves_holiday_and_removes_displaced_effect():
    history = pd.Series(100.,index=pd.date_range('2022-01-01','2026-08-31'))
    for year in range(2022,2026):
        history.loc[labor_day(year)] = 250.
    dates = pd.DatetimeIndex(['2026-09-01','2026-09-07'])
    np.testing.assert_allclose(event_replay(history,dates),[100,250])


def test_four_week_replay_preserves_buildup_and_post_event_payback():
    history = pd.Series(100.,index=pd.date_range('2022-01-01','2026-08-31'))
    holiday = labor_day(2025)
    history.loc[holiday-pd.Timedelta(days=5)] = 140.
    history.loc[holiday+pd.Timedelta(days=10)] = 60.
    dates = pd.DatetimeIndex([labor_day(2026)+pd.Timedelta(days=k) for k in (-5,10)])
    np.testing.assert_allclose(event_replay(history,dates,window=(-7,20)),[140,60])


def test_calibration_and_selection_ignore_unfinished_windows():
    rows = []
    for origin in RECENT[:12]:
        for model,prediction in [('accurate',100.),('bad',150.)]:
            dates = pd.date_range(origin+pd.Timedelta(days=1),periods=28)
            rows.append(pd.DataFrame({'target':'sales','model':model,'origin':origin,
                                      'ds':dates,'forecast':prediction,'actual':100.+np.arange(28)}))
    frame = pd.concat(rows,ignore_index=True)
    origin = RECENT[9]
    changed = frame.copy()
    changed.loc[changed.origin+pd.Timedelta(days=28)>origin,'actual'] = 1e9
    first, second = calibrate(frame), calibrate(changed)
    pd.testing.assert_frame_equal(first.loc[first.origin==origin,['p10','p50','p90']],
                                  second.loc[second.origin==origin,['p10','p50','p90']])
    name, ranks = choose_model(first,'sales',origin)
    name2, ranks2 = choose_model(second,'sales',origin)
    assert name==name2
    pd.testing.assert_frame_equal(ranks,ranks2)


def test_statistical_models_return_finite_full_horizon():
    dates = pd.date_range('2022-01-01','2024-08-31')
    history = pd.Series(100.+10*np.sin(np.arange(len(dates))*2*np.pi/7),index=dates)
    future = pd.date_range('2024-09-01',periods=28)
    for values in statistical_models(history,future):
        assert len(values)==28 and np.isfinite(values).all()
