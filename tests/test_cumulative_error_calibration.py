from __future__ import annotations

import pandas as pd

from calibrate_cumulative_error_paths import build_cumulative_paths, rolling_intervals


def test_rolling_intervals_use_prior_cutoffs_and_both_methods() -> None:
    rows = []
    for cutoff_index, cutoff in enumerate(pd.date_range("2026-01-01", periods=4, freq="7D")):
        for step in range(1, 3):
            forecast = 10.0
            actual = forecast + cutoff_index + step
            rows.append({"horizon": 2, "cutoff": cutoff, "ds": cutoff + pd.Timedelta(days=step), "horizon_step": step, "actual": actual, "forecast": forecast})
    paths = build_cumulative_paths(pd.DataFrame(rows))
    intervals = rolling_intervals(paths, min_calibration_windows=2)

    assert set(intervals["calibration_method"]) == {"additive", "relative"}
    assert set(intervals["calibration_window"]) == {"recent8", "all_prior"}
    assert intervals["cutoff"].min() == pd.Timestamp("2026-01-15")
    assert intervals["calibration_count"].min() == 2
    assert intervals["calibration_count"].max() == 3


def test_rolling_intervals_use_only_residuals_available_at_evaluated_cutoff() -> None:
    rows = []
    for cutoff_index, cutoff in enumerate(pd.date_range("2026-01-01", periods=11, freq="2D")):
        for step in range(1, 5):
            rows.append(
                {
                    "horizon": 4,
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(days=step),
                    "horizon_step": step,
                    "actual": 10.0 + cutoff_index,
                    "forecast": 10.0,
                }
            )
    preds = pd.DataFrame(rows)
    changed = preds.copy()
    changed.loc[(changed["cutoff"].eq("2026-01-19")) & changed["ds"].gt("2026-01-21"), "actual"] = 1_000.0

    baseline = rolling_intervals(build_cumulative_paths(preds), min_calibration_windows=2)
    modified = rolling_intervals(build_cumulative_paths(changed), min_calibration_windows=2)
    columns = ["calibration_method", "calibration_window", "calibration_count", "p025", "p10", "p50", "p90", "p975"]
    baseline_row = baseline[(baseline["cutoff"].eq("2026-01-21")) & baseline["horizon_step"].eq(4)][columns].reset_index(drop=True)
    modified_row = modified[(modified["cutoff"].eq("2026-01-21")) & modified["horizon_step"].eq(4)][columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(baseline_row, modified_row)
    assert set(baseline_row["calibration_count"]) == {8, 9}  # recent8 is selected after availability.
    boundary = baseline[(baseline["cutoff"].eq("2026-01-07")) & baseline["horizon_step"].eq(4)]
    assert set(boundary["calibration_count"]) == {2}  # ds == cutoff is available.
    assert baseline[(baseline["cutoff"].eq("2026-01-05")) & baseline["horizon_step"].eq(4)].empty
