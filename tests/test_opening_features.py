from __future__ import annotations

import pandas as pd

from run_bakeoff import add_opening_features


def test_add_opening_features_ramp_windows() -> None:
    frame = pd.DataFrame({"ds": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-01", "2024-04-01", "2024-06-30"])})
    openings = pd.DataFrame({"unit_id": ["u1"], "open_date": pd.to_datetime(["2024-01-01"])})

    out = add_opening_features(frame, openings)

    assert out["new_unit_opened_flag"].tolist() == [1, 0, 0, 0, 0]
    assert out["days_since_latest_opening"].tolist() == [0, 30, 31, 91, 181]
    assert out["grand_opening_0_30_days"].tolist() == [1, 1, 0, 0, 0]
    assert out["ramp_31_90_days"].tolist() == [0, 0, 1, 0, 0]
    assert out["ramp_91_180_days"].tolist() == [0, 0, 0, 1, 0]
    assert out["mature_180_plus_days"].tolist() == [0, 0, 0, 0, 1]


def test_add_opening_features_counts_recent_openings() -> None:
    frame = pd.DataFrame({"ds": pd.to_datetime(["2024-02-01"])})
    openings = pd.DataFrame(
        {
            "unit_id": ["u1", "u2", "u3"],
            "open_date": pd.to_datetime(["2023-12-15", "2024-01-15", "2024-02-01"]),
        }
    )

    out = add_opening_features(frame, openings)

    assert int(out.loc[0, "num_units_open"]) == 3
    assert int(out.loc[0, "num_new_units_last_30_days"]) == 2
    assert int(out.loc[0, "num_new_units_last_60_days"]) == 3

