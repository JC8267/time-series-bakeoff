from __future__ import annotations

import numpy as np
import pandas as pd

from run_market_bakeoff import load_panel
from run_production_forecast import build_horizon_rows


def test_load_panel_fills_gaps_with_zero_to_common_end(tmp_path) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "sub_market_no": [1, 1, 1, 2, 2],
            "ds": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-03", "2026-01-04"]),
            "y": [10.0, 11.0, 12.0, 5.0, 6.0],
            "open_date": pd.NaT,
            "total_sqm": np.nan,
            "unit_type": None,
            "has_physical": False,
        }
    ).to_csv(path, index=False)

    panel = load_panel(path)

    m1 = panel[panel["sub_market_no"] == 1].sort_values("ds")
    assert len(m1) == 5  # reindexed 01-01 .. 01-05
    assert m1["y"].tolist() == [10.0, 11.0, 0.0, 0.0, 12.0]
    m2 = panel[panel["sub_market_no"] == 2].sort_values("ds")
    assert m2["ds"].max() == panel["ds"].max()  # extended to the common end
    assert m2["y"].iloc[-1] == 0.0


def test_production_rows_without_timesfm_still_emit_seasonal_naive() -> None:
    naive = np.full(3, 100.0)
    qs = {"seasonal_naive": np.array([-10.0, -5.0, 5.0, 10.0]), "blend": np.zeros(4), "n_windows": 8}
    dates = pd.Series(pd.date_range("2026-07-01", periods=3, freq="D"))

    rows = build_horizon_rows(naive, None, 3, dates, qs, pd.Timestamp("2026-06-30"))

    assert len(rows) == 3
    assert all(r["model_name"] == "final_3d_seasonal_naive_production" for r in rows)

    with_challenger = build_horizon_rows(naive, np.full(3, 80.0), 3, dates, qs, pd.Timestamp("2026-06-30"))
    challenger = [r for r in with_challenger if r["model_name"].startswith("challenger")]
    assert len(challenger) == 3
    np.testing.assert_allclose([r["forecast"] for r in challenger], 0.75 * 100.0 + 0.25 * 80.0)
