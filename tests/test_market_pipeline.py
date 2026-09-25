from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from run_market_bakeoff import load_panel
from run_production_forecast import build_horizon_rows, discover_grids, grid_model_degradation, main, parse_horizons, resolve_grid


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

    rows = build_horizon_rows(naive, None, 3, dates, qs, pd.Timestamp("2026-06-30"), "run-1")

    assert len(rows) == 3
    assert all(r["model_name"] == "final_3d_seasonal_naive_production" for r in rows)
    assert [r["horizon_step"] for r in rows] == [1, 2, 3]
    assert all(r["horizon"] == 3 and r["grain"] == "daily" and r["run_id"] == "run-1" for r in rows)

    with_challenger = build_horizon_rows(naive, np.full(3, 80.0), 3, dates, qs, pd.Timestamp("2026-06-30"), "run-1")
    challenger = [r for r in with_challenger if r["model_name"].startswith("challenger")]
    assert len(challenger) == 3
    np.testing.assert_allclose([r["forecast"] for r in challenger], 0.75 * 100.0 + 0.25 * 80.0)


def test_custom_horizon_contract_and_covering_grid() -> None:
    grids = {28: "28.csv", 91: "91.csv", 365: "365.csv"}

    assert parse_horizons("91,45,45,1") == [1, 45, 91]
    assert resolve_grid(grids, 45) == (91, "91.csv")
    assert resolve_grid(grids, 91) == (91, "91.csv")
    with pytest.raises(ValueError, match="between 1 and 365"):
        parse_horizons("0,366")
    with pytest.raises(ValueError, match="covers horizon"):
        resolve_grid(grids, 366)


def test_grid_discovery_prefers_latest_training_cutoff(tmp_path, monkeypatch) -> None:
    for name, cutoff in (("old", "2026-04-01"), ("new", "2026-08-29")):
        output = tmp_path / f"outputs_multifold_28d_{name}_validation"
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({"horizon": 28, "training_cutoff": cutoff}), encoding="utf-8")
        (output / "multifold_grid_predictions.csv").touch()
    monkeypatch.chdir(tmp_path)

    assert discover_grids()[28].parent.name == "outputs_multifold_28d_new_validation"


def test_degraded_timesfm_grid_is_not_eligible_for_diagnostics(tmp_path) -> None:
    grid_path = tmp_path / "multifold_grid_predictions.csv"
    grid_path.write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"degraded_models": {"timesfm_2_5": "fell back to seasonal naive"}}),
        encoding="utf-8",
    )

    assert grid_model_degradation(grid_path, "timesfm_2_5") == "fell back to seasonal naive"


def test_production_main_emits_final_only_custom_horizon(tmp_path, monkeypatch) -> None:
    sales_dates = pd.date_range("2025-01-01", periods=400, freq="D")
    pd.DataFrame({"transaction_date": sales_dates, "sales": np.arange(400, dtype=float) + 1000}).to_csv(
        tmp_path / "sales.csv", index=False
    )
    grid_dir = tmp_path / "outputs_multifold_4d_validation"
    grid_dir.mkdir()
    (grid_dir / "manifest.json").write_text(json.dumps({"horizon": 4}), encoding="utf-8")
    grid_rows = []
    for cutoff in pd.to_datetime(["2025-12-01", "2025-12-08"]):
        for step in range(1, 5):
            grid_rows.append(
                {
                    "horizon": 4,
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(days=step),
                    "horizon_step": step,
                    "actual": 100.0 + step,
                    "model_name": "seasonal_naive_lag_364",
                    "forecast": 95.0 + step,
                }
            )
            grid_rows.append(
                {
                    "horizon": 4,
                    "cutoff": cutoff,
                    "ds": cutoff + pd.Timedelta(days=step),
                    "horizon_step": step,
                    "actual": 100.0 + step,
                    "model_name": "timesfm_2_5",
                    "forecast": 90.0 + step,
                }
            )
    pd.DataFrame(grid_rows).to_csv(grid_dir / "multifold_grid_predictions.csv", index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_production_forecast.py",
            "--sales-csv",
            "sales.csv",
            "--horizons",
            "3",
            "--opening-info",
            "",
            "--output-dir",
            "forecast",
        ],
    )

    main()

    forecast = pd.read_csv(tmp_path / "forecast" / "production_forecast.csv")
    manifest = json.loads((tmp_path / "forecast" / "manifest.json").read_text(encoding="utf-8"))
    assert len(forecast) == 3
    assert forecast["horizon_step"].tolist() == [1, 2, 3]
    assert set(["run_id", "forecast_origin", "grain", "horizon", "horizon_step"]).issubset(forecast.columns)
    assert forecast["model_name"].str.startswith("final_").all()
    assert not (tmp_path / "forecast" / "diagnostic_forecast.csv").exists()
    assert manifest["interval_sources"]["3"]["source_horizon"] == 4
    assert manifest["interval_sources"]["3"]["n_rows"] == 6
    assert manifest["challenger_requested"] is False

    timesfm = types.ModuleType("run_prophet_timesfm_benchmark")
    timesfm.load_timesfm = lambda context, horizon: object()
    timesfm.forecast_timesfm = lambda model, train, horizon, context: pd.DataFrame({"forecast": np.full(horizon, 80.0)})
    monkeypatch.setitem(sys.modules, "run_prophet_timesfm_benchmark", timesfm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_production_forecast.py",
            "--sales-csv",
            "sales.csv",
            "--horizons",
            "4",
            "--include-challenger",
            "--opening-info",
            "",
            "--output-dir",
            "forecast_with_diagnostics",
        ],
    )

    main()

    production = pd.read_csv(tmp_path / "forecast_with_diagnostics" / "production_forecast.csv")
    diagnostic = pd.read_csv(tmp_path / "forecast_with_diagnostics" / "diagnostic_forecast.csv")
    assert len(production) == 4
    assert production["model_name"].str.startswith("final_").all()
    assert len(diagnostic) == 8
    assert set(diagnostic["model_name"].str.split("_").str[0]) == {"final", "challenger"}
