from __future__ import annotations

import pandas as pd
import pytest

from run_bakeoff import load_sales
from run_multifold_final_validation import missing_data_quality_note


def test_load_sales_writes_missing_date_audit(tmp_path) -> None:
    sales_path = tmp_path / "sales.csv"
    audit_path = tmp_path / "audit.csv"
    pd.DataFrame(
        {
            "transaction_date": ["2024-01-01", "2024-01-03"],
            "sales": [10.0, 30.0],
        }
    ).to_csv(sales_path, index=False)

    out = load_sales(sales_path, fill_missing="interpolate", audit_path=audit_path)
    audit = pd.read_csv(audit_path)

    assert out["y"].tolist() == [10.0, 20.0, 30.0]
    assert out["data_gap_filled_flag"].tolist() == [0, 1, 0]
    assert audit["ds"].tolist() == ["2024-01-02"]


def test_load_sales_can_error_on_missing_dates(tmp_path) -> None:
    sales_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "transaction_date": ["2024-01-01", "2024-01-03"],
            "sales": [10.0, 30.0],
        }
    ).to_csv(sales_path, index=False)

    with pytest.raises(ValueError, match="Missing sales dates"):
        load_sales(sales_path, fill_missing="error")


def test_load_sales_weekly_grain_aggregates_complete_weeks(tmp_path) -> None:
    sales_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "transaction_date": pd.date_range("2024-01-01", "2024-01-17", freq="D"),
            "sales": [1.0] * 17,
        }
    ).to_csv(sales_path, index=False)

    out = load_sales(sales_path, grain="weekly", weekly_freq="W-WED")

    assert out["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-10", "2024-01-17"]
    assert out["y"].tolist() == [7.0, 7.0]
    assert out["data_gap_filled_flag"].tolist() == [0, 0]


def test_missing_data_quality_note_discloses_unweighted_fills() -> None:
    assert missing_data_quality_note(0, "interpolate") == "No missing daily sales dates were filled."

    note = missing_data_quality_note(2, "interpolate")

    assert "2 missing daily sales date(s)" in note
    assert "not down-weighted" in note
    assert "missing_sales_dates_audit.csv" in note
