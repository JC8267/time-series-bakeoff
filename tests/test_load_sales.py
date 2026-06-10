from __future__ import annotations

import pandas as pd
import pytest

from run_bakeoff import load_sales


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

