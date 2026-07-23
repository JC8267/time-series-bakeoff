from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from run_bakeoff import load_sales
from run_multifold_final_validation import (
    CACHE_ARTIFACTS,
    default_output_dir,
    main as validation_main,
    missing_data_quality_note,
    validation_cache_is_valid,
    validation_package_versions,
)


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


def test_validation_cache_requires_matching_manifest_and_artifacts(tmp_path) -> None:
    signature = {"version": 1, "horizon": 45}
    for name in CACHE_ARTIFACTS:
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"cache_signature": signature}), encoding="utf-8")

    assert validation_cache_is_valid(tmp_path, signature)
    assert not validation_cache_is_valid(tmp_path, {**signature, "horizon": 46})
    (tmp_path / "multifold_summary.csv").unlink()
    assert not validation_cache_is_valid(tmp_path, signature)
    assert default_output_dir(45, "daily").name == "outputs_multifold_45d_validation"


def test_validation_cache_tracks_timesfm_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "run_multifold_final_validation.distribution_version",
        lambda package: {
            "huggingface-hub": "0.35.3",
            "numpy": "2.0.2",
            "pandas": "2.2.3",
            "timesfm": "2.0.2",
            "torch": "2.8.0",
        }[package],
    )

    versions = validation_package_versions(["seasonal_naive_lag_364", "timesfm_2_5"])

    assert versions["timesfm"] == "2.0.2"
    assert versions["torch"] == "2.8.0"


def test_multifold_validation_reuses_matching_cache(tmp_path, monkeypatch) -> None:
    sales_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "transaction_date": pd.date_range("2024-01-01", periods=750, freq="D"),
            "sales": range(1000, 1750),
        }
    ).to_csv(sales_path, index=False)
    argv = [
        "run_multifold_final_validation.py",
        "--sales-csv",
        "sales.csv",
        "--horizon",
        "2",
        "--n-folds",
        "1",
        "--grid-step",
        "2",
        "--max-grid-cutoffs",
        "4",
        "--pool",
        "seasonal_naive_lag_364",
        "--output-dir",
        "validation",
        "--reuse-cache",
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)

    validation_main()
    first = json.loads((tmp_path / "validation" / "manifest.json").read_text(encoding="utf-8"))
    validation_main()
    second = json.loads((tmp_path / "validation" / "manifest.json").read_text(encoding="utf-8"))

    assert first["run_id"] == second["run_id"]
    assert first["run_mode"] == "offline_custom_pool_validation"
    assert first["training_cutoff"] == "2026-01-19"
