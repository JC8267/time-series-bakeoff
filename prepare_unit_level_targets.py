"""Prepare existing-unit and new-unit sales targets from unit-level daily sales."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_bakeoff import load_openings


def normalize_unit_sales(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rename = {
        "transaction_date": "ds",
        "date": "ds",
        "sales_date": "ds",
        "unit": "unit_id",
        "store_id": "unit_id",
        "sales": "y",
        "net_sales": "y",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    required = {"unit_id", "ds", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["unit_id"] = df["unit_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    return df.groupby(["unit_id", "ds"], as_index=False)["y"].sum()


def prepare_targets(unit_sales: pd.DataFrame, openings: pd.DataFrame) -> pd.DataFrame:
    open_dates = openings[["unit_id", "open_date"]].copy()
    open_dates["unit_id"] = open_dates["unit_id"].astype(str)
    df = unit_sales.merge(open_dates, on="unit_id", how="left")
    df["days_since_open"] = (df["ds"] - df["open_date"]).dt.days
    df["is_known_opening_unit"] = df["open_date"].notna()
    df["is_new_unit_180"] = df["is_known_opening_unit"] & df["days_since_open"].between(0, 180, inclusive="both")
    df["is_grand_opening_30"] = df["is_known_opening_unit"] & df["days_since_open"].between(0, 30, inclusive="both")
    df["is_mature_unit"] = ~df["is_known_opening_unit"] | (df["days_since_open"] > 180)

    daily = df.groupby("ds").agg(
        total_sales=("y", "sum"),
        existing_unit_sales=("y", lambda s: float(s[df.loc[s.index, "is_mature_unit"]].sum())),
        new_unit_sales_180=("y", lambda s: float(s[df.loc[s.index, "is_new_unit_180"]].sum())),
        grand_opening_sales_30=("y", lambda s: float(s[df.loc[s.index, "is_grand_opening_30"]].sum())),
        units_selling=("unit_id", "nunique"),
    ).reset_index()
    daily["new_unit_sales_share_180"] = daily["new_unit_sales_180"] / daily["total_sales"].replace(0, np.nan)
    return daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-sales-csv", required=True, help="Unit-level daily sales with unit_id, date, and sales columns.")
    parser.add_argument("--opening-info", default="opening_info.csv")
    parser.add_argument("--output-csv", default="outputs_unit_level/unit_level_targets.csv")
    args = parser.parse_args()

    unit_sales = normalize_unit_sales(Path(args.unit_sales_csv))
    openings = load_openings(Path(args.opening_info))
    targets = prepare_targets(unit_sales, openings)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_path, index=False)
    print(targets.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

