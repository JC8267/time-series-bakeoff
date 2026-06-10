"""Estimate new-unit ramp curves from unit-level daily sales."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_unit_level_targets import normalize_unit_sales
from run_bakeoff import load_openings


RAMP_BANDS = [
    ("grand_opening_0_30", 0, 30),
    ("early_ramp_31_90", 31, 90),
    ("stabilization_91_180", 91, 180),
    ("mature_181_365", 181, 365),
]


def estimate_ramp(unit_sales: pd.DataFrame, openings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    opening_dates = openings[["unit_id", "open_date"]].copy()
    opening_dates["unit_id"] = opening_dates["unit_id"].astype(str)
    df = unit_sales.merge(opening_dates, on="unit_id", how="inner")
    df["days_since_open"] = (df["ds"] - df["open_date"]).dt.days
    df = df[df["days_since_open"].between(0, 365, inclusive="both")].copy()
    if df.empty:
        raise ValueError("No unit-level sales rows matched opened units within 0-365 days after opening.")

    daily_curve = (
        df.groupby("days_since_open")
        .agg(mean_sales=("y", "mean"), median_sales=("y", "median"), unit_days=("y", "size"), units=("unit_id", "nunique"))
        .reset_index()
    )

    band_rows = []
    for name, start, end in RAMP_BANDS:
        band = df[df["days_since_open"].between(start, end, inclusive="both")]
        band_rows.append(
            {
                "ramp_band": name,
                "start_day": start,
                "end_day": end,
                "mean_daily_sales": float(band["y"].mean()) if not band.empty else 0.0,
                "median_daily_sales": float(band["y"].median()) if not band.empty else 0.0,
                "unit_days": int(len(band)),
                "units": int(band["unit_id"].nunique()),
            }
        )
    band_curve = pd.DataFrame(band_rows)
    mature = band_curve.loc[band_curve["ramp_band"] == "mature_181_365", "mean_daily_sales"].iloc[0]
    if mature:
        band_curve["mean_vs_mature_index"] = band_curve["mean_daily_sales"] / mature
    else:
        band_curve["mean_vs_mature_index"] = pd.NA
    return daily_curve, band_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-sales-csv", required=True, help="Unit-level daily sales with unit_id, date, and sales columns.")
    parser.add_argument("--opening-info", default="opening_info.csv")
    parser.add_argument("--output-dir", default="outputs_ramp_curve")
    args = parser.parse_args()

    unit_sales = normalize_unit_sales(Path(args.unit_sales_csv))
    openings = load_openings(Path(args.opening_info))
    daily_curve, band_curve = estimate_ramp(unit_sales, openings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_curve.to_csv(output_dir / "daily_opening_ramp_curve.csv", index=False)
    band_curve.to_csv(output_dir / "opening_ramp_band_summary.csv", index=False)
    print(band_curve.to_string(index=False))


if __name__ == "__main__":
    main()

