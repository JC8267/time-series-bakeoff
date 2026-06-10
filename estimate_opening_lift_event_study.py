"""Rough aggregate event study of new-unit openings on total daily sales.

For each opening with enough observed post-opening history, estimates the
total-sales lift in ramp bands (0-30, 31-90, 91-180, 181-365 days) as:

    counterfactual(band) = year-ago sales in the band window * pre-opening YoY ratio
    lift(band)           = actual band sales - counterfactual(band)

The pre-opening YoY ratio (default 91 days) absorbs chain-level growth, so the
lift is a difference-in-differences style estimate. Limitations: this is
aggregate data, so concurrent openings contaminate each other's windows (the
`overlapping_openings` column counts other units opening inside or up to a
year before the band window); estimates are rough and should be read as
percent-of-chain magnitudes, not unit P&L.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_bakeoff import load_openings, load_sales


BANDS = [(0, 30), (31, 90), (91, 180), (181, 365)]


def window_sum(y: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float(y.loc[start:end].sum())


def estimate(sales_csv: Path, opening_info: Path, pre_days: int) -> pd.DataFrame:
    df = load_sales(sales_csv)
    y = df.set_index("ds")["y"]
    data_end = y.index.max()
    openings = load_openings(opening_info)
    open_dates = list(openings["open_date"])

    rows = []
    for unit, open_date in zip(openings["unit_id"], openings["open_date"]):
        pre_start = open_date - pd.Timedelta(days=pre_days)
        pre_end = open_date - pd.Timedelta(days=1)
        pre = window_sum(y, pre_start, pre_end)
        pre_year_ago = window_sum(y, pre_start - pd.Timedelta(days=364), pre_end - pd.Timedelta(days=364))
        if pre_year_ago <= 0 or pre_start < y.index.min() + pd.Timedelta(days=364):
            continue
        ratio = pre / pre_year_ago

        for band_start, band_end in BANDS:
            ws = open_date + pd.Timedelta(days=band_start)
            we = open_date + pd.Timedelta(days=band_end)
            if we > data_end:
                continue
            actual = window_sum(y, ws, we)
            year_ago = window_sum(y, ws - pd.Timedelta(days=364), we - pd.Timedelta(days=364))
            counterfactual = ratio * year_ago
            overlapping = sum(
                1
                for other in open_dates
                if other != open_date and ws - pd.Timedelta(days=364) <= other <= we
            )
            rows.append(
                {
                    "unit_id": unit,
                    "open_date": open_date.date(),
                    "band": f"{band_start}-{band_end}",
                    "days": (we - ws).days + 1,
                    "pre_yoy_ratio": round(ratio, 4),
                    "lift_abs": actual - counterfactual,
                    "lift_pct_of_chain": (actual - counterfactual) / counterfactual if counterfactual else float("nan"),
                    "overlapping_openings": overlapping,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="daily_sales_data.csv")
    parser.add_argument("--opening-info", default="opening_info.csv")
    parser.add_argument("--pre-days", type=int, default=91)
    parser.add_argument("--output-dir", default="outputs_opening_event_study")
    args = parser.parse_args()

    result = estimate(Path(args.sales_csv), Path(args.opening_info), args.pre_days)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "opening_lift_by_band.csv", index=False)
    display = result[["unit_id", "open_date", "band", "pre_yoy_ratio", "lift_pct_of_chain", "overlapping_openings"]].copy()
    display["lift_pct_of_chain"] = (display["lift_pct_of_chain"] * 100).round(2)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
