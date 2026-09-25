"""Prepare daily channel and total unit series for the existing bakeoff runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"transaction_date", "units", "sales_channel"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="units_forecast/units_raw.csv")
    parser.add_argument("--cutoff", required=True, help="Last complete business date to include.")
    parser.add_argument("--output-dir", default="units_forecast/outputs_prepared")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    raw = pd.read_csv(args.input)
    if not REQUIRED_COLUMNS.issubset(raw.columns):
        raise ValueError(f"{args.input} must contain {sorted(REQUIRED_COLUMNS)}.")

    raw["transaction_date"] = pd.to_datetime(raw["transaction_date"], errors="coerce")
    raw["units"] = pd.to_numeric(raw["units"], errors="coerce")
    if raw[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Units data contains a missing or invalid date, units value, or channel.")

    excluded = raw[raw["transaction_date"] > cutoff].sort_values(["transaction_date", "sales_channel"])
    daily = (
        raw[raw["transaction_date"] <= cutoff]
        .groupby(["transaction_date", "sales_channel"], as_index=False)["units"]
        .sum()
    )
    channels = sorted(daily["sales_channel"].unique())
    if not channels or any(not str(channel).isalnum() for channel in channels):
        raise ValueError(f"Expected simple non-empty channel names, found {channels}.")

    wide = daily.pivot(index="transaction_date", columns="sales_channel", values="units")
    expected_dates = pd.date_range(wide.index.min(), cutoff, freq="D")
    wide = wide.reindex(expected_dates)
    if wide.isna().any().any():
        missing = wide.isna().stack()
        examples = missing[missing].index.tolist()[:10]
        raise ValueError(f"Missing channel-date rows through cutoff; first examples: {examples}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel in channels:
        wide[channel].rename("sales").rename_axis("transaction_date").to_csv(
            output_dir / f"{channel.lower()}.csv"
        )
    total = wide.sum(axis=1)
    total.rename("sales").rename_axis("transaction_date").to_csv(output_dir / "total.csv")
    excluded.to_csv(output_dir / "excluded_rows_audit.csv", index=False)

    pd.testing.assert_series_equal(total, wide.sum(axis=1))
    manifest = {
        "input": args.input,
        "cutoff": str(cutoff.date()),
        "date_min": str(wide.index.min().date()),
        "date_max": str(wide.index.max().date()),
        "days": len(wide),
        "channels": channels,
        "excluded_rows": len(excluded),
        "negative_days_by_channel": {channel: int((wide[channel] < 0).sum()) for channel in channels},
        "outputs": [f"{channel.lower()}.csv" for channel in channels] + ["total.csv", "excluded_rows_audit.csv"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
