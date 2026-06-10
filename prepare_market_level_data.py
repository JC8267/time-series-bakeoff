"""Prepare the trade-area (sub-market) daily sales panel.

Data semantics (confirmed 2026-06-10): `bu_level_sls.csv` rows are TRADE-AREA
sales combining the physical unit (once open) and e-commerce attributed to the
market area. E-commerce is backfilled for some markets regardless of when the
physical unit opened, so rows before `Opening Date` are an ecomm-only baseline,
not a data error. `complete_opening_info.csv` is an xlsx (despite the
extension) holding the physical-unit metadata: BU number, sqm, unit type,
opening date, relocation date.

Outputs (under --output-dir):
- market_daily_panel.csv: sub_market_no, ds, y, has_physical, open_date,
  days_since_physical_open, pre_opening_ecomm flag, total_sqm, unit_type
- market_metadata.csv: one row per market with coverage and baseline stats
- normalized_openings.csv: unit_id/open_date/sqm/unit_type (load_openings-compatible)
- dropped_rows_audit.csv: future-dated or otherwise excluded rows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_panel(sales_csv: Path, opening_xlsx: Path, max_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(sales_csv, parse_dates=["transaction_date"]).rename(
        columns={"transaction_date": "ds", "sales": "y"}
    )
    sales["y"] = pd.to_numeric(sales["y"], errors="coerce")
    if sales["y"].isna().any():
        raise ValueError("bu-level sales contains non-numeric values.")
    sales = sales.groupby(["sub_market_no", "ds"], as_index=False)["y"].sum()

    dropped = sales[sales["ds"] > max_date].copy()
    dropped["drop_reason"] = f"transaction_date after {max_date.date()}"
    sales = sales[sales["ds"] <= max_date].copy()

    meta = pd.read_excel(opening_xlsx).rename(
        columns={
            "Bu Number": "sub_market_no",
            "Total Sqm": "total_sqm",
            "Unit Type": "unit_type",
            "Opening Date": "open_date",
            "Relocation Date": "relocation_date",
        }
    )
    meta["open_date"] = pd.to_datetime(meta["open_date"])
    meta["relocation_date"] = pd.to_datetime(meta["relocation_date"])
    return sales, meta, dropped


def build_outputs(sales: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = sales.merge(
        meta[["sub_market_no", "open_date", "relocation_date", "total_sqm", "unit_type"]],
        on="sub_market_no",
        how="left",
    )
    panel["has_physical"] = panel["open_date"].notna()
    panel["days_since_physical_open"] = (panel["ds"] - panel["open_date"]).dt.days
    panel["pre_opening_ecomm"] = panel["has_physical"] & (panel["ds"] < panel["open_date"])

    market_rows = []
    missing_rows = []
    for market, g in panel.groupby("sub_market_no"):
        open_date = g["open_date"].iloc[0]
        pre = g[g["pre_opening_ecomm"]]
        # Dates inside a market's own span with no row at all: real closure days
        # are usually recorded as zeros, so absent dates may be extract gaps.
        # Downstream zero-fill (run_market_bakeoff.load_panel) treats them as
        # zero-sales days; this audit makes that assumption inspectable.
        expected = pd.date_range(g["ds"].min(), g["ds"].max(), freq="D")
        missing = expected.difference(g["ds"])
        missing_rows.extend({"sub_market_no": market, "missing_ds": d} for d in missing)
        market_rows.append(
            {
                "sub_market_no": market,
                "has_physical": bool(g["has_physical"].iloc[0]),
                "open_date": open_date.date() if pd.notna(open_date) else None,
                "unit_type": g["unit_type"].iloc[0],
                "total_sqm": g["total_sqm"].iloc[0],
                "first_ds": g["ds"].min().date(),
                "last_ds": g["ds"].max().date(),
                "rows": len(g),
                "missing_days_in_span": int(len(missing)),
                "pre_opening_days": int(len(pre)),
                "pre_opening_median_daily": float(pre["y"].median()) if len(pre) else None,
                "post_opening_days": int((~g["pre_opening_ecomm"]).sum()) if pd.notna(open_date) else None,
            }
        )
    return panel, pd.DataFrame(market_rows), pd.DataFrame(missing_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-csv", default="bu_level_sls.csv")
    parser.add_argument("--opening-info", default="complete_opening_info.csv", help="xlsx network metadata (despite the .csv extension).")
    parser.add_argument("--max-date", default=None, help="Drop rows after this date; defaults to yesterday (the last fully closed business day -- today's rows are usually a partial day).")
    parser.add_argument("--output-dir", default="outputs_market_prep")
    args = parser.parse_args()

    max_date = pd.Timestamp(args.max_date) if args.max_date else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    sales, meta, dropped = load_panel(Path(args.sales_csv), Path(args.opening_info), max_date)
    panel, markets, missing = build_outputs(sales, meta)

    # A final day far below the trailing same-weekday level usually means the
    # extract caught a partial business day; anchor forecasts one day earlier.
    daily_total = panel.groupby("ds")["y"].sum().sort_index()
    last_ds = daily_total.index.max()
    same_dow = daily_total[daily_total.index.dayofweek == last_ds.dayofweek].iloc[-5:-1]
    if len(same_dow) >= 3 and daily_total.iloc[-1] < 0.5 * same_dow.median():
        print(
            f"WARNING [prepare]: last panel day {last_ds.date()} totals less than half its trailing "
            f"same-weekday median -- likely a partial day. Re-run with --max-date {(last_ds - pd.Timedelta(days=1)).date()}.",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "market_daily_panel.csv", index=False)
    markets.to_csv(output_dir / "market_metadata.csv", index=False)
    dropped.to_csv(output_dir / "dropped_rows_audit.csv", index=False)
    missing.to_csv(output_dir / "missing_dates_audit.csv", index=False)
    print(f"missing market-dates inside spans: {len(missing)} (see missing_dates_audit.csv)")
    openings_out = meta[meta["open_date"].notna()][["sub_market_no", "open_date", "total_sqm", "unit_type"]].rename(
        columns={"sub_market_no": "unit_id"}
    )
    openings_out.to_csv(output_dir / "normalized_openings.csv", index=False)

    print(f"panel rows: {len(panel)} | markets: {markets.shape[0]} | dropped rows: {len(dropped)}")
    with_physical = markets[markets["has_physical"]]
    print(f"markets with physical unit: {len(with_physical)} | ecomm-only/unknown: {len(markets) - len(with_physical)}")
    backfilled = with_physical[with_physical["pre_opening_days"] >= 56]
    print(f"markets with >=56 days pre-opening ecomm baseline: {len(backfilled)}")
    print(backfilled[["sub_market_no", "open_date", "unit_type", "total_sqm", "pre_opening_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
