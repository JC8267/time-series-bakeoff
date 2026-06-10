"""Create adjusted opening-info files for forecast scenarios."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_bakeoff import load_openings


def original_date_column(path: Path) -> str:
    cols = pd.read_csv(path, nrows=0).columns
    for name in ["Opening_date", "opening_date", "open_date"]:
        if name in cols:
            return name
    return "open_date"


def build_scenario(openings_path: Path, mode: str, start: str | None, end: str | None) -> pd.DataFrame:
    raw = pd.read_csv(openings_path)
    date_col = original_date_column(openings_path)
    raw[date_col] = pd.to_datetime(raw[date_col])

    if mode == "as_is":
        scenario = raw.copy()
    elif mode == "remove_between":
        if not start or not end:
            raise ValueError("--start-date and --end-date are required for remove_between.")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        scenario = raw[~raw[date_col].between(start_ts, end_ts)].copy()
    elif mode == "remove_on_or_after":
        if not start:
            raise ValueError("--start-date is required for remove_on_or_after.")
        scenario = raw[raw[date_col] < pd.Timestamp(start)].copy()
    else:
        raise ValueError(f"Unsupported scenario mode: {mode}")

    scenario[date_col] = scenario[date_col].dt.strftime("%Y-%m-%d")
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening-info", default="opening_info.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--mode", choices=["as_is", "remove_between", "remove_on_or_after"], required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    openings_path = Path(args.opening_info)
    scenario = build_scenario(openings_path, args.mode, args.start_date, args.end_date)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario.to_csv(output_path, index=False)

    loaded = load_openings(output_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_opening_info": str(openings_path),
        "output_csv": str(output_path),
        "mode": args.mode,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "opening_count": int(len(loaded)),
        "opening_dates": [str(d.date()) for d in loaded["open_date"]],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

