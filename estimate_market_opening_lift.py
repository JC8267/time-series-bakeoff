"""Within-market opening lift from the trade-area panel.

For markets whose pre-opening rows are backfilled e-commerce (see
prepare_market_level_data.py), the physical-opening lift is identifiable
inside the market: the counterfactual is the market's own pre-opening ecomm
level, projected forward with a control index built from stable markets
(physical unit open at least a year before the subject's opening). This is
the identification the chain-level event study (estimate_opening_lift_event_
study.py) could not achieve.

    expected(t) = pre_baseline_rate * control_index(t)
    lift(t)     = actual(t) - expected(t)

Note: the lift bundles the physical unit's sales with any change in the
market's ecomm (halo or cannibalization); the panel cannot split the two.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BANDS = [(0, 30), (31, 90), (91, 180), (181, 365), (366, 10_000)]
MIN_PRE_DAYS = 56
BASELINE_GAP_DAYS = 14  # skip the fortnight before opening (pre-launch noise)


def control_series(panel: pd.DataFrame, subject_open: pd.Timestamp) -> pd.Series:
    stable = panel[
        panel["has_physical"]
        & (panel["open_date"] <= subject_open - pd.Timedelta(days=365))
    ]
    return stable.groupby("ds")["y"].sum()


def estimate(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, g in panel[panel["has_physical"]].groupby("sub_market_no"):
        g = g.sort_values("ds")
        open_date = g["open_date"].iloc[0]
        pre = g[(g["ds"] < open_date - pd.Timedelta(days=BASELINE_GAP_DAYS))]
        if len(pre) < MIN_PRE_DAYS or open_date > g["ds"].max():
            continue
        base_start = max(pre["ds"].min(), open_date - pd.Timedelta(days=364))
        base = pre[pre["ds"] >= base_start]
        control = control_series(panel, open_date)
        base_control = control.loc[base["ds"].min() : base["ds"].max()]
        if base["y"].sum() <= 0 or base_control.sum() <= 0:
            continue
        scale = base["y"].sum() / base_control.sum()

        post = g[g["ds"] >= open_date].set_index("ds")
        expected = (control * scale).reindex(post.index)
        for band_start, band_end in BANDS:
            ws = open_date + pd.Timedelta(days=band_start)
            we = open_date + pd.Timedelta(days=band_end)
            window = post.loc[ws:we]
            if len(window) < (band_end - band_start + 1) * 0.9 and band_end < 10_000:
                continue
            if window.empty:
                continue
            exp_sum = expected.loc[window.index].sum()
            act_sum = window["y"].sum()
            rows.append(
                {
                    "sub_market_no": market,
                    "open_date": open_date.date(),
                    "unit_type": g["unit_type"].iloc[0],
                    "total_sqm": g["total_sqm"].iloc[0],
                    "band": f"{band_start}-{band_end if band_end < 10_000 else 'end'}",
                    "days": len(window),
                    "baseline_days": len(base),
                    "expected_ecomm_sum": exp_sum,
                    "actual_sum": act_sum,
                    "lift_abs": act_sum - exp_sum,
                    "lift_multiple_of_ecomm": act_sum / exp_sum - 1 if exp_sum else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="outputs_market_prep/market_daily_panel.csv")
    parser.add_argument("--output-dir", default="outputs_market_opening_lift")
    args = parser.parse_args()

    panel = pd.read_csv(Path(args.panel), parse_dates=["ds", "open_date"])
    result = estimate(panel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "market_opening_lift.csv", index=False)
    display = result[["sub_market_no", "open_date", "unit_type", "total_sqm", "band", "days", "lift_multiple_of_ecomm"]].copy()
    display["lift_multiple_of_ecomm"] = display["lift_multiple_of_ecomm"].round(2)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
