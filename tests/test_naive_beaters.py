from __future__ import annotations

import pandas as pd
import pytest

from explore_naive_beaters import BENCHMARK_MODEL, CHALLENGER_MODEL, challenger_scorecard


def test_challenger_scorecard_separates_preregistered_holdouts() -> None:
    metrics = pd.DataFrame(
        [
            {"horizon": 28, "cutoff": pd.Timestamp("2026-03-04"), "model_name": BENCHMARK_MODEL, "wape": 0.10, "bias_pct": 0.01},
            {"horizon": 28, "cutoff": pd.Timestamp("2026-03-04"), "model_name": CHALLENGER_MODEL, "wape": 0.09, "bias_pct": 0.00},
            {"horizon": 28, "cutoff": pd.Timestamp("2026-04-11"), "model_name": BENCHMARK_MODEL, "wape": 0.08, "bias_pct": 0.01},
            {"horizon": 28, "cutoff": pd.Timestamp("2026-04-11"), "model_name": CHALLENGER_MODEL, "wape": 0.07, "bias_pct": 0.00},
        ]
    )

    scorecard = challenger_scorecard(metrics, pd.Timestamp("2026-04-01")).set_index("evidence_split")

    assert scorecard.loc["exploration", "windows"] == 1
    assert scorecard.loc["holdout", "windows"] == 1
    assert scorecard.loc["holdout", "wins_vs_naive"] == 1
    assert scorecard.loc["holdout", "mean_wape_diff"] == pytest.approx(-0.01)
