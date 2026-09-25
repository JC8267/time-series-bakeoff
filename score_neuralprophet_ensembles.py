"""Score frozen NeuralProphet blends from completed same-window predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_new_model_screen import sha256_file


def score(predictions: pd.DataFrame, registration: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["horizon", "cutoff", "ds", "actual"]
    wide = predictions.groupby(keys, as_index=False).first()
    if len(wide) * 2 != len(predictions) or wide.isna().any().any():
        raise ValueError("Both NeuralProphet configurations must cover the same complete windows.")
    formulas = {"seasonal_naive_lag_364": {"seasonal_naive_lag_364": 1.0},
                "blend_snl75_tfm25": {"seasonal_naive_lag_364": 0.75, "timesfm_2_5": 0.25}}
    formulas.update(registration["formulas"])
    for name, weights in formulas.items():
        if abs(sum(weights.values()) - 1.0) > 1e-12:
            raise ValueError(f"Weights do not sum to one: {name}")
        wide[name] = sum(weight * wide[component] for component, weight in weights.items())
    rows = []
    for (horizon, cutoff), window in wide.groupby(["horizon", "cutoff"]):
        denom = window["actual"].abs().sum()
        for name in formulas:
            error = window[name] - window["actual"]
            rows.append({"horizon": horizon, "cutoff": cutoff, "model_name": name,
                         "wape": error.abs().sum() / denom, "bias_pct": error.sum() / denom})
    folds = pd.DataFrame(rows)
    summary = []
    for horizon, window in folds.groupby("horizon"):
        wape = window.pivot(index="cutoff", columns="model_name", values="wape")
        bias = window.groupby("model_name")["bias_pct"].mean()
        for name in formulas:
            delta = wape[name] - wape["seasonal_naive_lag_364"]
            summary.append({"horizon": horizon, "model_name": name, "windows": len(wape),
                            "mean_wape": wape[name].mean(), "mean_bias_pct": bias[name],
                            "wins_vs_naive": int((delta < 0).sum()), "worst_wape_diff_vs_naive": delta.max()})
    return folds, pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", type=Path, default=Path("outputs_neuralprophet_screen_sep2026"))
    parser.add_argument("--registration", type=Path, default=Path("neuralprophet_ensemble_registration.json"))
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    if registration["status"] != "frozen_before_scoring" or sha256_file(Path(registration["source_screen"])) != registration["source_screen_sha256"]:
        raise ValueError("Frozen source registration does not match.")
    manifest = json.loads((args.screen_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["registration_sha256"] != registration["source_screen_sha256"] or any(
        status["status"] != "completed" for status in manifest["model_status"].values()
    ):
        raise ValueError("Both registered configurations must complete before ensemble scoring.")
    predictions = pd.read_csv(args.screen_dir / "predictions.csv", parse_dates=["cutoff", "ds"])
    folds, summary = score(predictions, registration)
    folds.to_csv(args.screen_dir / "ensemble_fold_metrics.csv", index=False)
    summary.to_csv(args.screen_dir / "ensemble_scorecard.csv", index=False)
    (args.screen_dir / "ensemble_manifest.json").write_text(json.dumps({
        "source_run_id": manifest["run_id"], "registration": str(args.registration),
        "registration_sha256": sha256_file(args.registration), "source_registration_sha256": manifest["registration_sha256"],
        "artifacts": ["ensemble_fold_metrics.csv", "ensemble_scorecard.csv"]
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
