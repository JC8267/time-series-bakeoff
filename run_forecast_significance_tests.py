"""Paired significance checks for overlapping forecast validation windows."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import append_experiment


def normal_two_sided_pvalue(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def infer_metadata(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def comparison_differences(fold_metrics_path: Path, candidate: str, baseline: str) -> pd.DataFrame:
    metrics = pd.read_csv(fold_metrics_path, parse_dates=["fold_cutoff"])
    wide = metrics.pivot_table(index="fold_cutoff", columns="model_name", values="wape").reset_index()
    missing = [name for name in [candidate, baseline] if name not in wide.columns]
    if missing:
        raise ValueError(f"{fold_metrics_path} is missing model(s): {missing}")
    out = wide[["fold_cutoff", candidate, baseline]].copy()
    out["loss_diff"] = out[candidate] - out[baseline]
    return out.sort_values("fold_cutoff").reset_index(drop=True)


def hac_standard_error(values: np.ndarray, max_lag: int) -> tuple[float, float]:
    n = len(values)
    centered = values - values.mean()
    if n < 2:
        return np.nan, np.nan
    max_lag = max(0, min(max_lag, n - 1))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_var += 2.0 * weight * gamma
    long_run_var = max(long_run_var, 0.0)
    se = math.sqrt(long_run_var / n)
    return se, long_run_var


def moving_block_bootstrap(values: np.ndarray, block_length: int, n_bootstrap: int, seed: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.asarray([])
    block_length = max(1, min(block_length, n))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            block = [values[(start + offset) % n] for offset in range(block_length)]
            sampled.extend(block)
        means[i] = float(np.mean(sampled[:n]))
    return means


def summarize_comparison(
    name: str,
    run_dir: Path,
    candidate: str,
    baseline: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    metadata = infer_metadata(run_dir)
    fold_metrics_path = run_dir / "multifold_fold_metrics.csv"
    diffs = comparison_differences(fold_metrics_path, candidate, baseline)
    values = diffs["loss_diff"].to_numpy(dtype=float)
    n = len(values)
    horizon = int(metadata.get("horizon", n))
    grid_step = int(metadata.get("grid_step", metadata.get("grid_step_days", 1)))
    block_length = max(1, math.ceil(horizon / grid_step)) if grid_step else n
    hac_lag = max(0, block_length - 1)
    se, long_run_var = hac_standard_error(values, hac_lag)
    mean_diff = float(values.mean()) if n else np.nan
    z = mean_diff / se if se and np.isfinite(se) and se > 0 else np.nan
    p_value = normal_two_sided_pvalue(z) if np.isfinite(z) else np.nan
    boot = moving_block_bootstrap(values, block_length, n_bootstrap, seed)
    ci_low, ci_high = (np.nan, np.nan)
    if len(boot) and block_length < n:
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    warnings = []
    if block_length >= n:
        warnings.append("block_length_ge_sample_size_bootstrap_ci_degenerate")
    if n < 2 * block_length:
        warnings.append("less_than_two_independent_blocks_approx")
    if n < 10:
        warnings.append("small_window_count")
    row = {
        "comparison": name,
        "run_dir": str(run_dir),
        "candidate": candidate,
        "baseline": baseline,
        "windows": n,
        "horizon": horizon,
        "grid_step": grid_step,
        "block_length": block_length,
        "candidate_wins": int((values < 0).sum()),
        "mean_wape_diff": mean_diff,
        "hac_lag": min(hac_lag, max(n - 1, 0)),
        "hac_se": se,
        "hac_z": z,
        "hac_p_value": p_value,
        "block_bootstrap_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "block_bootstrap_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "long_run_variance": long_run_var,
        "warning": ";".join(warnings),
    }
    diffs.insert(0, "comparison", name)
    return row, diffs


def parse_comparison(text: str) -> tuple[str, Path, str, str]:
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 4:
        raise ValueError("Comparison must be name|run_dir|candidate|baseline")
    return parts[0], Path(parts[1]), parts[2], parts[3]


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Forecast Significance Tests",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Negative mean WAPE difference means the candidate beat the baseline.",
        "",
        "| Comparison | Candidate | Baseline | Windows | Block length | Wins | Mean diff | HAC p-value | Block CI | Warning |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        ci = (
            ""
            if pd.isna(row.block_bootstrap_ci_low)
            else f"[{row.block_bootstrap_ci_low:.5f}, {row.block_bootstrap_ci_high:.5f}]"
        )
        lines.append(
            f"| {row.comparison} | {row.candidate} | {row.baseline} | {row.windows} | {row.block_length} | "
            f"{row.candidate_wins}/{row.windows} | {row.mean_wape_diff:+.5f} | {row.hac_p_value:.3f} | {ci} | {row.warning} |"
        )
    (output_dir / "significance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="Comparison as name|run_dir|candidate|baseline.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs_forecast_significance_tests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    diff_frames = []
    for text in args.comparison:
        name, run_dir, candidate, baseline = parse_comparison(text)
        row, diffs = summarize_comparison(name, run_dir, candidate, baseline, args.n_bootstrap, args.seed)
        rows.append(row)
        diff_frames.append(diffs)
    summary = pd.DataFrame(rows)
    details = pd.concat(diff_frames, ignore_index=True)
    summary.to_csv(output_dir / "significance_summary.csv", index=False)
    details.to_csv(output_dir / "paired_window_differences.csv", index=False)
    write_report(summary, output_dir)

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": args.comparison,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "outputs": ["significance_summary.csv", "paired_window_differences.csv", "significance_report.md"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_experiment(output_dir, "run_forecast_significance_tests.py", manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
