"""Lightweight experiment registry helpers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_COLUMNS = [
    "registered_at",
    "run_id",
    "output_dir",
    "script",
    "created_at",
    "horizons",
    "models",
    "headline",
]


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def append_experiment(output_dir: Path, script: str, manifest: dict[str, Any], headline: str = "", registry_path: Path = Path("experiments.csv")) -> None:
    row = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_id": compact(manifest.get("run_id") or manifest.get("created_at") or output_dir.name),
        "output_dir": str(output_dir),
        "script": script,
        "created_at": compact(manifest.get("created_at")),
        "horizons": compact(manifest.get("horizons")),
        "models": compact(manifest.get("models")),
        "headline": headline,
    }
    exists = registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

