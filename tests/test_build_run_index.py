import json
import subprocess
import sys
from pathlib import Path


def test_build_run_index_discovers_nested_manifests(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs_nested" / "Online"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"horizons": [365], "models": ["example"], "training_cutoff": "2026-07-23"}),
        encoding="utf-8",
    )
    output_path = tmp_path / "RUN_INDEX.md"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "build_run_index.py"),
            "--root",
            str(tmp_path),
            "--output-md",
            str(output_path),
        ],
        check=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "`outputs_nested/Online`" in text
    assert "| 365 | 1 | 2026-07-23 |" in text
