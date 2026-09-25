"""Build a markdown index of output directories with manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest(output_dir: Path) -> dict:
    for name in ["manifest.json", "phase6_manifest.json", "phase5_manifest.json"]:
        path = output_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-md", default="RUN_INDEX.md")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for output_root in sorted(root.glob("outputs*")):
        if not output_root.is_dir():
            continue
        output_dirs = [output_root]
        if not load_manifest(output_root):
            output_dirs = sorted(
                {path.parent for name in ["manifest.json", "phase6_manifest.json", "phase5_manifest.json"] for path in output_root.rglob(name)}
            )
        for output_dir in output_dirs:
            manifest = load_manifest(output_dir)
            rows.append(
                {
                    "dir": output_dir.relative_to(root).as_posix(),
                    "created_at": manifest.get("created_at", ""),
                    "horizons": ",".join(str(h) for h in manifest.get("horizons", [])) if isinstance(manifest.get("horizons"), list) else "",
                    "models": len(manifest.get("models", [])) if isinstance(manifest.get("models"), list) else "",
                    "training_cutoff": manifest.get("training_cutoff", ""),
                }
            )

    lines = [
        "# Run Index",
        "",
        "Generated from output-directory manifests.",
        "",
        "| Output dir | Created at | Horizons | Model count | Training cutoff |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['dir']}` | {row['created_at']} | {row['horizons']} | {row['models']} | {row['training_cutoff']} |")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_md} with {len(rows)} runs.")


if __name__ == "__main__":
    main()
