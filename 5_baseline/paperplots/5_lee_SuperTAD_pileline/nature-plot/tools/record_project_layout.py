"""Write a deterministic, portable SHA-256 inventory for nature-plot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_NAMES = {".DS_Store"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(project=PROJECT, exclude=None):
    project = Path(project).resolve()
    excluded = {Path(path).resolve() for path in (exclude or ())}
    records = []
    for path in sorted(project.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        relative = path.relative_to(project)
        if relative.name in EXCLUDED_NAMES or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def write_inventory(output, project=PROJECT):
    output = Path(output)
    if not output.is_absolute():
        output = Path(project) / output
    records = inventory(project, exclude=(output,))
    payload = {
        "schema_version": 1,
        "project_root": ".",
        "file_count": len(records),
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    payload = write_inventory(args.output)
    print(f"Recorded {payload['file_count']} files in {args.output}.")


if __name__ == "__main__":
    main()
