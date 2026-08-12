#!/usr/bin/env python3
"""Convenience entry point for the isolated FLAMINGO evaluation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def discover_experiment_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir.parent,
            script_dir.parent / "2_callLoop_apa",
            script_dir.parent.parent / "2_callLoop_apa",
            Path.cwd().resolve(),
        ]
    relative = Path("4_test_corrected_benchmark/scripts/evaluate_flamingo.py")
    for candidate in candidates:
        if (candidate / relative).is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate evaluate_flamingo.py; checked {[str(path) for path in candidates]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument(
        "--stage",
        choices=["preflight", "loops", "apa", "heldout", "support", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = discover_experiment_root(args.experiment_root)
    evaluator = root / "4_test_corrected_benchmark/scripts/evaluate_flamingo.py"
    command = [sys.executable, str(evaluator), "--stage", args.stage]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

