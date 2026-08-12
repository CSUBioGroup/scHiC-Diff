#!/usr/bin/env python3
"""Audit target and trial SuperTAD TSVs without requiring window overlap."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
import config as cfg  # noqa: E402


AUDIT_FIELDS = [
    "kind",
    "cell_type",
    "trial_id",
    "path",
    "status",
    "valid",
    "n_lines",
    "n_domains",
    "n_invalid",
    "n_drawable",
    "n_enclosing",
    "n_outside",
]

SUMMARY_FIELDS = [
    "kind",
    "cell_type",
    "n_expected",
    "n_valid",
    "n_invalid",
    "n_drawable_outputs",
    "n_no_boundary_in_window",
    "complete",
    "all_valid",
]


def audit_tad_file(
    path: Path,
    n_bins: int,
    window: tuple[int, int],
) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "status": "missing",
        "valid": False,
        "n_lines": 0,
        "n_domains": 0,
        "n_invalid": 0,
        "n_drawable": 0,
        "n_enclosing": 0,
        "n_outside": 0,
    }
    if not path.is_file():
        return result

    with path.open() as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    result["n_lines"] = len(lines)
    if not lines:
        result["status"] = "empty"
        return result

    domains: list[tuple[int, int]] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 8:
            result["n_invalid"] += 1
            continue
        try:
            start_one_based = int(parts[1])
            end_one_based = int(parts[5])
        except ValueError:
            result["n_invalid"] += 1
            continue
        if not (
            1 <= start_one_based <= end_one_based <= int(n_bins)
        ):
            result["n_invalid"] += 1
            continue
        domains.append((start_one_based - 1, end_one_based - 1))

    result["n_domains"] = len(domains)
    if result["n_invalid"] or not domains:
        result["status"] = "invalid"
        return result

    start, end = window
    if start < 0 or end > n_bins or start >= end:
        raise ValueError(f"invalid window {window} for {n_bins} bins")
    for domain_start, domain_end in domains:
        if domain_start < start and domain_end > end - 1:
            result["n_enclosing"] += 1
            continue
        clipped_start = max(domain_start, start)
        clipped_end = min(domain_end, end - 1)
        if clipped_end - clipped_start >= 1:
            result["n_drawable"] += 1
        else:
            result["n_outside"] += 1

    result["valid"] = True
    result["status"] = (
        "valid_drawable"
        if result["n_drawable"] > 0
        else "valid_no_boundary_in_window"
    )
    return result


def summarize_rows(
    rows: list[dict[str, Any]],
    kind: str,
    cell_type: str,
    n_expected: int,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["kind"] == kind and row["cell_type"] == cell_type
    ]
    n_valid = sum(bool(row["valid"]) for row in selected)
    n_drawable = sum(row["status"] == "valid_drawable" for row in selected)
    n_no_boundary = sum(
        row["status"] == "valid_no_boundary_in_window" for row in selected
    )
    complete = len(selected) == n_expected
    return {
        "kind": kind,
        "cell_type": cell_type,
        "n_expected": n_expected,
        "n_valid": n_valid,
        "n_invalid": len(selected) - n_valid,
        "n_drawable_outputs": n_drawable,
        "n_no_boundary_in_window": n_no_boundary,
        "complete": complete,
        "all_valid": bool(complete and n_valid == n_expected),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="FLAMINGO_fixed_contact")
    parser.add_argument("--supertad-root", type=Path, default=Path(cfg.SUPERTAD_DIR))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cell-types", nargs="+", default=cfg.CELL_TYPES)
    parser.add_argument("--n-trials", type=int, default=cfg.N_TRIALS)
    parser.add_argument("--n-bins", type=int, default=cfg.N_BINS)
    parser.add_argument("--window-start", type=int, default=cfg.PDGFRA_SUB_BINS[0])
    parser.add_argument("--window-end", type=int, default=cfg.PDGFRA_SUB_BINS[1])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    method_dir = args.supertad_root / args.method
    output_dir = args.output_dir or (method_dir / "audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = (args.window_start, args.window_end)
    rows: list[dict[str, Any]] = []

    for cell_type in args.cell_types:
        target_path = args.supertad_root / "target" / f"{cell_type}_target.tsv"
        rows.append({
            "kind": "target",
            "cell_type": cell_type,
            "trial_id": "",
            **audit_tad_file(target_path, args.n_bins, window),
        })
        for trial_id in range(args.n_trials):
            trial_path = (
                method_dir / "trials" / f"{cell_type}_trial{trial_id:03d}.tsv"
            )
            rows.append({
                "kind": "trial",
                "cell_type": cell_type,
                "trial_id": trial_id,
                **audit_tad_file(trial_path, args.n_bins, window),
            })

    summary_rows: list[dict[str, Any]] = []
    for cell_type in args.cell_types:
        summary_rows.append(summarize_rows(rows, "target", cell_type, 1))
        summary_rows.append(
            summarize_rows(rows, "trial", cell_type, args.n_trials)
        )

    overall_pass = all(row["all_valid"] for row in summary_rows)
    status_counts = Counter(row["status"] for row in rows)
    write_csv(output_dir / "tad_output_audit.csv", rows, AUDIT_FIELDS)
    write_csv(
        output_dir / "tad_output_audit_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    provenance = {
        "method": args.method,
        "supertad_root": str(args.supertad_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "cell_types": list(args.cell_types),
        "n_trials_per_cell_type": int(args.n_trials),
        "n_bins": int(args.n_bins),
        "window": [int(args.window_start), int(args.window_end)],
        "status_counts": dict(status_counts),
        "overall_pass": bool(overall_pass),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "tad_output_audit_provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)

    for row in summary_rows:
        print(
            f"{row['kind']:6s} {row['cell_type']:5s}: "
            f"valid={row['n_valid']}/{row['n_expected']}, "
            f"drawable={row['n_drawable_outputs']}, "
            f"no-window-boundary={row['n_no_boundary_in_window']}"
        )
    print(f"Status counts: {dict(status_counts)}")
    print(f"Audit outputs: {output_dir}")
    if not overall_pass:
        raise SystemExit("SuperTAD output audit failed")


if __name__ == "__main__":
    main()
