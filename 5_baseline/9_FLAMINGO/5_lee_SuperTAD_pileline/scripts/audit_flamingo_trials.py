#!/usr/bin/env python3
"""Audit every corrected FLAMINGO pseudo-bulk trial before SuperTAD."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import config as cfg  # noqa: E402
from evaluate_flamingo_contact_maps import (  # noqa: E402
    distance_profile_spearman,
    full_matrix_pcc,
)


AUDIT_FIELDS = [
    "trial_id",
    "cell_type",
    "matrix_path",
    "rho_distance",
    "pcc_49x49",
    "pcc_8x8",
    "recorded_pcc_49x49",
    "recorded_pcc_8x8",
    "pcc_csv_match",
    "finite",
    "symmetric",
    "nonnegative",
    "gate_pass",
]

SUMMARY_FIELDS = [
    "cell_type",
    "n_trials",
    "n_unique_trials",
    "n_pass",
    "n_fail",
    "rho_min",
    "rho_max",
    "pcc_49x49_min",
    "pcc_49x49_max",
    "pcc_8x8_min",
    "pcc_8x8_max",
    "complete",
    "all_pass",
]


def audit_trial_matrix(
    matrix: np.ndarray,
    target: np.ndarray,
    subregion: tuple[int, int] = cfg.PDGFRA_SUB_BINS,
) -> dict[str, Any]:
    predicted = np.asarray(matrix, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[0] != predicted.shape[1]:
        raise ValueError(f"trial matrix must be square, got {predicted.shape}")
    if expected.shape != predicted.shape:
        raise ValueError(
            f"target shape mismatch: {expected.shape} != {predicted.shape}"
        )

    start, end = subregion
    if start < 0 or end > predicted.shape[0] or start >= end:
        raise ValueError(f"invalid subregion {subregion} for {predicted.shape}")

    rho_distance = distance_profile_spearman(predicted)
    pcc_49x49 = full_matrix_pcc(predicted, expected)
    pcc_8x8 = full_matrix_pcc(
        predicted[start:end, start:end],
        expected[start:end, start:end],
    )
    finite = bool(np.all(np.isfinite(predicted)))
    symmetric = bool(np.allclose(predicted, predicted.T))
    nonnegative = bool(np.all(predicted >= 0.0))
    gate_pass = bool(
        np.isfinite(rho_distance)
        and np.isfinite(pcc_49x49)
        and np.isfinite(pcc_8x8)
        and rho_distance < 0.0
        and pcc_49x49 > 0.0
        and pcc_8x8 > 0.0
        and finite
        and symmetric
        and nonnegative
    )
    return {
        "rho_distance": float(rho_distance),
        "pcc_49x49": float(pcc_49x49),
        "pcc_8x8": float(pcc_8x8),
        "finite": finite,
        "symmetric": symmetric,
        "nonnegative": nonnegative,
        "gate_pass": gate_pass,
    }


def _finite_extreme(rows: list[dict[str, Any]], field: str, reducer) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(reducer(finite)) if finite.size else float("nan")


def summarize_cell_type(
    rows: list[dict[str, Any]],
    expected_trial_ids: set[int],
) -> dict[str, Any]:
    trial_ids = [int(row["trial_id"]) for row in rows]
    unique_ids = set(trial_ids)
    complete = bool(
        len(rows) == len(expected_trial_ids)
        and len(unique_ids) == len(trial_ids)
        and unique_ids == expected_trial_ids
    )
    n_pass = sum(bool(row["gate_pass"]) for row in rows)
    return {
        "n_trials": len(rows),
        "n_unique_trials": len(unique_ids),
        "n_pass": n_pass,
        "n_fail": len(rows) - n_pass,
        "rho_min": _finite_extreme(rows, "rho_distance", np.min),
        "rho_max": _finite_extreme(rows, "rho_distance", np.max),
        "pcc_49x49_min": _finite_extreme(rows, "pcc_49x49", np.min),
        "pcc_49x49_max": _finite_extreme(rows, "pcc_49x49", np.max),
        "pcc_8x8_min": _finite_extreme(rows, "pcc_8x8", np.min),
        "pcc_8x8_max": _finite_extreme(rows, "pcc_8x8", np.max),
        "complete": complete,
        "all_pass": bool(complete and n_pass == len(rows)),
    }


def load_recorded_pcc(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    recorded: dict[tuple[str, int], tuple[float, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["cell_type"], int(row["trial_id"]))
            if key in recorded:
                raise ValueError(f"duplicate PCC row for {key}")
            recorded[key] = (
                float(row["pcc_49x49_full"]),
                float(row["pcc_8x8_full"]),
            )
    return recorded


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="FLAMINGO_fixed_contact")
    parser.add_argument("--target-dir", type=Path, default=Path(cfg.TARGET_DIR))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cell-types", nargs="+", default=cfg.CELL_TYPES)
    parser.add_argument("--n-trials", type=int, default=cfg.N_TRIALS)
    parser.add_argument("--subregion-start", type=int, default=cfg.PDGFRA_SUB_BINS[0])
    parser.add_argument("--subregion-end", type=int, default=cfg.PDGFRA_SUB_BINS[1])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    trials_dir = Path(cfg.TRIALS_DIR) / args.method
    matrices_dir = trials_dir / "matrices"
    pcc_path = trials_dir / "pcc_results.csv"
    output_dir = args.output_dir or (trials_dir / "audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    recorded = load_recorded_pcc(pcc_path)
    expected_trial_ids = set(range(args.n_trials))
    audit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for cell_type in args.cell_types:
        target_path = args.target_dir / f"{cell_type}_target.npz"
        target = sparse.load_npz(target_path).toarray().astype(np.float64)
        cell_rows: list[dict[str, Any]] = []

        for trial_id in range(args.n_trials):
            matrix_path = matrices_dir / f"{cell_type}_trial{trial_id:03d}.npz"
            if not matrix_path.is_file():
                raise FileNotFoundError(f"missing trial matrix: {matrix_path}")
            key = (cell_type, trial_id)
            if key not in recorded:
                raise KeyError(f"missing PCC result for {key}")

            matrix = sparse.load_npz(matrix_path).toarray().astype(np.float64)
            metrics = audit_trial_matrix(
                matrix,
                target,
                subregion=(args.subregion_start, args.subregion_end),
            )
            recorded_49, recorded_8 = recorded[key]
            pcc_csv_match = bool(
                np.isclose(metrics["pcc_49x49"], recorded_49, atol=1e-6, rtol=0.0)
                and np.isclose(metrics["pcc_8x8"], recorded_8, atol=1e-6, rtol=0.0)
            )
            row = {
                "trial_id": trial_id,
                "cell_type": cell_type,
                "matrix_path": str(matrix_path.resolve()),
                **metrics,
                "recorded_pcc_49x49": recorded_49,
                "recorded_pcc_8x8": recorded_8,
                "pcc_csv_match": pcc_csv_match,
            }
            row["gate_pass"] = bool(row["gate_pass"] and pcc_csv_match)
            cell_rows.append(row)
            audit_rows.append(row)

        summary = summarize_cell_type(cell_rows, expected_trial_ids)
        summary_rows.append({"cell_type": cell_type, **summary})

    expected_keys = {
        (cell_type, trial_id)
        for cell_type in args.cell_types
        for trial_id in range(args.n_trials)
    }
    unexpected_pcc_rows = sorted(set(recorded) - expected_keys)
    overall_pass = bool(
        not unexpected_pcc_rows
        and len(recorded) == len(expected_keys)
        and all(row["all_pass"] for row in summary_rows)
    )

    write_csv(output_dir / "trial_audit.csv", audit_rows, AUDIT_FIELDS)
    write_csv(output_dir / "trial_audit_summary.csv", summary_rows, SUMMARY_FIELDS)
    provenance = {
        "method": args.method,
        "trials_dir": str(trials_dir.resolve()),
        "matrices_dir": str(matrices_dir.resolve()),
        "pcc_results": str(pcc_path.resolve()),
        "target_dir": str(args.target_dir.resolve()),
        "cell_types": list(args.cell_types),
        "n_trials_per_cell_type": int(args.n_trials),
        "subregion": [int(args.subregion_start), int(args.subregion_end)],
        "unexpected_pcc_rows": [list(key) for key in unexpected_pcc_rows],
        "overall_pass": overall_pass,
        "gate": "rho_distance < 0 and PCC49 > 0 and PCC8 > 0 and finite and symmetric and nonnegative and PCC CSV match",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "trial_audit_provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)

    for row in summary_rows:
        print(
            f"{row['cell_type']}: {row['n_pass']}/{row['n_trials']} pass; "
            f"rho=[{row['rho_min']:.4f}, {row['rho_max']:.4f}], "
            f"PCC49=[{row['pcc_49x49_min']:.4f}, {row['pcc_49x49_max']:.4f}], "
            f"PCC8=[{row['pcc_8x8_min']:.4f}, {row['pcc_8x8_max']:.4f}]"
        )
    print(f"Audit outputs: {output_dir}")
    if not overall_pass:
        raise SystemExit("FLAMINGO trial audit failed")


if __name__ == "__main__":
    main()
