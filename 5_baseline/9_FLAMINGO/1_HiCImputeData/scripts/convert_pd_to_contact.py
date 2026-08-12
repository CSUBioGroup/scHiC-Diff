#!/usr/bin/env python3
"""Convert completed PD/distance tensors back to IF/contact tensors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = BASE_DIR / "input_distance"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output_distance"
DEFAULT_CONTACT_ROOT = DEFAULT_OUTPUT_ROOT / "contact_from_pd"
DEFAULT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.tsv"
ALPHA = 0.25


def pd_to_contact(distance: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    contact = np.zeros(distance.shape, dtype=np.float64)
    mask = np.isfinite(distance) & (distance > 0)
    contact[mask] = np.power(distance[mask], -1.0 / alpha)
    contact[~np.isfinite(contact)] = 0.0
    contact[contact < 0] = 0.0
    return contact


def tensor_to_feature_matrix(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError(f"Tensor must be cells x beads x beads, got {tensor.shape}")
    tril_i, tril_j = np.tril_indices(tensor.shape[1], k=-1)
    return tensor[:, tril_i, tril_j]


def read_dataset_ids(manifest: Path, output_root: Path) -> list[str]:
    if manifest.exists():
        with manifest.open(newline="") as handle:
            return [row["dataset_id"] for row in csv.DictReader(handle, delimiter="\t")]
    return sorted(
        p.name
        for p in output_root.iterdir()
        if p.is_dir() and (p / "completed_tensor.npy").exists()
    )


def _write_cell_matrices(matrix_dir: Path, tensor: np.ndarray, prefix: str) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for idx, matrix in enumerate(tensor, start=1):
        np.savetxt(matrix_dir / f"{prefix}_Cell_{idx:03d}.txt", matrix, fmt="%.10g", delimiter="\t")


def convert_one(dataset_id: str, input_root: Path, output_root: Path, contact_root: Path, write_txt: bool) -> dict[str, object]:
    dataset_output = output_root / dataset_id
    completed_path = dataset_output / "completed_tensor.npy"
    if not completed_path.exists():
        raise FileNotFoundError(f"Missing completed tensor: {completed_path}")

    dataset_contact = contact_root / dataset_id
    dataset_contact.mkdir(parents=True, exist_ok=True)
    completed_pd = np.real(np.load(completed_path)).astype(np.float64)
    completed_contact = pd_to_contact(completed_pd)
    np.save(dataset_contact / "completed_contact_tensor.npy", completed_contact)
    np.save(dataset_contact / "high_resolution_contact.npy", completed_contact)

    npz_dir = contact_root / "npz_lower_tri"
    npz_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(
        npz_dir / f"{dataset_id}_flamingo_contact_lower_tri.npz",
        sparse.csr_matrix(tensor_to_feature_matrix(completed_contact)),
    )

    dataset_input = input_root / dataset_id
    for stem in ["observed_distance_tensor", "truth_distance_tensor"]:
        src = dataset_input / f"{stem}.npy"
        if src.exists():
            contact = pd_to_contact(np.load(src).astype(np.float64))
            contact_name = stem.replace("distance", "contact")
            np.save(dataset_contact / f"{contact_name}.npy", contact)
            sparse.save_npz(
                npz_dir / f"{dataset_id}_{contact_name}_lower_tri.npz",
                sparse.csr_matrix(tensor_to_feature_matrix(contact)),
            )

    if write_txt:
        _write_cell_matrices(dataset_contact / "completed_contact_matrices", completed_contact, "Contact")

    positive = completed_contact[np.isfinite(completed_contact) & (completed_contact > 0)]
    summary = {
        "dataset_id": dataset_id,
        "n_cells": int(completed_contact.shape[0]),
        "n_beads": int(completed_contact.shape[1]),
        "nnz": int(np.count_nonzero(completed_contact)),
        "min_positive": float(positive.min()) if positive.size else float("nan"),
        "max": float(np.nanmax(completed_contact)),
        "mean_positive": float(positive.mean()) if positive.size else float("nan"),
        "source": str(completed_path),
    }
    with (dataset_contact / "conversion_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def convert_all(input_root: Path, output_root: Path, contact_root: Path, manifest: Path, write_txt: bool) -> None:
    dataset_ids = read_dataset_ids(manifest, output_root)
    summaries = []
    for dataset_id in dataset_ids:
        summaries.append(convert_one(dataset_id, input_root, output_root, contact_root, write_txt=write_txt))
    pd.DataFrame(summaries).to_csv(contact_root / "conversion_summary.csv", index=False)
    print(f"Converted {len(summaries)} datasets to {contact_root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contact-root", type=Path, default=DEFAULT_CONTACT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-txt", action="store_true", help="Skip per-cell text matrix export")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    convert_all(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        contact_root=args.contact_root.resolve(),
        manifest=args.manifest.resolve(),
        write_txt=not args.no_txt,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
