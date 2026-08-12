#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


N_BINS = 61
N_FEATURES = N_BINS * (N_BINS - 1) // 2
N_CELLS = 100

DEFAULT_RAW_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "0_gtData/0_downsampled_HiCImputeData"
)
DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "3_HiCImpute/1_HiCImputeData/input"
)
DATASETS = [
    "K562_T1_1k", "K562_T1_2k", "K562_T1_4k", "K562_T1_7k",
    "K562_T2_1k", "K562_T2_2k", "K562_T2_4k", "K562_T2_7k",
    "K562_T3_1k", "K562_T3_2k", "K562_T3_4k", "K562_T3_7k",
]
CELL_RE = re.compile(r"cell_(\d+)_chr19\.txt$")


def r_colmajor_permutation(n: int) -> np.ndarray:
    iu, ju = np.triu_indices(n, k=1)
    return np.lexsort((iu, ju)).astype(np.int64)


def cell_sort_key(path: Path) -> int:
    match = CELL_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected cell filename: {path}")
    return int(match.group(1))


def load_upper_tri_vector_from_triplets(path: Path, n_beads: int) -> np.ndarray:
    iu, ju = np.triu_indices(n_beads, k=1)
    if path.stat().st_size == 0:
        return np.zeros(len(iu), dtype=np.float64)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 3)
    bin_a = data[:, 0].astype(np.int64)
    bin_b = data[:, 1].astype(np.int64)
    counts = data[:, 2].astype(np.float64)
    lo = np.minimum(bin_a, bin_b)
    hi = np.maximum(bin_a, bin_b)
    matrix = np.zeros((n_beads, n_beads), dtype=np.float64)
    np.add.at(matrix, (lo, hi), counts)
    matrix = matrix + matrix.T
    return matrix[iu, ju]


def load_cells_matrix(cell_files: list[Path], n_beads: int) -> np.ndarray:
    rows = np.empty((len(cell_files), N_FEATURES), dtype=np.float64)
    for idx, cell_file in enumerate(cell_files):
        rows[idx] = load_upper_tri_vector_from_triplets(cell_file, n_beads)
        if (idx + 1) % 20 == 0 or idx + 1 == len(cell_files):
            print(f"  loaded {idx + 1}/{len(cell_files)} cells", flush=True)
    return rows


def write_fortran_bin(path: Path, cells_by_features: np.ndarray) -> None:
    features_by_cells = np.asarray(cells_by_features, dtype="<f8").T
    path.parent.mkdir(parents=True, exist_ok=True)
    features_by_cells.ravel(order="F").tofile(path)


def write_names(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def prepare_dataset(dataset: str, raw_root: Path, input_root: Path, overwrite: bool) -> None:
    out_dir = input_root / dataset
    marker = out_dir / ".complete"
    if marker.exists() and not overwrite:
        print(f"[prepare] {dataset}: skip (.complete exists)", flush=True)
        return

    source_dir = raw_root / dataset
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    cell_files = sorted(source_dir.glob("cell_*_chr19.txt"), key=cell_sort_key)
    if len(cell_files) != N_CELLS:
        raise ValueError(f"{dataset}: expected {N_CELLS} cell files, got {len(cell_files)}")

    print(f"[prepare] {dataset}: loading {N_CELLS} sparse cells", flush=True)
    counts_numpy = load_cells_matrix(cell_files, N_BINS)

    order = r_colmajor_permutation(N_BINS)
    counts_r = counts_numpy[:, order]
    bulk_r = counts_r.sum(axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_fortran_bin(out_dir / "schic.bin", counts_r)
    np.asarray(bulk_r, dtype="<f8").tofile(out_dir / "bulk.bin")
    np.save(out_dir / "feature_order.npy", order)

    obs_names = [f"cell_{idx}" for idx in range(1, N_CELLS + 1)]
    write_names(out_dir / "obs_names.txt", obs_names)

    iu, ju = np.triu_indices(N_BINS, k=1)
    var_names = [f"chr19_{iu[idx]}_{ju[idx]}" for idx in range(N_FEATURES)]
    write_names(out_dir / "var_names.txt", var_names)

    meta = {
        "dataset_id": dataset,
        "source_raw_dir": str(source_dir),
        "n_cells": N_CELLS,
        "n_features": N_FEATURES,
        "n_beads": N_BINS,
        "schic_layout": "features_by_cells",
        "input_format": "float64 little-endian binary, Fortran order",
        "feature_order_input": "R upper.tri column-major (sorted by column then row)",
        "feature_order_evaluation": "numpy row-major np.triu_indices(n, k=1)",
        "feature_order_npy": "permutation where numpy_row_major[:, order] == R col-major",
        "bulk": "sum of observed cells per feature in R col-major order",
        "expected": None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    marker.write_text("complete\n")
    print(f"[prepare] {dataset}: done -> {out_dir} counts_sum={counts_r.sum():.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HiCImputeData for HiCImpute")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = args.datasets or DATASETS
    args.input_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        prepare_dataset(dataset, args.raw_root, args.input_root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
