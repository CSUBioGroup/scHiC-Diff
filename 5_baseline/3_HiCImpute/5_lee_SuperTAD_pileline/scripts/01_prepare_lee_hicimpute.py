#!/usr/bin/env python3
"""Prepare Lee SuperTAD pipeline BEDPE data for HiCImpute R MCMCImpute.

Input:  per_cell_bedpe/<CellType>_cell_<NNNN>.txt
        5-col BEDPE:  chrA  startA  chrB  startB  contact

Output per cell type:
  schic.bin          - float64 little-endian, Fortran order, (n_features, n_cells)
  bulk.bin           - float64 little-endian, (n_features,)
  feature_order.npy  - permutation: numpy_row_major[:, order] == R col-major
  obs_names.txt      - cell names
  var_names.txt      - feature names chr4_i_j
  metadata.json      - dataset metadata
  .complete          - marker file

Data geometry (from metadata.json):
  chromosome      = chr4
  resolution      = 10000
  n_bins          = 49
  region_start    = 54890000
  n_features      = 49*48/2 = 1176
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "3_HiCImpute/5_lee_SuperTAD_pileline/input_lee"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "3_HiCImpute/5_lee_SuperTAD_pileline/hicimpute_input"
)
DEFAULT_METADATA = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "3_HiCImpute/5_lee_SuperTAD_pileline/input_lee/metadata.json"
)

CELL_TYPES = ["Astro", "Endo", "ODC", "OPC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Lee SuperTAD BEDPE for HiCImpute R."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cell-types", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def r_colmajor_permutation(n: int) -> np.ndarray:
    """Permutation where numpy_row_major[:, order] == R upper.tri column-major."""
    iu, ju = np.triu_indices(n, k=1)
    return np.lexsort((iu, ju)).astype(np.int64)


def load_upper_tri_vector_from_bedpe(bedpe_path: Path, n_bins: int,
                                     region_start: int, resolution: int) -> np.ndarray:
    """Read a BEDPE file and return the numpy row-major upper-tri feature vector."""
    iu, ju = np.triu_indices(n_bins, k=1)
    n_features = len(iu)
    if bedpe_path.stat().st_size == 0:
        return np.zeros(n_features, dtype=np.float64)
    data = np.loadtxt(bedpe_path, dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        return np.zeros(n_features, dtype=np.float64)
    starts_a = data[:, 1].astype(np.int64)
    starts_b = data[:, 3].astype(np.int64)
    counts = data[:, 4].astype(np.float64)
    bin_a = (starts_a - region_start) // resolution
    bin_b = (starts_b - region_start) // resolution
    lo = np.minimum(bin_a, bin_b)
    hi = np.maximum(bin_a, bin_b)
    matrix = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(matrix, (lo, hi), counts)
    matrix = matrix + matrix.T
    return matrix[iu, ju]


def write_fortran_bin(path: Path, cells_by_features: np.ndarray) -> None:
    """Write (n_cells, n_features) array as Fortran-order (n_features, n_cells) binary."""
    features_by_cells = np.asarray(cells_by_features, dtype="<f8").T
    path.parent.mkdir(parents=True, exist_ok=True)
    features_by_cells.ravel(order="F").tofile(path)


def write_names(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def prepare_cell_type(cell_type: str, input_root: Path, output_root: Path,
                      chrom: str, n_bins: int, resolution: int,
                      region_start: int, overwrite: bool) -> None:
    n_features = n_bins * (n_bins - 1) // 2
    out_dir = output_root / cell_type
    marker = out_dir / ".complete"
    if marker.exists() and not overwrite:
        print(f"[prepare] {cell_type}: skip (.complete exists)", flush=True)
        return

    bedpe_dir = input_root / "per_cell_bedpe"
    pattern = re.compile(rf"^{cell_type}_cell_(\d+)\.txt$")
    cell_files = []
    for p in sorted(bedpe_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            cell_files.append((int(m.group(1)), p))
    cell_files.sort(key=lambda t: t[0])
    if not cell_files:
        raise FileNotFoundError(f"No BEDPE files for {cell_type} in {bedpe_dir}")
    n_cells = len(cell_files)
    print(f"[prepare] {cell_type}: {n_cells} cells, n_bins={n_bins}, "
          f"n_features={n_features}", flush=True)

    # Load all cells into (n_cells, n_features) numpy row-major triu
    counts_numpy = np.zeros((n_cells, n_features), dtype=np.float64)
    for idx, (bedpe_id, bedpe_path) in enumerate(cell_files):
        counts_numpy[idx] = load_upper_tri_vector_from_bedpe(
            bedpe_path, n_bins, region_start, resolution
        )
        if (idx + 1) % 200 == 0 or idx + 1 == n_cells:
            print(f"  loaded {idx + 1}/{n_cells} cells", flush=True)

    # Permute to R column-major upper.tri order
    order = r_colmajor_permutation(n_bins)
    counts_r = counts_numpy[:, order]
    bulk_r = counts_r.sum(axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_fortran_bin(out_dir / "schic.bin", counts_r)
    np.asarray(bulk_r, dtype="<f8").tofile(out_dir / "bulk.bin")
    np.save(out_dir / "feature_order.npy", order)

    obs_names = [f"{cell_type}_cell_{bedpe_id:04d}" for _, (bedpe_id, _) in
                 enumerate(cell_files)]
    write_names(out_dir / "obs_names.txt", obs_names)

    iu, ju = np.triu_indices(n_bins, k=1)
    var_names = [f"{chrom}_{iu[k]}_{ju[k]}" for k in range(n_features)]
    write_names(out_dir / "var_names.txt", var_names)

    meta = {
        "dataset_id": cell_type,
        "source_bedpe_dir": str(bedpe_dir),
        "n_cells": n_cells,
        "n_features": n_features,
        "n_beads": n_bins,
        "chromosome": chrom,
        "resolution": resolution,
        "region_start": region_start,
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
    print(f"[prepare] {cell_type}: done -> {out_dir} "
          f"counts_sum={counts_r.sum():.1f}", flush=True)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with open(args.metadata) as f:
        meta = json.load(f)
    chrom = meta["chromosome"]
    resolution = meta["resolution"]
    n_bins = meta["n_bins"]
    region_start = meta["region_start"]

    cell_types = args.cell_types or list(meta.get("cell_types", {}).keys())
    if not cell_types:
        cell_types = CELL_TYPES

    print(f"[prepare] chrom={chrom}, res={resolution}, n_bins={n_bins}, "
          f"cell_types={cell_types}", flush=True)

    for ct in cell_types:
        prepare_cell_type(ct, args.input_root, args.output_root,
                          chrom, n_bins, resolution, region_start,
                          args.overwrite)

    print("[prepare] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())