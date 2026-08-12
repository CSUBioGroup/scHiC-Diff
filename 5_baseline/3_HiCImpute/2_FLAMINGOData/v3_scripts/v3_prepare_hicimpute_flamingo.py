#!/usr/bin/env python3
"""Prepare FLAMINGO v3 raw data for HiCImpute (R MCMCImpute).

Reads raw 500x500 symmetric contact matrices from
``1_RawData/3_fixed_flamnigoGen/sim_<stem>/`` and writes per-dataset binary
files consumed by ``v3_run_hicimpute_flamingo.R``.

Key fix vs. the old v3_prepare_hicimpute.py:
  The old script wrote features in numpy row-major upper-triangle order
  (``np.triu_indices(n, k=1)``), but HiCImpute's ``MCMCImpute`` fills each
  cell's matrix via ``m[upper.tri(m)] <- single[, k]`` which is R
  column-major.  The mismatch scrambled the spatial structure that
  ``neivar``/``correctfac`` rely on, yielding PCC ~0.07.

  This script extracts the upper triangle in numpy row-major order
  (matching the h5ad/evaluation convention) and then REORDERS features to
  R column-major order so R reconstructs the correct (i, j) positions.

Outputs (per dataset, under ``<input-root>/<stem>/``):
  * ``schic.bin``                -- observed counts (features x cells),
                                    float64 LE, Fortran (column-major)
                                    order, features in R col-major order.
  * ``expected.bin``             -- per-cell GT (features x cells), same
                                    layout/order.
  * ``bulk.bin``                 -- per-feature sum of observed cells
                                    (R col-major order).
  * ``feature_order.npy``        -- permutation ``order`` such that
                                    numpy_row_major[:, order] == R col-major.
                                    ``v3_run_hicimpute_flamingo.R`` uses
                                    the inverse to restore numpy row-major
                                    in the output NPZ.
  * ``obs_names.txt`` / ``var_names.txt`` -- text lists.
  * ``metadata.json`` / ``.complete``     -- bookkeeping.

Raw file naming convention (verified against the h5ad):
  ``gt_contact_data/type_{1..3}_cell_{1..500}_contact.txt``
  Sorting: type_1_cell_1, type_1_cell_2, ..., type_1_cell_500,
           type_2_cell_1, ..., type_2_cell_500,
           type_3_cell_1, ..., type_3_cell_500.
  This maps 1:1 to h5ad obs index 0..1499 (type_1_cell_1 -> cell 0).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

N_BINS = 500
N_FEATURES = N_BINS * (N_BINS - 1) // 2  # 124750
N_CELLS_PER_TYPE = 500
N_TYPES = 3
N_CELLS = N_TYPES * N_CELLS_PER_TYPE  # 1500

DEFAULT_RAW_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/"
    "1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/"
    "3_fixed_flamnigoGen"
)

DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/"
    "5_baseline/3_HiCImpute/2_FLAMINGOData/v3_inputData"
)

DATASETS = [
    "v3_hybrid_W0p5_500cells_level0",
    "v3_hybrid_W0p6_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0_r0p01",
    "v3_hybrid_W0p7_500cells_level0_r0p05",
    "v3_hybrid_W0p8_500cells_level0",
    "v3_hybrid_W0p9_500cells_level0",
]


def r_colmajor_permutation(n: int) -> np.ndarray:
    """Permutation ``order`` so that ``numpy_row_major[:, order]`` becomes
    R ``upper.tri`` column-major order.

    numpy row-major upper-tri pairs: ``np.triu_indices(n, k=1)`` returns
    ``(i, j)`` ordered row-major (by i, then j).
    R ``upper.tri`` column-major order is: by column j, then row i.
    For a symmetric contact matrix the values are equal at (i, j) and
    (j, i), but the feature *positions* differ, so the permutation matters.
    """
    iu, ju = np.triu_indices(n, k=1)
    order = np.lexsort((iu, ju))  # primary key ju, secondary iu
    return order.astype(np.int64)


def natural_cell_sort_key(path: str) -> tuple[int, int]:
    """Sort ``type_{t}_cell_{c}_contact.txt`` by (type, cell_id)."""
    m = re.search(r"type_(\d+)_cell_(\d+)_contact", os.path.basename(path))
    if m is None:
        return (999, 999)
    return (int(m.group(1)), int(m.group(2)))


def natural_cell_sort_key_plain(path: str) -> tuple[int, int]:
    """Sort ``type_{t}_cell_{c}.txt`` (downsampled, no _contact suffix)."""
    m = re.search(r"type_(\d+)_cell_(\d+)", os.path.basename(path))
    if m is None:
        return (999, 999)
    return (int(m.group(1)), int(m.group(2)))


def load_upper_tri_vector(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle (k=1) in numpy row-major order."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)].astype(np.float64)


def load_cells_matrix(cell_files: list[str], n_beads: int) -> np.ndarray:
    """Load all cells -> (n_cells, n_features) numpy row-major."""
    n_feat = n_beads * (n_beads - 1) // 2
    cells = np.empty((len(cell_files), n_feat), dtype=np.float64)
    for k, f in enumerate(cell_files):
        mat = np.loadtxt(f)
        cells[k] = load_upper_tri_vector(mat)
        if k % 200 == 0:
            print(f"  loaded {k+1}/{len(cell_files)} cells", flush=True)
    return cells


def write_fortran_bin(path: Path, cells_by_features: np.ndarray) -> None:
    """Write features-by-cells matrix in Fortran (column-major) order.

    R ``matrix(values, nrow=n_features, ncol=n_cells)`` fills column-major,
    so the file must contain all of column 0 (cell 0), then column 1, ...
    which is the F-order ravel of a features x cells matrix.
    ``np.asfortranarray(...).tofile()`` does NOT write F-order bytes
    (``.tofile`` always ravel order='C'); we ravel explicitly instead.
    """
    feats_by_cells = np.asarray(cells_by_features, dtype="<f8").T
    path.parent.mkdir(parents=True, exist_ok=True)
    feats_by_cells.ravel(order="F").tofile(path)


def write_names(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as h:
        for v in values:
            h.write(f"{v}\n")


def prepare_dataset(stem: str, raw_root: Path, input_root: Path, overwrite: bool) -> None:
    out_dir = input_root / stem
    marker = out_dir / ".complete"
    if marker.exists() and not overwrite:
        print(f"[prepare] {stem}: skip (.complete exists)", flush=True)
        return

    sim_dir = raw_root / f"sim_{stem}"
    if not sim_dir.is_dir():
        raise FileNotFoundError(f"Raw sim dir not found: {sim_dir}")

    gt_files = sorted(
        glob.glob(str(sim_dir / "gt_contact_data" / "*.txt")),
        key=natural_cell_sort_key,
    )
    ds_files = sorted(
        glob.glob(str(sim_dir / "downsampled_contact_data" / "*.txt")),
        key=natural_cell_sort_key_plain,
    )
    if len(gt_files) != N_CELLS:
        raise ValueError(f"{stem}: expected {N_CELLS} gt files, got {len(gt_files)}")
    if len(ds_files) != N_CELLS:
        raise ValueError(f"{stem}: expected {N_CELLS} downsampled files, got {len(ds_files)}")

    print(f"[prepare] {stem}: loading {N_CELLS} downsampled (observed) cells", flush=True)
    counts_numpy = load_cells_matrix(ds_files, N_BINS)  # (1500, 124750) row-major

    print(f"[prepare] {stem}: loading {N_CELLS} GT cells", flush=True)
    gt_numpy = load_cells_matrix(gt_files, N_BINS)  # (1500, 124750) row-major

    order = r_colmajor_permutation(N_BINS)  # numpy_rowmajor_index -> r_colmajor_position

    counts_r = counts_numpy[:, order]
    gt_r = gt_numpy[:, order]
    bulk_r = counts_r.sum(axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_fortran_bin(out_dir / "schic.bin", counts_r)
    write_fortran_bin(out_dir / "expected.bin", gt_r)
    np.asarray(bulk_r, dtype="<f8").tofile(out_dir / "bulk.bin")
    np.save(out_dir / "feature_order.npy", order)

    obs_names = [f"type_{t}_cell_{c}" for t in range(1, N_TYPES + 1)
                 for c in range(1, N_CELLS_PER_TYPE + 1)]
    write_names(out_dir / "obs_names.txt", obs_names)

    iu, ju = np.triu_indices(N_BINS, k=1)
    var_names = [f"chrFLAMINGO_{iu[k]}_{ju[k]}" for k in range(N_FEATURES)]
    write_names(out_dir / "var_names.txt", var_names)

    meta = {
        "dataset_id": stem,
        "source_raw_dir": str(sim_dir),
        "n_cells": N_CELLS,
        "n_features": N_FEATURES,
        "n_beads": N_BINS,
        "schic_layout": "features_by_cells",
        "input_format": "float64 little-endian binary, Fortran (column-major) order",
        "feature_order_input": "R upper.tri column-major (lexsort by col=j then row=i)",
        "feature_order_evaluation": "numpy row-major (np.triu_indices(n, k=1))",
        "feature_order_npy": "permutation: numpy_rowmajor[:, order] == R col-major",
        "bulk": "sum of observed cells per feature (R col-major order)",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    marker.write_text("complete\n")
    print(f"[prepare] {stem}: done -> {out_dir} "
          f"(counts sum={counts_r.sum():.1f}, gt sum={gt_r.sum():.1f})", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root.mkdir(parents=True, exist_ok=True)
    stems = args.datasets if args.datasets else DATASETS
    for stem in stems:
        prepare_dataset(stem, args.raw_root, args.input_root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())