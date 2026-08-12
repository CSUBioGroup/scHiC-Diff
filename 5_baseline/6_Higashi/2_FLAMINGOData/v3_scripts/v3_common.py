#!/usr/bin/env python3
"""Shared helpers for the FLAMINGO v3 paramsweep h5ad -> baseline imputation pipelines.

The 7 paramsweep datasets live under
``5_paramsweep_datasets/`` as ``v3_hybrid_<tag>_scdiff2.h5ad`` (1500 cells,
124750 features, single chromosome ``chrFLAMINGO`` with 500 beads).  Each h5ad
exposes two CSR layers:

* ``layers['counts']`` -- observed, noisy contacts (the model input)
* ``layers['gt']``      -- per-cell ground truth (1500 rows, one per cell)

Feature ``k`` maps to the upper-triangle matrix coordinate ``(i, j)`` with
``i < j`` over an N x N contact matrix (N=500) in row-major order, matching the
``var_names`` pattern ``chrFLAMINGO_i_j``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


N_BINS = 500
N_FEATURES = N_BINS * (N_BINS - 1) // 2  # 124750
CHROM_NAME = "chrFLAMINGO"
DEFAULT_DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/"
    "3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
)


@dataclass(frozen=True)
class V3Dataset:
    stem: str           # e.g. v3_hybrid_W0p7_500cells_level0_r0p05
    h5ad: Path          # .../<stem>_scdiff2.h5ad
    n_cells: int        # 1500
    n_features: int     # 124750
    n_beads: int        # 500


def discover_datasets(data_dir: Path = DEFAULT_DATA_DIR) -> list[V3Dataset]:
    out = []
    for h5 in sorted(data_dir.glob("v3_hybrid_*_scdiff2.h5ad")):
        stem = h5.name[: -len("_scdiff2.h5ad")]
        n_cells, n_features = _h5ad_shape(h5)
        out.append(V3Dataset(stem, h5, n_cells, n_features,
                              lower_triangle_size_to_n(n_features)))
    return out


def lower_triangle_size_to_n(size: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * size)) / 2)
    if n * (n - 1) // 2 != size:
        raise ValueError(f"{size} is not a lower-triangle feature count")
    return n


def _h5ad_shape(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as h:
        n_cells = len(h["obs/_index"])
        n_features = len(h["var/_index"])
    return n_cells, n_features


def csr_from_group(handle: h5py.File, key: str) -> sparse.csr_matrix:
    group = handle[key]
    data = np.asarray(group["data"], dtype=np.float64)
    indices = np.asarray(group["indices"], dtype=np.int32)
    indptr = np.asarray(group["indptr"], dtype=np.int64)
    n_cells = len(indptr) - 1
    n_features = len(handle["var/_index"])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_features))
    matrix.data = np.nan_to_num(matrix.data, nan=0.0, posinf=0.0, neginf=0.0)
    matrix.data[matrix.data < 0] = 0.0
    matrix.eliminate_zeros()
    return matrix


def load_layer(path: Path, layer: str = "counts") -> sparse.csr_matrix:
    """Load a CSR layer from an h5ad. Falls back to X if the layer is absent."""
    with h5py.File(path, "r") as h:
        key = f"layers/{layer}"
        if key in h:
            return csr_from_group(h, key)
        return csr_from_group(h, "X")


def load_layer_dense(path: Path, layer: str = "counts") -> np.ndarray:
    mat = load_layer(path, layer)
    return mat.toarray().astype(np.float64, copy=False)


def upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def feature_to_bins(n_beads: int) -> tuple[np.ndarray, np.ndarray]:
    """Map feature index k -> (bin1, bin2) with bin1 < bin2 (upper-tri row-major)."""
    iu, ju = np.triu_indices(n_beads, k=1)
    return iu.astype(np.int64), ju.astype(np.int64)


def parse_tag(stem: str) -> str:
    m = re.match(r"^v3_hybrid_(.+)$", stem)
    return m.group(1) if m else stem


DATASETS = discover_datasets()