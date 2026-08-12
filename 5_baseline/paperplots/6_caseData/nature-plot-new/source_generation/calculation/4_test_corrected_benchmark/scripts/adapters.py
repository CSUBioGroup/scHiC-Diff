"""Input validation and feature-layout adapters for the corrected benchmark."""

from __future__ import annotations

from math import isqrt
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def infer_bins(n_features: int, include_diagonal: bool) -> int:
    """Infer matrix width from an upper-triangle feature count."""
    if n_features <= 0:
        raise ValueError(f"feature count must be positive, got {n_features}")
    discriminant = 1 + 8 * n_features
    root = isqrt(discriminant)
    if root * root != discriminant:
        raise ValueError(f"feature count {n_features} is not triangular")
    numerator = (-1 + root) if include_diagonal else (1 + root)
    if numerator % 2:
        raise ValueError(f"feature count {n_features} does not map to integer bins")
    n_bins = numerator // 2
    expected = n_bins * (n_bins + 1) // 2 if include_diagonal else n_bins * (n_bins - 1) // 2
    if expected != n_features:
        raise ValueError(f"feature count {n_features} is invalid for n_bins={n_bins}")
    return n_bins


def load_csr_npz(path: str | Path) -> sparse.csr_matrix:
    """Load either a scipy sparse NPZ or the project's equivalent CSR payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    required = {"data", "indices", "indptr", "shape"}
    if not required.issubset(payload.files):
        missing = sorted(required.difference(payload.files))
        raise ValueError(f"CSR archive {path} is missing keys: {missing}")
    matrix = sparse.csr_matrix(
        (payload["data"], payload["indices"], payload["indptr"]),
        shape=tuple(int(v) for v in payload["shape"]),
    )
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"CSR archive {path} contains non-finite values")
    return matrix


def _off_diagonal_to_full_column_map(n_bins: int) -> np.ndarray:
    full_rows, full_cols = np.triu_indices(n_bins, k=0)
    positions = np.full((n_bins, n_bins), -1, dtype=np.int64)
    positions[full_rows, full_cols] = np.arange(full_rows.size, dtype=np.int64)
    off_rows, off_cols = np.triu_indices(n_bins, k=1)
    return positions[off_rows, off_cols]


def standardize_feature_matrix(
    matrix: sparse.spmatrix | np.ndarray,
    include_diagonal: bool,
    n_bins: int = 100,
) -> sparse.csr_matrix:
    """Return CSR features in NumPy triu(k=0) order, inserting zero diagonal if needed."""
    matrix = sparse.csr_matrix(matrix)
    if not np.isfinite(matrix.data).all():
        raise ValueError("feature matrix contains non-finite values")

    expected_input = n_bins * (n_bins + 1) // 2 if include_diagonal else n_bins * (n_bins - 1) // 2
    if matrix.shape[1] != expected_input:
        raise ValueError(
            f"feature count mismatch: got {matrix.shape[1]}, expected {expected_input} "
            f"for n_bins={n_bins}, include_diagonal={include_diagonal}"
        )
    if include_diagonal:
        return matrix.tocsr(copy=True)

    column_map = _off_diagonal_to_full_column_map(n_bins)
    coo = matrix.tocoo(copy=False)
    standardized = sparse.csr_matrix(
        (coo.data, (coo.row, column_map[coo.col])),
        shape=(matrix.shape[0], n_bins * (n_bins + 1) // 2),
    )
    standardized.sort_indices()
    return standardized


def load_canonical_names(h5ad_path: str | Path) -> np.ndarray:
    """Load canonical observation names without materializing the H5AD matrix."""
    dataset = ad.read_h5ad(Path(h5ad_path), backed="r")
    try:
        names = dataset.obs_names.astype(str).to_numpy(copy=True)
    finally:
        dataset.file.close()
    if names.size != np.unique(names).size:
        raise ValueError("canonical H5AD contains duplicate observation names")
    return names


def load_npz_cellnames(npz_path: str | Path) -> np.ndarray:
    payload = np.load(Path(npz_path), allow_pickle=True)
    if "cellnames" not in payload.files:
        raise ValueError(f"NPZ archive has no cellnames: {npz_path}")
    names = np.asarray(payload["cellnames"], dtype=str)
    if names.ndim != 1:
        raise ValueError(f"cellnames must be one-dimensional: {npz_path}")
    return names


def validate_named_archive(canonical_names: np.ndarray, named_npz_path: str | Path) -> bool:
    archive_names = load_npz_cellnames(named_npz_path)
    if not np.array_equal(np.asarray(canonical_names, dtype=str), archive_names):
        mismatch = np.flatnonzero(np.asarray(canonical_names, dtype=str) != archive_names)
        first = int(mismatch[0]) if mismatch.size else None
        raise ValueError(f"named archive does not match canonical cell order; first mismatch={first}")
    return True


def map_group_names(canonical_names: np.ndarray, group_names: np.ndarray) -> np.ndarray:
    canonical_names = np.asarray(canonical_names, dtype=str)
    group_names = np.asarray(group_names, dtype=str)
    if group_names.ndim != 1:
        raise ValueError("group names must be one-dimensional")
    if group_names.size != np.unique(group_names).size:
        raise ValueError("group names contain duplicates")
    name_to_index = {name: idx for idx, name in enumerate(canonical_names)}
    missing = [name for name in group_names if name not in name_to_index]
    if missing:
        raise ValueError(f"group contains {len(missing)} names absent from canonical order; first={missing[0]}")
    return np.asarray([name_to_index[name] for name in group_names], dtype=np.int64)


def validate_method(
    path: str | Path,
    include_diagonal: bool,
    expected_cells: int = 7466,
    n_bins: int = 100,
) -> dict[str, object]:
    """Validate one real method archive and return an auditable summary."""
    matrix = load_csr_npz(path)
    if matrix.shape[0] != expected_cells:
        raise ValueError(f"cell count mismatch for {path}: {matrix.shape[0]} != {expected_cells}")
    inferred = infer_bins(matrix.shape[1], include_diagonal=include_diagonal)
    if inferred != n_bins:
        raise ValueError(f"bin count mismatch for {path}: {inferred} != {n_bins}")
    standardized_features = n_bins * (n_bins + 1) // 2
    data_min = float(min(0.0, matrix.data.min())) if matrix.data.size else 0.0
    data_max = float(max(0.0, matrix.data.max())) if matrix.data.size else 0.0
    return {
        "path": str(Path(path).resolve()),
        "cells": int(matrix.shape[0]),
        "input_features": int(matrix.shape[1]),
        "standard_features": standardized_features,
        "include_diagonal": bool(include_diagonal),
        "nnz": int(matrix.nnz),
        "min": data_min,
        "max": data_max,
        "finite": True,
    }
