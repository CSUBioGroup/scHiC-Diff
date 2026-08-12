"""Canonical cell-group mapping, repeated sampling, and matrix aggregation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from adapters import map_group_names, standardize_feature_matrix


def build_group_indices(
    canonical_names: np.ndarray,
    early_neuron_names: np.ndarray,
) -> dict[str, np.ndarray]:
    early = np.sort(map_group_names(canonical_names, early_neuron_names))
    all_indices = np.arange(len(canonical_names), dtype=np.int64)
    mask = np.ones(len(canonical_names), dtype=bool)
    mask[early] = False
    return {
        "earlyNeurons": early,
        "nonEarlyNeurons": all_indices[mask],
    }


def sample_indices(
    group_indices: np.ndarray,
    count: int,
    seed: int,
    deduplicate_full_group: bool = True,
) -> np.ndarray:
    group_indices = np.asarray(group_indices, dtype=np.int64)
    if group_indices.ndim != 1:
        raise ValueError("group indices must be one-dimensional")
    if np.unique(group_indices).size != group_indices.size:
        raise ValueError("group indices contain duplicates")
    if count <= 0:
        raise ValueError(f"sample count must be positive, got {count}")
    if count > group_indices.size:
        raise ValueError(f"sample count {count} exceeds group size {group_indices.size}")
    if deduplicate_full_group and count == group_indices.size:
        return np.sort(group_indices.copy())
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(group_indices, count, replace=False).astype(np.int64))


def index_sha256(indices: np.ndarray) -> str:
    normalized = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def aggregate_standard_vectors(
    matrix: sparse.spmatrix | np.ndarray,
    indices: np.ndarray,
    include_diagonal: bool,
    n_bins: int = 100,
) -> np.ndarray:
    matrix = sparse.csr_matrix(matrix)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("cannot aggregate an empty cell subset")
    if indices.min() < 0 or indices.max() >= matrix.shape[0]:
        raise IndexError("cell subset contains an out-of-range row")
    mean_vector = np.asarray(matrix[indices].mean(axis=0)).reshape(1, -1)
    standardized = standardize_feature_matrix(
        mean_vector,
        include_diagonal=include_diagonal,
        n_bins=n_bins,
    )
    return standardized.toarray()[0]


def write_subset_manifest(
    groups: dict[str, np.ndarray],
    counts: list[int],
    seeds: list[int],
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    written_full_paths: set[Path] = set()

    for group_name, group_indices in groups.items():
        group_indices = np.asarray(group_indices, dtype=np.int64)
        for count in counts:
            if count > group_indices.size:
                raise ValueError(f"count {count} exceeds group {group_name} size {group_indices.size}")
            is_full = count == group_indices.size
            for seed in seeds:
                selected = sample_indices(group_indices, count=count, seed=seed)
                if is_full:
                    subset_path = output_dir / f"{group_name}_{count}cells_full.npy"
                else:
                    subset_path = output_dir / f"{group_name}_{count}cells_seed{seed}.npy"
                if not is_full or subset_path not in written_full_paths:
                    np.save(subset_path, selected)
                    if is_full:
                        written_full_paths.add(subset_path)
                records.append(
                    {
                        "group": group_name,
                        "group_size": int(group_indices.size),
                        "cell_count": int(count),
                        "seed": int(seed),
                        "reused_full_group": bool(is_full),
                        "index_sha256": index_sha256(selected),
                        "subset_path": str(subset_path.resolve()),
                    }
                )

    manifest = pd.DataFrame.from_records(records)
    manifest.to_csv(output_dir / "subset_manifest.csv", index=False)
    return manifest
