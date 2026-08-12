#!/usr/bin/env python3
"""
Build a hicluster-compatible two-stage SVD embedding.

This reproduces the data-flow used by `hicluster embedding`:

  chr*.npz or h5ad grouped by chromosome
    -> per-chromosome SVD
    -> decomp/chr*_decomp.npz
    -> decomp/total_chrom_decomp_concat.npz
    -> final SVD
    -> decomp/total_decomp.npz

Input matrices must be cell x features, and all chromosomes must have the same
cell order as the label file used for plotting.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DEFAULT_CHROM_DIR = Path("imputed_data/method")
DEFAULT_OUTPUT_DIR = Path("two_stage_compare/method")
DEFAULT_LABELS = Path("cell_labels.csv")


def infer_chrom_name(path_or_name) -> str:
    path = Path(path_or_name)
    candidates = [path.stem, *reversed(path.parts)]
    for name in candidates:
        match = re.search(r"(?:^|[_-])chr([0-9]+|X|Y|M|MT)$", name, re.IGNORECASE)
        if match:
            return f"chr{match.group(1).upper()}"
        match = re.search(r"chr([0-9]+|X|Y|M|MT)", name, re.IGNORECASE)
        if match:
            return f"chr{match.group(1).upper()}"
    return path.stem


def chrom_sort_key(path_or_name) -> tuple[int, object]:
    name = infer_chrom_name(path_or_name)
    match = re.search(r"(?:^|[_-])chr([0-9]+|X|Y|M|MT)$", name, re.IGNORECASE)
    if not match:
        match = re.search(r"chr([0-9]+|X|Y|M|MT)", name, re.IGNORECASE)
    if not match:
        return (2, name)
    chrom = match.group(1).upper()
    if chrom.isdigit():
        return (0, int(chrom))
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return (1, special.get(chrom, chrom))


def load_npz_matrix(path: Path, npz_key: Optional[str] = None):
    try:
        return sparse.load_npz(path).tocsr(), "scipy_sparse_npz"
    except Exception:
        with np.load(path, allow_pickle=False) as data:
            key = npz_key
            if key is None:
                key = "arr_0" if "arr_0" in data.files else ("X" if "X" in data.files else data.files[0])
            if key not in data.files:
                raise KeyError(f"NPZ key {key!r} not found in {path}; available keys: {data.files}")
            matrix = data[key]
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D matrix in {path}, got shape {matrix.shape}.")
        return matrix, f"dense_npz:{key}"


def iter_chrom_npz(chrom_npz_dir: Path, chrom_glob: str, npz_key: Optional[str]):
    paths = sorted(chrom_npz_dir.glob(chrom_glob), key=chrom_sort_key)
    if not paths:
        raise FileNotFoundError(f"No chromosome NPZ files matched {chrom_npz_dir / chrom_glob}")
    seen_chroms = set()
    for path in paths:
        chrom = infer_chrom_name(path)
        if chrom in seen_chroms:
            raise ValueError(f"Duplicate chromosome name {chrom!r} inferred from {path}")
        seen_chroms.add(chrom)
        matrix, kind = load_npz_matrix(path, npz_key=npz_key)
        yield chrom, matrix, str(path), kind


def iter_chrom_h5ad(chrom_h5ad_dir: Path, chrom_h5ad_glob: str):
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Reading chromosome .h5ad inputs requires anndata.") from exc

    paths = sorted(chrom_h5ad_dir.glob(chrom_h5ad_glob), key=chrom_sort_key)
    if not paths:
        raise FileNotFoundError(f"No chromosome H5AD files matched {chrom_h5ad_dir / chrom_h5ad_glob}")
    seen_chroms = set()
    for path in paths:
        chrom = infer_chrom_name(path)
        if chrom in seen_chroms:
            raise ValueError(f"Duplicate chromosome name {chrom!r} inferred from {path}")
        seen_chroms.add(chrom)
        adata = ad.read_h5ad(path, backed=None)
        matrix = adata.X
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
        yield chrom, matrix, str(path), "h5ad:X"


def coded_mouse_chrom_map(n_chroms: int) -> Optional[dict[str, str]]:
    if n_chroms == 20:
        names = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
    elif n_chroms == 21:
        names = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
    else:
        return None
    names = sorted(names)
    return {str(i): name for i, name in enumerate(names)}


def iter_h5ad_chroms(input_h5ad: Path, chromosome_column: str, auto_code_map: bool):
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Reading .h5ad input requires anndata.") from exc

    adata = ad.read_h5ad(input_h5ad, backed=None)
    if chromosome_column not in adata.var.columns:
        raise ValueError(
            f"{input_h5ad} var does not contain chromosome column {chromosome_column!r}. "
            f"Available columns: {list(adata.var.columns)}"
        )
    chrom_values = adata.var[chromosome_column].astype(str)
    unique_values = sorted(chrom_values.unique().tolist(), key=lambda value: int(value) if value.isdigit() else value)
    chrom_map = None
    if auto_code_map and all(value.isdigit() for value in unique_values):
        expected_codes = [str(i) for i in range(len(unique_values))]
        if unique_values == expected_codes:
            chrom_map = coded_mouse_chrom_map(len(unique_values))

    if chrom_map is None:
        chrom_pairs = [(value, value) for value in unique_values]
    else:
        chrom_pairs = [(chrom_map[value], value) for value in unique_values]

    chrom_pairs = sorted(chrom_pairs, key=lambda item: chrom_sort_key(item[0]))
    for chrom, raw_value in chrom_pairs:
        col_idx = np.flatnonzero(chrom_values.to_numpy() == raw_value)
        matrix = adata.X[:, col_idx]
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
        yield str(chrom), matrix, f"{input_h5ad}::{chromosome_column}={raw_value}", "h5ad:X"


def validate_labels(labels_path: Optional[Path], expected_cells: int) -> None:
    if labels_path is None:
        return
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = pd.read_csv(labels_path)
    if len(labels) != expected_cells:
        raise ValueError(
            f"Cell count mismatch: matrix has {expected_cells} rows, labels has {len(labels)} rows."
        )


def fit_svd(matrix, dim: int, norm_sig: bool, random_state: Optional[int]):
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}.")
    effective_dim = min(dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if effective_dim < 1:
        raise ValueError(f"Matrix shape {matrix.shape} is too small for SVD.")

    kwargs = {"n_components": effective_dim, "algorithm": "arpack"}
    if random_state is not None:
        kwargs["random_state"] = random_state
    model = TruncatedSVD(**kwargs)
    decomp = model.fit_transform(matrix)

    if norm_sig:
        valid = model.singular_values_ > 0
        decomp = decomp[:, valid]
        singular_values = model.singular_values_[valid]
        decomp = decomp / singular_values[None, :]

    return decomp.astype(np.float32, copy=False), model, effective_dim


def save_npz(path: Path, array: np.ndarray, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, arr_0=array)
    else:
        np.savez(path, array)


def parse_random_state(value: str) -> Optional[int]:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create hicluster-compatible two-stage SVD total_decomp.npz."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chrom-npz-dir", type=Path, help="Directory containing per-chromosome chr*.npz matrices.")
    source.add_argument("--chrom-h5ad-dir", type=Path, help="Directory containing per-chromosome chr*.h5ad matrices.")
    source.add_argument("--input-h5ad", type=Path, help="Input h5ad with adata.X and a chromosome column in adata.var.")
    parser.add_argument("--chrom-glob", default="chr*.npz", help="Glob used with --chrom-npz-dir.")
    parser.add_argument("--chrom-h5ad-glob", default="*chr*.h5ad", help="Glob used with --chrom-h5ad-dir.")
    parser.add_argument("--chromosome-column", default="chromosome", help="adata.var column used with --input-h5ad.")
    parser.add_argument(
        "--no-auto-h5ad-code-map",
        action="store_true",
        help="Do not map integer-coded h5ad chromosome values to mouse chr names.",
    )
    parser.add_argument("--npz-key", default=None, help="Array key for dense NPZ inputs.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Optional label CSV for cell-count validation.")
    parser.add_argument("--no-label-check", action="store_true", help="Skip label row-count validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--dim", type=int, default=20, help="SVD dimensions for each stage.")
    parser.add_argument("--random-state", type=parse_random_state, default=None, help="Random state; default None matches hicluster.")
    parser.add_argument("--no-norm-sig", action="store_true", help="Disable division by singular values.")
    parser.add_argument("--compressed", action="store_true", help="Use compressed NPZ output instead of hicluster-style np.savez.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    decomp_dir = output_dir / "decomp"
    decomp_dir.mkdir(parents=True, exist_ok=True)

    if args.chrom_npz_dir is not None:
        chrom_iterable = iter_chrom_npz(args.chrom_npz_dir, args.chrom_glob, args.npz_key)
        source = str(args.chrom_npz_dir)
    elif args.chrom_h5ad_dir is not None:
        chrom_iterable = iter_chrom_h5ad(args.chrom_h5ad_dir, args.chrom_h5ad_glob)
        source = str(args.chrom_h5ad_dir)
    else:
        chrom_iterable = iter_h5ad_chroms(
            args.input_h5ad,
            args.chromosome_column,
            auto_code_map=not args.no_auto_h5ad_code_map,
        )
        source = str(args.input_h5ad)

    chrom_decomps: list[np.ndarray] = []
    chrom_meta = []
    expected_cells: Optional[int] = None
    norm_sig = not args.no_norm_sig

    for chrom, matrix, matrix_source, matrix_kind in chrom_iterable:
        if expected_cells is None:
            expected_cells = matrix.shape[0]
            labels_path = None if args.no_label_check else args.labels
            validate_labels(labels_path, expected_cells)
        elif matrix.shape[0] != expected_cells:
            raise ValueError(f"{chrom} has {matrix.shape[0]} cells; expected {expected_cells}.")

        decomp, model, effective_dim = fit_svd(
            matrix=matrix,
            dim=args.dim,
            norm_sig=norm_sig,
            random_state=args.random_state,
        )
        chrom_out = decomp_dir / f"{chrom}_decomp.npz"
        save_npz(chrom_out, decomp, compressed=args.compressed)
        chrom_decomps.append(decomp)
        chrom_meta.append(
            {
                "chrom": chrom,
                "source": matrix_source,
                "input_kind": matrix_kind,
                "matrix_shape": list(matrix.shape),
                "effective_dim": effective_dim,
                "decomp_shape": list(decomp.shape),
                "singular_values": model.singular_values_.astype(float).tolist(),
                "output": str(chrom_out),
            }
        )
        print(f"{chrom}: matrix={matrix.shape}, decomp={decomp.shape}, saved={chrom_out}")

    if expected_cells is None:
        raise RuntimeError("No chromosome matrices were loaded.")

    total_chrom = np.concatenate(chrom_decomps, axis=1).astype(np.float32, copy=False)
    total_chrom_path = decomp_dir / "total_chrom_decomp_concat.npz"
    save_npz(total_chrom_path, total_chrom, compressed=args.compressed)
    print(f"total_chrom_decomp_concat: {total_chrom.shape}, saved={total_chrom_path}")

    total_decomp, final_model, final_effective_dim = fit_svd(
        matrix=total_chrom,
        dim=args.dim,
        norm_sig=norm_sig,
        random_state=args.random_state,
    )
    total_decomp_path = decomp_dir / "total_decomp.npz"
    save_npz(total_decomp_path, total_decomp, compressed=args.compressed)
    print(f"total_decomp: {total_decomp.shape}, saved={total_decomp_path}")

    metadata = {
        "source": source,
        "output_dir": str(output_dir),
        "dim": args.dim,
        "norm_sig": norm_sig,
        "random_state": args.random_state,
        "compressed": args.compressed,
        "n_cells": expected_cells,
        "n_chromosomes": len(chrom_meta),
        "total_chrom_decomp_concat": {
            "path": str(total_chrom_path),
            "shape": list(total_chrom.shape),
        },
        "total_decomp": {
            "path": str(total_decomp_path),
            "shape": list(total_decomp.shape),
            "effective_dim": final_effective_dim,
            "singular_values": final_model.singular_values_.astype(float).tolist(),
        },
        "chromosomes": chrom_meta,
    }
    metadata_path = decomp_dir / "two_stage_svd_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
