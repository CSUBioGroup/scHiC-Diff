#!/usr/bin/env python3
"""
Build a 20-dimensional SVD embedding for HiRES 1Mb clustering plots.

Primary input format:
  - scipy sparse NPZ storing a cell x features matrix.

Also supported:
  - h5ad files, using adata.X.
  - dense NPZ files containing a 2D array under arr_0 or a user-given key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DEFAULT_INPUT = Path("input/cell_by_features.npz")
DEFAULT_LABELS = Path("cell_labels.csv")
DEFAULT_OUTPUT = Path("svd_embedding/method/final_svd_decomp.npz")
DEFAULT_MODEL_OUTPUT = Path("svd_embedding/method/final_svd_model.lib")
DEFAULT_META_OUTPUT = Path("svd_embedding/method/final_svd_metadata.json")


def load_matrix(input_path: Path, npz_key: Optional[str] = None):
    suffix = input_path.suffix.lower()
    if suffix == ".h5ad":
        try:
            import anndata as ad
        except ImportError as exc:
            raise RuntimeError("Reading .h5ad input requires anndata to be installed.") from exc
        adata = ad.read_h5ad(input_path, backed=None)
        matrix = adata.X
        if not sparse.issparse(matrix):
            matrix = sparse.csr_matrix(matrix)
        return matrix, "h5ad:X"

    if suffix != ".npz":
        raise ValueError(f"Unsupported input format: {input_path}. Use .npz or .h5ad.")

    try:
        matrix = sparse.load_npz(input_path)
        return matrix.tocsr(), "scipy_sparse_npz"
    except Exception:
        with np.load(input_path) as data:
            key = npz_key
            if key is None:
                key = "arr_0" if "arr_0" in data.files else ("X" if "X" in data.files else data.files[0])
            if key not in data.files:
                raise KeyError(f"NPZ key {key!r} not found. Available keys: {data.files}")
            matrix = data[key]
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D matrix in {input_path}, got shape {matrix.shape}.")
        return sparse.csr_matrix(matrix), f"dense_npz:{key}"


def validate_cell_count(matrix, labels_path: Optional[Path]) -> Optional[int]:
    if labels_path is None:
        return None
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = pd.read_csv(labels_path)
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            "Cell count mismatch: "
            f"matrix has {matrix.shape[0]} rows, labels has {len(labels)} rows."
        )
    return len(labels)


def compute_svd(matrix, dim: int, random_state: int, norm_sig: bool) -> Tuple[np.ndarray, TruncatedSVD]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D cell x features matrix, got {matrix.shape}.")
    max_dim = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_dim < 1:
        raise ValueError(f"Matrix shape {matrix.shape} is too small for SVD.")
    if dim > max_dim:
        print(f"[warn] Requested dim={dim}; reducing to valid dim={max_dim}.")
        dim = max_dim

    model = TruncatedSVD(n_components=dim, algorithm="arpack", random_state=random_state)
    embedding = model.fit_transform(matrix)

    if norm_sig:
        valid = model.singular_values_ > 1e-8
        embedding = embedding[:, valid]
        embedding = embedding / model.singular_values_[valid][None, :]

    return embedding.astype(np.float32, copy=False), model


def save_model(model: TruncatedSVD, model_output: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Saving the SVD model requires joblib. Use --no-save-model to skip it.") from exc
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)


def write_metadata(
    meta_output: Path,
    input_path: Path,
    input_kind: str,
    output_path: Path,
    matrix_shape,
    embedding_shape,
    model: TruncatedSVD,
    norm_sig: bool,
    labels_path: Optional[Path],
) -> None:
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(input_path),
        "input_kind": input_kind,
        "labels": str(labels_path) if labels_path else None,
        "output": str(output_path),
        "matrix_shape": list(matrix_shape),
        "embedding_shape": list(embedding_shape),
        "norm_sig": norm_sig,
        "singular_values": model.singular_values_.astype(float).tolist(),
        "explained_variance_ratio": model.explained_variance_ratio_.astype(float).tolist(),
    }
    meta_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 20d SVD embedding from a HiRES 1Mb cell x features matrix."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .npz or .h5ad matrix.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="cell_labels.csv for row-count validation.")
    parser.add_argument("--no-label-check", action="store_true", help="Skip labels row-count validation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output compressed NPZ embedding.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT, help="Output joblib SVD model.")
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_META_OUTPUT, help="Output JSON metadata.")
    parser.add_argument("--dim", type=int, default=20, help="Number of SVD dimensions.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for TruncatedSVD.")
    parser.add_argument("--npz-key", type=str, default=None, help="Array key for dense NPZ input.")
    parser.add_argument("--no-norm-sig", action="store_true", help="Do not divide SVD embedding by singular values.")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save the fitted SVD model.")
    parser.add_argument("--no-metadata", action="store_true", help="Do not write metadata JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix, input_kind = load_matrix(args.input, npz_key=args.npz_key)
    print(f"Loaded matrix: shape={matrix.shape}, format={input_kind}, dtype={matrix.dtype}")

    labels_path = None if args.no_label_check else args.labels
    validate_cell_count(matrix, labels_path)

    embedding, model = compute_svd(
        matrix=matrix,
        dim=args.dim,
        random_state=args.random_state,
        norm_sig=not args.no_norm_sig,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, arr_0=embedding)
    print(f"Saved SVD embedding: {args.output} shape={embedding.shape}")

    if not args.no_save_model:
        save_model(model, args.model_output)
        print(f"Saved SVD model: {args.model_output}")

    if not args.no_metadata:
        write_metadata(
            meta_output=args.metadata_output,
            input_path=args.input,
            input_kind=input_kind,
            output_path=args.output,
            matrix_shape=matrix.shape,
            embedding_shape=embedding.shape,
            model=model,
            norm_sig=not args.no_norm_sig,
            labels_path=labels_path,
        )
        print(f"Saved metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
