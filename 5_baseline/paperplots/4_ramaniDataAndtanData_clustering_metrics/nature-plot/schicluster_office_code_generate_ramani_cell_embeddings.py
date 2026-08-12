#!/usr/bin/env python3
"""Generate 64D/128D Ramani embeddings with the scHiCluster two-stage SVD flow."""

import argparse
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.decomposition import TruncatedSVD


CHROMS = tuple([f"chr{index}" for index in range(1, 23)] + ["chrX"])
DIMENSIONS = (64, 128)
EXPECTED_CELL_TYPES = {"HeLa", "HAP1", "GM12878", "K562"}
NATURE_DIR = Path(__file__).resolve().parent


def portable_path(path):
    return os.path.relpath(Path(path).resolve(), NATURE_DIR)


def infer_n_bins_from_triu_k1(n_features):
    if not isinstance(n_features, (int, np.integer)) or n_features <= 0:
        raise ValueError("strict-upper-triangle width must be positive")
    discriminant = 1 + 8 * int(n_features)
    root = math.isqrt(discriminant)
    if root * root != discriminant or (1 + root) % 2:
        raise ValueError(f"{n_features} is not a valid triu_k1 width")
    n_bins = (1 + root) // 2
    if n_bins * (n_bins - 1) // 2 != n_features:
        raise ValueError(f"{n_features} is not a valid triu_k1 width")
    return n_bins


def select_distance_band(matrix, max_bin_distance=1):
    if not sparse.issparse(matrix) or matrix.ndim != 2:
        raise ValueError("input must be a two-dimensional SciPy sparse matrix")
    if int(max_bin_distance) < 1:
        raise ValueError("max_bin_distance must be positive")
    matrix = matrix.tocsr()
    n_bins = infer_n_bins_from_triu_k1(matrix.shape[1])
    rows, columns = np.triu_indices(n_bins, k=1)
    offsets = columns - rows
    selected_offsets = np.unique(offsets[offsets <= int(max_bin_distance)])
    selected = matrix[:, offsets <= int(max_bin_distance)].tocsr()
    if selected.data.size and not np.isfinite(selected.data).all():
        raise ValueError("selected contact features contain non-finite values")
    return selected, n_bins, selected_offsets


def fit_norm_sig_svd(matrix, requested_dim, random_state=100):
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError("SVD input must be two-dimensional")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.size and not np.isfinite(values).all():
        raise ValueError("SVD input contains non-finite values")
    effective_dim = min(
        int(requested_dim), matrix.shape[0] - 1, matrix.shape[1] - 1
    )
    if effective_dim < 1:
        raise ValueError(f"SVD dimension is unreachable for shape {matrix.shape}")
    model = TruncatedSVD(
        n_components=effective_dim,
        algorithm="arpack",
        random_state=int(random_state),
    )
    scores = model.fit_transform(matrix)
    singular_values = np.asarray(model.singular_values_, dtype=np.float64)
    positive = singular_values > 0
    embedding = scores[:, positive] / singular_values[positive][None, :]
    if not np.isfinite(embedding).all():
        raise ValueError("norm_sig SVD embedding contains non-finite values")
    return embedding, singular_values[positive], effective_dim


def load_labels(path):
    table = pd.read_csv(path, sep="\t")
    if not {"cell_id", "celltype"}.issubset(table.columns):
        raise ValueError("label table must contain cell_id and celltype")
    table = table.loc[:, ["cell_id", "celltype"]].copy()
    if table.isna().any().any() or table["cell_id"].duplicated().any():
        raise ValueError("label table contains missing or duplicate values")
    if set(table["celltype"].astype(str)) != EXPECTED_CELL_TYPES:
        raise ValueError("label table does not contain the four Ramani types")
    return table


def load_manifest(path):
    path = Path(path).resolve()
    table = pd.read_csv(path)
    required = {"condition_id", "display_name", "chrom_dir", "chrom_pattern"}
    if not required.issubset(table.columns):
        raise ValueError(f"manifest lacks columns {sorted(required - set(table.columns))}")
    if table["condition_id"].duplicated().any() or len(table) != 8:
        raise ValueError("manifest must contain eight unique conditions")
    records = []
    for row in table.itertuples(index=False):
        chrom_dir = (path.parent / row.chrom_dir).resolve()
        chrom_paths = {
            chrom: chrom_dir / str(row.chrom_pattern).format(chrom=chrom)
            for chrom in CHROMS
        }
        missing = [str(value) for value in chrom_paths.values() if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"{row.condition_id} missing inputs: {missing[:3]}")
        records.append({
            "condition_id": str(row.condition_id),
            "display_name": str(row.display_name),
            "is_imputed": str(row.condition_id).lower() != "raw",
            "chrom_paths": chrom_paths,
        })
    return records


def build_embedding(condition, labels, dim, output_dir, svd_seed=100):
    block_embeddings = []
    block_details = []
    for chrom in CHROMS:
        input_path = condition["chrom_paths"][chrom]
        matrix = sparse.load_npz(input_path).tocsr()
        if matrix.shape[0] != len(labels):
            raise ValueError(
                f"{condition['condition_id']} {chrom} has {matrix.shape[0]} rows"
            )
        selected, n_bins, offsets = select_distance_band(
            matrix, max_bin_distance=1
        )
        selected = selected.astype(np.float64, copy=False) * 100_000
        block_embedding, singular, effective_dim = fit_norm_sig_svd(
            selected, requested_dim=dim, random_state=svd_seed
        )
        block_embeddings.append(block_embedding)
        block_details.append({
            "chrom": chrom,
            "input_path": portable_path(input_path),
            "input_shape": [int(value) for value in matrix.shape],
            "n_bins": int(n_bins),
            "selected_bin_offsets": [int(value) for value in offsets],
            "selected_shape": [int(value) for value in selected.shape],
            "requested_svd_components": int(dim),
            "effective_svd_components": int(effective_dim),
            "singular_values": singular.astype(float).tolist(),
        })

    concatenated = np.concatenate(block_embeddings, axis=1)
    final_embedding, final_singular, final_effective = fit_norm_sig_svd(
        concatenated, requested_dim=dim, random_state=svd_seed
    )
    if final_embedding.shape != (len(labels), dim) or final_effective != dim:
        raise RuntimeError(
            f"final embedding shape {final_embedding.shape} differs from "
            f"({len(labels)}, {dim})"
        )

    condition_dir = Path(output_dir) / "embeddings" / f"dim_{dim}" / condition[
        "condition_id"
    ]
    condition_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = condition_dir / "total_decomp.npz"
    np.savez_compressed(
        embedding_path,
        data=final_embedding,
        cells=np.asarray(labels["cell_id"].astype(str).tolist(), dtype=str),
        cell_types=np.asarray(labels["celltype"].astype(str).tolist(), dtype=str),
        source=np.asarray("scHiCluster official-style two-stage SVD"),
        dimensions=np.asarray(dim, dtype=np.int64),
    )
    table = pd.DataFrame(
        final_embedding,
        columns=[f"SVD_{index}" for index in range(1, dim + 1)],
    )
    table.insert(0, "celltype", labels["celltype"].astype(str).to_numpy())
    table.insert(0, "cell_id", labels["cell_id"].astype(str).to_numpy())
    table.to_csv(condition_dir / "total_decomp.tsv.gz", sep="\t", index=False)

    metadata = {
        "condition_id": condition["condition_id"],
        "display_name": condition["display_name"],
        "is_imputed": condition["is_imputed"],
        "n_cells": len(labels),
        "chromosomes": list(CHROMS),
        "resolution_bp": 1_000_000,
        "maximum_distance_bp": 1_000_000,
        "scale_factor": 100_000,
        "norm_sig": True,
        "svd_algorithm": "arpack",
        "svd_seed": int(svd_seed),
        "requested_dimension": int(dim),
        "concatenated_shape": [int(value) for value in concatenated.shape],
        "final_shape": [int(value) for value in final_embedding.shape],
        "final_singular_values": final_singular.astype(float).tolist(),
        "embedding_path": portable_path(embedding_path),
        "block_details": block_details,
    }
    (condition_dir / "two_stage_svd_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return final_embedding, metadata


def run(manifest_path, labels_path, output_dir, dimensions=DIMENSIONS, svd_seed=100):
    manifest_path = Path(manifest_path).resolve()
    labels_path = Path(labels_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path)
    conditions = load_manifest(manifest_path)
    validation_rows = []
    for dim in tuple(int(value) for value in dimensions):
        for condition in conditions:
            embedding, metadata = build_embedding(
                condition=condition,
                labels=labels,
                dim=dim,
                output_dir=output_dir,
                svd_seed=svd_seed,
            )
            for detail in metadata["block_details"]:
                validation_rows.append({
                    "condition_id": condition["condition_id"],
                    "display_name": condition["display_name"],
                    "is_imputed": condition["is_imputed"],
                    "embedding_dim": dim,
                    "chrom": detail["chrom"],
                    "input_path": detail["input_path"],
                    "input_rows": detail["input_shape"][0],
                    "input_features": detail["input_shape"][1],
                    "selected_features": detail["selected_shape"][1],
                    "selected_offsets": ",".join(
                        str(value) for value in detail["selected_bin_offsets"]
                    ),
                    "effective_svd_components": detail[
                        "effective_svd_components"
                    ],
                    "final_rows": embedding.shape[0],
                    "final_columns": embedding.shape[1],
                    "finite": bool(np.isfinite(embedding).all()),
                    "status": "pass",
                })
            print(
                f"Completed {condition['display_name']}: {embedding.shape}",
                flush=True,
            )

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(
        output_dir / "schicluster_office_code_Ramani_embedding_validation.csv",
        index=False,
    )
    config = {
        "workflow_name": "schicluster_office_code",
        "scope": "Ramani only",
        "manifest_path": portable_path(manifest_path),
        "labels_path": portable_path(labels_path),
        "output_dir": portable_path(output_dir),
        "conditions": [condition["condition_id"] for condition in conditions],
        "dimensions": [int(value) for value in dimensions],
        "n_cells": len(labels),
        "chromosomes": list(CHROMS),
        "resolution_bp": 1_000_000,
        "maximum_distance_bp": 1_000_000,
        "scale_factor": 100_000,
        "norm_sig": True,
        "svd_algorithm": "arpack",
        "svd_seed": int(svd_seed),
        "log1p": False,
        "pca": False,
        "umap": False,
        "z_score": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output_dir / "schicluster_office_code_Ramani_embedding_run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return validation, config


def build_parser():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "RamaniData_clustering_input_paths.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=script_dir / "test/config/ramani_cells.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results/schicluster_office_code_Ramani",
    )
    parser.add_argument("--dimensions", default="64,128")
    parser.add_argument("--svd-seed", type=int, default=100)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dimensions = tuple(int(value.strip()) for value in args.dimensions.split(","))
    run(
        manifest_path=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
        dimensions=dimensions,
        svd_seed=args.svd_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
