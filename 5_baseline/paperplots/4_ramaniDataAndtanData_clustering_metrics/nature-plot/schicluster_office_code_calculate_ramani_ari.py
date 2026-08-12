#!/usr/bin/env python3
"""Calculate Ramani ARI with the downstream expression in scHiCluster example.py."""

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI


SOURCE_DIMS = (64, 128)
NDIMS = (2, 5, 10, 20, 50)
NATURE_DIR = Path(__file__).resolve().parent


def portable_path(path):
    return os.path.relpath(Path(path).resolve(), NATURE_DIR)


def load_labels(path):
    labels = pd.read_csv(path, sep="\t")
    if not {"cell_id", "celltype"}.issubset(labels.columns):
        raise ValueError("label table must contain cell_id and celltype")
    labels = labels.loc[:, ["cell_id", "celltype"]].copy()
    if len(labels) != 626 or labels["cell_id"].duplicated().any():
        raise ValueError("Ramani label table must contain 626 unique cells")
    return labels


def load_manifest(path):
    table = pd.read_csv(path)
    required = {"condition_id", "display_name"}
    if not required.issubset(table.columns) or len(table) != 8:
        raise ValueError("manifest must contain eight named conditions")
    return table.loc[:, ["condition_id", "display_name"]].copy()


def load_embedding(path, labels, source_dim):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"embedding does not exist: {path}")
    with np.load(path) as archive:
        required = {"data", "cells", "cell_types", "dimensions"}
        if not required.issubset(archive.files):
            raise ValueError(f"embedding lacks keys {sorted(required - set(archive.files))}")
        embedding = np.asarray(archive["data"])
        cells = archive["cells"].astype(str)
        cell_types = archive["cell_types"].astype(str)
        dimensions = int(archive["dimensions"])
    if embedding.shape != (626, int(source_dim)) or dimensions != int(source_dim):
        raise ValueError(f"unexpected embedding shape or dimension in {path}")
    if not np.isfinite(embedding).all():
        raise ValueError(f"non-finite embedding values in {path}")
    if not np.array_equal(cells, labels["cell_id"].astype(str).to_numpy()):
        raise ValueError(f"cell ID order mismatch in {path}")
    if not np.array_equal(
        cell_types, labels["celltype"].astype(str).to_numpy()
    ):
        raise ValueError(f"cell type order mismatch in {path}")
    return embedding


def evaluate_embedding(embedding, labels, ndims=NDIMS):
    rows = []
    predictions = {}
    for ndim in tuple(int(value) for value in ndims):
        fitted = KMeans(n_clusters=4, n_init=200).fit(embedding[:, :ndim])
        predicted = np.asarray(fitted.labels_, dtype=np.int64)
        rows.append({"ndim": ndim, "ari": float(ARI(labels, predicted))})
        predictions[ndim] = predicted
    return rows, predictions


def run(embedding_root, labels_path, manifest_path, output_dir):
    embedding_root = Path(embedding_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path)
    manifest = load_manifest(manifest_path)
    result_rows = []
    validation_rows = []
    saved_predictions = {}

    for source_dim in SOURCE_DIMS:
        for row in manifest.itertuples(index=False):
            input_path = (
                embedding_root
                / "embeddings"
                / f"dim_{source_dim}"
                / str(row.condition_id)
                / "total_decomp.npz"
            )
            embedding = load_embedding(input_path, labels, source_dim)
            ari_rows, predictions = evaluate_embedding(
                embedding, labels["celltype"].astype(str).to_numpy()
            )
            is_imputed = str(row.condition_id).lower() != "raw"
            for ari_row in ari_rows:
                result_rows.append({
                    "condition_id": str(row.condition_id),
                    "display_name": str(row.display_name),
                    "is_imputed": is_imputed,
                    "source_embedding_dim": source_dim,
                    "ndim": ari_row["ndim"],
                    "ari": ari_row["ari"],
                    "n_cells": len(labels),
                    "n_clusters": 4,
                    "n_init": 200,
                    "random_state": None,
                })
                key = (
                    f"dim{source_dim}__{row.condition_id}__"
                    f"first{ari_row['ndim']}"
                )
                saved_predictions[key] = predictions[ari_row["ndim"]]
            validation_rows.append({
                "condition_id": str(row.condition_id),
                "display_name": str(row.display_name),
                "source_embedding_dim": source_dim,
                "input_path": portable_path(input_path),
                "input_rows": embedding.shape[0],
                "input_columns": embedding.shape[1],
                "cell_ids_exact_match": True,
                "cell_types_exact_match": True,
                "finite": bool(np.isfinite(embedding).all()),
                "status": "pass",
            })
            print(
                f"Completed {row.display_name}: source_dim={source_dim}",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    expected_rows = len(manifest) * len(SOURCE_DIMS) * len(NDIMS)
    if len(results) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} ARIs, observed {len(results)}")
    if results[["condition_id", "source_embedding_dim", "ndim"]].duplicated().any():
        raise RuntimeError("duplicate ARI combinations detected")
    if not np.isfinite(results["ari"]).all() or not results["ari"].between(-1, 1).all():
        raise RuntimeError("invalid ARI value detected")

    wide = results.pivot(
        index=["condition_id", "display_name", "is_imputed"],
        columns=["source_embedding_dim", "ndim"],
        values="ari",
    )
    wide.columns = [f"SVD{dim}_first{ndim}" for dim, ndim in wide.columns]
    wide = wide.reset_index()
    results.to_csv(
        output_dir / "schicluster_office_code_Ramani_ARI_long.csv", index=False
    )
    wide.to_csv(
        output_dir / "schicluster_office_code_Ramani_ARI_wide.csv", index=False
    )
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "schicluster_office_code_Ramani_ARI_validation.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "schicluster_office_code_Ramani_KMeans_labels.npz",
        **saved_predictions,
    )
    config = {
        "workflow_name": "schicluster_office_code",
        "scope": "Ramani only",
        "official_example": portable_path(
            Path(
                "/Users/wuhaoliu/Downloads/02_First_Review/First_Review/"
                "00_compare_methods/6_scHicluster/example/example.py"
            )
        ),
        "official_expression": (
            "ARI(label, KMeans(n_clusters=nc, n_init=200)"
            ".fit(embedding[:, :ndim]).labels_)"
        ),
        "embedding_root": portable_path(embedding_root),
        "labels_path": portable_path(labels_path),
        "manifest_path": portable_path(manifest_path),
        "output_dir": portable_path(output_dir),
        "source_embedding_dims": list(SOURCE_DIMS),
        "evaluated_prefix_components": list(NDIMS),
        "n_cells": len(labels),
        "n_clusters": 4,
        "n_init": 200,
        "random_state": None,
        "additional_dimensionality_reduction": False,
        "log1p": False,
        "pca": False,
        "umap": False,
        "z_score": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output_dir / "schicluster_office_code_Ramani_ARI_run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return results, wide


def build_parser():
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "results/schicluster_office_code_Ramani"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=default_root)
    parser.add_argument(
        "--labels",
        type=Path,
        default=script_dir / "test/config/ramani_cells.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "RamaniData_clustering_input_paths.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=default_root)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _, wide = run(
        embedding_root=args.embedding_root,
        labels_path=args.labels,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print("\nRamani scHiCluster-style ARI")
    print(wide.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
