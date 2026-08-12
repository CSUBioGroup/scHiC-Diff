#!/usr/bin/env python3
"""Calculate formal Tan clustering ARI and aligned confusion matrices."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix


WORKFLOW_DIR = Path(".")
DEFAULT_MANIFEST = Path("TanData_confusion_input_paths.csv")
DEFAULT_OUTPUT_DIR = Path("results")

METHOD_ORDER = (
    "Raw",
    "scHiCluster",
    "HiCImpute",
    "Higashi-nbr0",
    "Higashi-nbr5",
    "scVI-3D",
    "Tensor-FLAMINGO",
    "scHiC-Diff",
)
SEGMENT_ORDER = ("2050", "160190")
SEGMENT_LABELS = {"2050": "20–50", "160190": "160–190"}
CELL_TYPES = ("GM12878", "PBMC")
TRUE_LABELS = np.asarray([0] * 14 + [1] * 18, dtype=int)


def load_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"method": str, "segment": str, "path": str})
    required = {"method", "segment", "path"}
    if set(table.columns) != required:
        raise ValueError(f"manifest columns must be exactly {sorted(required)}")
    expected = {(method, segment) for method in METHOD_ORDER for segment in SEGMENT_ORDER}
    actual = set(zip(table["method"], table["segment"]))
    if actual != expected or len(table) != len(expected):
        raise ValueError("manifest must contain each method/segment pair exactly once")
    return table


def load_feature_matrix(manifest_path: Path, relative_path: str) -> tuple[np.ndarray, Path]:
    input_path = (manifest_path.parent / relative_path).resolve()
    matrix = load_npz(input_path).toarray().astype(np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(TRUE_LABELS):
        raise ValueError(f"{relative_path}: expected 32 rows, observed {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{relative_path}: non-finite values detected")
    if np.min(matrix) < 0:
        raise ValueError(f"{relative_path}: negative values detected")
    return matrix, input_path


def cluster_and_evaluate(
    matrix: np.ndarray,
    n_init: int = 100,
    random_state: int = 0,
    log_transform: bool = True,
) -> dict[str, np.ndarray | float]:
    transformed = np.log1p(matrix) if log_transform else matrix
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(transformed)
    labels = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
    ).fit_predict(embedding[:, :1])

    counts = confusion_matrix(TRUE_LABELS, labels, labels=(0, 1))
    _, column_order = linear_sum_assignment(-counts)
    aligned_counts = counts[:, column_order]
    aligned_labels = np.empty_like(labels)
    for aligned_index, original_index in enumerate(column_order):
        aligned_labels[labels == original_index] = aligned_index

    row_totals = aligned_counts.sum(axis=1, keepdims=True)
    fractions = np.divide(
        aligned_counts,
        row_totals,
        out=np.zeros_like(aligned_counts, dtype=float),
        where=row_totals != 0,
    )
    return {
        "ari": float(adjusted_rand_score(TRUE_LABELS, labels)),
        "counts": aligned_counts,
        "fractions": fractions,
        "aligned_labels": aligned_labels,
        "pca_explained_variance": pca.explained_variance_ratio_,
    }


def calculate(
    manifest_path: Path,
    output_dir: Path,
    n_init: int,
    random_state: int,
    log_transform: bool = True,
):
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    confusion_rows = []
    assignment_rows = []
    input_paths = {}

    for segment in SEGMENT_ORDER:
        for method in METHOD_ORDER:
            row = manifest[
                (manifest["method"] == method) & (manifest["segment"] == segment)
            ].iloc[0]
            matrix, input_path = load_feature_matrix(manifest_path, row["path"])
            result = cluster_and_evaluate(
                matrix,
                n_init=n_init,
                random_state=random_state,
                log_transform=log_transform,
            )
            relative_input = os.path.relpath(input_path, WORKFLOW_DIR.resolve())
            input_paths[f"{method}:{segment}"] = relative_input

            summary_rows.append(
                {
                    "method": method,
                    "segment": segment,
                    "segment_label": SEGMENT_LABELS[segment],
                    "ARI": result["ari"],
                    "PC1_explained_variance": result["pca_explained_variance"][0],
                    "PC2_explained_variance": result["pca_explained_variance"][1],
                    "n_cells": matrix.shape[0],
                    "n_features": matrix.shape[1],
                    "input_path": relative_input,
                }
            )
            for true_index, true_name in enumerate(CELL_TYPES):
                for predicted_index, predicted_name in enumerate(CELL_TYPES):
                    confusion_rows.append(
                        {
                            "method": method,
                            "segment": segment,
                            "true_cell_type": true_name,
                            "predicted_cell_type": predicted_name,
                            "count": int(result["counts"][true_index, predicted_index]),
                            "fraction": result["fractions"][true_index, predicted_index],
                        }
                    )
            for cell_index, (true_label, predicted_label) in enumerate(
                zip(TRUE_LABELS, result["aligned_labels"])
            ):
                assignment_rows.append(
                    {
                        "method": method,
                        "segment": segment,
                        "cell_index": cell_index,
                        "true_cell_type": CELL_TYPES[int(true_label)],
                        "predicted_cell_type": CELL_TYPES[int(predicted_label)],
                    }
                )
            print(f"[{segment}] {method:<16} ARI={result['ari']:.4f} shape={matrix.shape}")

    summary_path = output_dir / "TanData_confusion_summary.csv"
    confusion_path = output_dir / "TanData_confusion_matrices.csv"
    assignments_path = output_dir / "TanData_cluster_assignments.csv"
    config_path = output_dir / "TanData_confusion_run_config.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)
    pd.DataFrame(assignment_rows).to_csv(assignments_path, index=False)
    config = {
        "manifest": os.path.relpath(manifest_path.resolve(), WORKFLOW_DIR.resolve()),
        "method_order": list(METHOD_ORDER),
        "segment_order": list(SEGMENT_ORDER),
        "cell_type_order": list(CELL_TYPES),
        "true_cell_counts": {"GM12878": 14, "PBMC": 18},
        "transform": "log1p" if log_transform else "none",
        "pca_components": 2,
        "kmeans_components_used": 1,
        "kmeans_n_clusters": 2,
        "kmeans_n_init": n_init,
        "kmeans_random_state": random_state,
        "cluster_alignment": "Hungarian maximum-count assignment",
        "input_paths": input_paths,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return summary_path, confusion_path, assignments_path, config_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-init", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Use nonnegative input values directly instead of applying log1p.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = calculate(
        args.manifest,
        args.output_dir,
        args.n_init,
        args.random_state,
        log_transform=not args.no_log,
    )
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
