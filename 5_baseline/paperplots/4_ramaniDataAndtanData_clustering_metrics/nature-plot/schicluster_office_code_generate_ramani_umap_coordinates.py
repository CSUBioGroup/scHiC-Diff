#!/usr/bin/env python3
"""Generate deterministic UMAP coordinates from configured Ramani embeddings.

The UMAP coordinates produced here are visualization-only. Ramani ARI values are
read from the separate scHiCluster-style K-means evaluation and are never
calculated from these two-dimensional coordinates. The embedding source and
prefix dimension for each method are defined in the shared plot configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from schicluster_office_code_calculate_ramani_plot_ari import (
    load_labels,
)
from schicluster_office_code_ramani_plot_config import METHOD_CONFIGS

N_NEIGHBORS = 30
MIN_DIST = 0.3
RANDOM_STATE = 0
NATURE_DIR = Path(__file__).resolve().parent


def portable_path(path):
    return os.path.relpath(Path(path).resolve(), NATURE_DIR)


def load_manifest(path):
    table = pd.read_csv(path)
    required = {"condition_id", "display_name"}
    if not required.issubset(table.columns) or len(table) != 8:
        raise ValueError("manifest must contain eight named conditions")
    table = table.loc[:, ["condition_id", "display_name"]].copy()
    if table["condition_id"].duplicated().any():
        raise ValueError("manifest condition IDs must be unique")
    return table


def load_embedding(path, expected_dim):
    with np.load(path) as archive:
        required = {"data", "cells", "cell_types", "dimensions"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"embedding lacks keys {sorted(missing)}")
        embedding = np.asarray(archive["data"], dtype=np.float64)
        cells = archive["cells"].astype(str)
        cell_types = archive["cell_types"].astype(str)
        dimensions = int(archive["dimensions"])
    if embedding.shape != (626, int(expected_dim)):
        raise ValueError(f"unexpected embedding shape in {path}: {embedding.shape}")
    if dimensions != int(expected_dim):
        raise ValueError(f"unexpected source dimension in {path}: {dimensions}")
    if not np.isfinite(embedding).all():
        raise ValueError(f"embedding contains non-finite values: {path}")
    if len(cells) != 626 or len(cell_types) != 626:
        raise ValueError(f"unexpected metadata length in {path}")
    return embedding, cells, cell_types


def fit_display_umap(matrix):
    # Some conda installations cannot write numba's package-local cache. Keep
    # that cache in the system temporary directory without changing UMAP math.
    cache_dir = Path(tempfile.gettempdir()) / "ramani_office_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    from umap import UMAP

    model = UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    coordinates = model.fit_transform(np.asarray(matrix, dtype=np.float64))
    if coordinates.shape != (626, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("UMAP did not produce a finite 626 x 2 coordinate matrix")
    return coordinates


def run(
    embedding_root,
    manifest_path,
    labels_path,
    output_path,
):
    embedding_root = Path(embedding_root).resolve()
    output_path = Path(output_path).resolve()
    manifest = load_manifest(manifest_path)
    expected_manifest = {
        (row["condition_id"], row["display_name"]) for row in METHOD_CONFIGS
    }
    observed_manifest = set(zip(manifest["condition_id"], manifest["display_name"]))
    if observed_manifest != expected_manifest:
        raise ValueError("manifest methods differ from the shared plot configuration")
    labels = load_labels(labels_path)
    frames = []
    reference_cells = None
    reference_types = None

    for config in METHOD_CONFIGS:
        condition_id = config["condition_id"]
        display_name = config["display_name"]
        source_dim = config["source_embedding_dim"]
        input_components = config["main_ndim"]
        embedding_path = (
            embedding_root
            / "embeddings"
            / f"dim_{source_dim}"
            / condition_id
            / "total_decomp.npz"
        )
        if not embedding_path.is_file():
            raise FileNotFoundError(f"embedding does not exist: {embedding_path}")
        embedding, cells, cell_types = load_embedding(embedding_path, source_dim)
        if reference_cells is None:
            reference_cells = cells
            reference_types = cell_types
        elif not np.array_equal(cells, reference_cells) or not np.array_equal(
            cell_types, reference_types
        ):
            raise ValueError(f"cell order mismatch in {embedding_path}")

        coordinates = fit_display_umap(embedding[:, :input_components])
        frames.append(
            pd.DataFrame(
                {
                    "condition_id": condition_id,
                    "method": display_name,
                    "cell_id": cells,
                    "cell_type": cell_types,
                    "UMAP1": coordinates[:, 0],
                    "UMAP2": coordinates[:, 1],
                    "source_embedding_dim": source_dim,
                    "input_components": input_components,
                    "source_kind": config["source_kind"],
                    "input_path": portable_path(embedding_path),
                }
            )
        )
        print(
            f"Completed display UMAP: {display_name} "
            f"(SVD{source_dim}, first {input_components})",
            flush=True,
        )

    result = pd.concat(frames, ignore_index=True)
    if len(result) != 8 * 626 or result[["method", "cell_id"]].duplicated().any():
        raise RuntimeError("unexpected coordinate table size or duplicate cells")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    import sklearn
    import umap

    config = {
        "workflow_name": "schicluster_office_code_Ramani_display_umap",
        "purpose": "two-dimensional visualization only; not used for K-means or ARI",
        "embedding_root": portable_path(embedding_root),
        "manifest_path": portable_path(manifest_path),
        "output_path": portable_path(output_path),
        "source_embedding_dim_by_condition": {
            row["condition_id"]: row["source_embedding_dim"]
            for row in METHOD_CONFIGS
        },
        "input_components_by_condition": {
            row["condition_id"]: row["main_ndim"] for row in METHOD_CONFIGS
        },
        "labels_path": portable_path(labels_path),
        "n_components": 2,
        "n_neighbors": N_NEIGHBORS,
        "min_dist": MIN_DIST,
        "metric": "euclidean",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "n_cells_per_method": 626,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "umap_learn": umap.__version__,
    }
    config_path = output_path.with_name(
        "schicluster_office_code_Ramani_display_UMAP_run_config.json"
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser():
    result_dir = NATURE_DIR / "results/schicluster_office_code_Ramani"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=result_dir)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=NATURE_DIR / "RamaniData_clustering_input_paths.csv",
    )
    parser.add_argument(
        "--labels", type=Path, default=NATURE_DIR / "test/config/ramani_cells.tsv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=result_dir
        / "schicluster_office_code_Ramani_cluster_coordinates.csv",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run(
        args.embedding_root,
        args.manifest,
        args.labels,
        args.output,
    )
    print(f"Saved {args.output} ({len(result)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
