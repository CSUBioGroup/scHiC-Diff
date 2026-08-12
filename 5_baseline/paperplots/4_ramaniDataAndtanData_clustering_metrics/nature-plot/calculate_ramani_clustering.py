#!/usr/bin/env python3
"""Calculate standardized Ramani UMAP coordinates and clustering metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


WORKFLOW_DIR = Path(".")
DEFAULT_MANIFEST = Path("RamaniData_clustering_input_paths.csv")
DEFAULT_CELL_LIST = Path(
    "../../../../1_Dataset/2-Ramani-GSE84920-ML1-ML3/"
    "upper_npz/1000000bp/ML1_ML3_cell_list.txt"
)
DEFAULT_RESULTS_DIR = Path("results")
CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX"])
DEFAULT_VIZ_N_NEIGHBORS = 30
DEFAULT_VIZ_MIN_DIST = 0.3
DEFAULT_SCHICLUSTER_NPZ = Path("../test/output/scHiCluster_embedding.npz")


def portable_path(path: Path) -> str:
    """Store paths relative to this workflow so results remain relocatable."""
    relative = os.path.relpath(path.resolve(), start=WORKFLOW_DIR.resolve())
    return Path(relative).as_posix()


@dataclass(frozen=True)
class InputSpec:
    condition_id: str
    display_name: str
    chrom_dir: Path
    chrom_pattern: str
    plot_cluster: bool
    calculate_ari: bool
    notes: str


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: Path) -> list[InputSpec]:
    required = {
        "condition_id",
        "display_name",
        "chrom_dir",
        "chrom_pattern",
        "plot_cluster",
        "calculate_ari",
        "notes",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    base_dir = path.resolve().parent
    specs = [
        InputSpec(
            condition_id=row["condition_id"].strip(),
            display_name=row["display_name"].strip(),
            chrom_dir=(base_dir / row["chrom_dir"].strip()).resolve(),
            chrom_pattern=row["chrom_pattern"].strip(),
            plot_cluster=parse_bool(row["plot_cluster"]),
            calculate_ari=parse_bool(row["calculate_ari"]),
            notes=row["notes"].strip(),
        )
        for row in rows
    ]
    ids = [spec.condition_id for spec in specs]
    names = [spec.display_name for spec in specs]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("condition_id and display_name must be unique")
    return specs


def validate_contract(specs: list[InputSpec]) -> None:
    plotted = [spec for spec in specs if spec.plot_cluster]
    evaluated = [spec for spec in specs if spec.calculate_ari]
    if len(plotted) != 8:
        raise ValueError(f"Expected 8 cluster conditions, found {len(plotted)}")
    if len(evaluated) != 7:
        raise ValueError(f"Expected 7 ARI methods, found {len(evaluated)}")
    raw = [spec for spec in specs if spec.condition_id == "raw"]
    if len(raw) != 1 or raw[0].calculate_ari:
        raise ValueError("Raw must appear once and must be excluded from formal ARI")


def load_cells(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        cells = [line.strip() for line in handle if line.strip()]
    if len(cells) != 626:
        raise ValueError(f"Expected 626 cells, found {len(cells)} in {path}")
    cell_types = np.asarray([cell.split("_")[0] for cell in cells], dtype=object)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(cell_types)
    if len(encoder.classes_) != 4:
        raise ValueError(f"Expected 4 cell types, found {encoder.classes_.tolist()}")
    return cells, cell_types, labels


def parse_integer_range(raw: str) -> list[int]:
    """Parse a mixed spec like '1-10,20,50' into a deduplicated ordered list."""
    dims: list[int] = []
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "-" in segment:
            low, high = segment.split("-", 1)
            dims.extend(range(int(low), int(high) + 1))
        else:
            dims.append(int(segment))
    seen: set[int] = set()
    return [d for d in dims if not (d in seen or seen.add(d))]


def build_chromosome_feature(
    spec: InputSpec,
    chrom: str,
    expected_rows: int,
    svd_dim: int,
    svd_seed: int,
    log_transform: bool = True,
) -> tuple[str, str, np.ndarray, dict[str, object]]:
    path = spec.chrom_dir / spec.chrom_pattern.format(chrom=chrom)
    if not path.exists():
        raise FileNotFoundError(f"{spec.display_name}: missing {path}")
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{spec.display_name} {chrom}: {matrix.shape[0]} rows, "
            f"expected {expected_rows}"
        )
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{spec.display_name} {chrom}: non-finite values found")

    negative_count = int(np.count_nonzero(matrix.data < 0))
    original_min = float(matrix.data.min()) if matrix.nnz else 0.0
    original_max = float(matrix.data.max()) if matrix.nnz else 0.0
    if negative_count:
        raise ValueError(
            f"{spec.display_name} {chrom}: found {negative_count} negative "
            "contact values; fair evaluation does not clip inputs"
        )

    # Each chromosome is independent, so these reductions can be distributed
    # across CPU processes without changing the validated feature definition.
    dense = matrix.toarray().astype(np.float64, copy=False)
    transformed = np.log1p(dense) if log_transform else dense
    reduced = TruncatedSVD(
        n_components=svd_dim,
        random_state=svd_seed,
    ).fit_transform(transformed)
    validation = {
        "condition_id": spec.condition_id,
        "method": spec.display_name,
        "chromosome": chrom,
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "nnz_input": int(matrix.nnz),
        "negative_values": negative_count,
        "svd_input_dtype": str(transformed.dtype),
        "original_min": original_min,
        "original_max": original_max,
        "path": portable_path(path),
    }
    return (
        spec.condition_id,
        chrom,
        reduced.astype(np.float32, copy=False),
        validation,
    )


def to_embedding(features: np.ndarray) -> np.ndarray:
    """Global PCA on concatenated per-chrom reductions (scHiCluster raw_pca style)."""
    n_components = min(features.shape[0], features.shape[1]) - 1
    return PCA(n_components=n_components).fit_transform(features).astype(np.float32)


def fit_umap_and_kmeans(
    features: np.ndarray,
    dimension: int,
    seed: int,
    n_clusters: int,
    kmeans_n_init: int,
) -> tuple[np.ndarray, np.ndarray]:
    import umap

    embedding = umap.UMAP(
        n_components=dimension,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(features)
    predicted = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=kmeans_n_init,
        random_state=seed,
    ).fit_predict(embedding)
    return embedding, predicted


def fit_visualization_umap(
    features: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Build the dedicated 2D embedding used only for the cluster figure."""
    import umap

    return umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=1,
    ).fit_transform(features)


def evaluate_metric_task(
    features: np.ndarray,
    labels_true: np.ndarray,
    spec: InputSpec,
    seed: int,
    dimension: int,
    kmeans_n_init: int,
) -> dict[str, object]:
    embedding, predicted = fit_umap_and_kmeans(
        features,
        dimension=dimension,
        seed=seed,
        n_clusters=4,
        kmeans_n_init=kmeans_n_init,
    )
    return {
        "condition_id": spec.condition_id,
        "method": spec.display_name,
        "seed": seed,
        "umap_dimension": dimension,
        "ARI": float(metrics.adjusted_rand_score(labels_true, predicted)),
        "NMI": float(metrics.normalized_mutual_info_score(labels_true, predicted)),
        "silhouette": float(silhouette_score(embedding, predicted)),
        "n_cells": len(labels_true),
        "n_clusters": 4,
        "formal_comparison": spec.calculate_ari,
    }


def summarize_ari(
    condition_id: str,
    method: str,
    rows: list[dict[str, object]],
    fixed_dim: int,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    fixed = frame[frame["umap_dimension"] == fixed_dim]
    picked = frame.loc[frame.groupby("seed")["silhouette"].idxmax()]
    maxima = frame.loc[frame.groupby("seed")["ARI"].idxmax()]
    mean_ari_by_dimension = frame.groupby("umap_dimension")["ARI"].mean()
    best_mean_dimension = int(mean_ari_by_dimension.idxmax())
    by_best_mean = frame[frame["umap_dimension"] == best_mean_dimension]

    def summary_row(
        rule: str,
        selected: pd.DataFrame,
        reportable: bool,
        selected_dimensions: str,
    ) -> dict[str, object]:
        return {
            "condition_id": condition_id,
            "method": method,
            "rule": rule,
            "ARI_mean": float(selected["ARI"].mean()),
            "ARI_std": float(selected["ARI"].std(ddof=0)),
            "n_seeds": int(selected["seed"].nunique()),
            "selected_dimensions": selected_dimensions,
            "reportable": reportable,
        }

    return [
        summary_row(
            f"fixed_dim={fixed_dim}",
            fixed,
            True,
            str(fixed_dim),
        ),
        summary_row(
            "unsupervised_dim_by_silhouette",
            picked,
            True,
            ",".join(str(value) for value in picked.sort_values("seed")["umap_dimension"]),
        ),
        summary_row(
            "best_mean_ARI_over_dimensions",
            by_best_mean,
            False,
            str(best_mean_dimension),
        ),
        summary_row(
            "max_ARI_over_dimensions",
            maxima,
            False,
            ",".join(str(value) for value in maxima.sort_values("seed")["umap_dimension"]),
        ),
    ]


def calculate(args: argparse.Namespace) -> None:
    specs = read_manifest(args.manifest)
    validate_contract(specs)
    cells, cell_types, labels_true = load_cells(args.cell_list)
    seeds = parse_integer_range(args.seeds)
    dimensions = parse_integer_range(args.umap_dimensions)
    if args.fixed_dim not in dimensions:
        raise ValueError("fixed_dim must be included in umap_dimensions")
    if args.viz_seed not in seeds:
        raise ValueError("viz_seed must be included in seeds")
    if args.n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    coordinate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    feature_sets: dict[str, np.ndarray] = {}

    parallel_options = {
        "n_jobs": args.n_jobs,
        "backend": "loky",
        "batch_size": 1,
        "max_nbytes": "100K",
        "mmap_mode": "r",
        "verbose": 5,
    }
    with Parallel(**parallel_options) as parallel:
        chromosome_tasks = [
            (spec, chrom) for spec in specs for chrom in CHROMS
        ]
        print(
            f"[features] reducing {len(chromosome_tasks)} chromosomes "
            f"with {args.n_jobs} processes",
            flush=True,
        )
        chromosome_results = parallel(
            delayed(build_chromosome_feature)(
                spec,
                chrom,
                expected_rows=len(cells),
                svd_dim=args.svd_dim,
                svd_seed=args.svd_seed,
                log_transform=not args.no_log,
            )
            for spec, chrom in chromosome_tasks
        )

        validation_rows = [result[3] for result in chromosome_results]
        chromosome_features = {
            (condition_id, chrom): reduced
            for condition_id, chrom, reduced, _ in chromosome_results
        }
        expected_results = len(specs) * len(CHROMS)
        if len(chromosome_features) != expected_results:
            raise ValueError(
                f"Expected {expected_results} chromosome reductions, "
                f"found {len(chromosome_features)}"
            )
        for chrom in CHROMS:
            feature_counts = {
                int(row["features"])
                for row in validation_rows
                if row["chromosome"] == chrom
            }
            if len(feature_counts) != 1:
                raise ValueError(
                    f"{chrom}: inconsistent feature counts across methods: "
                    f"{sorted(feature_counts)}"
                )

        for spec in specs:
            if spec.condition_id == "scHiCluster" and args.schicluster_npz.exists():
                native_embedding = np.load(
                    args.schicluster_npz, allow_pickle=True
                )["embedding"].astype(np.float32, copy=False)
                if native_embedding.shape[0] != len(cells):
                    raise ValueError(
                        f"scHiCluster native embedding has "
                        f"{native_embedding.shape[0]} rows, "
                        f"expected {len(cells)}"
                    )
                embedding = native_embedding
                print(
                    f"[scHiCluster] native embedding {embedding.shape} "
                    f"(used directly)",
                    flush=True,
                )
            else:
                features = np.hstack(
                    [
                        chromosome_features[(spec.condition_id, chrom)]
                        for chrom in CHROMS
                    ]
                ).astype(np.float32, copy=False)
                embedding = to_embedding(features)
                print(
                    f"[{spec.display_name}] concat={features.shape} "
                    f"-> PCA {embedding.shape}",
                    flush=True,
                )
            feature_sets[spec.condition_id] = embedding
            np.savez_compressed(
                feature_dir / f"{spec.condition_id}_features.npz",
                data=embedding,
                cells=np.asarray(cells, dtype=object),
                cell_types=cell_types,
                svd_dim=args.svd_dim,
                svd_seed=args.svd_seed,
            )

        print(
            f"[visualization] fitting {len(specs)} dedicated 2D UMAPs "
            f"with {args.n_jobs} processes",
            flush=True,
        )
        visualization_embeddings = parallel(
            delayed(fit_visualization_umap)(
                feature_sets[spec.condition_id],
                seed=args.viz_seed,
                n_neighbors=args.viz_n_neighbors,
                min_dist=args.viz_min_dist,
            )
            for spec in specs
        )
        for spec, viz_embedding in zip(specs, visualization_embeddings):
            if not spec.plot_cluster:
                continue
            for index, (cell, cell_type) in enumerate(zip(cells, cell_types)):
                coordinate_rows.append(
                    {
                        "condition_id": spec.condition_id,
                        "method": spec.display_name,
                        "cell_index": index,
                        "cell_id": cell,
                        "cell_type": cell_type,
                        "UMAP1": float(viz_embedding[index, 0]),
                        "UMAP2": float(viz_embedding[index, 1]),
                        "umap_dimension": 2,
                        "seed": args.viz_seed,
                        "n_neighbors": args.viz_n_neighbors,
                        "min_dist": args.viz_min_dist,
                    }
                )

        evaluated_specs = [
            spec
            for spec in specs
            if spec.calculate_ari or spec.condition_id == "raw"
        ]
        metric_tasks = [
            (spec, seed, dimension)
            for spec in evaluated_specs
            for seed in seeds
            for dimension in dimensions
        ]
        print(
            f"[metrics] evaluating {len(metric_tasks)} UMAP/KMeans combinations "
            f"with {args.n_jobs} processes",
            flush=True,
        )
        metric_rows = parallel(
            delayed(evaluate_metric_task)(
                feature_sets[spec.condition_id],
                labels_true,
                spec,
                seed,
                dimension,
                args.kmeans_n_init,
            )
            for spec, seed, dimension in metric_tasks
        )

    for spec in evaluated_specs:
        method_metric_rows = [
            row for row in metric_rows if row["condition_id"] == spec.condition_id
        ]
        method_summaries = summarize_ari(
            spec.condition_id,
            spec.display_name,
            method_metric_rows,
            args.fixed_dim,
        )
        for row in method_summaries:
            row["formal_comparison"] = spec.calculate_ari
        summary_rows.extend(method_summaries)

    pd.DataFrame(validation_rows).to_csv(
        args.output_dir / "RamaniData_input_validation.csv", index=False
    )
    pd.DataFrame(coordinate_rows).to_csv(
        args.output_dir / "RamaniData_cluster_coordinates.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "RamaniData_ARI_per_seed_dimension.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "RamaniData_ARI_summary.csv", index=False
    )
    config = {
        "path_base": "nature-plot directory",
        "manifest": portable_path(args.manifest),
        "cell_list": portable_path(args.cell_list),
        "cluster_conditions": 8,
        "ari_methods": 7,
        "metric_conditions": 8,
        "raw_metrics_role": "reference only; excluded from seven-method formal comparison",
        "chromosomes": list(CHROMS),
        "preprocessing": (
            "reject negatives; dense float64 "
            f"{'without transform' if args.no_log else 'log1p'}; "
            "per-chrom TruncatedSVD; hstack float32; global PCA to embedding "
            "(scHiCluster uses native embedding directly)"
        ),
        "svd_dim": args.svd_dim,
        "svd_seed": args.svd_seed,
        "schicluster_npz": (
            portable_path(args.schicluster_npz)
            if args.schicluster_npz.exists()
            else None
        ),
        "umap_dimensions": dimensions,
        "seeds": seeds,
        "fixed_dim": args.fixed_dim,
        "viz_seed": args.viz_seed,
        "viz_n_neighbors": args.viz_n_neighbors,
        "viz_min_dist": args.viz_min_dist,
        "kmeans_n_init": args.kmeans_n_init,
        "n_jobs": args.n_jobs,
        "primary_ari_rule": "max_ARI_over_dimensions",
    }
    (args.output_dir / "RamaniData_clustering_run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote results to {args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cell-list", type=Path, default=DEFAULT_CELL_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--umap-dimensions", default="1-10,20,50")
    parser.add_argument("--fixed-dim", type=int, default=2)
    parser.add_argument("--viz-seed", type=int, default=0)
    parser.add_argument(
        "--viz-n-neighbors", type=int, default=DEFAULT_VIZ_N_NEIGHBORS
    )
    parser.add_argument("--viz-min-dist", type=float, default=DEFAULT_VIZ_MIN_DIST)
    parser.add_argument("--svd-dim", type=int, default=5)
    parser.add_argument("--svd-seed", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=200)
    parser.add_argument(
        "--schicluster-npz",
        type=Path,
        default=DEFAULT_SCHICLUSTER_NPZ,
        help="Precomputed scHiCluster native embedding; used directly instead of "
        "rebuilding features from chrom_npz.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Use nonnegative input values directly instead of applying log1p.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    calculate(parse_args())
