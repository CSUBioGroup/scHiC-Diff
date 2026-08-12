#!/usr/bin/env python3
"""HiRES/scHiCluster paper-style two-stage SVD for supplied 1 Mb matrices."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_samples


SUPPORTED_LAYOUTS = {"triu_k0", "triu_k1", "band_k1"}
SUPPORTED_INPUT_FORMATS = {"scipy_npz", "h5ad"}

RED_CELLTYPES = {
    "blood",
    "early mesoderm",
    "exe mesoderm",
    "early mesenchyme",
    "intermediate mesoderm",
    "myocyte",
    "myocytes",
    "mix late mesenchyme",
    "mixed late mesenchyme",
}

BLUE_CELLTYPES = {
    "neural ectoderm",
    "nmp",
    "neural tube",
    "notochord",
    "radial glia",
    "radial glias",
    "opc",
    "oligodendrocytes and progenitors",
    "early neuron",
    "early neurons",
    "schwann cell precursor",
    "schwann cell precursors",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    input_format: str
    layout: str
    input_pattern: str
    cell_order_path: str
    source_normalization: str


def mouse_chromosomes() -> List[str]:
    return ["chr{}".format(i) for i in range(1, 20)] + ["chrX"]


def infer_n_bins(n_features: int, layout: str) -> int:
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            "Unsupported layout {!r}; expected one of {}".format(
                layout,
                sorted(SUPPORTED_LAYOUTS),
            )
        )
    if n_features < 1:
        raise ValueError("Feature count must be positive, got {}".format(n_features))
    if layout == "band_k1":
        return n_features + 1

    discriminant = 1 + 8 * n_features
    root = math.isqrt(discriminant)
    if root * root != discriminant:
        raise ValueError(
            "{} features is not a valid {} flattened triangle".format(
                n_features,
                layout,
            )
        )

    if layout == "triu_k0":
        n_bins = (-1 + root) // 2
        expected = n_bins * (n_bins + 1) // 2
    else:
        n_bins = (1 + root) // 2
        expected = n_bins * (n_bins - 1) // 2
    if n_bins < 2 or expected != n_features:
        raise ValueError(
            "{} features is not a valid {} flattened triangle".format(
                n_features,
                layout,
            )
        )
    return n_bins


def extract_first_off_diagonal(matrix, layout: str):
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError("Expected a cell-by-feature matrix, got ndim={}".format(getattr(matrix, "ndim", None)))
    n_bins = infer_n_bins(matrix.shape[1], layout)
    if layout == "band_k1":
        return matrix, n_bins

    triangle_k = 0 if layout == "triu_k0" else 1
    rows, cols = np.triu_indices(n_bins, k=triangle_k)
    selected = np.flatnonzero(cols - rows == 1)
    result = matrix[:, selected]
    if result.shape[1] != n_bins - 1:
        raise RuntimeError(
            "First off-diagonal extraction returned {} features; expected {}".format(
                result.shape[1],
                n_bins - 1,
            )
        )
    return result, n_bins


def validate_cell_order(
    canonical_ids: Sequence[str],
    observed_ids: Sequence[str],
    context: str,
) -> None:
    canonical = [str(value) for value in canonical_ids]
    observed = [str(value) for value in observed_ids]
    if len(canonical) != len(observed):
        raise ValueError(
            "{}: {} canonical cells but matrix order witness has {} rows".format(
                context,
                len(canonical),
                len(observed),
            )
        )
    for index, (expected, actual) in enumerate(zip(canonical, observed), start=1):
        if expected != actual:
            raise ValueError(
                "{}: cell order mismatch at row {}: expected {!r}, observed {!r}".format(
                    context,
                    index,
                    expected,
                    actual,
                )
            )


def fit_svd_norm_sig(
    matrix,
    n_components: int = 20,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError("Expected a 2D matrix for SVD")
    min_shape = min(matrix.shape)
    if n_components < 1 or n_components >= min_shape:
        raise ValueError(
            "cannot compute {} components with arpack for matrix shape {}; "
            "n_components must be smaller than {}".format(
                n_components,
                matrix.shape,
                min_shape,
            )
        )

    model = TruncatedSVD(
        n_components=n_components,
        algorithm="arpack",
        random_state=random_state,
    )
    transformed = model.fit_transform(matrix)
    singular_values = np.asarray(model.singular_values_, dtype=np.float64)
    valid = np.isfinite(singular_values) & (singular_values > 0)
    if not valid.any():
        raise ValueError("SVD produced no positive finite singular values")
    transformed = transformed[:, valid] / singular_values[valid][None, :]
    if not np.isfinite(transformed).all():
        raise ValueError("norm_sig produced non-finite values")
    return transformed.astype(np.float32, copy=False), singular_values[valid]


def _normalize_celltype(value: object) -> str:
    return str(value).strip().casefold()


def assign_red_blue(celltypes: Iterable[object]) -> np.ndarray:
    lineages = []
    for celltype in celltypes:
        normalized = _normalize_celltype(celltype)
        if normalized in RED_CELLTYPES:
            lineages.append("Red")
        elif normalized in BLUE_CELLTYPES:
            lineages.append("Blue")
        else:
            lineages.append(None)
    return np.asarray(lineages, dtype=object)


def compute_stage_silhouettes(
    embedding: np.ndarray,
    labels: pd.DataFrame,
    n_dims: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"cell_id", "stage", "celltype"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError("Labels are missing required columns: {}".format(sorted(missing)))
    if embedding.ndim != 2:
        raise ValueError("Embedding must be two-dimensional")
    if embedding.shape[0] != len(labels):
        raise ValueError(
            "embedding has {} rows but labels has {}".format(
                embedding.shape[0],
                len(labels),
            )
        )
    if n_dims < 1 or n_dims > embedding.shape[1]:
        raise ValueError(
            "Requested {} silhouette dimensions from embedding width {}".format(
                n_dims,
                embedding.shape[1],
            )
        )

    labels_work = labels.reset_index(drop=True).copy()
    if labels_work[["cell_id", "stage", "celltype"]].isna().any().any():
        raise ValueError("cell_id, stage, and celltype must not contain missing values")
    labels_work["lineage"] = assign_red_blue(labels_work["celltype"])

    summary_rows: List[Dict[str, object]] = []
    per_cell_frames: List[pd.DataFrame] = []
    for stage in pd.unique(labels_work["stage"]):
        stage_mask = labels_work["stage"].eq(stage).to_numpy()
        lineage_mask = labels_work["lineage"].notna().to_numpy()
        used_indices = np.flatnonzero(stage_mask & lineage_mask)
        used_labels = labels_work.iloc[used_indices].copy()
        n_red = int(used_labels["lineage"].eq("Red").sum())
        n_blue = int(used_labels["lineage"].eq("Blue").sum())
        base = {
            "stage": stage,
            "n_red": n_red,
            "n_blue": n_blue,
            "n_used": int(len(used_indices)),
        }
        if n_red < 2:
            summary_rows.append(
                dict(
                    base,
                    status="skipped",
                    reason="Red has fewer than 2 cells",
                    mean_silhouette=np.nan,
                )
            )
            continue
        if n_blue < 2:
            summary_rows.append(
                dict(
                    base,
                    status="skipped",
                    reason="Blue has fewer than 2 cells",
                    mean_silhouette=np.nan,
                )
            )
            continue

        scores = silhouette_samples(
            embedding[used_indices, :n_dims],
            used_labels["lineage"].to_numpy(),
            metric="euclidean",
        )
        summary_rows.append(
            dict(
                base,
                status="ok",
                reason="",
                mean_silhouette=float(scores.mean()),
            )
        )
        used_labels["silhouette"] = scores
        per_cell_frames.append(
            used_labels[["cell_id", "stage", "celltype", "lineage", "silhouette"]]
        )

    summary_columns = [
        "stage",
        "status",
        "reason",
        "n_red",
        "n_blue",
        "n_used",
        "mean_silhouette",
    ]
    per_cell_columns = ["cell_id", "stage", "celltype", "lineage", "silhouette"]
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    if per_cell_frames:
        per_cell = pd.concat(per_cell_frames, axis=0, ignore_index=True)
    else:
        per_cell = pd.DataFrame(columns=per_cell_columns)
    return summary, per_cell


def _package_versions() -> Dict[str, str]:
    versions = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "anndata", "schicluster"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def load_canonical_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    if "cell_id" not in labels.columns:
        if "cellname" not in labels.columns:
            raise ValueError("Labels must contain cell_id or cellname")
        labels = labels.copy()
        labels["cell_id"] = labels["cellname"].astype(str)
    for column in ("cell_id", "stage", "celltype"):
        if column not in labels.columns:
            raise ValueError("Labels are missing required column {!r}".format(column))
        labels[column] = labels[column].astype(str)
    if labels["cell_id"].duplicated().any():
        duplicated = labels.loc[labels["cell_id"].duplicated(), "cell_id"].iloc[0]
        raise ValueError("Duplicate cell_id in labels: {!r}".format(duplicated))
    return labels.reset_index(drop=True)


def load_cell_order(path: Path) -> List[str]:
    table = pd.read_csv(path, sep="\t", header=None, dtype=str, usecols=[0])
    return table.iloc[:, 0].tolist()


def load_method_spec(manifest_path: Path, method: str) -> MethodSpec:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "method",
        "input_format",
        "layout",
        "input_pattern",
        "cell_order_path",
        "source_normalization",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError("Manifest is missing columns: {}".format(sorted(missing)))
    selected = manifest.loc[manifest["method"] == method]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one manifest row for method {!r}, found {}".format(
                method,
                len(selected),
            )
        )
    row = selected.iloc[0]
    spec = MethodSpec(**{field: row[field] for field in MethodSpec.__dataclass_fields__})
    if spec.layout not in SUPPORTED_LAYOUTS:
        raise ValueError("Unsupported layout in manifest: {!r}".format(spec.layout))
    if spec.input_format not in SUPPORTED_INPUT_FORMATS:
        raise ValueError("Unsupported input format in manifest: {!r}".format(spec.input_format))
    return spec


def load_npz_matrix(path: Path):
    try:
        return sparse.load_npz(path).tocsr(), "scipy_sparse_npz"
    except ValueError:
        with np.load(path, allow_pickle=False) as archive:
            if "arr_0" in archive.files:
                key = "arr_0"
            elif "X" in archive.files:
                key = "X"
            elif len(archive.files) == 1:
                key = archive.files[0]
            else:
                raise ValueError(
                    "Dense NPZ {} has ambiguous keys {}".format(path, archive.files)
                )
            matrix = archive[key]
        if matrix.ndim != 2:
            raise ValueError("Expected 2D matrix in {}, got {}".format(path, matrix.shape))
        return matrix, "dense_npz:{}".format(key)


def load_h5ad_matrix(path: Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Reading H5AD requires anndata") from exc
    adata = ad.read_h5ad(path)
    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    return matrix, adata.obs_names.astype(str).tolist(), "h5ad:X"


def _chromosome_path(pattern: str, chrom: str) -> Path:
    path = Path(pattern.format(chrom=chrom))
    if not path.exists():
        raise FileNotFoundError("Missing chromosome input: {}".format(path))
    return path


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, arr_0=np.asarray(array, dtype=np.float32))


def run_pipeline(
    manifest_path: Path,
    method: str,
    labels_path: Path,
    output_dir: Path,
    dim: int = 20,
    silhouette_dims: int = 15,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    spec = load_method_spec(manifest_path, method)
    labels = load_canonical_labels(labels_path)
    canonical_ids = labels["cell_id"].tolist()

    if spec.cell_order_path:
        witness_path = Path(spec.cell_order_path)
        if not witness_path.exists():
            raise FileNotFoundError("Cell-order witness not found: {}".format(witness_path))
        witness_ids = load_cell_order(witness_path)
        validate_cell_order(canonical_ids, witness_ids, "{} cell-order witness".format(method))

    decomp_dir = output_dir / "decomp"
    metrics_dir = output_dir / "metrics"
    decomp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    chromosome_embeddings: List[np.ndarray] = []
    chromosome_metadata: List[Dict[str, object]] = []
    for chrom in mouse_chromosomes():
        input_path = _chromosome_path(spec.input_pattern, chrom)
        if spec.input_format == "scipy_npz":
            matrix, input_kind = load_npz_matrix(input_path)
            observed_ids = None
        else:
            matrix, observed_ids, input_kind = load_h5ad_matrix(input_path)
            validate_cell_order(canonical_ids, observed_ids, "{} {} obs_names".format(method, chrom))
        if matrix.shape[0] != len(labels):
            raise ValueError(
                "{} {} has {} rows but labels has {}".format(
                    method,
                    chrom,
                    matrix.shape[0],
                    len(labels),
                )
            )

        selected, n_bins = extract_first_off_diagonal(matrix, spec.layout)
        chromosome_embedding, singular_values = fit_svd_norm_sig(
            selected,
            n_components=dim,
            random_state=random_state,
        )
        if chromosome_embedding.shape[1] != dim:
            raise ValueError(
                "{} {} produced {} positive SVD components; expected {}".format(
                    method,
                    chrom,
                    chromosome_embedding.shape[1],
                    dim,
                )
            )
        chrom_output = decomp_dir / "{}_decomp.npz".format(chrom)
        _save_array(chrom_output, chromosome_embedding)
        chromosome_embeddings.append(chromosome_embedding)
        chromosome_metadata.append(
            {
                "chromosome": chrom,
                "input_path": str(input_path),
                "input_kind": input_kind,
                "input_shape": list(matrix.shape),
                "layout": spec.layout,
                "n_bins": int(n_bins),
                "selected_distance_bins": [1],
                "selected_feature_count": int(selected.shape[1]),
                "embedding_shape": list(chromosome_embedding.shape),
                "singular_values": singular_values.astype(float).tolist(),
                "output_path": str(chrom_output),
            }
        )
        print(
            "{} {}: input={} k1={} svd={}".format(
                method,
                chrom,
                matrix.shape,
                selected.shape,
                chromosome_embedding.shape,
            ),
            flush=True,
        )

    concatenated = np.concatenate(chromosome_embeddings, axis=1).astype(np.float32, copy=False)
    expected_width = len(mouse_chromosomes()) * dim
    if concatenated.shape != (len(labels), expected_width):
        raise RuntimeError(
            "Concatenated embedding shape {} does not equal expected {}".format(
                concatenated.shape,
                (len(labels), expected_width),
            )
        )
    concat_path = decomp_dir / "total_chrom_decomp_concat.npz"
    _save_array(concat_path, concatenated)

    final_embedding, final_singular_values = fit_svd_norm_sig(
        concatenated,
        n_components=dim,
        random_state=random_state,
    )
    if final_embedding.shape != (len(labels), dim):
        raise ValueError(
            "Final SVD shape {} does not equal expected {}".format(
                final_embedding.shape,
                (len(labels), dim),
            )
        )
    final_path = decomp_dir / "total_decomp.npz"
    _save_array(final_path, final_embedding)

    final_table = pd.DataFrame(
        final_embedding,
        columns=["SVD_{}".format(i) for i in range(1, dim + 1)],
    )
    final_table.insert(0, "cell_id", canonical_ids)
    final_table_path = decomp_dir / "total_decomp.tsv.gz"
    final_table.to_csv(final_table_path, sep="\t", index=False, compression="gzip")

    silhouette_summary, silhouette_per_cell = compute_stage_silhouettes(
        final_embedding,
        labels,
        n_dims=silhouette_dims,
    )
    silhouette_summary_path = metrics_dir / "red_blue_silhouette_by_stage.tsv"
    silhouette_per_cell_path = metrics_dir / "red_blue_silhouette_per_cell.tsv"
    silhouette_summary.to_csv(silhouette_summary_path, sep="\t", index=False)
    silhouette_per_cell.to_csv(silhouette_per_cell_path, sep="\t", index=False)

    pipeline_metadata: Dict[str, object] = {
        "method": method,
        "method_spec": asdict(spec),
        "labels_path": str(labels_path),
        "n_cells": int(len(labels)),
        "chromosomes": mouse_chromosomes(),
        "resolution_bp": 1_000_000,
        "maximum_distance_bp": 1_000_000,
        "selected_bin_offsets": [1],
        "per_chromosome_svd_components": int(dim),
        "per_chromosome_norm_sig": True,
        "concatenated_shape": list(concatenated.shape),
        "final_svd_components": int(dim),
        "final_norm_sig": True,
        "final_shape": list(final_embedding.shape),
        "silhouette_dimensions": list(range(1, silhouette_dims + 1)),
        "silhouette_metric": "euclidean",
        "random_state": random_state,
        "source_normalization": spec.source_normalization,
        "package_versions": _package_versions(),
        "outputs": {
            "total_chrom_decomp_concat": str(concat_path),
            "total_decomp": str(final_path),
            "total_decomp_table": str(final_table_path),
            "red_blue_silhouette_by_stage": str(silhouette_summary_path),
            "red_blue_silhouette_per_cell": str(silhouette_per_cell_path),
        },
        "chromosome_details": chromosome_metadata,
        "excluded_steps": [
            "top_20_percent_filter",
            "additional_sqrtvc",
            "kmeans",
            "ari_nmi",
            "normalization_parameter_sweep",
        ],
    }
    metadata_path = decomp_dir / "two_stage_svd_metadata.json"
    metadata_path.write_text(
        json.dumps(pipeline_metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print("Concatenated embedding: {}".format(concatenated.shape), flush=True)
    print("Final embedding: {}".format(final_embedding.shape), flush=True)
    print("Metadata: {}".format(metadata_path), flush=True)
    return pipeline_metadata


def _parse_random_state(value: str) -> Optional[int]:
    if value.strip().lower() in {"none", "null", ""}:
        return None
    return int(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HiRES paper-style two-stage SVD on one 1 Mb method.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--silhouette-dims", type=int, default=15)
    parser.add_argument(
        "--random-state",
        type=_parse_random_state,
        default=None,
        help="Default None matches schicluster's TruncatedSVD call.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        manifest_path=args.manifest,
        method=args.method,
        labels_path=args.labels,
        output_dir=args.output_dir,
        dim=args.dim,
        silhouette_dims=args.silhouette_dims,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
