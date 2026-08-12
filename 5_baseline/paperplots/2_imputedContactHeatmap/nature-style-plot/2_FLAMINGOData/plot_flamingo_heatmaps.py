#!/usr/bin/env python3
"""Render the FLAMINGOData 7x9 contact-map comparison figure."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy import sparse
from scipy.sparse import load_npz


SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from gr_style import (  # noqa: E402
    C_GRID,
    C_HERO,
    C_INK,
    FS_ANNOT,
    FS_LABEL,
    GR_DOUBLE_COL_IN,
    LW_SPINE,
    apply_gr_style,
    get_cmap,
    save_gr,
)


DEFAULT_MATRIX_PATHS = SCRIPT_DIR / "FLAMINGOData_heatmap_matrix_paths.tsv"
DEFAULT_METRICS = SCRIPT_DIR / "FLAMINGOData_PCC_MAE_metrics.tsv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures" / "main_fig"
OUTPUT_STEM = "FLAMINGOData_heatmap_grid_main_7x9"

N_BEADS = 500
N_FEATURES = N_BEADS * (N_BEADS - 1) // 2
HEADER_FONTSIZE = 7.5

METHODS = (
    "True",
    "Raw",
    "scVI-3D",
    "HiCImpute",
    "scHiCluster",
    "Higashi_nbr0",
    "Higashi_nbr5",
    "Tensor-FLAMINGO",
    "scHiC-Diff",
)

COLUMN_SPECS = (
    ("Raw", "Input"),
    ("True", "GT"),
    ("scHiCluster", "scHiCluster"),
    ("HiCImpute", "HiCImpute"),
    ("Higashi_nbr0", "Higashi-0"),
    ("Higashi_nbr5", "Higashi-5"),
    ("scVI-3D", "scVI-3D"),
    ("Tensor-FLAMINGO", "T-FLAMINGO"),
    ("scHiC-Diff", "scHiC-Diff"),
)

DATASETS = (
    "v3_hybrid_W0p5_500cells_level0",
    "v3_hybrid_W0p6_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0",
    "v3_hybrid_W0p8_500cells_level0",
    "v3_hybrid_W0p9_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0_r0p01",
    "v3_hybrid_W0p7_500cells_level0_r0p05",
)

DATASET_LABELS = {
    "v3_hybrid_W0p5_500cells_level0": "W=0.5",
    "v3_hybrid_W0p6_500cells_level0": "W=0.6",
    "v3_hybrid_W0p7_500cells_level0": "W=0.7",
    "v3_hybrid_W0p8_500cells_level0": "W=0.8",
    "v3_hybrid_W0p9_500cells_level0": "W=0.9",
    "v3_hybrid_W0p7_500cells_level0_r0p01": "P=1%",
    "v3_hybrid_W0p7_500cells_level0_r0p05": "P=5%",
}

TENSOR_SUBDIRS = {
    "v3_hybrid_W0p5_500cells_level0": "w0p5_r005_contact",
    "v3_hybrid_W0p6_500cells_level0": "w0p6_r005_contact",
    "v3_hybrid_W0p7_500cells_level0": "w0p7_r005_contact",
    "v3_hybrid_W0p8_500cells_level0": "w0p8_r005_contact",
    "v3_hybrid_W0p9_500cells_level0": "w0p9_r005_contact",
    "v3_hybrid_W0p7_500cells_level0_r0p01": "w0p7_r0p01_contact",
    "v3_hybrid_W0p7_500cells_level0_r0p05": "w0p7_r0p05_contact",
}


@dataclass(frozen=True)
class MatrixPathRecord:
    method: str
    path_template: str
    feature_order: str
    notes: str


@dataclass(frozen=True)
class ResolvedInput:
    method: str
    dataset: str
    path: Path
    feature_order: str


def display_path(path: Path) -> Path:
    return Path(os.path.relpath(path, start=STYLE_DIR))


@lru_cache(maxsize=None)
def read_matrix_paths(path: str) -> dict[str, MatrixPathRecord]:
    table_path = Path(path)
    records: dict[str, MatrixPathRecord] = {}
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"method", "path_template", "feature_order", "notes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Matrix-path table is missing columns: {table_path}")
        for row in reader:
            method = row["method"]
            if method in records:
                raise ValueError(f"Duplicate method in matrix-path table: {method}")
            records[method] = MatrixPathRecord(
                method=method,
                path_template=row["path_template"],
                feature_order=row["feature_order"],
                notes=row["notes"],
            )

    missing = set(METHODS) - set(records)
    extra = set(records) - set(METHODS)
    if missing or extra:
        raise ValueError(
            f"Matrix-path method mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return {method: records[method] for method in METHODS}


def resolve_input(
    records: dict[str, MatrixPathRecord],
    matrix_paths_path: Path,
    method: str,
    dataset: str,
) -> ResolvedInput:
    record = records[method]
    format_values = {"data_name": dataset}
    if method == "Tensor-FLAMINGO":
        format_values["subdir"] = TENSOR_SUBDIRS[dataset]
    path = Path(record.path_template.format(**format_values))
    if not path.is_absolute():
        path = matrix_paths_path.resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return ResolvedInput(method, dataset, path, record.feature_order)


def resolve_all_inputs(
    matrix_paths_path: Path,
) -> tuple[dict[str, MatrixPathRecord], dict[tuple[str, str], ResolvedInput]]:
    matrix_paths_path = matrix_paths_path.absolute()
    records = read_matrix_paths(str(matrix_paths_path))
    resolved = {
        (method, dataset): resolve_input(
            records,
            matrix_paths_path,
            method,
            dataset,
        )
        for dataset in DATASETS
        for method in METHODS
    }
    return records, resolved


@lru_cache(maxsize=None)
def read_pcc_index(path: str) -> dict[tuple[str, str], float]:
    metrics_path = Path(path)
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"method", "dataset", "transform", "pcc_all_mean"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"PCC/MAE metrics table is missing columns: {metrics_path}")
        rows = list(reader)

    transforms = {row["transform"] for row in rows}
    if transforms != {"raw"}:
        raise ValueError(f"Heatmap metrics must use raw values, got {sorted(transforms)}")

    index: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["method"], row["dataset"])
        if key in index:
            raise ValueError(f"Duplicate metric row: {key}")
        value = float(row["pcc_all_mean"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite pcc_all_mean: {key}")
        index[key] = value

    expected = {
        (method, dataset)
        for method, _display_name in COLUMN_SPECS[2:]
        for dataset in DATASETS
    }
    missing = expected - set(index)
    extra = set(index) - expected
    if missing or extra:
        raise ValueError(
            f"Metrics-table key mismatch: missing={sorted(missing)[:3]}, "
            f"extra={sorted(extra)[:3]}"
        )
    return index


def feature_vector_to_symmetric(
    vector: np.ndarray,
    feature_order: str,
    n_beads: int = N_BEADS,
) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    expected = n_beads * (n_beads - 1) // 2
    if values.size != expected:
        raise ValueError(f"Expected {expected} features, got {values.size}")
    matrix = np.zeros((n_beads, n_beads), dtype=np.float64)
    if feature_order == "tril":
        indices = np.tril_indices(n_beads, k=-1)
    elif feature_order == "triu":
        indices = np.triu_indices(n_beads, k=1)
    else:
        raise ValueError(f"Unsupported vector feature order: {feature_order}")
    matrix[indices] = values
    return matrix + matrix.T


def triangle_encoded_matrix_to_symmetric(
    matrix: np.ndarray,
    *,
    storage_order: str,
    feature_order: str,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square encoded matrix, got {values.shape}")
    if storage_order == "tril":
        storage_indices = np.tril_indices(values.shape[0], k=-1)
    elif storage_order == "triu":
        storage_indices = np.triu_indices(values.shape[0], k=1)
    else:
        raise ValueError(f"Unsupported storage order: {storage_order}")
    return feature_vector_to_symmetric(
        values[storage_indices],
        feature_order,
        values.shape[0],
    )


def normalize_contacts(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square contact matrix, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Contact matrix has no positive contacts")
    return values / total * 10_000.0


def dense_row(value) -> np.ndarray:
    if sparse.issparse(value) or hasattr(value, "toarray"):
        return np.asarray(value.toarray(), dtype=np.float64).reshape(-1)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def parse_h5ad_pairs(var_names) -> tuple[np.ndarray, np.ndarray, int]:
    rows = np.empty(len(var_names), dtype=np.int64)
    columns = np.empty(len(var_names), dtype=np.int64)
    prefix = "chrFLAMINGO_"
    for feature_index, name in enumerate(var_names):
        text = str(name)
        if not text.startswith(prefix):
            raise ValueError(f"Unexpected FLAMINGO feature name: {text}")
        parts = text[len(prefix) :].split("_")
        if len(parts) != 2:
            raise ValueError(f"Unexpected FLAMINGO feature name: {text}")
        rows[feature_index], columns[feature_index] = int(parts[0]), int(parts[1])
    n_beads = int(max(rows.max(), columns.max())) + 1
    return rows, columns, n_beads


def load_h5ad_gt_and_counts(path: Path, cell_index: int) -> dict[str, np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if not 0 <= cell_index < adata.n_obs:
            raise IndexError(f"Cell {cell_index} outside h5ad with {adata.n_obs} cells")
        if "gt" not in adata.layers:
            raise KeyError(f"Missing layers['gt'] in {path}")
        sources = {
            "gt": adata.layers["gt"],
            "counts": adata.layers["counts"] if "counts" in adata.layers else adata.X,
        }
        rows, columns, n_beads = parse_h5ad_pairs(adata.var_names)
        if n_beads != N_BEADS:
            raise ValueError(f"Expected {N_BEADS} beads in {path}, got {n_beads}")

        matrices = {}
        for layer, source in sources.items():
            vector = dense_row(source[cell_index, :])
            if vector.size != rows.size:
                raise ValueError(
                    f"h5ad vector/pair mismatch: {vector.size} vs {rows.size}"
                )
            matrix = np.zeros((n_beads, n_beads), dtype=np.float64)
            matrix[rows, columns] = vector
            matrix[columns, rows] = vector
            matrices[layer] = matrix
        return matrices
    finally:
        adata.file.close()


def load_sparse_cell(path: Path, feature_order: str, cell_index: int) -> np.ndarray:
    array = load_npz(path)
    if array.shape[1] == N_FEATURES:
        if not 0 <= cell_index < array.shape[0]:
            raise IndexError(f"Cell {cell_index} outside sparse matrix {array.shape}")
        vector = dense_row(array.getrow(cell_index))
    elif array.shape[0] == N_FEATURES:
        if not 0 <= cell_index < array.shape[1]:
            raise IndexError(f"Cell {cell_index} outside sparse matrix {array.shape}")
        vector = dense_row(array.getcol(cell_index))
    else:
        raise ValueError(
            f"Expected one sparse dimension to be {N_FEATURES}: {path} {array.shape}"
        )
    return feature_vector_to_symmetric(vector, feature_order)


def load_tensor_cell(path: Path, feature_order: str, cell_index: int) -> np.ndarray:
    tensor = np.load(path, mmap_mode="r")
    if tensor.ndim != 3 or tensor.shape[1:] != (N_BEADS, N_BEADS):
        raise ValueError(f"Unexpected Tensor-FLAMINGO shape: {path} {tensor.shape}")
    if not 0 <= cell_index < tensor.shape[0]:
        raise IndexError(f"Cell {cell_index} outside tensor with {tensor.shape[0]} cells")
    matrix = np.asarray(tensor[cell_index], dtype=np.float64)
    if feature_order == "tri_tensor":
        return matrix
    if feature_order == "tri_tensor_tril_encoded_triu":
        return triangle_encoded_matrix_to_symmetric(
            matrix,
            storage_order="tril",
            feature_order="triu",
        )
    raise ValueError(f"Unsupported tensor feature order: {feature_order}")


def load_matrices(
    resolved: dict[tuple[str, str], ResolvedInput],
    cell_index: int,
) -> dict[tuple[str, str], np.ndarray]:
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for dataset in DATASETS:
        raw_input = resolved[("Raw", dataset)]
        true_input = resolved[("True", dataset)]
        if raw_input.path != true_input.path:
            raise ValueError(f"Raw and True h5ad paths differ for {dataset}")
        h5ad_matrices = load_h5ad_gt_and_counts(true_input.path, cell_index)
        matrices[(dataset, "Raw")] = normalize_contacts(h5ad_matrices["counts"])
        matrices[(dataset, "True")] = normalize_contacts(h5ad_matrices["gt"])
        print(f"LOADED\tRaw+True\t{dataset}\t{display_path(true_input.path)}")

        for method, _display_name in COLUMN_SPECS[2:]:
            source = resolved[(method, dataset)]
            if source.feature_order.startswith("tri_tensor"):
                matrix = load_tensor_cell(source.path, source.feature_order, cell_index)
            else:
                matrix = load_sparse_cell(source.path, source.feature_order, cell_index)
            matrices[(dataset, method)] = normalize_contacts(matrix)
            print(f"LOADED\t{method}\t{dataset}\t{display_path(source.path)}")
    return matrices


def global_gt_vmax(
    matrices: dict[tuple[str, str], np.ndarray],
    percentile: float,
) -> float:
    positive_values = []
    for dataset in DATASETS:
        matrix = matrices[(dataset, "True")]
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        if positive.size:
            positive_values.append(positive)
    if not positive_values:
        raise ValueError("All normalized GT matrices are empty")
    vmax = float(np.percentile(np.concatenate(positive_values), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError(f"Invalid global GT vmax: {vmax}")
    return vmax


def render_grid(
    matrices: dict[tuple[str, str], np.ndarray],
    pcc_index: dict[tuple[str, str], float],
    output_dir: Path,
    vmax_percentile: float,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    apply_gr_style()
    cmap = get_cmap("hic_fall")
    norm = Normalize(vmin=0.0, vmax=global_gt_vmax(matrices, vmax_percentile))

    nrows = len(DATASETS)
    ncols = len(COLUMN_SPECS)
    figure_width = GR_DOUBLE_COL_IN
    left, right, top, bottom = 0.46, 0.47, 0.31, 0.08
    gap_x = gap_y = 0.035
    panel = (figure_width - left - right - gap_x * (ncols - 1)) / ncols
    figure_height = top + bottom + panel * nrows + gap_y * (nrows - 1)

    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        nrows,
        ncols,
        left=left / figure_width,
        right=1.0 - right / figure_width,
        top=1.0 - top / figure_height,
        bottom=bottom / figure_height,
        wspace=gap_x / panel,
        hspace=gap_y / panel,
    )

    last_image = None
    for row_index, dataset in enumerate(DATASETS):
        for column_index, (method, display_name) in enumerate(COLUMN_SPECS):
            axis = fig.add_subplot(grid[row_index, column_index])
            last_image = axis.imshow(
                matrices[(dataset, method)],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_xticks([])
            axis.set_yticks([])

            is_hero = method == "scHiC-Diff"
            for spine in axis.spines.values():
                spine.set_edgecolor(C_HERO if is_hero else C_GRID)
                spine.set_linewidth(LW_SPINE)
                spine.set_zorder(10)

            if row_index == 0:
                axis.set_title(
                    display_name,
                    fontsize=HEADER_FONTSIZE,
                    pad=3,
                    color=C_HERO if is_hero else C_INK,
                    fontweight="bold" if is_hero else "normal",
                )
            if column_index == 0:
                axis.set_ylabel(DATASET_LABELS[dataset], fontsize=FS_LABEL, labelpad=3)

            if method not in {"Raw", "True"}:
                pcc = pcc_index[(method, dataset)]
                axis.text(
                    0.02,
                    0.08,
                    f"r = {pcc:.3f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="center",
                    fontsize=FS_ANNOT,
                    color=C_HERO if is_hero else C_INK,
                    fontweight="bold" if is_hero else "normal",
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                    },
                    zorder=5,
                )

    if last_image is None:
        raise RuntimeError("No heatmap was rendered")
    colorbar_left = figure_width - right + 0.10
    colorbar_axis = fig.add_axes(
        [
            colorbar_left / figure_width,
            bottom / figure_height,
            0.075 / figure_width,
            (figure_height - top - bottom) / figure_height,
        ]
    )
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=FS_ANNOT, width=LW_SPINE, length=2)
    colorbar.outline.set_edgecolor(C_GRID)
    colorbar.outline.set_linewidth(LW_SPINE)
    colorbar.set_label("Contacts per 10,000", fontsize=FS_LABEL)

    paths = save_gr(
        fig,
        OUTPUT_STEM,
        output_dir,
        raster_dpi=dpi,
        formats=formats,
    )
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-paths", type=Path, default=DEFAULT_MATRIX_PATHS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "tiff"),
        default=("pdf", "png"),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _records, resolved = resolve_all_inputs(args.matrix_paths)
    print(f"VALIDATED\t{len(resolved)} matrix paths")
    pcc_index = read_pcc_index(str(args.metrics.absolute()))
    expected_pcc_rows = len(DATASETS) * (len(COLUMN_SPECS) - 2)
    print(f"VALIDATED\t{expected_pcc_rows} PCC records")
    if args.validate_only:
        return

    matrices = load_matrices(resolved, args.cell_index)
    output_paths = render_grid(
        matrices,
        pcc_index,
        args.output_dir,
        args.vmax_percentile,
        tuple(args.formats),
        args.dpi,
    )
    for path in output_paths:
        print(f"SAVED\t{display_path(path)}")


if __name__ == "__main__":
    main()
