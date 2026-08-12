#!/usr/bin/env python3
"""Prepare Ramani chromosomes as Tensor-FLAMINGO raw-contact inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


COMMON_DIR = (
    Path("/public/home/hpc254701055/2_projects/10_schicdiff")
    / "1_scHiC/5_baseline/paperplots/4_ramani_clustering_metrics/scripts"
)
sys.path.insert(0, str(COMMON_DIR.parent))
from scripts import ramani_imputation_common as common  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
INPUT_SUBDIR = "contact_matrices"
TRANSFORM = "raw contact scale; upper triangle expanded to symmetric matrix"


@dataclass(frozen=True)
class ChromRecord:
    dataset: str
    chrom: str
    source_npz: str
    aligned_npz: str
    input_dir: str
    n_cells: int
    n_features: int
    n_bins: int
    input_subdir: str
    transform: str


def vector_to_square(vector: np.ndarray, n_bins: int) -> np.ndarray:
    matrix = np.zeros((n_bins, n_bins), dtype=np.float32)
    upper = np.triu_indices(n_bins, k=1)
    values = np.asarray(vector).reshape(-1).astype(np.float32, copy=False)
    if values.size != upper[0].size:
        raise ValueError(f"Expected {upper[0].size} upper-triangle values for {n_bins} bins, got {values.size}")
    values = values.copy()
    values[~np.isfinite(values)] = 0.0
    values[values < 0.0] = 0.0
    matrix[upper] = values
    matrix[(upper[1], upper[0])] = values
    np.fill_diagonal(matrix, 0.0)
    return matrix


def clean_csr(matrix) -> sparse.csr_matrix:
    out = matrix.tocsr().astype(np.float32, copy=True)
    out.data[~np.isfinite(out.data)] = 0.0
    out.data[out.data < 0.0] = 0.0
    out.eliminate_zeros()
    return out


def build_records(source_dir: Path, input_root: Path, cell_list: Path, chroms: list[str]) -> list[ChromRecord]:
    cells = common.load_cell_list(cell_list)
    aligned_dir = input_root / "raw_626_chrom_npz"
    records: list[ChromRecord] = []
    for chrom in chroms:
        source_npz = source_dir / f"{chrom}.npz"
        matrix = sparse.load_npz(source_npz)
        n_bins = common.n_bins_from_upper_triangle_size(matrix.shape[1])
        records.append(
            ChromRecord(
                dataset=chrom,
                chrom=chrom,
                source_npz=str(source_npz),
                aligned_npz=str(aligned_dir / f"{chrom}.npz"),
                input_dir=str(input_root / chrom),
                n_cells=len(cells),
                n_features=int(matrix.shape[1]),
                n_bins=int(n_bins),
                input_subdir=INPUT_SUBDIR,
                transform=TRANSFORM,
            )
        )
    return records


def write_manifest(records: list[ChromRecord], manifest: Path) -> Path:
    if not records:
        raise ValueError("No chromosomes selected")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return manifest


def select_records(records: list[ChromRecord], dataset: str | None, task_id: int | None) -> list[ChromRecord]:
    if dataset is not None and task_id is not None:
        raise ValueError("Use either --dataset or --task-id, not both")
    if dataset is not None:
        selected = [record for record in records if record.dataset == dataset or record.chrom == dataset]
        if not selected:
            raise ValueError(f"Dataset not found: {dataset}")
        return selected
    if task_id is not None:
        if task_id < 0 or task_id >= len(records):
            raise IndexError(f"task id {task_id} outside 0-{len(records) - 1}")
        return [records[task_id]]
    return records


def write_input_index(path: Path, cells: list[str]) -> None:
    rows = []
    for idx, cell in enumerate(cells):
        rows.append(
            {
                "cell_idx": idx,
                "cell_number": idx + 1,
                "cell_name": cell,
                "input_file": f"RawCount_Cell_{idx + 1:04d}.txt",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def prepare_record(record: ChromRecord, cells: list[str], cell_list: Path, manifest: Path, force: bool) -> None:
    input_dir = Path(record.input_dir)
    matrix_dir = input_dir / record.input_subdir
    marker = input_dir / ".complete"
    if marker.exists() and not force:
        return

    started = time.time()
    matrix = clean_csr(sparse.load_npz(record.aligned_npz))
    common.assert_row_count(matrix, len(cells), record.chrom)
    if matrix.shape[1] != record.n_features:
        raise ValueError(f"{record.chrom}: feature count {matrix.shape[1]} != expected {record.n_features}")

    matrix_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(matrix.shape[0]):
        out_path = matrix_dir / f"RawCount_Cell_{idx + 1:04d}.txt"
        square = vector_to_square(matrix.getrow(idx).toarray().ravel(), record.n_bins)
        np.savetxt(out_path, square, fmt="%.10g", delimiter="\t")

    sparse.save_npz(input_dir / "observed_contact_features.npz", matrix)
    write_input_index(input_dir / "input_file_index.tsv", cells)
    metadata = asdict(record)
    metadata.update(
        {
            "cell_list": str(cell_list),
            "prepared_at_epoch": time.time(),
            "prepare_seconds": time.time() - started,
            "observed_contact_nnz": int(matrix.nnz),
        }
    )
    (input_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (input_dir / "manifest.tsv").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    marker.write_text("complete\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> Path:
    chroms = common.chroms_from_arg(args.chroms)
    cells = common.load_cell_list(args.cell_list)
    aligned_dir = args.input_root / "raw_626_chrom_npz"
    records = build_records(args.source_dir, args.input_root, args.cell_list, chroms)
    manifest = write_manifest(records, args.manifest or (args.input_root / "manifest.tsv"))
    selected = select_records(records, args.dataset, args.task_id)
    common.filter_630_to_626(
        args.source_dir,
        aligned_dir,
        args.cell_list,
        [record.chrom for record in selected],
        force=args.force,
    )
    for record in selected:
        prepare_record(record, cells, args.cell_list, manifest, force=args.force)
    print(manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=common.DEFAULT_RAW_CHROM_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cell-list", type=Path, default=common.DEFAULT_CELL_LIST)
    parser.add_argument("--chroms", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
