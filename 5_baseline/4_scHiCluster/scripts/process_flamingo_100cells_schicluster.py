#!/usr/bin/env python3
"""Prepare 100-cell FLAMINGO data for scHiCluster and run imputation."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz, vstack


DEFAULT_DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/"
    "scHiC-Diff-master/data/SimuData/2_FLAMINGOData/100cells"
)
DEFAULT_WORK_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_scHiCDiff"
)
DEFAULT_HICLUSTER = Path("/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster")

FILENAME_RE = re.compile(
    r"^(?P<prefix>beads_(?P<bins>\d+)_W_(?P<w>[^_]+)_level_(?P<level>\d+))"
    r"_consensus_(?P<consensus>\d+)_slice_(?P<slice>\d+)$"
)


@dataclass(frozen=True)
class CellInput:
    cell_id: int
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FLAMINGO 100-cell square matrices to scHiCluster input, run imputation, and collect NPZ outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names to process; empty means all.")
    parser.add_argument("--max-cells-per-dataset", type=int, default=0, help="Use the first N cells per dataset; 0 means all.")
    parser.add_argument("--chrom", default="chr19")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--rp", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--step-size", type=int, default=500)
    parser.add_argument("--output-format", default="npz", choices=("npz", "hdf5"))
    parser.add_argument("--hicluster", type=Path, default=DEFAULT_HICLUSTER)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-impute", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    return parser.parse_args()


def mode_name(pad: int, std: float, rp: float) -> str:
    return f"pad{pad}_std{std:g}_rp{rp:g}_sqrtvc"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def dataset_from_stem(stem: str) -> str:
    match = FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"Unexpected FLAMINGO filename stem: {stem}")
    return f"{match.group('prefix')}_T{match.group('consensus')}"


def n_bins_from_dataset(dataset: str) -> int:
    match = re.match(r"^beads_(\d+)_", dataset)
    if not match:
        raise ValueError(f"Cannot parse bin count from dataset name: {dataset}")
    return int(match.group(1))


def slice_id_from_stem(stem: str) -> int:
    match = FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"Unexpected FLAMINGO filename stem: {stem}")
    return int(match.group("slice"))


def square_to_schicluster_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {matrix.shape}")
    row, col = np.triu_indices(matrix.shape[0], k=1)
    value = matrix[row, col]
    keep = value != 0
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack((row[keep], col[keep], value[keep]))


def discover_inputs(square_dir: Path) -> dict[str, list[CellInput]]:
    require(square_dir)
    grouped: dict[str, list[CellInput]] = defaultdict(list)
    for path in sorted(square_dir.glob("*.txt")):
        dataset = dataset_from_stem(path.stem)
        grouped[dataset].append(CellInput(slice_id_from_stem(path.stem), path))
    return {name: sorted(cells, key=lambda item: item.cell_id) for name, cells in sorted(grouped.items())}


def select_datasets(all_inputs: dict[str, list[CellInput]], requested: str) -> dict[str, list[CellInput]]:
    if not requested:
        return all_inputs
    names = [item.strip() for item in requested.split(",") if item.strip()]
    missing = [name for name in names if name not in all_inputs]
    if missing:
        raise ValueError(f"Requested datasets not found: {', '.join(missing)}")
    return {name: all_inputs[name] for name in names}


def write_schicluster_cell(source: Path, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return
    matrix = np.loadtxt(source)
    rows = square_to_schicluster_rows(matrix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rows.size == 0:
        dest.write_text("")
    else:
        np.savetxt(dest, rows, fmt=["%d", "%d", "%.10g"], delimiter="\t")


def prepare_inputs(
    grouped: dict[str, list[CellInput]],
    work_dir: Path,
    chrom: str,
    max_cells_per_dataset: int,
    overwrite: bool,
) -> None:
    input_root = work_dir / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    for dataset, cells in grouped.items():
        selected = cells[:max_cells_per_dataset] if max_cells_per_dataset else cells
        n_bins = n_bins_from_dataset(dataset)
        dataset_dir = input_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / f"simu_{chrom}.chrom.sizes").write_text(f"{chrom}\t{n_bins - 1}\n")
        for output_idx, cell in enumerate(selected, start=1):
            dest = dataset_dir / f"cell_{output_idx}_{chrom}.txt"
            write_schicluster_cell(cell.source, dest, overwrite)
        print(f"prepared {dataset}: {len(selected)} cells, {n_bins} bins", flush=True)


def expected_output_path(
    work_dir: Path,
    dataset: str,
    cell_id: int,
    chrom: str,
    mode: str,
    output_format: str,
) -> Path:
    return work_dir / "output" / "1_imputed_npz" / dataset / f"cell_{cell_id}_{chrom}_{mode}.{output_format}"


def run_imputation(
    grouped: dict[str, list[CellInput]],
    work_dir: Path,
    chrom: str,
    resolution: int,
    pad: int,
    std: float,
    rp: float,
    tol: float,
    window_size: int,
    step_size: int,
    output_format: str,
    hicluster: Path,
    max_cells_per_dataset: int,
    overwrite: bool,
) -> None:
    mode = mode_name(pad, std, rp)
    require(hicluster)
    for dataset, cells in grouped.items():
        selected = cells[:max_cells_per_dataset] if max_cells_per_dataset else cells
        n_bins = n_bins_from_dataset(dataset)
        input_dir = work_dir / "input" / dataset
        chrom_file = input_dir / f"simu_{chrom}.chrom.sizes"
        require(chrom_file)
        output_dir = work_dir / "output" / "1_imputed_npz" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for cell_id in range(1, len(selected) + 1):
            out_path = expected_output_path(work_dir, dataset, cell_id, chrom, mode, output_format)
            if out_path.exists() and not overwrite:
                print(f"skip existing {out_path}", flush=True)
                continue
            cmd = [
                str(hicluster),
                "impute-cell",
                "--indir",
                f"{input_dir}/",
                "--outdir",
                f"{output_dir}/",
                "--cell",
                f"cell_{cell_id}",
                "--chrom",
                chrom,
                "--res",
                str(resolution),
                "--chrom_file",
                str(chrom_file),
                "--pad",
                str(pad),
                "--std",
                str(std),
                "--rp",
                str(rp),
                "--tol",
                str(tol),
                "--window_size",
                str(window_size),
                "--step_size",
                str(step_size),
                "--output_dist",
                str(n_bins),
                "--output_format",
                output_format,
                "--mode",
                mode,
            ]
            print(f"impute {dataset} cell_{cell_id}", flush=True)
            subprocess.run(cmd, check=True)
            require(out_path)


def lower_triangle_rows(paths: list[Path], n_bins: int) -> coo_matrix:
    lower_idx = np.tril_indices(n_bins, k=-1)
    upper_idx_for_lower = (lower_idx[1], lower_idx[0])
    rows = []
    for path in paths:
        matrix = load_npz(path).toarray()
        if matrix.shape != (n_bins, n_bins):
            raise ValueError(f"{path} has shape {matrix.shape}; expected {(n_bins, n_bins)}")
        rows.append(coo_matrix(matrix[upper_idx_for_lower].reshape(1, -1)))
    return vstack(rows, format="coo")


def collect_outputs(
    grouped: dict[str, list[CellInput]],
    work_dir: Path,
    chrom: str,
    pad: int,
    std: float,
    rp: float,
    max_cells_per_dataset: int,
) -> None:
    mode = mode_name(pad, std, rp)
    out_dir = work_dir / "output" / "2_lower_tri_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, cells in grouped.items():
        selected = cells[:max_cells_per_dataset] if max_cells_per_dataset else cells
        n_bins = n_bins_from_dataset(dataset)
        paths = [
            expected_output_path(work_dir, dataset, cell_id, chrom, mode, "npz")
            for cell_id in range(1, len(selected) + 1)
        ]
        for path in paths:
            require(path)
        stacked = lower_triangle_rows(paths, n_bins)
        out_path = out_dir / f"{dataset}_scHiCluster_imputed.npz"
        save_npz(out_path, stacked)
        print(f"collected {out_path}: shape={stacked.shape}", flush=True)


def write_manifest(grouped: dict[str, list[CellInput]], work_dir: Path, max_cells_per_dataset: int) -> None:
    manifest = work_dir / "manifest.tsv"
    with manifest.open("w") as handle:
        handle.write("dataset\tn_bins\tn_cells\n")
        for dataset, cells in grouped.items():
            n_cells = min(len(cells), max_cells_per_dataset) if max_cells_per_dataset else len(cells)
            handle.write(f"{dataset}\t{n_bins_from_dataset(dataset)}\t{n_cells}\n")
    print(f"wrote {manifest}", flush=True)


def main() -> int:
    args = parse_args()
    square_dir = args.data_dir / "sim" / "0_square_matrix"
    all_inputs = discover_inputs(square_dir)
    grouped = select_datasets(all_inputs, args.datasets)
    if not grouped:
        raise ValueError(f"No input matrices found in {square_dir}")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        write_manifest(grouped, args.work_dir, args.max_cells_per_dataset)
        prepare_inputs(grouped, args.work_dir, args.chrom, args.max_cells_per_dataset, args.overwrite)
    if not args.skip_impute:
        run_imputation(
            grouped,
            args.work_dir,
            args.chrom,
            args.resolution,
            args.pad,
            args.std,
            args.rp,
            args.tol,
            args.window_size,
            args.step_size,
            args.output_format,
            args.hicluster,
            args.max_cells_per_dataset,
            args.overwrite,
        )
    if not args.skip_collect:
        if args.output_format != "npz":
            raise ValueError("Collecting lower-triangle features requires --output-format npz")
        collect_outputs(grouped, args.work_dir, args.chrom, args.pad, args.std, args.rp, args.max_cells_per_dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
