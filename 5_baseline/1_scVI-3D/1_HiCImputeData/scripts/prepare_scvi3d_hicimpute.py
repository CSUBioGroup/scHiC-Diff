#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np


CHROM_NAME = "chr19"
RESOLUTION = 1
N_BINS = 61
N_CELLS = 100

DEFAULT_RAW_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "0_gtData/0_downsampled_HiCImputeData"
)
DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "1_scVI-3D/1_HiCImputeData/input"
)
DATASETS = [
    "K562_T1_1k", "K562_T1_2k", "K562_T1_4k", "K562_T1_7k",
    "K562_T2_1k", "K562_T2_2k", "K562_T2_4k", "K562_T2_7k",
    "K562_T3_1k", "K562_T3_2k", "K562_T3_4k", "K562_T3_7k",
]
CELL_RE = re.compile(r"cell_(\d+)_chr19\.txt$")


def cell_sort_key(path: Path) -> int:
    match = CELL_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected cell filename: {path}")
    return int(match.group(1))


def infer_cell_type(dataset: str) -> str:
    parts = dataset.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def read_triplets(path: Path) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.empty((0, 3), dtype=np.int64)
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, 3)
    return data


def write_cell(args_tuple) -> tuple[int, int]:
    cell_idx, raw_path, dest_dir, overwrite = args_tuple
    dest = dest_dir / f"cell_{cell_idx + 1}.txt"
    if dest.exists() and not overwrite:
        return cell_idx, -1

    triplets = read_triplets(raw_path)
    if triplets.shape[0] == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("")
        return cell_idx, 0

    rows = np.column_stack([
        np.full(triplets.shape[0], CHROM_NAME, dtype=object),
        triplets[:, 0].astype(np.int64),
        np.full(triplets.shape[0], CHROM_NAME, dtype=object),
        triplets[:, 1].astype(np.int64),
        triplets[:, 2].astype(np.int64),
    ])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dest, rows, fmt=["%s", "%d", "%s", "%d", "%d"], delimiter="\t")
    return cell_idx, int(triplets.shape[0])


def prepare_dataset(
    dataset: str,
    raw_root: Path,
    input_root: Path,
    workers: int,
    overwrite: bool,
) -> None:
    source_dir = raw_root / dataset
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    raw_files = sorted(source_dir.glob("cell_*_chr19.txt"), key=cell_sort_key)
    if len(raw_files) != N_CELLS:
        raise ValueError(f"{dataset}: expected {N_CELLS} cell files, got {len(raw_files)}")

    dest_dir = input_root / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "genome.txt").write_text(f"{CHROM_NAME}\t{N_BINS - 1}\n")

    cell_type = infer_cell_type(dataset)
    with (dest_dir / "cell_summary.txt").open("w") as handle:
        handle.write("name\tbatch\tcell_type\n")
        for cell_id in range(1, N_CELLS + 1):
            handle.write(f"cell_{cell_id}.txt\tbatch1\t{cell_type}\n")

    items = [(idx, raw_files[idx], dest_dir, overwrite) for idx in range(N_CELLS)]
    if workers <= 1:
        results = [write_cell(item) for item in items]
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(write_cell, items, chunksize=max(1, N_CELLS // (workers * 4)))

    written = sum(1 for _, n_rows in results if n_rows >= 0)
    skipped = sum(1 for _, n_rows in results if n_rows == -1)
    total_rows = sum(n_rows for _, n_rows in results if n_rows > 0)
    print(
        f"[scvi-prep] {dataset}: cells={N_CELLS} written={written} skipped={skipped} rows={total_rows}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HiCImputeData for scVI-3D")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    datasets = args.datasets or DATASETS
    args.input_root.mkdir(parents=True, exist_ok=True)
    print(f"[scvi-prep] datasets={datasets} workers={workers}", flush=True)
    for dataset in datasets:
        prepare_dataset(dataset, args.raw_root, args.input_root, workers, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
