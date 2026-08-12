#!/usr/bin/env python3
"""Validate 100-cell FLAMINGO scHiCluster imputation outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scipy.sparse import load_npz


DEFAULT_WORK_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_scHiCDiff"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FLAMINGO 100-cell scHiCluster outputs.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--chrom", default="chr19")
    parser.add_argument("--mode", default="pad1_std1_rp0.5_sqrtvc")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def n_bins_from_dataset(dataset: str) -> int:
    match = re.match(r"^beads_(\d+)_", dataset)
    if not match:
        raise ValueError(f"Cannot parse bin count from dataset name: {dataset}")
    return int(match.group(1))


def read_manifest(path: Path) -> list[tuple[str, int, int]]:
    require(path)
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if line_no == 1:
            if line != "dataset\tn_bins\tn_cells":
                raise ValueError(f"Unexpected manifest header: {line}")
            continue
        dataset, n_bins, n_cells = line.split("\t")
        rows.append((dataset, int(n_bins), int(n_cells)))
    return rows


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    rows = read_manifest(work_dir / "manifest.tsv")
    if len(rows) != 36:
        raise ValueError(f"Expected 36 datasets, found {len(rows)}")

    total_cells = 0
    for dataset, n_bins, n_cells in rows:
        if n_bins != n_bins_from_dataset(dataset):
            raise ValueError(f"{dataset} manifest n_bins={n_bins} does not match name")
        total_cells += n_cells
        n_features = n_bins * (n_bins - 1) // 2
        chrom_file = work_dir / "input" / dataset / f"simu_{args.chrom}.chrom.sizes"
        require(chrom_file)
        expected_chrom = f"{args.chrom}\t{n_bins - 1}\n"
        if chrom_file.read_text() != expected_chrom:
            raise ValueError(f"Unexpected chrom sizes content in {chrom_file}")

        input_files = sorted((work_dir / "input" / dataset).glob(f"cell_*_{args.chrom}.txt"))
        if len(input_files) != n_cells:
            raise ValueError(f"{dataset} has {len(input_files)} input files; expected {n_cells}")

        imputed_dir = work_dir / "output" / "1_imputed_npz" / dataset
        require(imputed_dir)
        imputed_files = sorted(imputed_dir.glob(f"cell_*_{args.chrom}_{args.mode}.npz"))
        if len(imputed_files) != n_cells:
            raise ValueError(f"{dataset} has {len(imputed_files)} imputed files; expected {n_cells}")
        first = load_npz(imputed_dir / f"cell_1_{args.chrom}_{args.mode}.npz")
        if first.shape != (n_bins, n_bins):
            raise ValueError(f"{dataset} first imputed matrix shape {first.shape}; expected {(n_bins, n_bins)}")

        lower_path = work_dir / "output" / "2_lower_tri_npz" / f"{dataset}_scHiCluster_imputed.npz"
        require(lower_path)
        lower = load_npz(lower_path)
        if lower.shape != (n_cells, n_features):
            raise ValueError(f"{lower_path} shape {lower.shape}; expected {(n_cells, n_features)}")

    print(f"Validation OK: {len(rows)} datasets, {total_cells} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
