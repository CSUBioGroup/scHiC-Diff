#!/usr/bin/env python3
"""Collect Higashi Ramani hdf5 outputs into plotting-ready chrom_npz/embedding."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


COMMON_DIR = (
    Path("/public/home/hpc254701055/2_projects/10_schicdiff")
    / "1_scHiC/5_baseline/paperplots/4_ramani_clustering_metrics/scripts"
)
sys.path.insert(0, str(COMMON_DIR.parent))
from scripts import ramani_imputation_common as common  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"


def hdf5_to_upper_tri(path: Path, chrom: str, n_cells: int, n_bins: int) -> np.ndarray:
    upper = np.triu_indices(n_bins, k=1)
    out = np.zeros((n_cells, upper[0].size), dtype=np.float32)
    with h5py.File(path, "r") as handle:
        group = handle[chrom] if chrom in handle and isinstance(handle[chrom], h5py.Group) else handle
        if "coordinates" in group:
            coords = np.asarray(group["coordinates"])
            for cell_id in range(n_cells):
                key = str(cell_id) if str(cell_id) in group else f"cell_{cell_id}"
                if key not in group:
                    continue
                values = np.asarray(group[key]).reshape(-1)
                if values.size == 0:
                    continue
                cell_coords = coords[: values.size]
                rows = np.minimum(cell_coords[:, 0], cell_coords[:, 1]).astype(int)
                cols = np.maximum(cell_coords[:, 0], cell_coords[:, 1]).astype(int)
                index = rows * n_bins - rows * (rows + 1) // 2 + (cols - rows - 1)
                out[cell_id, index] = values
        else:
            for cell_id in range(n_cells):
                if f"cell_{cell_id}" in group:
                    key = f"cell_{cell_id}"
                else:
                    key = str(cell_id)
                if key not in group:
                    continue
                matrix = np.asarray(group[key])
                if matrix.ndim == 2:
                    out[cell_id, :] = matrix[upper]
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def select_higashi_hdf5(dataset_root: Path, chrom: str, neighbor_num: int, pattern: str) -> Path:
    if pattern:
        return Path(pattern.format(dataset_root=str(dataset_root), chrom=chrom, nbr=neighbor_num))
    temp_dir = dataset_root / "temp"
    candidates = [
        temp_dir / f"{chrom}_ramani_higashi_{chrom}_nbr{neighbor_num}_nbr_{neighbor_num}_impute.hdf5",
        temp_dir / f"{chrom}_ramani_higashi_{chrom}_nbr{neighbor_num}_nbr_{neighbor_num}_impute.h5",
        temp_dir / f"{chrom}_exp_zinb3_nbr_{neighbor_num}_impute.hdf5",
        temp_dir / f"{chrom}_hicimpute_higashi_nbr_{neighbor_num}_impute.hdf5",
    ]
    candidates.extend(sorted(temp_dir.glob(f"*nbr_{neighbor_num}_impute.hdf5")))
    candidates.extend(sorted(temp_dir.glob(f"*nbr_{neighbor_num}_impute.h5")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def collect(args: argparse.Namespace) -> Path:
    rows = read_manifest(args.manifest)
    out_base = args.output_root / f"higashi_nbr{args.neighbor_num}"
    chrom_dir = out_base / "chrom_npz"
    chrom_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        chrom = row["chrom"]
        dataset_root = Path(row["dataset_root"])
        path = select_higashi_hdf5(dataset_root, chrom, args.neighbor_num, args.hdf5_pattern)
        if not path.exists():
            raise FileNotFoundError(f"{chrom}: no Higashi hdf5 found, tried {path}")
        matrix = hdf5_to_upper_tri(path, chrom, int(row["n_cells"]), int(row["n_bins"]))
        sparse.save_npz(chrom_dir / f"{chrom}.npz", sparse.csr_matrix(matrix))
    summary = common.validate_chrom_npz(chrom_dir)
    common.save_validation_json(summary, out_base / "ramani_higashi_validation.json")
    if args.make_embedding:
        embedding = out_base / "ramani_embedding.npz"
        common.save_embedding_from_chrom_npz(chrom_dir, embedding, per_chrom_dim=args.per_chrom_dim, seed=args.seed)
        common.write_manifest_snippet(
            out_base / "ramani_method_manifest_row.csv",
            method=f"higashi_nbr{args.neighbor_num}",
            display_name=f"Higashi ({args.neighbor_num} nbr)",
            source_type="embedding",
            source_path=embedding,
            notes=f"Generated from 6_Higashi/3_ramaniData nbr{args.neighbor_num}.",
        )
    print(chrom_dir)
    return chrom_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbor-num", type=int, choices=[0, 5], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hdf5-pattern", default="")
    parser.add_argument("--make-embedding", action="store_true")
    parser.add_argument("--per-chrom-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    collect(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
