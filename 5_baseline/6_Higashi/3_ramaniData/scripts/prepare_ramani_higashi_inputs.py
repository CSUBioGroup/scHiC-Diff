#!/usr/bin/env python3
"""Build per-chromosome Ramani Higashi configs for nbr0 or nbr5."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
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
CHROM_INDEX = 0


def vector_to_rows(vector: np.ndarray, cell_id: int, n_bins: int, min_delta: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices(n_bins, k=1)
    values = np.asarray(vector).reshape(-1)
    if values.size != rows.size:
        raise ValueError(f"Expected {rows.size} upper-triangle values, got {values.size}")
    mask = np.isfinite(values) & (values > 0) & ((cols - rows) >= min_delta)
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.int64), np.empty((0,), dtype=np.float32)
    coords = np.column_stack(
        [
            np.full(mask.sum(), cell_id, dtype=np.int64),
            np.full(mask.sum(), CHROM_INDEX, dtype=np.int64),
            rows[mask].astype(np.int64),
            cols[mask].astype(np.int64),
        ]
    )
    return coords, values[mask].astype(np.float32, copy=False)


def write_label_info(path: Path, cells: list[str]) -> None:
    labels = {
        "cell_name": cells,
        "cell_type": [cell.split("_", 1)[0] for cell in cells],
    }
    with path.open("wb") as handle:
        pickle.dump(labels, handle, protocol=4)


def build_config(dataset_root: Path, chrom: str, n_bins: int, neighbor_num: int, args: argparse.Namespace) -> dict[str, object]:
    data_dir = dataset_root / "data"
    temp_dir = dataset_root / "temp"
    chrom_name = chrom
    return {
        "data_dir": str(data_dir.resolve()),
        "temp_dir": str(temp_dir.resolve()),
        "genome_reference_path": str((data_dir / "ramani.chrom.sizes").resolve()),
        "chrom_list": [chrom_name],
        "impute_list": [chrom_name],
        "resolution": args.resolution,
        "resolution_cell": args.resolution,
        "minimum_distance": 0,
        "maximum_distance": -1,
        "minimum_impute_distance": 0,
        "maximum_impute_distance": -1,
        "local_transfer_range": 1,
        "dimensions": args.dimensions,
        "embedding_name": f"ramani_higashi_{chrom}_nbr{neighbor_num}",
        "loss_mode": args.loss_mode,
        "neighbor_num": neighbor_num,
        "cpu_num": args.cpu_num,
        "cpu_num_torch": max(1, min(args.cpu_num, 4)),
        "gpu_num": args.gpu_num,
        "embedding_epoch": args.embedding_epoch,
        "no_nbr_epoch": args.no_nbr_epoch,
        "with_nbr_epoch": args.with_nbr_epoch,
        "impute_no_nbr": True,
        "impute_with_nbr": neighbor_num > 0,
        "correct_be_impute": False,
        "precompute_weighted_nbr": True,
        "input_format": "higashi_v1",
        "structured": False,
        "contact_header": ["cell_id", "chrom1", "chrom2", "pos1", "pos2", "count"],
        "header_included": True,
    }


def prepare(args: argparse.Namespace) -> Path:
    chroms = common.chroms_from_arg(args.chroms)
    cells = common.load_cell_list(args.cell_list)
    aligned_dir = args.input_root / "raw_626_chrom_npz"
    common.filter_630_to_626(args.source_dir, aligned_dir, args.cell_list, chroms, force=args.force)
    rows_for_manifest: list[dict[str, object]] = []
    for chrom in chroms:
        matrix = sparse.load_npz(aligned_dir / f"{chrom}.npz").tocsr()
        common.assert_row_count(matrix, len(cells), chrom)
        n_bins = common.n_bins_from_upper_triangle_size(matrix.shape[1])
        dataset_root = args.input_root / f"{chrom}_nbr{args.neighbor_num}"
        data_dir = dataset_root / "data"
        temp_dir = dataset_root / "temp"
        data_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "raw").mkdir(parents=True, exist_ok=True)
        if args.force or not (dataset_root / "config.JSON").exists():
            coord_parts = []
            weight_parts = []
            for cell_id in range(matrix.shape[0]):
                coords, weights = vector_to_rows(matrix.getrow(cell_id).toarray().ravel(), cell_id, n_bins, args.min_delta)
                if coords.size:
                    coord_parts.append(coords)
                    weight_parts.append(weights)
            if not coord_parts:
                raise ValueError(f"{chrom}: no positive contacts after filtering")
            np.save(temp_dir / "data.npy", np.concatenate(coord_parts, axis=0), allow_pickle=True)
            np.save(temp_dir / "weight.npy", np.concatenate(weight_parts).astype(np.float32), allow_pickle=True)
            np.save(temp_dir / "chrom_start_end.npy", np.array([[0, n_bins]], dtype=np.int64))
            (data_dir / "ramani.chrom.sizes").write_text(f"{chrom}\t{n_bins * args.resolution}\n", encoding="utf-8")
            write_label_info(data_dir / "label_info.pickle", cells)
            config = build_config(dataset_root, chrom, n_bins, args.neighbor_num, args)
            (dataset_root / "config.JSON").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows_for_manifest.append(
            {
                "chrom": chrom,
                "neighbor_num": args.neighbor_num,
                "config": str((dataset_root / "config.JSON").resolve()),
                "dataset_root": str(dataset_root.resolve()),
                "n_cells": len(cells),
                "n_features": int(matrix.shape[1]),
                "n_bins": int(n_bins),
            }
        )
    manifest = args.input_root / f"ramani_higashi_nbr{args.neighbor_num}_manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_for_manifest[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_for_manifest)
    print(manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbor-num", type=int, choices=[0, 5], required=True)
    parser.add_argument("--source-dir", type=Path, default=common.DEFAULT_RAW_CHROM_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--cell-list", type=Path, default=common.DEFAULT_CELL_LIST)
    parser.add_argument("--chroms", default=None)
    parser.add_argument("--resolution", type=int, default=1_000_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--embedding-epoch", type=int, default=60)
    parser.add_argument("--no-nbr-epoch", type=int, default=45)
    parser.add_argument("--with-nbr-epoch", type=int, default=30)
    parser.add_argument("--cpu-num", type=int, default=4)
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--loss-mode", default="zinb")
    parser.add_argument("--min-delta", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
