#!/usr/bin/env python3
"""Prepare FLAMINGO 100-cell data for HiCImpute and collect outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = (
    Path("/public/home/hpc254701055/2_projects/10_schicdiff")
    / "1_scHiC/3_DiffusionModel/scHiC-Diff-master/data/SimuData/2_FLAMINGOData/100cells"
)
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
DEFAULT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.tsv"


@dataclass(frozen=True)
class FlamingoDataset:
    dataset_id: str
    sim_npz: str
    gt_npz: str
    n_cells: int
    n_features: int
    n_beads: int


def lower_triangle_size_to_n(size: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * size)) / 2)
    if n * (n - 1) // 2 != size:
        raise ValueError(f"{size} is not a lower-triangle feature count")
    return n


def _dataset_sort_key(dataset_id: str) -> tuple[int, float, int, int]:
    match = re.fullmatch(r"beads_(\d+)_W_([0-9.]+)_level_(\d+)_T(\d+)", dataset_id)
    if not match:
        return (999999, 999.0, 999999, 999999)
    beads, width, level, timepoint = match.groups()
    return (int(beads), float(width), int(level), int(timepoint))


def load_npz_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    data = np.asarray(archive["data"], dtype=np.float64)
    obs_names = np.asarray(archive["obs_names"])
    var_names = np.asarray(archive["var_names"])
    if data.ndim != 2:
        raise ValueError(f"{path} data must be 2D, got {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0.0
    return data, obs_names, var_names


def discover_datasets(data_root: Path) -> list[FlamingoDataset]:
    sim_root = data_root / "sim" / "1_lower_tri_feature" / "npz"
    gt_root = data_root / "gt" / "1_lower_tri_feature" / "npz"
    records: list[FlamingoDataset] = []
    for sim_npz in sorted(sim_root.glob("*.npz"), key=lambda p: _dataset_sort_key(p.stem)):
        gt_npz = gt_root / sim_npz.name
        if not gt_npz.exists():
            raise FileNotFoundError(f"Missing gt npz for {sim_npz}: {gt_npz}")
        sim, _, sim_vars = load_npz_matrix(sim_npz)
        gt, _, gt_vars = load_npz_matrix(gt_npz)
        if gt.shape[1] != sim.shape[1]:
            raise ValueError(f"Feature mismatch for {sim_npz.stem}: sim={sim.shape}, gt={gt.shape}")
        if len(sim_vars) != sim.shape[1] or len(gt_vars) != gt.shape[1]:
            raise ValueError(f"var_names length mismatch for {sim_npz.stem}")
        if not np.array_equal(sim_vars, gt_vars):
            raise ValueError(f"sim/gt var_names differ for {sim_npz.stem}")
        records.append(
            FlamingoDataset(
                dataset_id=sim_npz.stem,
                sim_npz=str(sim_npz.resolve()),
                gt_npz=str(gt_npz.resolve()),
                n_cells=int(sim.shape[0]),
                n_features=int(sim.shape[1]),
                n_beads=lower_triangle_size_to_n(int(sim.shape[1])),
            )
        )
    if not records:
        raise ValueError(f"No FLAMINGO npz files found under {sim_root}")
    return records


def write_manifest(records: list[FlamingoDataset], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(manifest: Path) -> list[FlamingoDataset]:
    with manifest.open(newline="") as handle:
        return [
            FlamingoDataset(
                dataset_id=row["dataset_id"],
                sim_npz=row["sim_npz"],
                gt_npz=row["gt_npz"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_beads=int(row["n_beads"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def select_record(records: list[FlamingoDataset], dataset: str | None, task_id: int | None) -> FlamingoDataset:
    if dataset is not None:
        for record in records:
            if record.dataset_id == dataset:
                return record
        raise ValueError(f"Dataset not found: {dataset}")
    if task_id is None:
        raise ValueError("Either --dataset or --task-id is required")
    if task_id < 0 or task_id >= len(records):
        raise IndexError(f"task id {task_id} outside 0-{len(records) - 1}")
    return records[task_id]


def write_hicimpute_matrix_bin(path: Path, cells_by_features: np.ndarray) -> None:
    """Write a feature-by-cell matrix for R with column-major layout."""
    matrix = np.asarray(cells_by_features, dtype="<f8").T
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asfortranarray(matrix).tofile(path)


def _write_names(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def prepare_one(record: FlamingoDataset, input_root: Path, force: bool = False) -> Path:
    input_dir = input_root / record.dataset_id
    complete_marker = input_dir / ".complete"
    if complete_marker.exists() and not force:
        return input_dir

    sim, obs_names, var_names = load_npz_matrix(Path(record.sim_npz))
    gt, gt_obs_names, _ = load_npz_matrix(Path(record.gt_npz))
    if sim.shape != (record.n_cells, record.n_features):
        raise ValueError(f"Unexpected sim shape for {record.dataset_id}: {sim.shape}")
    if gt.shape[1] != record.n_features:
        raise ValueError(f"Unexpected gt shape for {record.dataset_id}: {gt.shape}")

    bulk = sim.sum(axis=0)
    gt_for_cells = np.repeat(gt[:1, :], record.n_cells, axis=0)

    write_hicimpute_matrix_bin(input_dir / "schic_features_by_cells.bin", sim)
    write_hicimpute_matrix_bin(input_dir / "expected_features_by_cells.bin", gt_for_cells)
    np.asarray(bulk, dtype="<f8").tofile(input_dir / "bulk_vector.bin")
    _write_names(input_dir / "obs_names.txt", obs_names)
    _write_names(input_dir / "gt_obs_names.txt", gt_obs_names)
    _write_names(input_dir / "var_names.txt", var_names)

    metadata = asdict(record)
    metadata.update({
        "input_format": "float64 little-endian binary",
        "schic_layout": "features_by_cells",
        "source_sim_layout": "cells_by_features",
        "feature_order": "FLAMINGO lower triangle, matching R upper.tri contact-pair order",
        "bulk": "sum of observed simulated cells per feature",
    })
    with (input_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    complete_marker.write_text("complete\n")
    return input_dir


def collect_one(record: FlamingoDataset, output_root: Path, mode: str = "Impute_All") -> Path:
    rds_path = output_root / "rds" / f"{record.dataset_id}_hicimpute_result.rds"
    bin_path = output_root / "bin" / f"{record.dataset_id}_{mode}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing exported binary for {record.dataset_id}: {bin_path}")
    values = np.fromfile(bin_path, dtype="<f8")
    expected_size = record.n_features * record.n_cells
    if values.size != expected_size:
        raise ValueError(f"{bin_path} has {values.size} values, expected {expected_size}")
    features_by_cells = values.reshape((record.n_features, record.n_cells), order="F")
    cells_by_features = features_by_cells.T

    sim, obs_names, var_names = load_npz_matrix(Path(record.sim_npz))
    if cells_by_features.shape != sim.shape:
        raise ValueError(f"Shape mismatch after collect: {cells_by_features.shape} vs {sim.shape}")

    out_dir = output_root / "npz_lower_tri"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.dataset_id}_hicimpute_{mode}_lower_tri.npz"
    sparse.save_npz(out_path, sparse.csr_matrix(cells_by_features))

    dense_dir = output_root / "npz_dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dense_dir / f"{record.dataset_id}_hicimpute_{mode}.npz",
        data=cells_by_features,
        obs_names=obs_names,
        var_names=var_names,
        source_rds=str(rds_path),
    )
    return out_path


def validate_outputs(manifest: Path, output_root: Path, mode: str = "Impute_All") -> pd.DataFrame:
    rows = []
    for record in read_manifest(manifest):
        npz_path = output_root / "npz_lower_tri" / f"{record.dataset_id}_hicimpute_{mode}_lower_tri.npz"
        rds_path = output_root / "rds" / f"{record.dataset_id}_hicimpute_result.rds"
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "n_cells": record.n_cells,
                "n_features": record.n_features,
                "n_beads": record.n_beads,
                "rds_exists": rds_path.exists(),
                "npz_exists": npz_path.exists(),
                "npz_path": str(npz_path),
            }
        )
    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / f"validation_summary_{mode}.tsv", sep="\t", index=False)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-manifest")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p = sub.add_parser("prep")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("collect")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--mode", choices=["Impute_All", "Impute_SZ"], default="Impute_All")

    p = sub.add_parser("validate")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--mode", choices=["Impute_All", "Impute_SZ"], default="Impute_All")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "write-manifest":
        records = discover_datasets(args.data_root.resolve())
        write_manifest(records, args.manifest.resolve())
        print(f"Wrote {len(records)} datasets to {args.manifest.resolve()}")
    elif args.command == "prep":
        record = select_record(read_manifest(args.manifest.resolve()), args.dataset, args.task_id)
        print(prepare_one(record, args.input_root.resolve(), force=args.force))
    elif args.command == "collect":
        record = select_record(read_manifest(args.manifest.resolve()), args.dataset, args.task_id)
        print(collect_one(record, args.output_root.resolve(), mode=args.mode))
    elif args.command == "validate":
        df = validate_outputs(args.manifest.resolve(), args.output_root.resolve(), mode=args.mode)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
