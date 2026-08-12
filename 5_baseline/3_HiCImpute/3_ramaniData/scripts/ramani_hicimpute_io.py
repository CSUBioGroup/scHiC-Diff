#!/usr/bin/env python3
"""Prepare and collect Ramani HiCImpute inputs/outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
DEFAULT_MANIFEST = DEFAULT_INPUT_ROOT / "ramani_hicimpute_manifest.tsv"


@dataclass(frozen=True)
class ChromRecord:
    chrom: str
    source_npz: str
    input_dir: str
    n_cells: int
    n_features: int
    n_bins: int


def _write_f64_matrix(path: Path, cells_by_features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asfortranarray(cells_by_features.T.astype("<f8", copy=False)).tofile(path)


def _write_names(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def write_manifest(records: list[ChromRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


def read_manifest(path: Path) -> list[ChromRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            ChromRecord(
                chrom=row["chrom"],
                source_npz=row["source_npz"],
                input_dir=row["input_dir"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_bins=int(row["n_bins"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def build_inputs(
    source_dir: Path,
    input_root: Path,
    manifest: Path,
    cell_list: Path,
    chroms: list[str],
    force: bool,
) -> Path:
    aligned_dir = input_root / "raw_626_chrom_npz"
    common.filter_630_to_626(source_dir, aligned_dir, cell_list, chroms, force=force)
    cells = common.load_cell_list(cell_list)
    records: list[ChromRecord] = []
    for chrom in chroms:
        matrix = sparse.load_npz(aligned_dir / f"{chrom}.npz").tocsr()
        common.assert_row_count(matrix, len(cells), chrom)
        n_bins = common.n_bins_from_upper_triangle_size(matrix.shape[1])
        input_dir = input_root / "hicimpute_binary" / chrom
        marker = input_dir / ".complete"
        if force or not marker.exists():
            dense = matrix.toarray().astype(np.float64, copy=False)
            bulk = np.asarray(dense.sum(axis=0), dtype="<f8")
            _write_f64_matrix(input_dir / "schic_features_by_cells.bin", dense)
            _write_f64_matrix(input_dir / "expected_features_by_cells.bin", dense)
            bulk.tofile(input_dir / "bulk_vector.bin")
            _write_names(input_dir / "obs_names.txt", cells)
            _write_names(input_dir / "var_names.txt", [f"{chrom}:{i}" for i in range(matrix.shape[1])])
            metadata = {
                "chrom": chrom,
                "source_npz": str(aligned_dir / f"{chrom}.npz"),
                "n_cells": len(cells),
                "n_features": int(matrix.shape[1]),
                "n_bins": int(n_bins),
                "layout": "features_by_cells float64 little-endian for R HiCImpute",
            }
            (input_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            marker.write_text("complete\n", encoding="utf-8")
        records.append(
            ChromRecord(
                chrom=chrom,
                source_npz=str(aligned_dir / f"{chrom}.npz"),
                input_dir=str(input_dir),
                n_cells=len(cells),
                n_features=int(matrix.shape[1]),
                n_bins=int(n_bins),
            )
        )
    return write_manifest(records, manifest)


def collect_outputs(
    manifest: Path,
    output_root: Path,
    mode: str,
    make_embedding: bool,
    per_chrom_dim: int,
    seed: int,
) -> Path:
    chrom_dir = output_root / "chrom_npz"
    chrom_dir.mkdir(parents=True, exist_ok=True)
    for record in read_manifest(manifest):
        bin_path = output_root / "bin" / f"{record.chrom}_{mode}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing exported HiCImpute binary: {bin_path}")
        values = np.fromfile(bin_path, dtype="<f8")
        expected = record.n_features * record.n_cells
        if values.size != expected:
            raise ValueError(f"{bin_path}: {values.size} values != expected {expected}")
        features_by_cells = values.reshape((record.n_features, record.n_cells), order="F")
        cells_by_features = np.nan_to_num(features_by_cells.T, nan=0.0, posinf=0.0, neginf=0.0)
        cells_by_features[cells_by_features < 0] = 0.0
        sparse.save_npz(chrom_dir / f"{record.chrom}.npz", sparse.csr_matrix(cells_by_features))
    summary = common.validate_chrom_npz(chrom_dir)
    common.save_validation_json(summary, output_root / "ramani_hicimpute_validation.json")
    if make_embedding:
        embedding = output_root / "ramani_embedding.npz"
        common.save_embedding_from_chrom_npz(
            chrom_dir,
            embedding,
            per_chrom_dim=per_chrom_dim,
            seed=seed,
        )
        common.write_manifest_snippet(
            output_root / "ramani_method_manifest_row.csv",
            method="HiCImpute",
            display_name="HiCImpute",
            source_type="embedding",
            source_path=embedding,
            notes="Generated from 3_HiCImpute/3_ramaniData.",
        )
    return chrom_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--source-dir", type=Path, default=common.DEFAULT_RAW_CHROM_DIR)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--cell-list", type=Path, default=common.DEFAULT_CELL_LIST)
    p.add_argument("--chroms", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: print(build_inputs(
        a.source_dir, a.input_root, a.manifest, a.cell_list, common.chroms_from_arg(a.chroms), a.force
    )))

    p = sub.add_parser("collect")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--mode", default="Impute_All")
    p.add_argument("--make-embedding", action="store_true")
    p.add_argument("--per-chrom-dim", type=int, default=5)
    p.add_argument("--seed", type=int, default=10)
    p.set_defaults(func=lambda a: print(collect_outputs(
        a.manifest, a.output_root, a.mode, a.make_embedding, a.per_chrom_dim, a.seed
    )))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
