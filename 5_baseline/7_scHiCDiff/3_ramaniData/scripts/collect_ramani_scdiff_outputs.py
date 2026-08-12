#!/usr/bin/env python3
"""Collect scHiC-Diff Ramani h5ad/npz outputs into plotting-ready files."""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"


def load_h5ad_x(path: Path):
    import anndata as ad

    adata = ad.read_h5ad(path)
    matrix = adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect(args: argparse.Namespace) -> Path:
    rows = read_manifest(args.manifest)
    chrom_dir = args.output_root / "chrom_npz"
    chrom_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        chrom = row["chrom"]
        if args.result_pattern:
            result_path = Path(args.result_pattern.format(chrom=chrom))
        else:
            result_path = args.output_root / "scdiff_results" / f"{chrom}_imputed.h5ad"
        if not result_path.exists():
            raise FileNotFoundError(f"{chrom}: missing scHiC-Diff result {result_path}")
        if result_path.suffix == ".npz":
            matrix = sparse.load_npz(result_path).tocsr()
        elif result_path.suffix == ".npy":
            matrix = sparse.csr_matrix(np.load(result_path))
        elif result_path.suffix == ".h5ad":
            matrix = load_h5ad_x(result_path)
        else:
            raise ValueError(f"Unsupported scHiC-Diff result type: {result_path}")
        sparse.save_npz(chrom_dir / f"{chrom}.npz", matrix)
    summary = common.validate_chrom_npz(chrom_dir)
    common.save_validation_json(summary, args.output_root / "ramani_scdiff_validation.json")
    if args.make_embedding:
        embedding = args.output_root / "ramani_embedding.npz"
        common.save_embedding_from_chrom_npz(
            chrom_dir, embedding, per_chrom_dim=args.per_chrom_dim, seed=100, log1p=True,
        )
        common.write_manifest_snippet(
            args.output_root / "ramani_method_manifest_row.csv",
            method="scHiC-Diff",
            display_name="scHiC-Diff",
            source_type="embedding",
            source_path=embedding,
            notes="Generated from 7_scHiCDiff/3_ramaniData v1.2 (3000 epoch). log1p=True, SVD seed=100.",
        )
    print(chrom_dir)
    return chrom_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "input" / "ramani_scdiff_h5ad_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-pattern", default="")
    parser.add_argument("--make-embedding", action="store_true")
    parser.add_argument("--per-chrom-dim", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    collect(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
