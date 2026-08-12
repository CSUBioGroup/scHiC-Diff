#!/usr/bin/env python3
"""Collect scVI-3D Ramani per-cell full matrices into plotting-ready files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


COMMON_DIR = (
    Path("/public/home/hpc254701055/2_projects/10_schicdiff")
    / "1_scHiC/5_baseline/paperplots/4_ramani_clustering_metrics/scripts"
)
sys.path.insert(0, str(COMMON_DIR.parent))
from scripts import ramani_imputation_common as common  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FULL_NPZ = BASE_DIR.parent / "2_Ramani" / "full_npz"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"


def collect(args: argparse.Namespace) -> Path:
    chroms = common.chroms_from_arg(args.chroms)
    cell_order = common.legacy_method_cell_order()
    common.save_cell_order(args.output_root / "legacy_626_cell_order.txt", cell_order)
    chrom_dir = args.output_root / "chrom_npz"
    common.export_full_npz_cells_to_chrom_npz(
        full_npz_root=args.full_npz_root,
        out_dir=chrom_dir,
        cell_order=cell_order,
        chroms=chroms,
        filename_pattern=args.filename_pattern,
        triangle="upper",
        force=args.force,
    )
    summary = common.validate_chrom_npz(chrom_dir, chroms=chroms)
    common.save_validation_json(summary, args.output_root / "ramani_scvi3d_validation.json")
    if args.make_embedding:
        embedding = args.output_root / "ramani_embedding.npz"
        common.save_embedding_from_chrom_npz(chrom_dir, embedding, chroms=chroms, per_chrom_dim=args.per_chrom_dim)
        common.write_manifest_snippet(
            args.output_root / "ramani_method_manifest_row.csv",
            method="scVI-3D",
            display_name="scVI-3D",
            source_type="embedding",
            source_path=embedding,
            notes="Generated from 1_scVI-3D/3_ramaniData.",
        )
    print(chrom_dir)
    return chrom_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-npz-root", type=Path, default=DEFAULT_FULL_NPZ)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--filename-pattern", default="{cell}_{chrom}.npz")
    parser.add_argument("--chroms", default=None)
    parser.add_argument("--make-embedding", action="store_true")
    parser.add_argument("--per-chrom-dim", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    collect(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
