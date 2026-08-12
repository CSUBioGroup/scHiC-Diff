#!/usr/bin/env python3
"""Create an isolated Ramani Higashi input tree for a training-updates control."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = BASE_DIR / "input"


def safe_symlink(src: Path, dest: Path, force: bool) -> None:
    if dest.exists() or dest.is_symlink():
        if not force:
            return
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.symlink_to(src)


def copy_tree(src: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if not force:
            return
        shutil.rmtree(dest)
    shutil.copytree(src, dest, symlinks=True)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_control(args: argparse.Namespace) -> Path:
    task_rows: list[dict[str, object]] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for neighbor_num in args.neighbor_nums:
        source_manifest = args.source_root / f"ramani_higashi_nbr{neighbor_num}_manifest.tsv"
        manifest_rows = load_manifest(source_manifest)
        control_rows: list[dict[str, object]] = []
        for chrom_index, row in enumerate(manifest_rows):
            chrom = row["chrom"]
            source_dataset_root = Path(row["dataset_root"])
            dataset_root = args.output_root / f"{chrom}_nbr{neighbor_num}"
            data_dir = dataset_root / "data"
            temp_dir = dataset_root / "temp"
            raw_dir = temp_dir / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            copy_tree(source_dataset_root / "data", data_dir, args.force)
            for name in ["data.npy", "weight.npy", "chrom_start_end.npy", "node_feats.hdf5", "sparse_nondiag_adj_nbr_1.npy"]:
                safe_symlink(source_dataset_root / "temp" / name, temp_dir / name, args.force)
            for raw_file in (source_dataset_root / "temp" / "raw").glob("*"):
                safe_symlink(raw_file, raw_dir / raw_file.name, args.force)

            config = json.loads(Path(row["config"]).read_text(encoding="utf-8"))
            config["data_dir"] = str(data_dir.resolve())
            config["temp_dir"] = str(temp_dir.resolve())
            config["genome_reference_path"] = str((data_dir / "ramani.chrom.sizes").resolve())
            config["cpu_num"] = args.cpu_num
            config["cpu_num_torch"] = args.cpu_num_torch
            config["embedding_name"] = f"ramani_higashi_{chrom}_nbr{neighbor_num}_u{args.training_updates}"
            config_path = dataset_root / "config.JSON"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            final_hdf5 = temp_dir / f"{chrom}_{config['embedding_name']}_nbr_{neighbor_num}_impute.hdf5"
            control_row = {
                "chrom": chrom,
                "neighbor_num": neighbor_num,
                "config": str(config_path.resolve()),
                "dataset_root": str(dataset_root.resolve()),
                "n_cells": row["n_cells"],
                "n_features": row["n_features"],
                "n_bins": row["n_bins"],
            }
            control_rows.append(control_row)
            task_rows.append(
                {
                    "nbr": neighbor_num,
                    "idx": chrom_index,
                    "chrom": chrom,
                    "config": str(config_path.resolve()),
                    "dataset_root": str(dataset_root.resolve()),
                    "final_hdf5": str(final_hdf5.resolve()),
                }
            )
        write_manifest(args.output_root / f"ramani_higashi_nbr{neighbor_num}_manifest.tsv", control_rows)
    task_manifest = args.output_root / f"ramani_higashi_updates{args.training_updates}_multi5_tasks.tsv"
    write_manifest(task_manifest, task_rows)
    print(task_manifest)
    return task_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-updates", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--neighbor-nums", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--cpu-num", type=int, default=2)
    parser.add_argument("--cpu-num-torch", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    make_control(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
