#!/usr/bin/env python3
"""Preserve and hash the code that generated the frozen plotting inputs."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


PLOTTING_FILES = (
    ("nature-plot/hires_1mb_panel_a/src/plot_nature_panel_a.py", "plot_nature_panel_a.py"),
    ("nature-plot/gr_stagefig.py", "gr_stagefig.py"),
    ("nature-plot/gr_panels_bcd.py", "gr_panels_bcd.py"),
    ("nature-plot/plot_call_loops_seed42.py", "plot_call_loops_seed42.py"),
    ("nature-plot/plot_call_loop_counts.py", "plot_call_loop_counts.py"),
    (
        "nature-plot/plot_apa_600kb_topn_with_flamingo.py",
        "plot_apa_600kb_topn_with_flamingo.py",
    ),
    (
        "nature-plot/plot_heldout_600kb_with_flamingo.py",
        "plot_heldout_600kb_with_flamingo.py",
    ),
    ("nature-plot/run_flamingo_evaluation.py", "run_flamingo_evaluation.py"),
)

CONFIG_FILES = (
    "benchmark.json",
    "benchmark_flamingo.json",
    "methods.json",
    "methods_flamingo.json",
    "loci.json",
)

EXTRA_CALCULATION_FILES = (
    (
        "2_callLoop_apa/1_experiment_loop_compare/scripts/call_loops_from_npz.py",
        "1_experiment_loop_compare/scripts/call_loops_from_npz.py",
    ),
    (
        "2_callLoop_apa/3_experiment_apa/scripts/selected_cells_to_hic.py",
        "3_experiment_apa/scripts/selected_cells_to_hic.py",
    ),
)

MANIFEST_NAME = "source_code_manifest.csv"
MANIFEST_FIELDS = ("destination", "source", "size_bytes", "sha256")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_source_code_map(case_root):
    case_root = Path(case_root).resolve()
    mapping = []
    for source_relative, destination_name in PLOTTING_FILES:
        mapping.append(
            (case_root / source_relative, Path("plotting") / destination_name)
        )

    benchmark_root = case_root / "2_callLoop_apa/4_test_corrected_benchmark"
    for source in sorted((benchmark_root / "scripts").glob("*.py")):
        mapping.append(
            (
                source,
                Path("calculation/4_test_corrected_benchmark/scripts") / source.name,
            )
        )
    for name in CONFIG_FILES:
        mapping.append(
            (
                benchmark_root / "configs" / name,
                Path("calculation/4_test_corrected_benchmark/configs") / name,
            )
        )
    for source_relative, destination_relative in EXTRA_CALCULATION_FILES:
        mapping.append(
            (case_root / source_relative, Path("calculation") / destination_relative)
        )
    return tuple(mapping)


def copy_source_code_bundle(destination, mapping):
    destination = Path(destination).resolve()
    missing = [source for source, _ in mapping if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing source-generation code: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    records = []
    for source, relative in mapping:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        digest = sha256_file(source)
        if sha256_file(target) != digest:
            raise IOError("source-code copy hash mismatch: {}".format(target))
        records.append(
            {
                "destination": relative.as_posix(),
                "source": str(source.resolve()),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / (MANIFEST_NAME + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(destination / MANIFEST_NAME)
    return records


def verify_source_code_bundle(destination):
    destination = Path(destination).resolve()
    manifest = destination / MANIFEST_NAME
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("source-code manifest is empty")
    verified = 0
    for row in rows:
        path = destination / row["destination"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("source-code size mismatch: {}".format(path))
        if sha256_file(path) != row["sha256"]:
            raise ValueError("source-code SHA-256 mismatch: {}".format(path))
        verified += 1
    return verified, len(rows)


def parse_args(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=project_root.parent)
    parser.add_argument(
        "--destination", type=Path, default=project_root / "source_generation"
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verify_only:
        verified, total = verify_source_code_bundle(args.destination)
        print("{}/{} source-code files verified".format(verified, total))
        return 0
    mapping = build_source_code_map(args.case_root)
    records = copy_source_code_bundle(args.destination, mapping)
    print("{} source-code files copied".format(len(records)))
    verified, total = verify_source_code_bundle(args.destination)
    print("{}/{} source-code files verified".format(verified, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
