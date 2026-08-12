#!/usr/bin/env python3
"""Copy the approved frozen plotting inputs into a hash-verified local bundle."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


COPY_MAP = (
    (
        Path("hires_1mb_panel_a/outputs/source_data_panel_a.csv"),
        Path("developmental_stage/source_data_panel_a.csv"),
    ),
    (
        Path("hires_1mb_panel_a/outputs/source_data_panel_a_summary.csv"),
        Path("developmental_stage/source_data_panel_a_summary.csv"),
    ),
    (
        Path("hires_1mb_panel_a/outputs/nature_panel_a_run_metadata.json"),
        Path("developmental_stage/nature_panel_a_run_metadata.json"),
    ),
    (
        Path("call_loops_seed42_panel_data_with_flamingo.npz"),
        Path("contact_loops/call_loops_seed42_panel_data_with_flamingo.npz"),
    ),
    (
        Path("call_loops_seed42_panel_counts_with_flamingo.csv"),
        Path("contact_loops/call_loops_seed42_panel_counts_with_flamingo.csv"),
    ),
    (
        Path("call_loops_seed42_source_manifest_with_flamingo.csv"),
        Path("contact_loops/call_loops_seed42_source_manifest_with_flamingo.csv"),
    ),
    (
        Path("call_loop_counts_three_seed_raw_with_flamingo.csv"),
        Path("contact_loops/call_loop_counts_three_seed_raw_with_flamingo.csv"),
    ),
    (
        Path("call_loop_counts_three_seed_summary_with_flamingo.csv"),
        Path("contact_loops/call_loop_counts_three_seed_summary_with_flamingo.csv"),
    ),
    (
        Path("call_loop_counts_three_seed_source_manifest_with_flamingo.csv"),
        Path("contact_loops/call_loop_counts_three_seed_source_manifest_with_flamingo.csv"),
    ),
    (
        Path("apa_600kb_top10_top20_top50_normed_matrices_with_flamingo.npz"),
        Path("heldout_apa/apa_600kb_top10_top20_top50_normed_matrices_with_flamingo.npz"),
    ),
    (
        Path("apa_600kb_top10_top20_top50_metrics_with_flamingo.csv"),
        Path("heldout_apa/apa_600kb_top10_top20_top50_metrics_with_flamingo.csv"),
    ),
    (
        Path("apa_600kb_top10_top20_top50_per_split_with_flamingo.csv"),
        Path("heldout_apa/apa_600kb_top10_top20_top50_per_split_with_flamingo.csv"),
    ),
    (
        Path("apa_600kb_top10_top20_top50_source_manifest_with_flamingo.csv"),
        Path("heldout_apa/apa_600kb_top10_top20_top50_source_manifest_with_flamingo.csv"),
    ),
    (
        Path("support_fraction_600kb_data_with_flamingo.csv"),
        Path("heldout_support/support_fraction_600kb_data_with_flamingo.csv"),
    ),
    (
        Path("panelB_600kb_raw_supported_counts_with_flamingo.csv"),
        Path("heldout_support/panelB_600kb_raw_supported_counts_with_flamingo.csv"),
    ),
    (
        Path("heldout_600kb_source_manifest_with_flamingo.csv"),
        Path("heldout_support/heldout_600kb_source_manifest_with_flamingo.csv"),
    ),
)

MANIFEST_NAME = "copied_data_manifest.csv"
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


def _preflight_sources(source_root, mapping):
    missing = []
    for source_relative, _ in mapping:
        path = source_root / source_relative
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "missing frozen source files: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )


def _copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise IOError(
            "copy SHA-256 mismatch: {} -> {}".format(source, destination)
        )
    return source_hash


def copy_bundle(source_root, destination, mapping=COPY_MAP):
    """Copy all mapped files and atomically write their provenance manifest."""

    source_root = Path(source_root).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    _preflight_sources(source_root, mapping)

    records = []
    for source_relative, destination_relative in mapping:
        source = source_root / source_relative
        target = destination / destination_relative
        digest = _copy_verified(source, target)
        records.append(
            {
                "destination": destination_relative.as_posix(),
                "source": str(source.resolve()),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / MANIFEST_NAME
    temporary = destination / (MANIFEST_NAME + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(manifest)
    return records


def verify_bundle(destination):
    """Verify every destination recorded by the copied-data manifest."""

    destination = Path(destination).expanduser().resolve()
    manifest = destination / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError("copied-data manifest does not exist: {}".format(manifest))
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("copied-data manifest is empty: {}".format(manifest))

    verified = 0
    for row in rows:
        relative = Path(row["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe manifest destination: {}".format(relative))
        path = destination / relative
        if not path.is_file():
            raise FileNotFoundError("copied data file does not exist: {}".format(path))
        observed_size = path.stat().st_size
        if observed_size != int(row["size_bytes"]):
            raise ValueError(
                "size mismatch for {}: {} != {}".format(
                    path, observed_size, row["size_bytes"]
                )
            )
        observed_hash = sha256_file(path)
        if observed_hash != row["sha256"]:
            raise ValueError(
                "SHA-256 mismatch for {}: {} != {}".format(
                    path, observed_hash, row["sha256"]
                )
            )
        verified += 1
    return verified, len(rows)


def parse_args(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root.parent / "nature-plot",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "data",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verify_only:
        verified, total = verify_bundle(args.destination)
        print("{}/{} files verified".format(verified, total))
        return 0
    records = copy_bundle(args.source_root, args.destination)
    print("{} files copied".format(len(records)))
    verified, total = verify_bundle(args.destination)
    print("{}/{} files verified".format(verified, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
