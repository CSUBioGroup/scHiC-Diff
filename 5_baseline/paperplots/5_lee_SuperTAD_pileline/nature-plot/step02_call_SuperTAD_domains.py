"""Run SuperTAD only for fixed Targets and selected representative trials."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

import TAD_method_comparison_config as config


def build_supertad_command(txt_path, binary=config.SUPERTAD_BIN):
    return [
        str(binary),
        config.SUPERTAD_MODE,
        str(txt_path),
        "--chrom1",
        config.CHROM,
        "--chrom1-start",
        str(config.REGION_START),
        "-r",
        str(config.RESOLUTION),
        "-h",
        str(config.SUPERTAD_HEIGHT),
    ]


def validate_supertad_tsv(path, n_bins=config.N_BINS):
    """Validate the eight-column SuperTAD domain output and return a summary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"empty SuperTAD TSV: {path}")
    domains = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"malformed blank row at {path}:{line_number}")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                raise ValueError(f"malformed SuperTAD row at {path}:{line_number}")
            try:
                start_bin = int(parts[1])
                end_bin = int(parts[5])
                int(parts[2])
                int(parts[3])
                int(parts[6])
                int(parts[7])
            except ValueError as error:
                raise ValueError(
                    f"malformed numeric field at {path}:{line_number}"
                ) from error
            if not (1 <= start_bin <= end_bin <= n_bins):
                raise ValueError(
                    f"out-of-range SuperTAD bins at {path}:{line_number}: "
                    f"{start_bin}-{end_bin}"
                )
            domains.append((start_bin, end_bin))
    if not domains:
        raise ValueError(f"empty SuperTAD TSV: {path}")
    return {"row_count": len(domains), "domains_1based": domains}


def _resolve_project_path(recorded_path, project_root):
    recorded = Path(recorded_path)
    if recorded.is_absolute():
        raise ValueError(f"representative matrix_path must be relative: {recorded}")
    project_root = Path(project_root).resolve()
    resolved = (project_root / recorded).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"representative matrix_path escapes project: {recorded}") from error
    return resolved


def load_representative_manifest(path, project_root=None, expected_count=None):
    """Load representative records and enforce matrix/trial identity."""
    path = Path(path)
    project_root = Path(project_root or Path.cwd())
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"method", "cell_type", "trial_id", "matrix_path"}
    if not rows:
        raise ValueError(f"representative manifest is empty: {path}")
    if not required.issubset(rows[0]):
        raise ValueError(f"representative manifest lacks fields: {sorted(required)}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} representative records, found {len(rows)}"
        )

    records = []
    seen = set()
    for row in rows:
        method = row["method"]
        cell_type = row["cell_type"]
        if not method or Path(method).name != method:
            raise ValueError(f"invalid method name in representative manifest: {method!r}")
        if cell_type not in config.CELL_TYPES:
            raise ValueError(f"unknown cell type in representative manifest: {cell_type}")
        try:
            trial_id = int(row["trial_id"])
        except ValueError as error:
            raise ValueError(f"invalid trial_id: {row['trial_id']!r}") from error
        matrix_path = _resolve_project_path(row["matrix_path"], project_root)
        expected_name = f"{cell_type}_trial{trial_id:03d}.npz"
        if matrix_path.name != expected_name:
            raise ValueError(
                f"matrix filename does not match cell_type/trial_id: "
                f"expected {expected_name}, found {matrix_path.name}"
            )
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        key = (method, cell_type)
        if key in seen:
            raise ValueError(f"duplicate representative record: {method}/{cell_type}")
        seen.add(key)
        record = dict(row)
        record.update(
            method=method,
            cell_type=cell_type,
            trial_id=trial_id,
            matrix_path=matrix_path,
        )
        records.append(record)
    return records


def build_run_jobs(
    representative_records,
    target_root=config.TARGET_ROOT,
    target_output_root=config.SUPERTAD_DOMAIN_ROOT / "target",
    representative_output_root=config.SUPERTAD_DOMAIN_ROOT / "representatives",
):
    """Return exactly four Target jobs plus one job per representative row."""
    target_root = Path(target_root)
    target_output_root = Path(target_output_root)
    representative_output_root = Path(representative_output_root)
    jobs = []
    for cell_type in config.CELL_TYPES:
        matrix_path = target_root / f"{cell_type}_target.npz"
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        jobs.append(
            {
                "kind": "target",
                "method": "Target",
                "cell_type": cell_type,
                "trial_id": None,
                "matrix_path": matrix_path,
                "output_path": target_output_root / f"{cell_type}_target.tsv",
            }
        )
    for record in representative_records:
        jobs.append(
            {
                "kind": "representative",
                "method": record["method"],
                "cell_type": record["cell_type"],
                "trial_id": int(record["trial_id"]),
                "matrix_path": Path(record["matrix_path"]),
                "output_path": (
                    representative_output_root
                    / record["method"]
                    / f'{record["cell_type"]}_trial{int(record["trial_id"]):03d}.tsv'
                ),
            }
        )
    return jobs


def run_supertad_matrix(
    matrix_path,
    output_path,
    temporary_root,
    executor=subprocess.run,
    binary=config.SUPERTAD_BIN,
    timeout=60,
):
    """Run one validated matrix and atomically publish its TSV."""
    matrix_path = Path(matrix_path)
    output_path = Path(output_path)
    temporary_root = Path(temporary_root)
    matrix = load_npz(str(matrix_path)).toarray()
    if matrix.shape != (config.N_BINS, config.N_BINS):
        raise ValueError(f"{matrix_path}: expected 49x49 matrix, found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{matrix_path}: matrix contains non-finite values")

    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(temporary_root)) as tmp:
        txt_path = Path(tmp) / f"{matrix_path.stem}.txt"
        np.savetxt(str(txt_path), matrix, fmt="%.6f")
        command = build_supertad_command(txt_path, binary=binary)
        completed = executor(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"SuperTAD exit status {completed.returncode} for {matrix_path}: "
                f"{detail[:300]}"
            )
        generated = Path(str(txt_path) + f".{config.SUPERTAD_MODE}.tsv")
        if not generated.is_file():
            raise RuntimeError(f"SuperTAD did not create expected output: {generated.name}")
        validation = validate_supertad_tsv(generated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generated, output_path)
        validation["output_path"] = output_path
        return validation


def _portable(path):
    return Path(os.path.relpath(Path(path).resolve(), Path.cwd().resolve())).as_posix()


def _target_cache_is_complete(target_root):
    target_root = Path(target_root)
    paths = [target_root / f"{cell_type}_target.tsv" for cell_type in config.CELL_TYPES]
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"partial Target SuperTAD cache: {target_root}")
    for path in paths:
        validate_supertad_tsv(path)
    return True


def call_official_SuperTAD_domains(force=False, executor=subprocess.run):
    """Call SuperTAD for Target and all official representative matrices."""
    manifest_path = config.REPRESENTATIVE_TRIALS_FILE
    final_representative_root = config.SUPERTAD_DOMAIN_ROOT / "representatives"
    expected_count = len(config.MAIN_METHOD_SOURCES) * len(config.CELL_TYPES)
    records = load_representative_manifest(
        manifest_path,
        project_root=Path.cwd(),
        expected_count=expected_count,
    )
    final_target_root = config.SUPERTAD_DOMAIN_ROOT / "target"
    target_cached = _target_cache_is_complete(final_target_root)
    if final_representative_root.exists() and not force:
        raise FileExistsError(
            f"refusing to replace existing SuperTAD output without --force: "
            f"{final_representative_root}"
        )

    stage_root = config.SUPERTAD_DOMAIN_ROOT / ".staging_official"
    if stage_root.exists():
        if not force:
            raise FileExistsError(f"stale SuperTAD staging directory: {stage_root}")
        shutil.rmtree(stage_root)
    stage_target_root = stage_root / "target"
    stage_representative_root = stage_root / "representatives"
    stage_root.mkdir(parents=True)

    all_jobs = build_run_jobs(
        records,
        target_output_root=stage_target_root,
        representative_output_root=stage_representative_root,
    )
    target_jobs = [job for job in all_jobs if job["kind"] == "target"]
    representative_jobs = [job for job in all_jobs if job["kind"] == "representative"]
    if target_cached and not force:
        target_jobs = []

    results = []
    try:
        for index, job in enumerate(target_jobs + representative_jobs, start=1):
            print(
                f"SuperTAD {index}/{len(target_jobs) + len(representative_jobs)}: "
                f"{job['method']}/{job['cell_type']}",
                flush=True,
            )
            validation = run_supertad_matrix(
                job["matrix_path"],
                job["output_path"],
                temporary_root=stage_root / "tmp",
                executor=executor,
            )
            final_output = (
                final_target_root / job["output_path"].name
                if job["kind"] == "target"
                else final_representative_root
                / job["method"]
                / job["output_path"].name
            )
            results.append(
                {
                    "kind": job["kind"],
                    "method": job["method"],
                    "cell_type": job["cell_type"],
                    "trial_id": job["trial_id"],
                    "matrix_path": _portable(job["matrix_path"]),
                    "output_path": _portable(final_output),
                    "row_count": validation["row_count"],
                }
            )

        if not target_jobs:
            for cell_type in config.CELL_TYPES:
                path = final_target_root / f"{cell_type}_target.tsv"
                results.append(
                    {
                        "kind": "target",
                        "method": "Target",
                        "cell_type": cell_type,
                        "trial_id": None,
                        "matrix_path": _portable(
                            config.TARGET_ROOT / f"{cell_type}_target.npz"
                        ),
                        "output_path": _portable(path),
                        "row_count": validate_supertad_tsv(path)["row_count"],
                    }
                )

        summary = {
            "scope": "official",
            "representative_manifest": _portable(manifest_path),
            "command_contract": {
                "mode": config.SUPERTAD_MODE,
                "chrom": config.CHROM,
                "chrom_start": config.REGION_START,
                "resolution": config.RESOLUTION,
                "height": config.SUPERTAD_HEIGHT,
            },
            "target_count": 4,
            "representative_count": len(representative_jobs),
            "records": sorted(
                results,
                key=lambda item: (
                    item["kind"] != "target",
                    item["method"],
                    item["cell_type"],
                ),
            ),
        }
        with (stage_representative_root / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

        if target_jobs:
            if final_target_root.exists():
                if not force:
                    raise FileExistsError(final_target_root)
                shutil.rmtree(final_target_root)
            final_target_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_target_root, final_target_root)
        if force and final_representative_root.exists():
            shutil.rmtree(final_representative_root)
        final_representative_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_representative_root, final_representative_root)
        if stage_root.exists():
            shutil.rmtree(stage_root)
        return summary
    except Exception:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config.validate_project_cwd()
    summary = call_official_SuperTAD_domains(force=args.force)
    print(
        f"Published {summary['representative_count']} representative SuperTAD calls "
        "for the official comparison."
    )


if __name__ == "__main__":
    main()
