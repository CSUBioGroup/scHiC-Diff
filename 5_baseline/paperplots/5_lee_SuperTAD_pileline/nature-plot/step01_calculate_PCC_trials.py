"""Step 01: calculate PCC trials and select representative pseudo-bulks.

The scientific sampling and PCC definitions match the approved TAD method
comparison. All trial metrics are retained, while only the middle observed
``pcc_8x8_full`` trial is reconstructed and saved for each cell type.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.stats import pearsonr

import TAD_method_comparison_config as config


PCC_FIELDS = (
    "trial_id",
    "cell_type",
    "seed",
    "pcc_8x8_full",
    "pcc_8x8_upper",
    "pcc_49x49_full",
    "n_sampled",
    "n_total",
)
REPRESENTATIVE_FIELDS = (
    "method",
    "input_key",
    "cell_type",
    "trial_id",
    "seed",
    "pcc_8x8_full",
    "pcc_8x8_upper",
    "pcc_49x49_full",
    "n_sampled",
    "n_total",
    "matrix_path",
    "sampled_indices",
    "sampled_files",
)


def compute_pcc_full(first, second):
    """Pearson correlation after zeroing the complete matrix diagonal."""
    first = np.asarray(first, dtype=float).copy()
    second = np.asarray(second, dtype=float).copy()
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("PCC matrices must be square and have identical shapes")
    np.fill_diagonal(first, 0.0)
    np.fill_diagonal(second, 0.0)
    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(pearsonr(first.ravel(), second.ravel())[0])


def compute_pcc_upper(first, second):
    """Pearson correlation over the strict upper triangle only."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("PCC matrices must be square and have identical shapes")
    upper = np.triu_indices(first.shape[0], k=1)
    first_values = first[upper]
    second_values = second[upper]
    if np.std(first_values) == 0 or np.std(second_values) == 0:
        return 0.0
    return float(pearsonr(first_values, second_values)[0])


def sample_indices(n_cells, trial_id, n_sample=config.N_SAMPLE, base_seed=config.BASE_SEED):
    """Return the exact legacy ``RandomState(seed).choice`` sample."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    size = min(int(n_sample), int(n_cells))
    rng = np.random.RandomState(int(base_seed) + int(trial_id))
    return rng.choice(int(n_cells), size=size, replace=False)


def load_dense_npz(path):
    return load_npz(str(path)).toarray()


def reconstruct_pseudo_bulk(paths, indices, n_bins=config.N_BINS, loader=load_dense_npz):
    """Load and sum only the selected cells, preserving their sampled order."""
    pseudo_bulk = np.zeros((n_bins, n_bins), dtype=float)
    for index in indices:
        matrix = np.asarray(loader(Path(paths[int(index)])), dtype=float)
        if matrix.shape != (n_bins, n_bins):
            raise ValueError(f"expected {(n_bins, n_bins)}, found {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("per-cell matrix contains non-finite values")
        pseudo_bulk += matrix
    return pseudo_bulk


def select_middle_observed(rows, cell_type, pcc_column="pcc_8x8_full"):
    """Select position ``n // 2`` after sorting observed trial PCC values."""
    table = pd.DataFrame(rows)
    subset = table[table["cell_type"] == cell_type].sort_values(pcc_column)
    if subset.empty:
        raise ValueError(f"no trial rows for {cell_type}")
    return subset.iloc[len(subset) // 2].to_dict()


def portable_path(path):
    """Serialize a path relative to the required project working directory."""
    return Path(os.path.relpath(Path(path).resolve(), Path.cwd().resolve())).as_posix()


def _rounded(value):
    return float(f"{float(value):.6f}")


def _write_pcc_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PCC_FIELDS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in ("pcc_8x8_full", "pcc_8x8_upper", "pcc_49x49_full"):
                output[field] = f'{row[field]:.6f}'
            writer.writerow(output)


def _representative_csv_row(record):
    row = dict(record)
    for field in ("pcc_8x8_full", "pcc_8x8_upper", "pcc_49x49_full"):
        row[field] = f'{record[field]:.6f}'
    row["sampled_indices"] = json.dumps(record["sampled_indices"], separators=(",", ":"))
    row["sampled_files"] = json.dumps(record["sampled_files"], separators=(",", ":"))
    return row


def write_representative_csv(path, records):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPRESENTATIVE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_representative_csv_row(record))


def _validate_dense_matrix(matrix, expected_shape, label):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != expected_shape:
        raise ValueError(f"{label}: expected {expected_shape}, found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label}: matrix contains non-finite values")
    return matrix


def compute_method_compact(
    method,
    input_key,
    source_dir,
    target_root,
    pcc_output_dir,
    representative_output_dir,
    cell_types=config.CELL_TYPES,
    expected_counts=None,
    n_trials=config.N_TRIALS,
    n_sample=config.N_SAMPLE,
    base_seed=config.BASE_SEED,
    n_bins=config.N_BINS,
    sub_bins=config.PDGFRA_SUB_BINS,
    force=False,
    published_pcc_dir=None,
    published_representative_dir=None,
):
    """Compute one method into two compact, atomically published directories."""
    source_dir = Path(source_dir)
    target_root = Path(target_root)
    pcc_output_dir = Path(pcc_output_dir)
    representative_output_dir = Path(representative_output_dir)
    published_pcc_dir = Path(published_pcc_dir or pcc_output_dir)
    published_representative_dir = Path(
        published_representative_dir or representative_output_dir
    )
    cell_types = tuple(cell_types)

    for output in (pcc_output_dir, representative_output_dir):
        if output.exists() and not force:
            raise FileExistsError(f"refusing to replace existing output without force: {output}")

    token = uuid.uuid4().hex
    pcc_stage = pcc_output_dir.parent / f".{pcc_output_dir.name}.staging-{token}"
    representative_stage = (
        representative_output_dir.parent
        / f".{representative_output_dir.name}.staging-{token}"
    )
    matrices_stage = representative_stage
    pcc_stage.mkdir(parents=True)
    matrices_stage.mkdir(parents=True)

    trial_rows = []
    representative_records = []
    start, end = sub_bins
    expected_shape = (n_bins, n_bins)

    try:
        for cell_type in cell_types:
            paths = sorted(source_dir.glob(f"{cell_type}_cell_*.npz"))
            n_total = len(paths)
            if expected_counts is not None and n_total != expected_counts[cell_type]:
                raise ValueError(
                    f"{method} {cell_type}: expected {expected_counts[cell_type]} "
                    f"per-cell files, found {n_total}"
                )
            if not paths:
                raise ValueError(f"{method} {cell_type}: no per-cell NPZ files")

            target_path = target_root / f"{cell_type}_target.npz"
            target = _validate_dense_matrix(
                load_dense_npz(target_path), expected_shape, str(target_path)
            )
            target_sub = target[start:end, start:end]

            cell_matrices = [
                _validate_dense_matrix(load_dense_npz(path), expected_shape, str(path))
                for path in paths
            ]
            cell_rows = []
            for trial_id in range(n_trials):
                indices = sample_indices(n_total, trial_id, n_sample, base_seed)
                pseudo_bulk = np.zeros(expected_shape, dtype=float)
                for index in indices:
                    pseudo_bulk += cell_matrices[int(index)]
                trial_sub = pseudo_bulk[start:end, start:end]
                row = {
                    "trial_id": trial_id,
                    "cell_type": cell_type,
                    "seed": base_seed + trial_id,
                    "pcc_8x8_full": _rounded(compute_pcc_full(trial_sub, target_sub)),
                    "pcc_8x8_upper": _rounded(compute_pcc_upper(trial_sub, target_sub)),
                    "pcc_49x49_full": _rounded(compute_pcc_full(pseudo_bulk, target)),
                    "n_sampled": len(indices),
                    "n_total": n_total,
                }
                cell_rows.append(row)
                trial_rows.append(row)

            selected = select_middle_observed(cell_rows, cell_type)
            trial_id = int(selected["trial_id"])
            selected_indices = sample_indices(n_total, trial_id, n_sample, base_seed)
            representative = reconstruct_pseudo_bulk(
                paths, selected_indices, n_bins=n_bins
            )
            matrix_name = f"{cell_type}_trial{trial_id:03d}.npz"
            matrix_stage_path = matrices_stage / matrix_name
            matrix_published_path = published_representative_dir / matrix_name
            save_npz(str(matrix_stage_path), csr_matrix(representative))

            representative_records.append(
                {
                    "method": method,
                    "input_key": input_key,
                    "cell_type": cell_type,
                    "trial_id": trial_id,
                    "seed": base_seed + trial_id,
                    "pcc_8x8_full": float(selected["pcc_8x8_full"]),
                    "pcc_8x8_upper": float(selected["pcc_8x8_upper"]),
                    "pcc_49x49_full": float(selected["pcc_49x49_full"]),
                    "n_sampled": int(selected["n_sampled"]),
                    "n_total": int(selected["n_total"]),
                    "matrix_path": portable_path(matrix_published_path),
                    "sampled_indices": [int(index) for index in selected_indices],
                    "sampled_files": [portable_path(paths[int(index)]) for index in selected_indices],
                }
            )

        expected_rows = len(cell_types) * n_trials
        if len(trial_rows) != expected_rows:
            raise RuntimeError(f"expected {expected_rows} PCC rows, found {len(trial_rows)}")
        metrics = np.asarray(
            [
                [row["pcc_8x8_full"], row["pcc_8x8_upper"], row["pcc_49x49_full"]]
                for row in trial_rows
            ]
        )
        if not np.isfinite(metrics).all():
            raise RuntimeError("PCC output contains non-finite values")
        if len(list(matrices_stage.glob("*.npz"))) != len(cell_types):
            raise RuntimeError("representative matrix count is incomplete")

        pcc_filename = f"{method}_PCC_trials.csv"
        information_filename = f"{method}_PCC_calculation_information.json"
        _write_pcc_csv(pcc_stage / pcc_filename, trial_rows)
        metadata = {
            "method": method,
            "input_key": input_key,
            "source_dir": portable_path(source_dir),
            "target_root": portable_path(target_root),
            "pcc_trials": portable_path(published_pcc_dir / pcc_filename),
            "representative_matrix_root": portable_path(
                published_representative_dir
            ),
            "n_trials": n_trials,
            "n_sample": n_sample,
            "base_seed": base_seed,
            "cell_types": list(cell_types),
            "expected_counts": expected_counts,
        }
        with (pcc_stage / information_filename).open("w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        if force:
            for output in (pcc_output_dir, representative_output_dir):
                if output.exists():
                    shutil.rmtree(output)
        pcc_output_dir.parent.mkdir(parents=True, exist_ok=True)
        representative_output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pcc_stage, pcc_output_dir)
        os.replace(representative_stage, representative_output_dir)
        return representative_records
    except Exception:
        for stage in (pcc_stage, representative_stage):
            if stage.exists():
                shutil.rmtree(stage)
        raise


def calculate_official_PCC_trials(force=False):
    """Calculate and publish compact PCC outputs for all official methods."""
    sources = config.MAIN_METHOD_SOURCES
    input_keys = config.MAIN_METHOD_INPUT_KEYS
    final_pcc_root = config.PCC_RESULTS_ROOT
    final_representative_root = config.REPRESENTATIVE_MATRIX_ROOT
    final_manifest = config.REPRESENTATIVE_TRIALS_FILE
    for output in (final_pcc_root, final_representative_root, final_manifest):
        if output.exists() and not force:
            raise FileExistsError(f"refusing to replace existing output without --force: {output}")

    stage_root = config.INTERMEDIATE_ROOT / ".staging_step01_official"
    if stage_root.exists():
        if not force:
            raise FileExistsError(f"stale staging directory exists: {stage_root}")
        shutil.rmtree(stage_root)
    stage_pcc_root = stage_root / "PCC_trials_by_method"
    stage_representative_root = stage_root / "representative_matrices"
    stage_manifest = stage_root / "selected_representative_trials.csv"
    stage_root.mkdir(parents=True)

    all_representatives = []
    try:
        for method, source in sources.items():
            input_key = input_keys[method]
            print(f"Calculating PCC trials: {method}", flush=True)
            records = compute_method_compact(
                method=method,
                input_key=input_key,
                source_dir=source,
                target_root=config.TARGET_ROOT,
                pcc_output_dir=stage_pcc_root / method,
                representative_output_dir=stage_representative_root / method,
                expected_counts=config.EXPECTED_CELL_COUNTS[input_key],
                published_pcc_dir=final_pcc_root / method,
                published_representative_dir=final_representative_root / method,
            )
            all_representatives.extend(records)

        write_representative_csv(stage_manifest, all_representatives)
        expected_representatives = len(sources) * len(config.CELL_TYPES)
        if len(all_representatives) != expected_representatives:
            raise RuntimeError(
                f"expected {expected_representatives} representatives, "
                f"found {len(all_representatives)}"
            )

        if force:
            for output in (final_pcc_root, final_representative_root):
                if output.exists():
                    shutil.rmtree(output)
            if final_manifest.exists():
                final_manifest.unlink()
        final_pcc_root.parent.mkdir(parents=True, exist_ok=True)
        final_representative_root.parent.mkdir(parents=True, exist_ok=True)
        final_manifest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_pcc_root, final_pcc_root)
        os.replace(stage_representative_root, final_representative_root)
        os.replace(stage_manifest, final_manifest)
        stage_root.rmdir()
        return all_representatives
    except Exception:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config.validate_project_cwd()
    records = calculate_official_PCC_trials(force=args.force)
    print(f"Published {len(records)} official representative trial records.")


if __name__ == "__main__":
    main()
