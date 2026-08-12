#!/usr/bin/env python3
"""Prepare and evaluate V3 h5ad contact-scale FLAMINGO LRTC inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SCHIC_ROOT = BASE_DIR.parents[2]
DEFAULT_H5AD_ROOT = (
    SCHIC_ROOT
    / "1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/"
    / "3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
)
DEFAULT_INPUT_PARENT = BASE_DIR / "v3ContactInput"
DEFAULT_OUTPUT_PARENT = BASE_DIR / "v3ContactOutput"
DEFAULT_MANIFEST = DEFAULT_INPUT_PARENT / "manifest.tsv"
DEFAULT_PATTERN = "v3_hybrid_*_500cells_level0*_scdiff2.h5ad"
MATRIX_SUBDIR = "contact_matrices"


@dataclass(frozen=True)
class H5adContactDataset:
    dataset: str
    dataset_id: str
    h5ad: str
    n_beads: int
    n_cells: int
    n_features: int
    observed_layer: str
    gt_layer: str
    transform: str


def lower_triangle_size_to_n(feature_count: int) -> int:
    n_float = (1.0 + math.sqrt(1.0 + 8.0 * feature_count)) / 2.0
    n = int(round(n_float))
    if n * (n - 1) // 2 != feature_count:
        raise ValueError(f"Invalid lower-triangle feature count: {feature_count}")
    return n


def filter_unmasked_h5ads(paths: list[Path]) -> list[Path]:
    return sorted(path for path in paths if "heldout_masked" not in path.name)


def dataset_id_from_h5ad(h5ad_path: Path) -> str:
    match = re.fullmatch(
        r"v3_hybrid_(W[0-9p]+)_500cells_level0(?:_(r[0-9p]+))?_scdiff2\.h5ad",
        h5ad_path.name,
    )
    if not match:
        raise ValueError(f"Cannot infer dataset id from {h5ad_path.name}")
    w_tag = match.group(1).lower()
    r_tag = match.group(2) or "r005"
    return f"{w_tag}_{r_tag}_contact"


def clean_csr(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        out = matrix.tocsr().astype(np.float64)
    else:
        out = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    out.data[~np.isfinite(out.data)] = 0.0
    out.data[out.data < 0.0] = 0.0
    out.eliminate_zeros()
    return out


def discover_dataset(
    h5ad_path: Path,
    dataset: str | None,
    observed_layer: str,
    gt_layer: str,
) -> H5adContactDataset:
    with h5py.File(h5ad_path, "r") as handle:
        if "layers" not in handle:
            raise ValueError(f"No layers group found in {h5ad_path}")
        if observed_layer not in handle["layers"]:
            raise ValueError(f"Observed layer '{observed_layer}' not found in {h5ad_path}")
        if gt_layer not in handle["layers"]:
            raise ValueError(f"GT layer '{gt_layer}' not found in {h5ad_path}")
        if "shape" in handle["X"].attrs:
            n_cells, n_features = [int(x) for x in handle["X"].attrs["shape"]]
        else:
            n_cells, n_features = [int(x) for x in handle["layers"][observed_layer].attrs["shape"]]
    n_beads = lower_triangle_size_to_n(n_features)
    dataset_id = dataset or dataset_id_from_h5ad(h5ad_path)
    return H5adContactDataset(
        dataset=dataset_id,
        dataset_id=dataset_id,
        h5ad=str(h5ad_path),
        n_beads=n_beads,
        n_cells=n_cells,
        n_features=n_features,
        observed_layer=observed_layer,
        gt_layer=gt_layer,
        transform="raw contact scale; missing entries are zero in observed layer",
    )


def discover_datasets(
    h5ad_paths: list[Path],
    dataset: str | None,
    observed_layer: str,
    gt_layer: str,
) -> list[H5adContactDataset]:
    if dataset is not None and len(h5ad_paths) != 1:
        raise ValueError("--dataset can only be used with exactly one --h5ad")
    return [discover_dataset(path, dataset, observed_layer, gt_layer) for path in filter_unmasked_h5ads(h5ad_paths)]


def write_manifest(records: list[H5adContactDataset], manifest: Path) -> None:
    if not records:
        raise ValueError("No h5ad records to write")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(manifest: Path) -> list[H5adContactDataset]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        H5adContactDataset(
            dataset=row.get("dataset", row["dataset_id"]),
            dataset_id=row["dataset_id"],
            h5ad=row["h5ad"],
            n_beads=int(row["n_beads"]),
            n_cells=int(row["n_cells"]),
            n_features=int(row["n_features"]),
            observed_layer=row["observed_layer"],
            gt_layer=row["gt_layer"],
            transform=row["transform"],
        )
        for row in rows
    ]


def select_record(records: list[H5adContactDataset], dataset: str | None, task_id: int | None) -> H5adContactDataset:
    if dataset is not None:
        for record in records:
            if record.dataset_id == dataset:
                return record
        raise ValueError(f"Dataset not found: {dataset}")
    if task_id is None:
        if len(records) == 1:
            return records[0]
        raise ValueError("Either --dataset or --task-id is required when manifest has multiple rows")
    if task_id < 0 or task_id >= len(records):
        raise IndexError(f"task id {task_id} outside 0-{len(records) - 1}")
    return records[task_id]


def feature_indices(n_beads: int, feature_order: str) -> tuple[np.ndarray, np.ndarray]:
    if feature_order == "triu":
        return np.triu_indices(n_beads, k=1)
    if feature_order == "tril":
        return np.tril_indices(n_beads, k=-1)
    raise ValueError(f"Unsupported feature order: {feature_order}")


def feature_row_to_matrix(
    row: np.ndarray,
    n_beads: int,
    feature_order: str = "triu",
) -> np.ndarray:
    expected = n_beads * (n_beads - 1) // 2
    if row.size != expected:
        raise ValueError(f"Expected {expected} features for {n_beads} beads, got {row.size}")
    feature_i, feature_j = feature_indices(n_beads, feature_order)
    matrix = np.zeros((n_beads, n_beads), dtype=np.float64)
    matrix[feature_i, feature_j] = row
    matrix[feature_j, feature_i] = row
    return matrix


def matrix_to_feature_row(
    matrix: np.ndarray,
    n_beads: int,
    feature_order: str = "triu",
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (n_beads, n_beads):
        raise ValueError(
            f"Expected ({n_beads}, {n_beads}) contact matrix, got {values.shape}"
        )
    return values[feature_indices(n_beads, feature_order)]


def lower_row_to_matrix(row: np.ndarray, n_beads: int) -> np.ndarray:
    """Backward-compatible legacy lower-triangle reconstruction wrapper."""
    return feature_row_to_matrix(row, n_beads, feature_order="tril")


def write_feature_tensor_npy(
    features: sparse.csr_matrix,
    n_beads: int,
    output_path: Path,
    feature_order: str = "triu",
) -> None:
    expected = n_beads * (n_beads - 1) // 2
    if features.shape[1] != expected:
        raise ValueError(f"Expected {expected} features for {n_beads} beads, got {features.shape[1]}")
    feature_i, feature_j = feature_indices(n_beads, feature_order)
    tensor = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(features.shape[0], n_beads, n_beads),
    )
    tensor[:] = 0.0
    for cell_idx in range(features.shape[0]):
        row = np.asarray(features.getrow(cell_idx).toarray()).ravel().astype(np.float32, copy=False)
        tensor[cell_idx, feature_i, feature_j] = row
        tensor[cell_idx, feature_j, feature_i] = row
    tensor.flush()


def write_full_contact_matrices(
    contact_features: sparse.csr_matrix,
    matrix_dir: Path,
    n_beads: int,
    feature_order: str = "triu",
) -> None:
    expected = n_beads * (n_beads - 1) // 2
    if contact_features.shape[1] != expected:
        raise ValueError(f"Expected {expected} features for {n_beads} beads, got {contact_features.shape[1]}")
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for cell_idx in range(contact_features.shape[0]):
        row = np.asarray(contact_features.getrow(cell_idx).toarray()).ravel()
        matrix = feature_row_to_matrix(row, n_beads, feature_order=feature_order)
        np.savetxt(matrix_dir / f"RawCount_Cell_{cell_idx + 1:04d}.txt", matrix, fmt="%.10g", delimiter="\t")


def prepare(record: H5adContactDataset, input_parent: Path, manifest: Path, force: bool = False) -> Path:
    input_dir = input_parent / record.dataset_id
    matrix_dir = input_dir / MATRIX_SUBDIR
    marker = input_dir / ".complete"
    if marker.exists() and not force:
        return input_dir

    started = time.time()
    adata = ad.read_h5ad(record.h5ad)
    try:
        observed = clean_csr(adata.layers[record.observed_layer])
        truth = clean_csr(adata.layers[record.gt_layer])
        obs_df = adata.obs.reset_index(drop=True)
    finally:
        if adata.isbacked:
            adata.file.close()
    if observed.shape != (record.n_cells, record.n_features):
        raise ValueError(f"Observed shape mismatch: {observed.shape}")
    if truth.shape != observed.shape:
        raise ValueError(f"Truth shape mismatch: {truth.shape} vs {observed.shape}")

    input_dir.mkdir(parents=True, exist_ok=True)
    write_full_contact_matrices(observed, matrix_dir, record.n_beads, feature_order="triu")
    sparse.save_npz(input_dir / "observed_contact_features.npz", observed)
    sparse.save_npz(input_dir / "truth_contact_features.npz", truth)
    write_feature_tensor_npy(
        observed,
        record.n_beads,
        input_dir / "observed_contact_tensor.npy",
        feature_order="triu",
    )
    write_feature_tensor_npy(
        truth,
        record.n_beads,
        input_dir / "truth_contact_tensor.npy",
        feature_order="triu",
    )

    rows = []
    for cell_idx in range(record.n_cells):
        row = {
            "cell_idx": cell_idx,
            "cell_number": cell_idx + 1,
            "input_file": f"RawCount_Cell_{cell_idx + 1:04d}.txt",
        }
        for col in ["cell_type", "batch", "cell_name", "type_id", "cell_index", "noise_level"]:
            if col in obs_df:
                row[col] = obs_df.iloc[cell_idx][col]
        rows.append(row)
    pd.DataFrame(rows).to_csv(input_dir / "input_file_index.csv", index=False)

    metadata = asdict(record) | {
        "prepared_at_epoch": time.time(),
        "prepare_seconds": time.time() - started,
        "observed_contact_nnz": int(observed.nnz),
        "truth_contact_nnz": int(truth.nnz),
        "input_subdir": MATRIX_SUBDIR,
        "feature_order": "numpy row-major triu",
    }
    with (input_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    (input_dir / "manifest.tsv").write_text(manifest.read_text())
    marker.write_text("complete\n")
    return input_dir


def _safe_pearson(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size < 2 or np.std(true) == 0.0 or np.std(pred) == 0.0:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def _safe_spearman(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size < 2 or np.std(true) == 0.0 or np.std(pred) == 0.0:
        return float("nan")
    return _safe_pearson(rankdata(true), rankdata(pred))


def metric_block(prefix: str, truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(truth) & np.isfinite(pred)
    x = truth[valid]
    y = pred[valid]
    if x.size == 0:
        return {
            f"n_{prefix}": 0,
            f"pcc_{prefix}": float("nan"),
            f"spearman_{prefix}": float("nan"),
            f"mae_{prefix}": float("nan"),
            f"rmse_{prefix}": float("nan"),
        }
    diff = y - x
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": _safe_pearson(x, y),
        f"spearman_{prefix}": _safe_spearman(x, y),
        f"mae_{prefix}": float(np.mean(np.abs(diff))),
        f"rmse_{prefix}": float(np.sqrt(np.mean(diff**2))),
    }


def evaluate(record: H5adContactDataset, input_parent: Path, output_parent: Path) -> None:
    started = time.time()
    input_dir = input_parent / record.dataset_id
    output_dir = output_parent / record.dataset_id
    completed_path = output_dir / "completed_tensor.npy"
    if not completed_path.exists():
        raise FileNotFoundError(completed_path)

    completed = np.load(completed_path, mmap_mode="r")
    if completed.shape != (record.n_cells, record.n_beads, record.n_beads):
        raise ValueError(f"Completed shape mismatch for {record.dataset_id}: {completed.shape}")
    feature_i, feature_j = feature_indices(record.n_beads, "triu")
    observed_contact = sparse.load_npz(input_dir / "observed_contact_features.npz").tocsr()
    truth_contact_csr = sparse.load_npz(input_dir / "truth_contact_features.npz").tocsr()
    index_df = pd.read_csv(input_dir / "input_file_index.csv")
    index_rows = index_df.to_dict(orient="records")

    rows = []
    for cell_idx in range(record.n_cells):
        pred_contact = np.asarray(completed[cell_idx][feature_i, feature_j], dtype=np.float64)
        pred_contact[~np.isfinite(pred_contact)] = 0.0
        pred_contact[pred_contact < 0.0] = 0.0
        obs_contact_row = np.asarray(observed_contact.getrow(cell_idx).toarray()).ravel()
        truth_contact = np.asarray(truth_contact_csr.getrow(cell_idx).toarray()).ravel()
        observed_mask = obs_contact_row > 0.0
        truth_mask = truth_contact > 0.0
        masks = {
            "all": truth_mask,
            "observed": observed_mask & truth_mask,
            "heldout": (~observed_mask) & truth_mask,
        }
        log_truth = np.log1p(np.maximum(truth_contact, 0.0))
        log_pred = np.log1p(np.maximum(pred_contact, 0.0))
        row = {"dataset_id": record.dataset_id, "cell_idx": int(cell_idx), "cell_number": int(cell_idx) + 1}
        meta = index_rows[cell_idx]
        for col in ["cell_type", "batch", "cell_name", "type_id", "cell_index", "noise_level"]:
            if col in meta:
                row[col] = meta[col]
        for label, mask in masks.items():
            row.update(metric_block(f"contact_{label}", truth_contact, pred_contact, mask))
            row.update(metric_block(f"log1p_contact_{label}", log_truth, log_pred, mask))
        rows.append(row)

    cell_df = pd.DataFrame(rows)
    group_cols = {"dataset_id", "cell_idx", "cell_number", "cell_type", "batch", "cell_name", "type_id", "cell_index", "noise_level"}
    summary = {"dataset_id": record.dataset_id, "n_cells": len(cell_df)}
    for col in cell_df.columns:
        if col in group_cols:
            continue
        summary[f"{col}_mean"] = float(cell_df[col].mean(skipna=True))
        summary[f"{col}_std"] = float(cell_df[col].std(skipna=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_df.to_csv(output_dir / "v3_h5ad_contact_lrtc_cell_level_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "v3_h5ad_contact_lrtc_summary_metrics.csv", index=False)
    if "cell_type" in cell_df.columns:
        metric_cols = [col for col in cell_df.columns if col not in group_cols]
        grouped = cell_df.groupby("cell_type", dropna=False)[metric_cols].agg(["mean", "std"])
        grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
        grouped.reset_index().to_csv(output_dir / "v3_h5ad_contact_lrtc_summary_by_cell_type.csv", index=False)
    with (output_dir / "evaluation_runtime.json").open("w") as handle:
        json.dump({"evaluation_seconds": time.time() - started, "evaluated_at_epoch": time.time()}, handle, indent=2)


def merge_outputs(output_parent: Path) -> None:
    summary_frames = []
    cell_type_frames = []
    process_rows = []
    for summary_path in sorted(output_parent.glob("*/v3_h5ad_contact_lrtc_summary_metrics.csv")):
        dataset_id = summary_path.parent.name
        summary_frames.append(pd.read_csv(summary_path))
        by_type = summary_path.parent / "v3_h5ad_contact_lrtc_summary_by_cell_type.csv"
        if by_type.exists():
            cell_type_frames.append(pd.read_csv(by_type).assign(dataset_id=dataset_id))
        process_path = summary_path.parent / "process_time.tsv"
        row = {"dataset_id": dataset_id}
        if process_path.exists():
            with process_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        row[key] = value
        process_rows.append(row)
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(
            output_parent / "all_v3_h5ad_contact_lrtc_summary_metrics.csv",
            index=False,
        )
    if cell_type_frames:
        pd.concat(cell_type_frames, ignore_index=True).to_csv(
            output_parent / "all_v3_h5ad_contact_lrtc_summary_by_cell_type.csv",
            index=False,
        )
    if process_rows:
        pd.DataFrame(process_rows).to_csv(output_parent / "all_v3_h5ad_contact_lrtc_process_times.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-manifest")
    p.add_argument("--h5ad", type=Path, nargs="*", default=None)
    p.add_argument("--h5ad-root", type=Path, default=DEFAULT_H5AD_ROOT)
    p.add_argument("--pattern", default=DEFAULT_PATTERN)
    p.add_argument("--dataset", default=None)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--observed-layer", default="counts")
    p.add_argument("--gt-layer", default="gt")

    p = sub.add_parser("prep")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-parent", type=Path, default=DEFAULT_INPUT_PARENT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-parent", type=Path, default=DEFAULT_INPUT_PARENT)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)

    p = sub.add_parser("merge")
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "write-manifest":
        if args.h5ad:
            h5ad_paths = args.h5ad
        else:
            h5ad_paths = sorted(args.h5ad_root.glob(args.pattern))
        records = discover_datasets(h5ad_paths, args.dataset, args.observed_layer, args.gt_layer)
        write_manifest(records, args.manifest)
    elif args.command == "prep":
        record = select_record(read_manifest(args.manifest), args.dataset, args.task_id)
        prepare(record, args.input_parent, args.manifest, force=args.force)
    elif args.command == "eval":
        record = select_record(read_manifest(args.manifest), args.dataset, args.task_id)
        evaluate(record, args.input_parent, args.output_parent)
    elif args.command == "merge":
        merge_outputs(args.output_parent)


if __name__ == "__main__":
    main()
