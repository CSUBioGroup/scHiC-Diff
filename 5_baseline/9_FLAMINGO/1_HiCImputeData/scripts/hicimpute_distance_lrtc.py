#!/usr/bin/env python3
"""Prepare and evaluate HiCImpute simulation distance-scale LRTC inputs."""

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
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SCHIC_ROOT = BASE_DIR.parents[2]
DEFAULT_NPZ_ROOT = (
    SCHIC_ROOT
    / "1_Dataset/1-HiCImpute_Simulation_Data/2_processed_data_dxy/1_lower_tri_feature/npz"
)
DEFAULT_INPUT_ROOT = BASE_DIR / "input_distance"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output_distance"
DEFAULT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.tsv"
ALPHA = 0.25


@dataclass(frozen=True)
class HiCImputeDataset:
    dataset_id: str
    sim_npz: str
    true_npz: str
    n_cells: int
    n_features: int
    n_beads: int


def lower_triangle_size_to_n(size: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * size)) / 2)
    if n * (n - 1) // 2 != size:
        raise ValueError(f"{size} is not a lower-triangle feature count")
    return n


def _dataset_sort_key(dataset_id: str) -> tuple[str, int, str]:
    match = re.match(r"(.+?)_(\d+)k(?:_(ALL))?$", dataset_id)
    if not match:
        return dataset_id, 0, ""
    prefix, depth, all_tag = match.groups()
    return prefix, int(depth), all_tag or ""


def _candidate_feature_counts(shape: tuple[int, int]) -> list[int]:
    out = []
    for dim in shape:
        try:
            lower_triangle_size_to_n(dim)
        except ValueError:
            continue
        out.append(dim)
    return out


def load_feature_matrix(path: Path) -> np.ndarray:
    matrix = sparse.load_npz(path).tocsr().astype(np.float64)
    dense = matrix.toarray()
    dense[~np.isfinite(dense)] = 0.0
    dense[dense < 0] = 0.0
    candidates = _candidate_feature_counts(dense.shape)
    if not candidates:
        raise ValueError(f"Cannot infer lower-triangle feature dimension from {path}: {dense.shape}")
    feature_count = max(candidates)
    if dense.shape[1] == feature_count:
        features = dense
    elif dense.shape[0] == feature_count:
        features = dense.T
    else:
        raise ValueError(f"Feature count {feature_count} not found in {path}: {dense.shape}")
    return features


def if_to_distance(values: np.ndarray) -> np.ndarray:
    distance = np.zeros(values.shape, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0)
    distance[mask] = np.power(values[mask], -ALPHA)
    return distance


def feature_matrix_to_tensor(features: np.ndarray, n_beads: int | None = None) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got {features.shape}")
    if n_beads is None:
        n_beads = lower_triangle_size_to_n(features.shape[1])
    expected = n_beads * (n_beads - 1) // 2
    if features.shape[1] != expected:
        raise ValueError(f"Expected {expected} features for {n_beads} beads, got {features.shape[1]}")
    tril_i, tril_j = np.tril_indices(n_beads, k=-1)
    tensor = np.zeros((features.shape[0], n_beads, n_beads), dtype=np.float64)
    tensor[:, tril_i, tril_j] = features
    tensor[:, tril_j, tril_i] = features
    return tensor


def tensor_to_feature_matrix(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError(f"Tensor must be cells x beads x beads, got {tensor.shape}")
    tril_i, tril_j = np.tril_indices(tensor.shape[1], k=-1)
    return tensor[:, tril_i, tril_j]


def discover_datasets(npz_root: Path) -> list[HiCImputeDataset]:
    records: list[HiCImputeDataset] = []
    for sim_npz in sorted(npz_root.rglob("*_sim.npz")):
        dataset_id = sim_npz.stem.removesuffix("_sim")
        true_npz = sim_npz.with_name(f"{dataset_id}_true.npz")
        if not true_npz.exists():
            raise FileNotFoundError(f"Missing true npz for {sim_npz}: {true_npz}")
        sim = load_feature_matrix(sim_npz)
        truth = load_feature_matrix(true_npz)
        if sim.shape != truth.shape:
            raise ValueError(f"Shape mismatch for {dataset_id}: sim={sim.shape}, true={truth.shape}")
        n_features = sim.shape[1]
        records.append(
            HiCImputeDataset(
                dataset_id=dataset_id,
                sim_npz=str(sim_npz.resolve()),
                true_npz=str(true_npz.resolve()),
                n_cells=sim.shape[0],
                n_features=n_features,
                n_beads=lower_triangle_size_to_n(n_features),
            )
        )
    return sorted(records, key=lambda r: _dataset_sort_key(r.dataset_id))


def write_manifest(records: list[HiCImputeDataset], manifest: Path) -> None:
    if not records:
        raise ValueError("No HiCImpute npz datasets found")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(manifest: Path) -> list[HiCImputeDataset]:
    with manifest.open(newline="") as handle:
        return [
            HiCImputeDataset(
                dataset_id=row["dataset_id"],
                sim_npz=row["sim_npz"],
                true_npz=row["true_npz"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_beads=int(row["n_beads"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def select_record(records: list[HiCImputeDataset], dataset: str | None, task_id: int | None) -> HiCImputeDataset:
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


def prepare_one(record: HiCImputeDataset, input_root: Path, force: bool = False) -> Path:
    input_dir = input_root / record.dataset_id
    matrix_dir = input_dir / "distance_matrices"
    complete_marker = input_dir / ".complete"
    if complete_marker.exists() and not force:
        return input_dir

    sim_if = load_feature_matrix(Path(record.sim_npz))
    true_if = load_feature_matrix(Path(record.true_npz))
    if sim_if.shape != (record.n_cells, record.n_features):
        raise ValueError(f"Unexpected sim shape for {record.dataset_id}: {sim_if.shape}")
    if true_if.shape != sim_if.shape:
        raise ValueError(f"Shape mismatch for {record.dataset_id}: {sim_if.shape} vs {true_if.shape}")

    observed_dist = feature_matrix_to_tensor(if_to_distance(sim_if), n_beads=record.n_beads)
    truth_dist = feature_matrix_to_tensor(if_to_distance(true_if), n_beads=record.n_beads)

    matrix_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell_idx, matrix in enumerate(observed_dist, start=1):
        dst_name = f"RawCount_Cell_{cell_idx:03d}.txt"
        np.savetxt(matrix_dir / dst_name, matrix, fmt="%.10g", delimiter="\t")
        n_observed = int(np.count_nonzero(matrix))
        rows.append(
            {
                "cell_idx": cell_idx - 1,
                "cell_number": cell_idx,
                "input_file": dst_name,
                "sim_npz": record.sim_npz,
                "true_npz": record.true_npz,
                "n_observed": n_observed,
                "observed_fraction": n_observed / matrix.size,
            }
        )

    np.save(input_dir / "observed_distance_tensor.npy", observed_dist)
    np.save(input_dir / "truth_distance_tensor.npy", truth_dist)
    pd.DataFrame(rows).to_csv(input_dir / "input_file_index.csv", index=False)
    with (input_dir / "metadata.json").open("w") as handle:
        json.dump(asdict(record) | {"alpha": ALPHA, "space": "distance_from_if"}, handle, indent=2)
    complete_marker.write_text("complete\n")
    return input_dir


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    value = fn(x, y)
    if hasattr(value, "statistic"):
        return float(value.statistic)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def _metrics(prefix: str, pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    x = pred[valid]
    y = truth[valid]
    if x.size == 0:
        return {
            f"n_{prefix}": 0,
            f"pcc_{prefix}": float("nan"),
            f"spearman_{prefix}": float("nan"),
            f"mae_{prefix}": float("nan"),
            f"rmse_{prefix}": float("nan"),
            f"relative_error_{prefix}": float("nan"),
        }
    diff = x - y
    denom = np.linalg.norm(y)
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": _safe_corr(pearsonr, x, y),
        f"spearman_{prefix}": _safe_corr(spearmanr, x, y),
        f"mae_{prefix}": float(np.mean(np.abs(diff))),
        f"rmse_{prefix}": float(np.sqrt(np.mean(diff**2))),
        f"relative_error_{prefix}": float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan"),
    }


def distance_metrics_for_cell(completed: np.ndarray, observed: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if completed.shape != observed.shape or completed.shape != truth.shape:
        raise ValueError(f"Shape mismatch: completed={completed.shape}, observed={observed.shape}, truth={truth.shape}")
    n = completed.shape[0]
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    truth_mask = lower & np.isfinite(truth) & (truth > 0)
    observed_mask = truth_mask & (observed > 0)
    heldout_mask = truth_mask & (observed == 0)
    out = {}
    out.update(_metrics("all", completed, truth, truth_mask))
    out.update(_metrics("observed", completed, truth, observed_mask))
    out.update(_metrics("heldout", completed, truth, heldout_mask))
    return out


def evaluate_one(record: HiCImputeDataset, input_root: Path, output_root: Path, npy_name: str = "completed_tensor.npy") -> None:
    input_dir = input_root / record.dataset_id
    output_dir = output_root / record.dataset_id
    completed_path = output_dir / npy_name
    if not completed_path.exists():
        raise FileNotFoundError(f"Missing completed tensor for {record.dataset_id}: {completed_path}")

    completed = np.real(np.load(completed_path)).astype(np.float64)
    observed = np.load(input_dir / "observed_distance_tensor.npy").astype(np.float64)
    truth = np.load(input_dir / "truth_distance_tensor.npy").astype(np.float64)
    if completed.shape != observed.shape or truth.shape != observed.shape:
        raise ValueError(f"Shape mismatch for {record.dataset_id}: completed={completed.shape}, observed={observed.shape}, truth={truth.shape}")

    rows = []
    for cell_idx in range(completed.shape[0]):
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "cell_idx": cell_idx,
                "cell_number": cell_idx + 1,
                **distance_metrics_for_cell(completed[cell_idx], observed[cell_idx], truth[cell_idx]),
            }
        )

    cell_df = pd.DataFrame(rows)
    summary = {"dataset_id": record.dataset_id, "group": "all_cells", "n_cells": int(len(cell_df))}
    for col in cell_df.columns:
        if col in {"dataset_id", "cell_idx", "cell_number"}:
            continue
        summary[f"{col}_mean"] = float(cell_df[col].mean(skipna=True))
        summary[f"{col}_std"] = float(cell_df[col].std(skipna=True))

    cell_df.to_csv(output_dir / "distance_cell_level_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "distance_summary_metrics.csv", index=False)

    npz_dir = output_root / "npz_lower_tri"
    npz_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(
        npz_dir / f"{record.dataset_id}_flamingo_distance_lower_tri.npz",
        sparse.csr_matrix(tensor_to_feature_matrix(completed)),
    )


def combine(output_root: Path, manifest: Path) -> None:
    frames = []
    summaries = []
    for record in read_manifest(manifest):
        cell_path = output_root / record.dataset_id / "distance_cell_level_metrics.csv"
        summary_path = output_root / record.dataset_id / "distance_summary_metrics.csv"
        if cell_path.exists():
            frames.append(pd.read_csv(cell_path))
        if summary_path.exists():
            summaries.append(pd.read_csv(summary_path))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_root / "all_distance_cell_level_metrics.csv", index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output_root / "all_distance_summary_metrics.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-manifest")
    p.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p = sub.add_parser("prep")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--npy-name", default="completed_tensor.npy")

    p = sub.add_parser("combine")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "write-manifest":
        records = discover_datasets(args.npz_root.resolve())
        write_manifest(records, args.manifest.resolve())
        print(f"Wrote {len(records)} datasets to {args.manifest.resolve()}")
    elif args.command == "prep":
        record = select_record(read_manifest(args.manifest.resolve()), args.dataset, args.task_id)
        print(prepare_one(record, args.input_root.resolve(), force=args.force))
    elif args.command == "eval":
        record = select_record(read_manifest(args.manifest.resolve()), args.dataset, args.task_id)
        evaluate_one(record, args.input_root.resolve(), args.output_root.resolve(), npy_name=args.npy_name)
    elif args.command == "combine":
        combine(args.output_root.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
