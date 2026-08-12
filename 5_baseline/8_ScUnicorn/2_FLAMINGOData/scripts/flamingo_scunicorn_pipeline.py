#!/usr/bin/env python3
"""FLAMINGO 100-cell ScUnicorn 10-fold data and inference utilities."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path("/public/home/hpc254701055/2_projects/10_schicdiff")
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "1_scHiC/3_DiffusionModel/scHiC-Diff-master/data/SimuData/2_FLAMINGOData/100cells"
)
SCUNICORN_TRAIN = (
    PROJECT_ROOT
    / "1_scHiC/4_ImputationCriteria/benchmark_criteria_master/8_ScUnicron/trainAll/train_scunicorn_k562.py"
)
DEFAULT_MANIFEST = BASE_DIR / "manifest.tsv"
DEFAULT_SPLIT_ROOT = BASE_DIR / "splits"
DEFAULT_DATASET_ROOT = BASE_DIR / "datasets"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"


@dataclass(frozen=True)
class FlamingoRecord:
    dataset_id: str
    sim_npz: str
    gt_npz: str
    n_cells: int
    n_features: int
    n_beads: int


def load_scunicorn_module():
    spec = importlib.util.spec_from_file_location("train_scunicorn_k562", SCUNICORN_TRAIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lower_triangle_size_to_n(size: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * size)) / 2)
    if n * (n - 1) // 2 != size:
        raise ValueError(f"{size} is not a lower-triangle feature count")
    return n


def dataset_sort_key(dataset_id: str) -> tuple[int, float, int, int]:
    match = re.fullmatch(r"beads_(\d+)_W_([0-9.]+)_level_(\d+)_T(\d+)", dataset_id)
    if not match:
        return (999999, 999.0, 999999, 999999)
    beads, width, level, timepoint = match.groups()
    return (int(beads), float(width), int(level), int(timepoint))


def load_source_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    data = np.asarray(archive["data"], dtype=np.float32)
    obs_names = np.asarray(archive["obs_names"])
    var_names = np.asarray(archive["var_names"])
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0.0
    if data.ndim != 2:
        raise ValueError(f"{path} data must be 2D, got {data.shape}")
    return data, obs_names, var_names


def discover_records(data_root: Path) -> list[FlamingoRecord]:
    sim_root = data_root / "sim/1_lower_tri_feature/npz"
    gt_root = data_root / "gt/1_lower_tri_feature/npz"
    records = []
    for sim_path in sorted(sim_root.glob("*.npz"), key=lambda p: dataset_sort_key(p.stem)):
        gt_path = gt_root / sim_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing gt npz for {sim_path}: {gt_path}")
        sim, _, sim_vars = load_source_npz(sim_path)
        gt, _, gt_vars = load_source_npz(gt_path)
        if gt.shape[1] != sim.shape[1]:
            raise ValueError(f"{sim_path.stem} feature mismatch: sim={sim.shape}, gt={gt.shape}")
        if not np.array_equal(sim_vars, gt_vars):
            raise ValueError(f"{sim_path.stem} sim/gt var_names differ")
        records.append(
            FlamingoRecord(
                dataset_id=sim_path.stem,
                sim_npz=str(sim_path.resolve()),
                gt_npz=str(gt_path.resolve()),
                n_cells=int(sim.shape[0]),
                n_features=int(sim.shape[1]),
                n_beads=lower_triangle_size_to_n(int(sim.shape[1])),
            )
        )
    if not records:
        raise ValueError(f"No sim npz files found under {sim_root}")
    return records


def write_manifest(records: list[FlamingoRecord], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> list[FlamingoRecord]:
    with path.open(newline="") as handle:
        return [
            FlamingoRecord(
                dataset_id=row["dataset_id"],
                sim_npz=row["sim_npz"],
                gt_npz=row["gt_npz"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_beads=int(row["n_beads"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def lower_triangle_to_matrix(vector: np.ndarray, nbin: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    expected = nbin * (nbin - 1) // 2
    if vector.size != expected:
        raise ValueError(f"Expected {expected} lower-triangle values, got {vector.size}")
    matrix = np.zeros((nbin, nbin), dtype=np.float32)
    lower_index = np.tril_indices(nbin, k=-1)
    matrix[lower_index] = vector
    matrix = matrix + matrix.T
    return matrix


def matrix_to_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got {matrix.shape}")
    return matrix[np.tril_indices(matrix.shape[0], k=-1)].astype(np.float32, copy=False)


def standardize_contact_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0] = 0.0
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def matrix_to_patches(matrix: np.ndarray, patch_size: int = 40):
    matrix = np.asarray(matrix, dtype=np.float32)
    rows, cols = matrix.shape
    padded_rows = int(np.ceil(rows / patch_size) * patch_size)
    padded_cols = int(np.ceil(cols / patch_size) * patch_size)
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[:rows, :cols] = matrix
    patches = []
    positions = []
    for row in range(0, padded_rows, patch_size):
        for col in range(0, padded_cols, patch_size):
            patches.append(padded[row : row + patch_size, col : col + patch_size])
            positions.append((row, col))
    return np.asarray(patches, dtype=np.float32)[:, None, :, :], positions, (rows, cols), (padded_rows, padded_cols)


def patches_to_matrix(patches: np.ndarray, positions, original_shape, padded_shape) -> np.ndarray:
    padded = np.zeros(padded_shape, dtype=np.float32)
    counts = np.zeros(padded_shape, dtype=np.float32)
    for patch, (row, col) in zip(patches, positions):
        patch_2d = np.asarray(patch, dtype=np.float32).squeeze()
        height, width = patch_2d.shape
        padded[row : row + height, col : col + width] += patch_2d
        counts[row : row + height, col : col + width] += 1.0
    counts[counts == 0] = 1.0
    rebuilt = padded / counts
    rows, cols = original_shape
    return rebuilt[:rows, :cols]


def make_folds(n_cells: int = 100, n_folds: int = 10, seed: int = 20260607):
    if n_cells % n_folds != 0:
        raise ValueError(f"n_cells must be divisible by n_folds: {n_cells} % {n_folds}")
    rng = np.random.default_rng(seed)
    cells = rng.permutation(np.arange(1, n_cells + 1, dtype=np.int32))
    test_blocks = [sorted(int(x) for x in block) for block in np.array_split(cells, n_folds)]
    all_cells = set(range(1, n_cells + 1))
    folds = []
    for fold_index, test_cells in enumerate(test_blocks):
        valid_cells = test_blocks[(fold_index + 1) % n_folds]
        train_cells = sorted(all_cells - set(test_cells) - set(valid_cells))
        folds.append({"fold_id": fold_index + 1, "train": train_cells, "valid": valid_cells, "test": test_cells})
    return folds


def write_folds(output_root: Path, n_cells: int, n_folds: int, seed: int) -> None:
    folds = make_folds(n_cells=n_cells, n_folds=n_folds, seed=seed)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "fold_cells.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("fold_id", "fold_name", "train_cells", "valid_cells", "test_cells"), delimiter="\t")
        writer.writeheader()
        for fold in folds:
            fold_name = f"fold_{fold['fold_id']:02d}"
            fold_dir = output_root / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            for key in ("train", "valid", "test"):
                (fold_dir / f"{key}_cells.txt").write_text("\n".join(str(x) for x in fold[key]) + "\n")
            writer.writerow(
                {
                    "fold_id": fold["fold_id"],
                    "fold_name": fold_name,
                    "train_cells": ",".join(map(str, fold["train"])),
                    "valid_cells": ",".join(map(str, fold["valid"])),
                    "test_cells": ",".join(map(str, fold["test"])),
                }
            )


def read_cells(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def split_for_cell(cell: int, train_cells: set[int], valid_cells: set[int], test_cells: set[int]) -> str | None:
    if cell in train_cells:
        return "train"
    if cell in valid_cells:
        return "valid"
    if cell in test_cells:
        return "test"
    return None


def compute_scale_factor(records: list[FlamingoRecord]) -> float:
    max_value = 0.0
    for record in records:
        for path in (record.sim_npz, record.gt_npz):
            data, _, _ = load_source_npz(Path(path))
            if data.size:
                max_value = max(max_value, float(np.max(data)))
    if max_value <= 0:
        raise ValueError("Cannot compute scale factor from all-zero matrices")
    return max_value


def append_cell(split_arrays, manifest_rows, record: FlamingoRecord, cell: int, split: str, sim_vector, gt_vector, scale_factor: float, patch_size: int):
    sim_matrix = standardize_contact_matrix(lower_triangle_to_matrix(sim_vector, record.n_beads))
    gt_matrix = standardize_contact_matrix(lower_triangle_to_matrix(gt_vector, record.n_beads))
    sim_patches, positions, _, _ = matrix_to_patches(sim_matrix / scale_factor, patch_size=patch_size)
    gt_patches, _, _, _ = matrix_to_patches(gt_matrix / scale_factor, patch_size=patch_size)
    split_arrays[split]["data"].append(sim_patches)
    split_arrays[split]["target"].append(gt_patches)
    manifest_rows.append(
        {
            "dataset": record.dataset_id,
            "cell": cell,
            "split": split,
            "n_beads": record.n_beads,
            "patch_count": int(sim_patches.shape[0]),
            "positions": ";".join(f"{row},{col}" for row, col in positions),
            "sim_sum": float(sim_matrix.sum()),
            "true_sum": float(gt_matrix.sum()),
        }
    )


def save_split(output_root: Path, split: str, arrays, scale_factor: float, patch_size: int):
    if arrays["data"]:
        data = np.concatenate(arrays["data"], axis=0).astype(np.float32, copy=False)
        target = np.concatenate(arrays["target"], axis=0).astype(np.float32, copy=False)
    else:
        data = np.empty((0, 1, patch_size, patch_size), dtype=np.float32)
        target = np.empty((0, 1, patch_size, patch_size), dtype=np.float32)
    np.savez(
        output_root / f"{split}.npz",
        data=data,
        target=target,
        scale_factor=np.float32(scale_factor),
        patch_size=np.int32(patch_size),
    )
    return data.shape


def build_fold_dataset(records: list[FlamingoRecord], fold_dir: Path, output_root: Path, patch_size: int = 40, datasets: set[str] | None = None):
    selected = [record for record in records if datasets is None or record.dataset_id in datasets]
    if not selected:
        raise ValueError("No records selected for fold dataset")
    scale_factor = compute_scale_factor(selected)
    train_cells = set(read_cells(fold_dir / "train_cells.txt"))
    valid_cells = set(read_cells(fold_dir / "valid_cells.txt"))
    test_cells = set(read_cells(fold_dir / "test_cells.txt"))
    split_arrays = {split: {"data": [], "target": []} for split in ("train", "valid", "test")}
    manifest_rows = []
    for record in selected:
        sim, _, _ = load_source_npz(Path(record.sim_npz))
        gt, _, _ = load_source_npz(Path(record.gt_npz))
        if gt.shape[0] == 1:
            gt = np.repeat(gt, record.n_cells, axis=0)
        for index in range(record.n_cells):
            cell = index + 1
            split = split_for_cell(cell, train_cells, valid_cells, test_cells)
            if split is None:
                continue
            append_cell(split_arrays, manifest_rows, record, cell, split, sim[index], gt[index], scale_factor, patch_size)
    output_root.mkdir(parents=True, exist_ok=True)
    shapes = {split: save_split(output_root, split, arrays, scale_factor, patch_size) for split, arrays in split_arrays.items()}
    with (output_root / "split_manifest.tsv").open("w", newline="") as handle:
        fieldnames = ("dataset", "cell", "split", "n_beads", "patch_count", "positions", "sim_sum", "true_sum")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_root / "normalization.tsv").open("w") as handle:
        handle.write("key\tvalue\n")
        handle.write(f"scale_factor\t{scale_factor:.10g}\n")
        handle.write(f"patch_size\t{patch_size}\n")
        handle.write(f"dataset_count\t{len(selected)}\n")
    return shapes


def load_model(checkpoint: Path, device):
    module = load_scunicorn_module()
    model = module.ScUnicorn(num_channels=64).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_matrix(model, matrix: np.ndarray, device, patch_size: int, batch_size: int) -> np.ndarray:
    patches, positions, original_shape, padded_shape = matrix_to_patches(matrix, patch_size=patch_size)
    loader = DataLoader(TensorDataset(torch.from_numpy(patches)), batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch.to(device))
            predictions.append(output.cpu().numpy())
    pred_patches = np.concatenate(predictions, axis=0)
    pred = patches_to_matrix(pred_patches, positions, original_shape, padded_shape)
    pred = np.maximum(pred, 0.0)
    pred = (pred + pred.T) / 2.0
    np.fill_diagonal(pred, 0.0)
    return pred


def run_inference(records: list[FlamingoRecord], checkpoint: Path, output_root: Path, datasets_root: Path, fold_name: str, patch_size: int, batch_size: int, cuda: int, test_only: bool = True):
    device = torch.device(f"cuda:{cuda}" if cuda >= 0 and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)
    model = load_model(checkpoint, device)
    norm_path = datasets_root / fold_name / "normalization.tsv"
    norm = {}
    with norm_path.open() as handle:
        next(handle)
        for line in handle:
            key, value = line.rstrip("\n").split("\t")
            norm[key] = value
    scale_factor = float(norm["scale_factor"])
    split_manifest = datasets_root / fold_name / "split_manifest.tsv"
    allowed_cells: dict[str, set[int]] = {}
    with split_manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (not test_only) or row["split"] == "test":
                allowed_cells.setdefault(row["dataset"], set()).add(int(row["cell"]))
    output_root.mkdir(parents=True, exist_ok=True)
    npz_root = output_root / fold_name / "npz_lower_tri"
    npz_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        sim, obs_names, var_names = load_source_npz(Path(record.sim_npz))
        out = np.zeros_like(sim, dtype=np.float32)
        cells = sorted(allowed_cells.get(record.dataset_id, set()))
        if not cells:
            continue
        for cell in cells:
            matrix = standardize_contact_matrix(lower_triangle_to_matrix(sim[cell - 1], record.n_beads)) / scale_factor
            pred = predict_matrix(model, matrix, device, patch_size=patch_size, batch_size=batch_size) * scale_factor
            out[cell - 1] = matrix_to_lower_triangle(pred)
        out_path = npz_root / f"{record.dataset_id}_scunicorn_{fold_name}_lower_tri.npz"
        sparse.save_npz(out_path, sparse.csr_matrix(out))
        dense_dir = output_root / fold_name / "npz_dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dense_dir / f"{record.dataset_id}_scunicorn_{fold_name}.npz",
            data=out,
            obs_names=obs_names,
            var_names=var_names,
        )
        print(f"[OK] {record.dataset_id}: cells={len(cells)} -> {out_path}", flush=True)


def combine_folds(records: list[FlamingoRecord], output_root: Path, split_root: Path, n_folds: int = 10):
    combined_root = output_root / "combined_10fold" / "npz_lower_tri"
    combined_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        combined = np.zeros((record.n_cells, record.n_features), dtype=np.float32)
        assigned = np.zeros(record.n_cells, dtype=bool)
        for fold_id in range(1, n_folds + 1):
            fold_name = f"fold_{fold_id:02d}"
            test_cells = read_cells(split_root / fold_name / "test_cells.txt")
            fold_path = output_root / fold_name / "npz_lower_tri" / f"{record.dataset_id}_scunicorn_{fold_name}_lower_tri.npz"
            fold_data = sparse.load_npz(fold_path).toarray().astype(np.float32, copy=False)
            rows = np.asarray(test_cells, dtype=np.int64) - 1
            if np.any(assigned[rows]):
                raise ValueError(f"{record.dataset_id} repeated cells in {fold_name}")
            combined[rows] = fold_data[rows]
            assigned[rows] = True
        if not assigned.all():
            missing = (np.flatnonzero(~assigned) + 1).tolist()
            raise ValueError(f"{record.dataset_id} missing combined cells: {missing}")
        out_path = combined_root / f"{record.dataset_id}_scunicorn_10fold_lower_tri.npz"
        sparse.save_npz(out_path, sparse.csr_matrix(combined))
        print(f"[OK] combined {record.dataset_id}: {combined.shape} -> {out_path}", flush=True)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-manifest")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p = sub.add_parser("make-splits")
    p.add_argument("--output-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--n-cells", type=int, default=100)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260607)

    p = sub.add_parser("build-fold")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--fold-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--patch-size", type=int, default=40)
    p.add_argument("--datasets", nargs="*", default=None)

    p = sub.add_parser("infer-fold")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--fold-name", required=True)
    p.add_argument("--patch-size", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cuda", type=int, default=0)

    p = sub.add_parser("combine-folds")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    p.add_argument("--n-folds", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "write-manifest":
        records = discover_records(args.data_root.resolve())
        write_manifest(records, args.manifest.resolve())
        print(f"[OK] wrote {len(records)} records: {args.manifest.resolve()}")
    elif args.command == "make-splits":
        write_folds(args.output_root.resolve(), args.n_cells, args.n_folds, args.seed)
        print(f"[OK] wrote splits: {args.output_root.resolve()}")
    elif args.command == "build-fold":
        records = read_manifest(args.manifest.resolve())
        shapes = build_fold_dataset(records, args.fold_dir.resolve(), args.output_root.resolve(), patch_size=args.patch_size, datasets=set(args.datasets) if args.datasets else None)
        for split, shape in shapes.items():
            print(f"[OK] {split}: {shape}")
    elif args.command == "infer-fold":
        records = read_manifest(args.manifest.resolve())
        run_inference(records, args.checkpoint.resolve(), args.output_root.resolve(), args.datasets_root.resolve(), args.fold_name, args.patch_size, args.batch_size, args.cuda)
    elif args.command == "combine-folds":
        records = read_manifest(args.manifest.resolve())
        combine_folds(records, args.output_root.resolve(), args.split_root.resolve(), n_folds=args.n_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
