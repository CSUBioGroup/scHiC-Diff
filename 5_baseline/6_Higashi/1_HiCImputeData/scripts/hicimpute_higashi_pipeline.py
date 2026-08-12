#!/usr/bin/env python3
"""HiCImpute simulation utilities for running Higashi imputation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path("/public/home/hpc254701055/2_projects/10_schicdiff")
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "1_scHiC/3_DiffusionModel/scHiC-Diff-master/data/SimuData/1_HiCImputeData"
)
DEFAULT_MANIFEST = BASE_DIR / "manifest.tsv"
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
CHROM_NAME = "chrHICIMPUTE"


@dataclass(frozen=True)
class HiCImputeRecord:
    dataset_id: str
    sim_h5ad: str
    gt_npz: str
    n_cells: int
    n_features: int
    n_beads: int


def dataset_sort_key(dataset_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"K562_T([123])_(\d+)k", dataset_id)
    if not match:
        return (999999, 999999)
    timepoint, depth = match.groups()
    return (int(timepoint), int(depth))


def lower_triangle_size_to_n(size: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * size)) / 2)
    if n * (n - 1) // 2 != size:
        raise ValueError(f"{size} is not a lower-triangle feature count")
    return n


def _decode_array(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def load_h5ad_csr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["X/data"], dtype=np.float32)
        indices = np.asarray(handle["X/indices"], dtype=np.int32)
        indptr = np.asarray(handle["X/indptr"], dtype=np.int32)
        n_cells = len(indptr) - 1
        n_features = len(handle["var/_index"])
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_features))
        obs_names = _decode_array(np.asarray(handle["obs/_index"]))
        var_names = _decode_array(np.asarray(handle["var/_index"]))
    dense = matrix.toarray().astype(np.float32, copy=False)
    dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
    dense[dense < 0] = 0.0
    return dense, obs_names, var_names


def load_gt_sparse(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = sparse.load_npz(path).toarray().astype(np.float32, copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0] = 0.0
    return matrix


def discover_records(data_root: Path) -> list[HiCImputeRecord]:
    sim_root = data_root / "sim"
    gt_root = data_root / "gt"
    records: list[HiCImputeRecord] = []
    for sim_path in sorted(sim_root.glob("*_sim.h5ad"), key=lambda p: dataset_sort_key(p.stem.removesuffix("_sim"))):
        dataset_id = sim_path.stem.removesuffix("_sim")
        gt_path = gt_root / f"{dataset_id}_true.npz"
        sim, _, _ = load_h5ad_csr(sim_path)
        gt = load_gt_sparse(gt_path)
        if sim.shape != gt.shape:
            raise ValueError(f"{dataset_id} shape mismatch: sim={sim.shape}, gt={gt.shape}")
        records.append(
            HiCImputeRecord(
                dataset_id=dataset_id,
                sim_h5ad=str(sim_path.resolve()),
                gt_npz=str(gt_path.resolve()),
                n_cells=int(sim.shape[0]),
                n_features=int(sim.shape[1]),
                n_beads=lower_triangle_size_to_n(int(sim.shape[1])),
            )
        )
    if not records:
        raise ValueError(f"No *_sim.h5ad files found under {sim_root}")
    return records


def write_manifest(records: list[HiCImputeRecord], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> list[HiCImputeRecord]:
    with path.open(newline="") as handle:
        return [
            HiCImputeRecord(
                dataset_id=row["dataset_id"],
                sim_h5ad=row["sim_h5ad"],
                gt_npz=row["gt_npz"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_beads=int(row["n_beads"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def selected_record(records: list[HiCImputeRecord], dataset_id: str) -> HiCImputeRecord:
    for record in records:
        if record.dataset_id == dataset_id:
            return record
    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def vector_to_higashi_rows(
    vector: np.ndarray,
    cell_id: int,
    chrom_index: int,
    n_beads: int,
    min_delta: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    expected = n_beads * (n_beads - 1) // 2
    if vector.size != expected:
        raise ValueError(f"Expected {expected} values, got {vector.size}")
    rows, cols = np.tril_indices(n_beads, k=-1)
    bin1 = np.minimum(rows, cols)
    bin2 = np.maximum(rows, cols)
    mask = np.isfinite(vector) & (vector > 0) & ((bin2 - bin1) >= min_delta)
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.int64), np.empty((0,), dtype=np.float32)
    out = np.column_stack(
        [
            np.full(mask.sum(), cell_id, dtype=np.int64),
            np.full(mask.sum(), chrom_index, dtype=np.int64),
            bin1[mask].astype(np.int64),
            bin2[mask].astype(np.int64),
        ]
    )
    return out.astype(np.int64, copy=False), vector[mask].astype(np.float32, copy=False)


def write_genome_file(path: Path, n_beads: int, resolution: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{CHROM_NAME}\t{n_beads * resolution}\n")


def write_label_info(path: Path, record: HiCImputeRecord) -> None:
    match = re.fullmatch(r"K562_(T[123])_(\d+k)", record.dataset_id)
    cell_type = match.group(1) if match else record.dataset_id
    labels = {
        "cell_name": [f"{record.dataset_id}_cell_{i + 1}" for i in range(record.n_cells)],
        "cell_type": [cell_type] * record.n_cells,
        "dataset_id": [record.dataset_id] * record.n_cells,
    }
    with path.open("wb") as handle:
        pickle.dump(labels, handle, protocol=4)


def build_config(
    record: HiCImputeRecord,
    dataset_root: Path,
    resolution: int,
    dimensions: int,
    embedding_epoch: int,
    no_nbr_epoch: int,
    with_nbr_epoch: int,
    neighbor_num: int,
    cpu_num: int,
    gpu_num: int,
    loss_mode: str,
) -> dict[str, object]:
    temp_dir = dataset_root / "temp"
    data_dir = dataset_root / "data"
    return {
        "data_dir": str(data_dir.resolve()),
        "temp_dir": str(temp_dir.resolve()),
        "genome_reference_path": str((data_dir / "hicimpute.chrom.sizes").resolve()),
        "chrom_list": [CHROM_NAME],
        "impute_list": [CHROM_NAME],
        "resolution": resolution,
        "resolution_cell": resolution,
        "minimum_distance": 0,
        "maximum_distance": -1,
        "minimum_impute_distance": 0,
        "maximum_impute_distance": -1,
        "local_transfer_range": 1,
        "dimensions": dimensions,
        "embedding_name": "hicimpute_higashi",
        "loss_mode": loss_mode,
        "neighbor_num": neighbor_num,
        "cpu_num": cpu_num,
        "cpu_num_torch": max(1, min(cpu_num, 4)),
        "gpu_num": gpu_num,
        "embedding_epoch": embedding_epoch,
        "no_nbr_epoch": no_nbr_epoch,
        "with_nbr_epoch": with_nbr_epoch,
        "impute_no_nbr": True,
        "impute_with_nbr": neighbor_num > 0,
        "correct_be_impute": False,
        "precompute_weighted_nbr": True,
        "input_format": "higashi_v1",
        "structured": False,
        "contact_header": ["cell_id", "chrom1", "chrom2", "pos1", "pos2", "count"],
        "header_included": True,
    }


def neighbor_dataset_dir_name(dataset_id: str, neighbor: int) -> str:
    return f"{dataset_id}_nbr{neighbor}"


def build_dataset(
    record: HiCImputeRecord,
    input_root: Path,
    dataset_dir_name: str | None = None,
    resolution: int = 1_000_000,
    dimensions: int = 64,
    embedding_epoch: int = 60,
    no_nbr_epoch: int = 45,
    with_nbr_epoch: int = 30,
    neighbor_num: int = 5,
    cpu_num: int = 4,
    gpu_num: int = 1,
    loss_mode: str = "zinb",
    min_delta: int = 1,
) -> Path:
    dataset_root = input_root / (dataset_dir_name or record.dataset_id)
    data_dir = dataset_root / "data"
    temp_dir = dataset_root / "temp"
    data_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "raw").mkdir(parents=True, exist_ok=True)

    sim, _, _ = load_h5ad_csr(Path(record.sim_h5ad))
    rows_parts = []
    weights_parts = []
    for cell_id, vector in enumerate(sim):
        rows, weights = vector_to_higashi_rows(
            vector,
            cell_id=cell_id,
            chrom_index=0,
            n_beads=record.n_beads,
            min_delta=min_delta,
        )
        if rows.size:
            rows_parts.append(rows)
            weights_parts.append(weights)
    if not rows_parts:
        raise ValueError(f"{record.dataset_id} has no positive contacts after filtering")
    rows_all = np.concatenate(rows_parts, axis=0)
    weights_all = np.concatenate(weights_parts, axis=0)
    np.save(temp_dir / "data.npy", rows_all, allow_pickle=True)
    np.save(temp_dir / "weight.npy", weights_all.astype(np.float32, copy=False), allow_pickle=True)
    np.save(temp_dir / "chrom_start_end.npy", np.array([[0, record.n_beads]], dtype=np.int64))

    write_genome_file(data_dir / "hicimpute.chrom.sizes", record.n_beads, resolution)
    write_label_info(data_dir / "label_info.pickle", record)

    config = build_config(
        record,
        dataset_root=dataset_root,
        resolution=resolution,
        dimensions=dimensions,
        embedding_epoch=embedding_epoch,
        no_nbr_epoch=no_nbr_epoch,
        with_nbr_epoch=with_nbr_epoch,
        neighbor_num=neighbor_num,
        cpu_num=cpu_num,
        gpu_num=gpu_num,
        loss_mode=loss_mode,
    )
    config_path = dataset_root / "config.JSON"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with (dataset_root / "input_summary.tsv").open("w") as handle:
        handle.write("key\tvalue\n")
        handle.write(f"dataset_id\t{record.dataset_id}\n")
        handle.write(f"n_cells\t{record.n_cells}\n")
        handle.write(f"n_beads\t{record.n_beads}\n")
        handle.write(f"n_features\t{record.n_features}\n")
        handle.write(f"positive_contacts\t{len(weights_all)}\n")
        handle.write(f"min_delta\t{min_delta}\n")
    return config_path


def hdf5_to_lower_tri(hdf5_path: Path, n_cells: int, n_beads: int) -> np.ndarray:
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    out = np.zeros((n_cells, n_beads * (n_beads - 1) // 2), dtype=np.float32)
    lower = np.tril_indices(n_beads, k=-1)
    with h5py.File(hdf5_path, "r") as handle:
        if CHROM_NAME in handle:
            group = handle[CHROM_NAME]
            if "coordinates" in group:
                coords = np.asarray(group["coordinates"])
                for cell_id in range(n_cells):
                    key = str(cell_id)
                    if key not in group:
                        continue
                    matrix = np.zeros((n_beads, n_beads), dtype=np.float32)
                    values = np.asarray(group[key], dtype=np.float32)
                    matrix[coords[:, 0].astype(int), coords[:, 1].astype(int)] = values
                    matrix = matrix + matrix.T
                    out[cell_id] = matrix[lower]
            else:
                for cell_id in range(n_cells):
                    key = str(cell_id)
                    if key in group:
                        matrix = np.asarray(group[key], dtype=np.float32)
                        out[cell_id] = matrix[lower]
        else:
            coords = np.asarray(handle["coordinates"])
            for cell_id in range(n_cells):
                key = f"cell_{cell_id}"
                if key not in handle:
                    continue
                matrix = np.zeros((n_beads, n_beads), dtype=np.float32)
                values = np.asarray(handle[key], dtype=np.float32)
                matrix[coords[:, 0].astype(int), coords[:, 1].astype(int)] = values
                matrix = matrix + matrix.T
                out[cell_id] = matrix[lower]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0] = 0.0
    return out


def output_npz_path(output_root: Path, dataset_id: str, neighbor: int = 0) -> Path:
    return output_root / "npz_lower_tri" / f"{dataset_id}_higashi_nbr_{neighbor}_lower_tri.npz"


def convert_output(
    record: HiCImputeRecord,
    input_root: Path,
    output_root: Path,
    neighbor: int = 0,
    dataset_dir_name: str | None = None,
) -> Path:
    dataset_root = input_root / (dataset_dir_name or record.dataset_id)
    hdf5_path = dataset_root / "temp" / f"{CHROM_NAME}_hicimpute_higashi_nbr_{neighbor}_impute.hdf5"
    data = hdf5_to_lower_tri(hdf5_path, record.n_cells, record.n_beads)
    out_path = output_npz_path(output_root, record.dataset_id, neighbor=neighbor)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_path, sparse.csr_matrix(data))
    dense_dir = output_root / "npz_dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dense_dir / f"{record.dataset_id}_higashi_nbr_{neighbor}.npz", data=data)
    return out_path


def inspect_output(record: HiCImputeRecord, output_root: Path, neighbor: int) -> dict[str, object]:
    path = output_npz_path(output_root, record.dataset_id, neighbor=neighbor)
    result: dict[str, object] = {
        "dataset_id": record.dataset_id,
        "neighbor": neighbor,
        "path": str(path),
        "expected_shape": f"{record.n_cells}x{record.n_features}",
    }
    if not path.exists():
        result.update({"status": "missing", "shape": "", "nnz": 0, "nan": ""})
        return result
    matrix = sparse.load_npz(path)
    shape = matrix.shape
    nan_count = int(np.isnan(matrix.data).sum()) if matrix.data.size else 0
    status = "ok" if shape == (record.n_cells, record.n_features) and nan_count == 0 else "bad"
    result.update(
        {
            "status": status,
            "shape": f"{shape[0]}x{shape[1]}",
            "nnz": int(matrix.nnz),
            "nan": nan_count,
        }
    )
    return result


def write_output_report(records: list[HiCImputeRecord], output_root: Path, report_path: Path, neighbors: list[int]) -> int:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_id", "neighbor", "status", "shape", "expected_shape", "nnz", "nan", "path"]
    bad_count = 0
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            for neighbor in neighbors:
                row = inspect_output(record, output_root, neighbor)
                writer.writerow({key: row[key] for key in fieldnames})
                if row["status"] != "ok":
                    bad_count += 1
    return bad_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-manifest")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p = sub.add_parser("build-dataset")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--dataset-dir-name", default=None)
    p.add_argument("--resolution", type=int, default=1_000_000)
    p.add_argument("--dimensions", type=int, default=64)
    p.add_argument("--embedding-epoch", type=int, default=60)
    p.add_argument("--no-nbr-epoch", type=int, default=45)
    p.add_argument("--with-nbr-epoch", type=int, default=30)
    p.add_argument("--neighbor-num", type=int, default=5)
    p.add_argument("--cpu-num", type=int, default=4)
    p.add_argument("--gpu-num", type=int, default=1)
    p.add_argument("--loss-mode", default="zinb", choices=["zinb", "rank", "regression", "classification"])
    p.add_argument("--min-delta", type=int, default=1)

    p = sub.add_parser("build-all")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--neighbors", type=int, nargs="+", default=[0, 5])
    p.add_argument("--resolution", type=int, default=1_000_000)
    p.add_argument("--dimensions", type=int, default=64)
    p.add_argument("--embedding-epoch", type=int, default=60)
    p.add_argument("--no-nbr-epoch", type=int, default=45)
    p.add_argument("--with-nbr-epoch", type=int, default=30)
    p.add_argument("--cpu-num", type=int, default=4)
    p.add_argument("--gpu-num", type=int, default=1)
    p.add_argument("--loss-mode", default="zinb", choices=["zinb", "rank", "regression", "classification"])
    p.add_argument("--min-delta", type=int, default=1)

    p = sub.add_parser("convert-output")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--dataset-dir-name", default=None)
    p.add_argument("--neighbor", type=int, default=0)

    p = sub.add_parser("verify-outputs")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report", type=Path, default=BASE_DIR / "output" / "higashi_output_report.tsv")
    p.add_argument("--neighbors", type=int, nargs="+", default=[0, 5])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "write-manifest":
        records = discover_records(args.data_root.resolve())
        write_manifest(records, args.manifest.resolve())
        print(f"[OK] wrote {len(records)} records: {args.manifest.resolve()}", flush=True)
    elif args.command == "build-dataset":
        record = selected_record(read_manifest(args.manifest.resolve()), args.dataset_id)
        config_path = build_dataset(
            record,
            input_root=args.input_root.resolve(),
            dataset_dir_name=args.dataset_dir_name,
            resolution=args.resolution,
            dimensions=args.dimensions,
            embedding_epoch=args.embedding_epoch,
            no_nbr_epoch=args.no_nbr_epoch,
            with_nbr_epoch=args.with_nbr_epoch,
            neighbor_num=args.neighbor_num,
            cpu_num=args.cpu_num,
            gpu_num=args.gpu_num,
            loss_mode=args.loss_mode,
            min_delta=args.min_delta,
        )
        print(f"[OK] {record.dataset_id}: {config_path}", flush=True)
    elif args.command == "build-all":
        records = read_manifest(args.manifest.resolve())
        for record in records:
            for neighbor in args.neighbors:
                config_path = build_dataset(
                    record,
                    input_root=args.input_root.resolve(),
                    dataset_dir_name=neighbor_dataset_dir_name(record.dataset_id, neighbor),
                    resolution=args.resolution,
                    dimensions=args.dimensions,
                    embedding_epoch=args.embedding_epoch,
                    no_nbr_epoch=args.no_nbr_epoch,
                    with_nbr_epoch=args.with_nbr_epoch,
                    neighbor_num=neighbor,
                    cpu_num=args.cpu_num,
                    gpu_num=args.gpu_num,
                    loss_mode=args.loss_mode,
                    min_delta=args.min_delta,
                )
                print(f"[OK] {record.dataset_id} nbr{neighbor}: {config_path}", flush=True)
    elif args.command == "convert-output":
        record = selected_record(read_manifest(args.manifest.resolve()), args.dataset_id)
        out_path = convert_output(
            record,
            args.input_root.resolve(),
            args.output_root.resolve(),
            neighbor=args.neighbor,
            dataset_dir_name=args.dataset_dir_name,
        )
        print(f"[OK] {record.dataset_id}: {out_path}", flush=True)
    elif args.command == "verify-outputs":
        records = read_manifest(args.manifest.resolve())
        bad_count = write_output_report(records, args.output_root.resolve(), args.report.resolve(), args.neighbors)
        print(f"[OK] wrote report: {args.report.resolve()}", flush=True)
        if bad_count:
            print(f"[ERROR] {bad_count} missing/bad outputs", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
