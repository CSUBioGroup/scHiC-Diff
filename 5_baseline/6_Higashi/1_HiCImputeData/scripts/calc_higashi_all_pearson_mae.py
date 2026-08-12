#!/usr/bin/env python3
"""Calculate ALL cell-wise Pearson and MAE for HiCImputeData Higashi outputs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = BASE_DIR / "manifest.tsv"
DEFAULT_IMPUTE_ROOT = BASE_DIR / "output" / "npz_lower_tri"
DEFAULT_OUTPUT_CSV = BASE_DIR / "output" / "metrics" / "higashi_hicimpute_ALL_Pearson_MAE.csv"


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    gt_npz: str
    n_cells: int
    n_features: int


def read_manifest(path: Path) -> list[DatasetRecord]:
    with path.open(newline="") as handle:
        records = []
        for row in csv.DictReader(handle, delimiter="\t"):
            records.append(
                DatasetRecord(
                    dataset_id=row["dataset_id"],
                    gt_npz=row["gt_npz"],
                    n_cells=int(row["n_cells"]),
                    n_features=int(row["n_features"]),
                )
            )
    return records


def dataset_sort_key(dataset_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"K562_T([123])_(\d+)k", dataset_id)
    if not match:
        return (999999, 999999)
    timepoint, depth = match.groups()
    return (int(timepoint), int(depth))


def parse_dataset_id(dataset_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"K562_(T[123])_(\d+k)", dataset_id)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def load_lower_tri_npz(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = load_npz(path).toarray().astype(np.float64, copy=False)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(data, index=None)


def cal_pcc_cell_wise(true: pd.DataFrame, impute_result: pd.DataFrame) -> tuple[float, float]:
    pearson_list = []
    for cell_id in range(true.shape[0]):
        true_vector = true.iloc[cell_id, :]
        result_vector = impute_result.iloc[cell_id, :]
        corr = np.corrcoef(true_vector, result_vector)
        pearson_list.append(corr[0][1])
    return float(np.mean(pearson_list)), float(np.std(pearson_list))


def cal_mae_cell_wise(true: pd.DataFrame, impute_result: pd.DataFrame) -> tuple[float, float]:
    mae_list = []
    for cell_id in range(true.shape[0]):
        true_vector = true.iloc[cell_id, :]
        result_vector = impute_result.iloc[cell_id, :]
        mae_list.append(np.mean(np.absolute(true_vector - result_vector)))
    return float(np.mean(mae_list)), float(np.std(mae_list))


def prediction_path(impute_root: Path, dataset_id: str, neighbor: int) -> Path:
    return impute_root / f"{dataset_id}_higashi_nbr_{neighbor}_lower_tri.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--impute-root", type=Path, default=DEFAULT_IMPUTE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--method-prefix", default="Higashi")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = sorted(read_manifest(args.manifest), key=lambda item: dataset_sort_key(item.dataset_id))
    results = []
    for record in records:
        true = load_lower_tri_npz(Path(record.gt_npz))
        if true.shape != (record.n_cells, record.n_features):
            raise ValueError(f"{record.dataset_id} true shape mismatch: {true.shape}")
        ctype, cdepth = parse_dataset_id(record.dataset_id)
        for neighbor in args.neighbors:
            method = f"{args.method_prefix} {neighbor} nbr"
            impute_result = load_lower_tri_npz(prediction_path(args.impute_root, record.dataset_id, neighbor))
            if impute_result.shape != true.shape:
                raise ValueError(
                    f"{record.dataset_id} nbr{neighbor} shape mismatch: "
                    f"imputed {impute_result.shape}, true {true.shape}"
                )
            pcc_mean, pcc_std = cal_pcc_cell_wise(true, impute_result)
            mae_mean, mae_std = cal_mae_cell_wise(true, impute_result)
            results.append(
                {
                    "method": method,
                    "data_name": record.dataset_id,
                    "ctype": ctype,
                    "cdepth": cdepth,
                    "pcc_mean": pcc_mean,
                    "pcc_std": pcc_std,
                    "mae_mean": mae_mean,
                    "mae_std": mae_std,
                }
            )
            print(
                f"{record.dataset_id} {method}: "
                f"pcc_mean={pcc_mean:.6f}, pcc_std={pcc_std:.6f}, "
                f"mae_mean={mae_mean:.6f}, mae_std={mae_std:.6f}",
                flush=True,
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved results to: {args.output_csv.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
