#!/usr/bin/env python3
"""Restore FLAMINGO HiCImpute Impute_All binaries to canonical triu NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz


DATASETS = (
    "v3_hybrid_W0p5_500cells_level0",
    "v3_hybrid_W0p6_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0_r0p01",
    "v3_hybrid_W0p7_500cells_level0_r0p05",
    "v3_hybrid_W0p8_500cells_level0",
    "v3_hybrid_W0p9_500cells_level0",
)

SCRIPT_DIR = Path(__file__).resolve().parent
HICIMPUTE_ROOT = SCRIPT_DIR.parent
DEFAULT_BIN_DIR = HICIMPUTE_ROOT / "v3_outputData/bin"
DEFAULT_INPUT_ROOT = HICIMPUTE_ROOT / "v3_inputData"
DEFAULT_OUTPUT_DIR = HICIMPUTE_ROOT / "v3_outputData/npz_triu_corrected"


def restore_numpy_triu(impute_r: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Restore R-column-major feature rows to NumPy row-major triu order."""
    values = np.asarray(impute_r)
    permutation = np.asarray(order, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != permutation.size:
        raise ValueError(
            f"Feature/order mismatch: matrix={values.shape}, order={permutation.size}"
        )
    if not np.array_equal(np.sort(permutation), np.arange(permutation.size)):
        raise ValueError("feature_order.npy is not a complete zero-based permutation")
    return values[np.argsort(permutation), :]


def convert_dataset(
    dataset: str,
    bin_dir: Path,
    input_root: Path,
    output_dir: Path,
    n_beads: int,
    n_cells: int,
    overwrite: bool,
) -> Path:
    n_features = n_beads * (n_beads - 1) // 2
    bin_path = bin_dir / f"{dataset}_Impute_All.bin"
    order_path = input_root / dataset / "feature_order.npy"
    output_path = output_dir / f"{dataset}_hicimpute_Impute_All_triu.npz"

    if output_path.exists() and not overwrite:
        print(f"SKIP\t{dataset}\t{output_path}", flush=True)
        return output_path
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)
    if not order_path.is_file():
        raise FileNotFoundError(order_path)

    expected_bytes = n_features * n_cells * np.dtype("<f8").itemsize
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected binary size for {dataset}: {actual_bytes} != {expected_bytes}"
        )

    impute_r = np.memmap(
        bin_path,
        dtype="<f8",
        mode="r",
        shape=(n_features, n_cells),
        order="F",
    )
    order = np.load(order_path)
    restored_features_by_cells = restore_numpy_triu(impute_r, order)
    if not np.isfinite(restored_features_by_cells).all():
        raise ValueError(f"Non-finite Impute_All values for {dataset}")

    sparse_output = csr_matrix(restored_features_by_cells.T)
    sparse_output.eliminate_zeros()
    if sparse_output.nnz == 0:
        raise ValueError(f"Corrected Impute_All is empty for {dataset}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp.npz")
    temporary_path.unlink(missing_ok=True)
    save_npz(temporary_path, sparse_output, compressed=True)
    temporary_path.replace(output_path)

    print(
        "SAVED\t{}\t{}\tshape={}\tnnz={}\tmin={:.6g}\tmax={:.6g}\tmean={:.6g}".format(
            dataset,
            output_path,
            sparse_output.shape,
            sparse_output.nnz,
            float(restored_features_by_cells.min()),
            float(restored_features_by_cells.max()),
            float(restored_features_by_cells.mean()),
        ),
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-beads", type=int, default=500)
    parser.add_argument("--n-cells", type=int, default=1500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_index < 0 or args.dataset_index >= len(DATASETS):
        raise IndexError(f"dataset-index must be in 0..{len(DATASETS) - 1}")
    convert_dataset(
        DATASETS[args.dataset_index],
        args.bin_dir,
        args.input_root,
        args.output_dir,
        args.n_beads,
        args.n_cells,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
