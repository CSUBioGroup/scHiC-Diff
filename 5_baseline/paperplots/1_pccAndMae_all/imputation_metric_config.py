#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent / "1_Dataset"
GT_ROOT = PROJECT_ROOT / "0_gtData"

FLAMINGO_PARAMSWEEP_ROOT = (
    DATA_ROOT
    / "5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
)
FLAMINGO_PARAMSWEEP_SCDIFF_ROOT = (
    FLAMINGO_PARAMSWEEP_ROOT / "training_results_v5fast_batched_test_bs1500_testbs9999"
)
HICIMPUTE_OBSERVED_ROOT = (
    DATA_ROOT / "1-HiCImpute_Simulation_Data/2_processed_data_dxy/1_lower_tri_feature/npz"
)

FAMILY_ORDER = ("FLAMINGOData", "HiCImputeData")
METHOD_ORDER = (
    "scVI-3D",
    "HiCImpute",
    "scHiCluster",
    "Higashi_nbr0",
    "Higashi_nbr5",
    "Tensor-FLAMINGO",
    "scHiC-Diff",
)
TENSOR_FLAMINGO_SUBDIR_MAP = {
    "v3_hybrid_W0p5_500cells_level0": "w0p5_r005_contact",
    "v3_hybrid_W0p6_500cells_level0": "w0p6_r005_contact",
    "v3_hybrid_W0p7_500cells_level0": "w0p7_r005_contact",
    "v3_hybrid_W0p7_500cells_level0_r0p01": "w0p7_r0p01_contact",
    "v3_hybrid_W0p7_500cells_level0_r0p05": "w0p7_r0p05_contact",
    "v3_hybrid_W0p8_500cells_level0": "w0p8_r005_contact",
    "v3_hybrid_W0p9_500cells_level0": "w0p9_r005_contact",
}


def expected_feature_count(n_beads: int) -> int:
    return n_beads * (n_beads - 1) // 2


DATASET_FAMILIES: dict[str, list[dict[str, Any]]] = {
    "FLAMINGOData": [
        {
            "name": "v3_hybrid_W0p5_500cells_level0",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p5_500cells_level0_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p5_500cells_level0_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
        {
            "name": "v3_hybrid_W0p6_500cells_level0",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p6_500cells_level0_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p6_500cells_level0_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
        {
            # Keep W=0.7 in the same paramsweep family used by the 7x9 heatmap.
            "name": "v3_hybrid_W0p7_500cells_level0",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p7_500cells_level0_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p7_500cells_level0_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
        {
            "name": "v3_hybrid_W0p7_500cells_level0_r0p01",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p7_500cells_level0_r0p01_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p7_500cells_level0_r0p01_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
        {
            "name": "v3_hybrid_W0p7_500cells_level0_r0p05",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p7_500cells_level0_r0p05_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p7_500cells_level0_r0p05_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
        {
            "name": "v3_hybrid_W0p8_500cells_level0",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p8_500cells_level0_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p8_500cells_level0_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
        {
            "name": "v3_hybrid_W0p9_500cells_level0",
            "gt_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p9_500cells_level0_scdiff2.h5ad",
            "observed_path": FLAMINGO_PARAMSWEEP_ROOT / "v3_hybrid_W0p9_500cells_level0_scdiff2.h5ad",
            "expected_shape": (1500, 124750),
            "n_beads": 500,
            "scdiff_root": FLAMINGO_PARAMSWEEP_SCDIFF_ROOT,
        },
    ],
    "HiCImputeData": [
        {
            "name": "K562_T1_1k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T1_1k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T1_1k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T1_2k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T1_2k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T1_2k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T1_4k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T1_4k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T1_4k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T1_7k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T1_7k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T1_7k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T2_1k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T2_1k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T2_1k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T2_2k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T2_2k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T2_2k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T2_4k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T2_4k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T2_4k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T2_7k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T2_7k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T2_7k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T3_1k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T3_1k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T3_1k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T3_2k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T3_2k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T3_2k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T3_4k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T3_4k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T3_4k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
        {
            "name": "K562_T3_7k",
            "gt_path": GT_ROOT / "1_Gt_HiCImputeData/K562_T3_7k_true.npz",
            "observed_path": HICIMPUTE_OBSERVED_ROOT / "K562_T3_7k_sim.npz",
            "expected_shape": (100, 1830),
            "n_beads": 61,
        },
    ],
}

METHOD_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "scVI-3D": {
        "FLAMINGOData": {
            "root": PROJECT_ROOT / "1_scVI-3D/2_FLAMINGOData/v3_outputData_earlystop_bs1500/npz_upper_tri",
            "filename_template": "{dataset}_scVI3D_imputed.npz",
            "loader_kind": "sparse_triangle",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "1_scVI-3D/1_HiCImputeData/output/npz_lower_tri",
            "filename_template": "{dataset}_scVI3D_imputed.npz",
            "loader_kind": "sparse_triangle",
        },
    },
    "HiCImpute": {
        "FLAMINGOData": {
            "root": PROJECT_ROOT / "3_HiCImpute/2_FLAMINGOData/v3_outputData/npz_triu_corrected",
            "filename_template": "{dataset}_hicimpute_Impute_All_triu.npz",
            "loader_kind": "sparse_triangle",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "3_HiCImpute/1_HiCImputeData/output/npz_lower_tri",
            "filename_template": "{dataset}_niter5000_burnin1000.npz",
            "loader_kind": "sparse_triangle",
        },
    },
    "scHiCluster": {
        "FLAMINGOData": {
            "root": PROJECT_ROOT / "4_scHiCluster/2_FLAMINGOData/v3_outputData/2_lower_tri_npz",
            "filename_template": "{dataset}_scHiCluster_imputed.npz",
            "loader_kind": "sparse_triangle",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "4_scHiCluster/1_HiCImputeDate/output/2_lower_tri_npz",
            "filename_template": "{dataset}_scHiCluster_imputed.npz",
            "loader_kind": "sparse_triangle",
        },
    },
    "Higashi_nbr0": {
        "FLAMINGOData": {
            "root": PROJECT_ROOT / "6_Higashi/2_FLAMINGOData/v3_epoch1000_outputData/npz_lower_tri",
            "filename_template": "{dataset}_higashi_nbr_0_lower_tri.npz",
            "loader_kind": "sparse_triangle",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "6_Higashi/1_HiCImputeData/output/npz_lower_tri",
            "filename_template": "{dataset}_higashi_nbr_0_lower_tri.npz",
            "loader_kind": "sparse_triangle",
        },
    },
    "Higashi_nbr5": {
        "FLAMINGOData": {
            "root": PROJECT_ROOT / "6_Higashi/2_FLAMINGOData/v3_epoch1000_outputData/npz_lower_tri",
            "filename_template": "{dataset}_higashi_nbr_5_lower_tri.npz",
            "loader_kind": "sparse_triangle",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "6_Higashi/1_HiCImputeData/output/npz_lower_tri",
            "filename_template": "{dataset}_higashi_nbr_5_lower_tri.npz",
            "loader_kind": "sparse_triangle",
        },
    },
    "Tensor-FLAMINGO": {
        "FLAMINGOData": {
            "root": PROJECT_ROOT / "9_FLAMINGO/2_FLAMINGOData/v3ContactOutput",
            "filename_template": "{subdir}/completed_tensor.npy",
            "loader_kind": "tensor_tril_encoded_triu",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "9_FLAMINGO/1_HiCImputeData/output_distance_best/contact_from_pd/npz_lower_tri",
            "filename_template": "{dataset}_flamingo_lower_tri.npz",
            "loader_kind": "sparse_triangle",
        },
    },
    "scHiC-Diff": {
        "FLAMINGOData": {
            "root": "scdiff_root",
            "filename_template": "{dataset}_scdiff2_v5fast_batched_test_bs1500_testbs9999/denoise_recon_inv.npz",
            "loader_kind": "sparse_triangle",
            "feature_order": "triu",
        },
        "HiCImputeData": {
            "root": PROJECT_ROOT / "7_scHiCDiff/1_HiCImputeData/output/npz_lower_tri",
            "filename_template": "{dataset}_scHiCDiff_imputed.npz",
            "loader_kind": "sparse_triangle",
        },
    },
}


def dataset_descriptors(dataset_family: str) -> list[dict[str, Any]]:
    if dataset_family not in DATASET_FAMILIES:
        raise KeyError(f"Unknown dataset family: {dataset_family}")
    return DATASET_FAMILIES[dataset_family]


def dataset_descriptor(dataset_family: str, dataset: str) -> dict[str, Any]:
    for descriptor in dataset_descriptors(dataset_family):
        if descriptor["name"] == dataset:
            return descriptor
    raise KeyError(f"Unknown dataset {dataset!r} for family {dataset_family!r}")


def method_config(method: str, dataset_family: str) -> dict[str, Any]:
    if method not in METHOD_CONFIGS:
        raise KeyError(f"Unknown method: {method}")
    if dataset_family not in METHOD_CONFIGS[method]:
        raise KeyError(f"Method {method!r} does not support family {dataset_family!r}")
    return METHOD_CONFIGS[method][dataset_family]


def resolve_imputed_path(dataset_family: str, dataset: str, method: str) -> Path:
    descriptor = dataset_descriptor(dataset_family, dataset)
    config = method_config(method, dataset_family)
    base_root = descriptor[config["root"]] if isinstance(config["root"], str) else config["root"]
    format_kwargs: dict[str, Any] = {"dataset": dataset}
    if method == "Tensor-FLAMINGO" and dataset_family == "FLAMINGOData":
        format_kwargs["subdir"] = TENSOR_FLAMINGO_SUBDIR_MAP[dataset]
    return Path(base_root) / config["filename_template"].format(**format_kwargs)


def _to_dense_2d(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        dense = value.toarray()
    else:
        dense = np.asarray(value)
    if dense.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {dense.shape}")
    return dense.astype(np.float64, copy=False)


def reorder_triangle_features(
    matrix: np.ndarray,
    n_beads: int,
    source_order: str,
    target_order: str,
) -> np.ndarray:
    """Reorder feature columns while preserving their symmetric coordinates."""
    supported = {"tril", "triu"}
    if source_order not in supported or target_order not in supported:
        raise ValueError(
            f"Unsupported triangle order: source={source_order}, target={target_order}"
        )
    values = np.asarray(matrix, dtype=np.float64)
    expected = expected_feature_count(n_beads)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(
            f"Expected (*, {expected}) triangle features for {n_beads} beads, got {values.shape}"
        )
    if source_order == target_order:
        return values

    source_indices = (
        np.tril_indices(n_beads, k=-1)
        if source_order == "tril"
        else np.triu_indices(n_beads, k=1)
    )
    target_indices = (
        np.tril_indices(n_beads, k=-1)
        if target_order == "tril"
        else np.triu_indices(n_beads, k=1)
    )
    coordinate_lookup = np.full((n_beads, n_beads), -1, dtype=np.int64)
    feature_ids = np.arange(expected, dtype=np.int64)
    coordinate_lookup[source_indices] = feature_ids
    coordinate_lookup[(source_indices[1], source_indices[0])] = feature_ids
    permutation = coordinate_lookup[target_indices]
    if np.any(permutation < 0):
        raise AssertionError("Triangle coordinate permutation is incomplete")
    return values[:, permutation]


def triu_to_tril_reorder(matrix: np.ndarray, n_beads: int) -> np.ndarray:
    """Backward-compatible wrapper for upper-to-lower feature conversion."""
    return reorder_triangle_features(matrix, n_beads, "triu", "tril")


def load_sparse_triangle_features(
    path: Path,
    n_beads: int | None = None,
    feature_order: str = "tril",
    target_order: str = "tril",
) -> np.ndarray:
    dense = load_npz(path).toarray()
    dense = np.asarray(dense, dtype=np.float64)
    if feature_order != target_order:
        if n_beads is None:
            raise ValueError("n_beads is required when converting triangle feature order")
        dense = reorder_triangle_features(
            dense,
            n_beads,
            source_order=feature_order,
            target_order=target_order,
        )
    elif feature_order not in {"tril", "triu"}:
        raise ValueError(f"Unsupported triangle order: {feature_order}")
    return dense


def load_h5ad_gt_and_observed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    adata = ad.read_h5ad(path)
    if "gt" not in adata.layers:
        raise ValueError(f"Missing layers['gt'] in {path}")
    gt = _to_dense_2d(adata.layers["gt"])
    observed_source = adata.layers["counts"] if "counts" in adata.layers else adata.X
    observed = _to_dense_2d(observed_source)
    if gt.shape != observed.shape:
        raise ValueError(f"GT/observed shape mismatch in {path}: {gt.shape} vs {observed.shape}")
    return gt, observed


def load_tensor_flamingo_triu_features(path: Path, n_beads: int) -> np.ndarray:
    """Decode canonical triu features stored at legacy tensor tril coordinates."""
    tensor = np.load(path, mmap_mode="r")
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor in {path}, got shape {tensor.shape}")
    if tensor.shape[1:] != (n_beads, n_beads):
        raise ValueError(f"Tensor shape mismatch in {path}: expected (*, {n_beads}, {n_beads}), got {tensor.shape}")
    lower_index = np.tril_indices(n_beads, k=-1)
    triu_features = tensor[:, lower_index[0], lower_index[1]]
    triu_features = np.asarray(triu_features, dtype=np.float64)
    if triu_features.shape[1] != expected_feature_count(n_beads):
        raise ValueError(f"Tensor feature extraction mismatch in {path}: {triu_features.shape}")
    return triu_features


def assert_expected_shape(array: np.ndarray, expected_shape: tuple[int, int], label: str) -> None:
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"{label} shape mismatch: expected {expected_shape}, got {array.shape}")


def validate_registry_paths() -> list[str]:
    missing: list[str] = []
    for dataset_family in FAMILY_ORDER:
        for descriptor in dataset_descriptors(dataset_family):
            if not descriptor["gt_path"].exists():
                missing.append(f"missing gt path: {descriptor['gt_path']}")
            if not descriptor["observed_path"].exists():
                missing.append(f"missing observed path: {descriptor['observed_path']}")
            for method in METHOD_ORDER:
                imputed_path = resolve_imputed_path(dataset_family, descriptor["name"], method)
                if not imputed_path.exists():
                    missing.append(f"missing imputed path: {imputed_path}")
    return missing
