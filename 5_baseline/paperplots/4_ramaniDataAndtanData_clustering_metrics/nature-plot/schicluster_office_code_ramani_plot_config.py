"""Shared method-specific embedding and prefix choices for Ramani plots."""

from __future__ import annotations


SWEEP_NDIMS = (2, 5, 10, 20, 50)

METHOD_CONFIGS = (
    {
        "condition_id": "raw",
        "display_name": "Raw",
        "source_embedding_dim": 64,
        "main_ndim": 50,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "scHiCluster",
        "display_name": "scHiCluster",
        "source_embedding_dim": 64,
        "main_ndim": 5,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "HiCImpute",
        "display_name": "HiCImpute",
        "source_embedding_dim": 64,
        "main_ndim": 2,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "higashi_nbr0",
        "display_name": "Higashi-nbr0",
        "source_embedding_dim": 128,
        "main_ndim": 10,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "higashi_nbr5",
        "display_name": "Higashi-nbr5",
        "source_embedding_dim": 128,
        "main_ndim": 10,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "scVI-3D",
        "display_name": "scVI-3D",
        "source_embedding_dim": 64,
        "main_ndim": 10,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "Tensor-FLAMINGO",
        "display_name": "Tensor-FLAMINGO",
        "source_embedding_dim": 64,
        "main_ndim": 20,
        "source_kind": "two_stage_svd_no_log1p",
    },
    {
        "condition_id": "scHiC-Diff",
        "display_name": "scHiC-Diff",
        "source_embedding_dim": 64,
        "main_ndim": 20,
        "source_kind": "two_stage_svd_no_log1p",
    },
)

CONFIG_BY_CONDITION = {row["condition_id"]: row for row in METHOD_CONFIGS}
CONFIG_BY_DISPLAY_NAME = {row["display_name"]: row for row in METHOD_CONFIGS}
METHOD_ORDER = tuple(row["display_name"] for row in METHOD_CONFIGS)
METHOD_SOURCE_EMBEDDING_DIMS = {
    row["display_name"]: row["source_embedding_dim"] for row in METHOD_CONFIGS
}
METHOD_REPORTED_NDIMS = {
    row["display_name"]: row["main_ndim"] for row in METHOD_CONFIGS
}


def validate_config():
    if len(CONFIG_BY_CONDITION) != len(METHOD_CONFIGS):
        raise ValueError("Ramani plot condition IDs must be unique")
    if len(CONFIG_BY_DISPLAY_NAME) != len(METHOD_CONFIGS):
        raise ValueError("Ramani plot display names must be unique")
    for row in METHOD_CONFIGS:
        if row["source_embedding_dim"] not in {64, 128}:
            raise ValueError(f"unsupported source dimension: {row}")
        if row["main_ndim"] not in SWEEP_NDIMS:
            raise ValueError(f"main dimension is not in the sweep: {row}")


validate_config()
