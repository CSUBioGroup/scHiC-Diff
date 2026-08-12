#!/usr/bin/env python3
"""Build the method-specific ARI table consumed by the Ramani paper figures."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from schicluster_office_code_ramani_plot_config import (
    METHOD_CONFIGS,
    SWEEP_NDIMS,
)


NATURE_DIR = Path(__file__).resolve().parent


def portable_path(path):
    return os.path.relpath(Path(path).resolve(), NATURE_DIR)


def load_labels(path):
    table = pd.read_csv(path, sep="\t")
    if not {"cell_id", "celltype"}.issubset(table.columns):
        raise ValueError("label table must contain cell_id and celltype")
    table = table.loc[:, ["cell_id", "celltype"]].copy()
    if len(table) != 626 or table["cell_id"].duplicated().any():
        raise ValueError("Ramani label table must contain 626 unique cells")
    return table


def select_standard_rows(official, config):
    selected = official[
        (official["condition_id"] == config["condition_id"])
        & (official["source_embedding_dim"] == config["source_embedding_dim"])
        & (official["ndim"].isin(SWEEP_NDIMS))
    ].copy()
    if len(selected) != len(SWEEP_NDIMS) or selected["ndim"].duplicated().any():
        raise ValueError(f"official ARI rows are incomplete for {config['display_name']}")
    selected["source_kind"] = config["source_kind"]
    return selected


def run(official_ari_path, labels_path, output_dir):
    official_ari_path = Path(official_ari_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    official = pd.read_csv(official_ari_path)
    labels = load_labels(labels_path)
    frames = []
    for config in METHOD_CONFIGS:
        frames.append(select_standard_rows(official, config))
    result = pd.concat(frames, ignore_index=True, sort=False)

    main_ndim_by_condition = {
        row["condition_id"]: row["main_ndim"] for row in METHOD_CONFIGS
    }
    result["selected_for_main"] = result["ndim"].eq(
        result["condition_id"].map(main_ndim_by_condition)
    )
    result["input_path"] = result["condition_id"].map(
        {row["condition_id"]: portable_path(official_ari_path) for row in METHOD_CONFIGS}
    )

    expected_pairs = {
        (row["condition_id"], ndim)
        for row in METHOD_CONFIGS
        for ndim in SWEEP_NDIMS
    }
    actual_pairs = set(zip(result["condition_id"], result["ndim"]))
    if actual_pairs != expected_pairs or len(result) != len(expected_pairs):
        raise RuntimeError("plot ARI table does not contain all configured rows")
    if result.groupby("condition_id")["selected_for_main"].sum().ne(1).any():
        raise RuntimeError("each method must have exactly one main ARI row")
    if not np.isfinite(result["ari"]).all():
        raise RuntimeError("plot ARI table contains non-finite values")

    output_path = output_dir / "schicluster_office_code_Ramani_plot_ARI_long.csv"
    result.to_csv(output_path, index=False)
    config_path = output_dir / "schicluster_office_code_Ramani_plot_ARI_run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_name": "schicluster_office_code_Ramani_plot_ARI",
                "official_ari_path": portable_path(official_ari_path),
                "labels_path": portable_path(labels_path),
                "output_path": portable_path(output_path),
                "method_config": list(METHOD_CONFIGS),
                "n_clusters": 4,
                "n_init": 200,
                "random_state": None,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def build_parser():
    result_dir = NATURE_DIR / "results/schicluster_office_code_Ramani"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-ari",
        type=Path,
        default=result_dir / "schicluster_office_code_Ramani_ARI_long.csv",
    )
    parser.add_argument(
        "--labels", type=Path, default=NATURE_DIR / "test/config/ramani_cells.tsv"
    )
    parser.add_argument("--output-dir", type=Path, default=result_dir)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run(args.official_ari, args.labels, args.output_dir)
    print(result[result["selected_for_main"]][
        ["display_name", "source_embedding_dim", "ndim", "ari"]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
