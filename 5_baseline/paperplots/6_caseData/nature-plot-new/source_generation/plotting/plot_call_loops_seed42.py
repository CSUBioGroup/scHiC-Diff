#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and plot the seed42 early-neuron call-loop comparison grid.

The approved main-text design is seven conditions by four aggregation depths
(10/100/200/476 cells). Every panel uses the seed42 contact matrix and the
matching seed42 loop-summit BEDPE. The script can rebuild a compact frozen
source archive from the corrected benchmark or replay the frozen archive after
delivery to ``nature-plot``. The rows are Raw plus six imputation methods,
including FLAMINGO.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_SLUGS = [
    "raw",
    "schicdiff",
    "flamingo",
    "scvi3d",
    "schicluster",
    "higashi_nbr0",
    "higashi_nbr5",
]
METHOD_LABELS = [
    "Raw",
    "scHiC-Diff",
    "FLAMINGO",
    "scVI-3D",
    "scHiCluster",
    "Higashi-0",
    "Higashi-5",
]
METHOD_LABEL_BY_SLUG = dict(zip(METHOD_SLUGS, METHOD_LABELS))
CELL_COUNTS = [10, 100, 200, 476]
GROUP = "earlyNeurons"
SEED = 42
N_BINS = 100
RESOLUTION_BP = 20_000
CALLER_MIN_DISTANCE_BP = 60_000
CALLER_MAX_DISTANCE_BP = 2_000_000

SOURCE_NPZ_NAME = "call_loops_seed42_panel_data_with_flamingo.npz"
PANEL_COUNTS_NAME = "call_loops_seed42_panel_counts_with_flamingo.csv"
SOURCE_MANIFEST_NAME = "call_loops_seed42_source_manifest_with_flamingo.csv"
OUTPUT_STEM = "call_loops_seed42_10_100_200_476_with_flamingo"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_experiment_root(explicit: str | Path | None = None) -> Path:
    """Find the ``2_callLoop_apa`` root from either staging or nature-plot."""
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir.parent,
            script_dir.parent / "2_callLoop_apa",
            script_dir.parent.parent / "2_callLoop_apa",
            Path.cwd().resolve(),
        ]
    marker = Path("4_test_corrected_benchmark/configs/benchmark.json")
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"cannot locate 2_callLoop_apa benchmark root; checked: {rendered}")


def discover_style_dir(
    experiment_root: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    """Locate the existing ``gr_panels_bcd.py`` and ``gr_stagefig.py`` pair."""
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        script_dir = Path(__file__).resolve().parent
        root = Path(experiment_root).resolve()
        candidates = [
            script_dir,
            root.parent / "nature-plot",
            script_dir.parent / "nature-plot",
            script_dir.parent.parent / "nature-plot",
        ]
    for candidate in candidates:
        if (candidate / "gr_panels_bcd.py").is_file() and (candidate / "gr_stagefig.py").is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"cannot locate gr_panels_bcd.py/gr_stagefig.py; checked: {rendered}")


def _add_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def load_summit_bins(
    path: str | Path,
    *,
    resolution: int = RESOLUTION_BP,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """Read a summit BEDPE and return integer ``(anchor1_bin, anchor2_bin)`` pairs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, 2), dtype=np.int32)
    frame = pd.read_csv(path, sep="\t", header=None, comment="#")
    if frame.empty:
        return np.empty((0, 2), dtype=np.int32)
    if frame.shape[1] < 4:
        raise ValueError(f"summit BEDPE must have at least four columns: {path}")
    coordinates = frame.iloc[:, :4].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int64)
    if (coordinates < 0).any():
        raise ValueError(f"summit BEDPE contains negative coordinates: {path}")
    if (coordinates[:, 1] <= coordinates[:, 0]).any() or (coordinates[:, 3] <= coordinates[:, 2]).any():
        raise ValueError(f"summit BEDPE contains empty/reversed anchors: {path}")
    mid1 = (coordinates[:, 0] + coordinates[:, 1]) // 2
    mid2 = (coordinates[:, 2] + coordinates[:, 3]) // 2
    bins = np.column_stack((mid1 // resolution, mid2 // resolution)).astype(np.int32)
    if (bins < 0).any() or (bins >= n_bins).any():
        raise ValueError(f"summit BEDPE maps outside 0..{n_bins - 1}: {path}")
    if (bins[:, 1] <= bins[:, 0]).any():
        raise ValueError(f"summit BEDPE anchors are not strictly upper-triangular: {path}")
    if len(np.unique(bins, axis=0)) != len(bins):
        raise ValueError(f"summit BEDPE contains duplicate bin pairs: {path}")
    return bins


def matrix_key(method: str, cell_count: int) -> str:
    return f"matrix__{method}__{int(cell_count)}"


def summit_key(method: str, cell_count: int) -> str:
    return f"summits__{method}__{int(cell_count)}"


def _validate_panel_payloads(
    matrices: dict[tuple[str, int], np.ndarray],
    summits: dict[tuple[str, int], np.ndarray],
) -> None:
    expected = {(method, count) for method in METHOD_SLUGS for count in CELL_COUNTS}
    if set(matrices) != expected:
        raise ValueError(f"matrix panel keys differ from approved design: missing={sorted(expected - set(matrices))}")
    if set(summits) != expected:
        raise ValueError(f"summit panel keys differ from approved design: missing={sorted(expected - set(summits))}")
    for key in sorted(expected):
        matrix = np.asarray(matrices[key])
        loop_bins = np.asarray(summits[key])
        if matrix.shape != (N_BINS, N_BINS) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid matrix for {key}: shape={matrix.shape}")
        if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-7):
            raise ValueError(f"contact matrix is not symmetric for {key}")
        if loop_bins.ndim != 2 or loop_bins.shape[1] != 2:
            raise ValueError(f"invalid summit array for {key}: shape={loop_bins.shape}")
        if loop_bins.size and ((loop_bins < 0).any() or (loop_bins >= N_BINS).any()):
            raise ValueError(f"summit array is out of bounds for {key}")


def write_source_archive(
    path: str | Path,
    matrices: dict[tuple[str, int], np.ndarray],
    summits: dict[tuple[str, int], np.ndarray],
) -> Path:
    """Write the 24 matrices and 24 loop arrays without pickle payloads."""
    _validate_panel_payloads(matrices, summits)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "method_slugs": np.asarray(METHOD_SLUGS, dtype="U32"),
        "method_labels": np.asarray(METHOD_LABELS, dtype="U32"),
        "cell_counts": np.asarray(CELL_COUNTS, dtype=np.int32),
        "group": np.asarray(GROUP),
        "seed": np.asarray(SEED, dtype=np.int32),
        "n_bins": np.asarray(N_BINS, dtype=np.int32),
        "resolution_bp": np.asarray(RESOLUTION_BP, dtype=np.int32),
        "caller_min_distance_bp": np.asarray(CALLER_MIN_DISTANCE_BP, dtype=np.int32),
        "caller_max_distance_bp": np.asarray(CALLER_MAX_DISTANCE_BP, dtype=np.int32),
    }
    for method in METHOD_SLUGS:
        for count in CELL_COUNTS:
            payload[matrix_key(method, count)] = np.asarray(matrices[(method, count)], dtype=np.float32)
            payload[summit_key(method, count)] = np.asarray(summits[(method, count)], dtype=np.int32)
    np.savez_compressed(path, **payload)
    return path


def load_source_archive(
    path: str | Path,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    """Load and strictly validate a frozen panel archive."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    matrices: dict[tuple[str, int], np.ndarray] = {}
    summits: dict[tuple[str, int], np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        archived_methods = archive["method_slugs"].astype(str).tolist()
        archived_labels = archive["method_labels"].astype(str).tolist()
        archived_counts = archive["cell_counts"].astype(int).tolist()
        metadata = {
            "group": str(archive["group"].item()),
            "seed": int(archive["seed"].item()),
            "n_bins": int(archive["n_bins"].item()),
            "resolution_bp": int(archive["resolution_bp"].item()),
            "caller_min_distance_bp": int(archive["caller_min_distance_bp"].item()),
            "caller_max_distance_bp": int(archive["caller_max_distance_bp"].item()),
        }
        if archived_methods != METHOD_SLUGS or archived_labels != METHOD_LABELS:
            raise ValueError("frozen source method order/labels do not match the approved figure")
        if archived_counts != CELL_COUNTS:
            raise ValueError("frozen source cell-count order does not match the approved figure")
        expected_metadata = {
            "group": GROUP,
            "seed": SEED,
            "n_bins": N_BINS,
            "resolution_bp": RESOLUTION_BP,
            "caller_min_distance_bp": CALLER_MIN_DISTANCE_BP,
            "caller_max_distance_bp": CALLER_MAX_DISTANCE_BP,
        }
        if metadata != expected_metadata:
            raise ValueError(f"unexpected frozen-source metadata: {metadata}")
        for method in METHOD_SLUGS:
            for count in CELL_COUNTS:
                matrices[(method, count)] = np.asarray(archive[matrix_key(method, count)], dtype=float)
                summits[(method, count)] = np.asarray(archive[summit_key(method, count)], dtype=np.int32)
    _validate_panel_payloads(matrices, summits)
    return matrices, summits, metadata


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one {label} row, found {len(selected)}")
    return selected.iloc[0]


def build_source_data(
    experiment_root: str | Path,
    data_dir: str | Path,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]]:
    """Reconstruct the approved seed42 matrices and package all provenance."""
    root = discover_experiment_root(experiment_root)
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = root / "4_test_corrected_benchmark"
    config_path = benchmark_dir / "configs/benchmark.json"
    flamingo_config_path = benchmark_dir / "configs/benchmark_flamingo.json"
    subset_manifest_path = benchmark_dir / "results/subsets/subset_manifest.csv"
    run_manifest_path = benchmark_dir / "results/manifests/loop_runs.csv"
    flamingo_run_manifest_path = benchmark_dir / "results_flamingo/manifests/loop_runs.csv"
    for required in (
        config_path,
        flamingo_config_path,
        subset_manifest_path,
        run_manifest_path,
        flamingo_run_manifest_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    scripts_dir = benchmark_dir / "scripts"
    _add_import_path(scripts_dir)
    adapters = importlib.import_module("adapters")
    group_matrices = importlib.import_module("build_group_matrices")
    benchmark = importlib.import_module("run_benchmark")
    config = benchmark.load_resolved_config(config_path)
    flamingo_config = benchmark.load_resolved_config(flamingo_config_path)

    if int(config["n_bins"]) != N_BINS:
        raise ValueError(f"benchmark n_bins={config['n_bins']} does not match {N_BINS}")
    loop_config = config["loop_calling"]
    observed_loop_metadata = (
        int(loop_config["resolution"]),
        int(loop_config["min_dist"]),
        int(loop_config["max_dist"]),
    )
    expected_loop_metadata = (RESOLUTION_BP, CALLER_MIN_DISTANCE_BP, CALLER_MAX_DISTANCE_BP)
    if observed_loop_metadata != expected_loop_metadata:
        raise ValueError(
            f"caller metadata {observed_loop_metadata} does not match {expected_loop_metadata}"
        )

    method_configs = {str(item["slug"]): item for item in config["methods"]}
    if len(flamingo_config["methods"]) != 1 or flamingo_config["methods"][0]["slug"] != "flamingo":
        raise ValueError("benchmark_flamingo.json must contain exactly the FLAMINGO method")
    method_configs["flamingo"] = flamingo_config["methods"][0]
    if set(method_configs) != set(METHOD_SLUGS):
        raise ValueError(
            f"benchmark methods differ from approved six conditions: {sorted(method_configs)}"
        )
    subsets = pd.read_csv(subset_manifest_path)
    runs = pd.read_csv(run_manifest_path)
    runs = runs.loc[runs["status"].eq("completed")].copy()
    flamingo_runs = pd.read_csv(flamingo_run_manifest_path)
    flamingo_runs = flamingo_runs.loc[flamingo_runs["status"].eq("completed")].copy()

    matrices: dict[tuple[str, int], np.ndarray] = {}
    summits: dict[tuple[str, int], np.ndarray] = {}
    panel_records: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = [
        {
            "role": "benchmark_config",
            "method": "",
            "cell_count": "",
            "seed": "",
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "semantic_sha256": "",
        },
        {
            "role": "subset_manifest",
            "method": "",
            "cell_count": "",
            "seed": "",
            "path": str(subset_manifest_path.resolve()),
            "sha256": sha256_file(subset_manifest_path),
            "semantic_sha256": "",
        },
        {
            "role": "loop_run_manifest",
            "method": "",
            "cell_count": "",
            "seed": "",
            "path": str(run_manifest_path.resolve()),
            "sha256": sha256_file(run_manifest_path),
            "semantic_sha256": "",
        },
        {
            "role": "flamingo_benchmark_config",
            "method": "flamingo",
            "cell_count": "",
            "seed": "",
            "path": str(flamingo_config_path.resolve()),
            "sha256": sha256_file(flamingo_config_path),
            "semantic_sha256": "",
        },
        {
            "role": "flamingo_loop_run_manifest",
            "method": "flamingo",
            "cell_count": "",
            "seed": "",
            "path": str(flamingo_run_manifest_path.resolve()),
            "sha256": sha256_file(flamingo_run_manifest_path),
            "semantic_sha256": "",
        },
    ]
    hashed_files: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        path = path.resolve()
        if path not in hashed_files:
            hashed_files[path] = sha256_file(path)
        return hashed_files[path]

    for method in METHOD_SLUGS:
        method_config = method_configs[method]
        input_path = Path(method_config["input_npz"])
        source = adapters.load_csr_npz(input_path)
        run_frame = flamingo_runs if method == "flamingo" else runs
        method_runs = run_frame.loc[run_frame["method"].eq(method)]
        input_hashes = method_runs["input_sha256"].dropna().astype(str).unique().tolist()
        if len(input_hashes) != 1:
            raise ValueError(f"method {method} has inconsistent input hashes: {input_hashes}")
        provenance_records.append(
            {
                "role": "method_input_npz",
                "method": method,
                "cell_count": "",
                "seed": "",
                "path": str(input_path.resolve()),
                "sha256": input_hashes[0],
                "semantic_sha256": "from loop_runs.csv",
            }
        )

        for count in CELL_COUNTS:
            subset = _one_row(
                subsets,
                subsets["group"].eq(GROUP)
                & subsets["cell_count"].astype(int).eq(count)
                & subsets["seed"].astype(int).eq(SEED),
                f"subset {GROUP}/{count}/seed{SEED}",
            )
            run = _one_row(
                run_frame,
                run_frame["method"].eq(method)
                & run_frame["group"].eq(GROUP)
                & run_frame["cell_count"].astype(int).eq(count)
                & run_frame["seed"].astype(int).eq(SEED),
                f"loop run {method}/{GROUP}/{count}/seed{SEED}",
            )
            if str(run["subset_sha256"]) != str(subset["index_sha256"]):
                raise ValueError(f"subset hash mismatch for {method}/{count}")
            subset_path = Path(str(subset["subset_path"]))
            if not subset_path.is_file():
                raise FileNotFoundError(subset_path)
            indices = np.load(subset_path, allow_pickle=False)
            if len(indices) != count:
                raise ValueError(f"subset length {len(indices)} != {count}: {subset_path}")
            vector = group_matrices.aggregate_standard_vectors(
                source,
                indices=indices,
                include_diagonal=bool(method_config["include_diagonal"]),
                n_bins=N_BINS,
            )
            matrix = benchmark.vector_to_symmetric_matrix(vector, N_BINS)
            summit_path = Path(str(run["run_dir"])) / "loops.loop_summit.bedpe"
            loop_bins = load_summit_bins(summit_path)
            expected_summits = int(run["summit_count"])
            if len(loop_bins) != expected_summits:
                raise ValueError(
                    f"summit count mismatch for {method}/{count}: {len(loop_bins)} != {expected_summits}"
                )

            matrices[(method, count)] = matrix
            summits[(method, count)] = loop_bins
            panel_records.append(
                {
                    "method": method,
                    "method_name": METHOD_LABEL_BY_SLUG[method],
                    "group": GROUP,
                    "cell_count": count,
                    "seed": SEED,
                    "subset_sha256": str(subset["index_sha256"]),
                    "input_sha256": input_hashes[0],
                    "loop_count": int(run["loop_count"]),
                    "summit_count": expected_summits,
                    "matrix_upper_offdiag_sum": float(np.triu(matrix, k=1).sum()),
                    "matrix_nonzero": int(np.count_nonzero(matrix)),
                    "caller_min_distance_bp": CALLER_MIN_DISTANCE_BP,
                    "caller_max_distance_bp": CALLER_MAX_DISTANCE_BP,
                }
            )
            provenance_records.extend(
                [
                    {
                        "role": "cell_subset_npy",
                        "method": method,
                        "cell_count": count,
                        "seed": SEED,
                        "path": str(subset_path.resolve()),
                        "sha256": cached_hash(subset_path),
                        "semantic_sha256": str(subset["index_sha256"]),
                    },
                    {
                        "role": "loop_summit_bedpe",
                        "method": method,
                        "cell_count": count,
                        "seed": SEED,
                        "path": str(summit_path.resolve()),
                        "sha256": cached_hash(summit_path),
                        "semantic_sha256": "",
                    },
                ]
            )

    _validate_panel_payloads(matrices, summits)
    write_source_archive(data_dir / SOURCE_NPZ_NAME, matrices, summits)
    panel_frame = pd.DataFrame(panel_records)
    panel_frame["method"] = pd.Categorical(panel_frame["method"], METHOD_SLUGS, ordered=True)
    panel_frame = panel_frame.sort_values(["method", "cell_count"]).reset_index(drop=True)
    panel_frame["method"] = panel_frame["method"].astype(str)
    panel_frame.to_csv(data_dir / PANEL_COUNTS_NAME, index=False)
    pd.DataFrame(provenance_records).to_csv(data_dir / SOURCE_MANIFEST_NAME, index=False)
    return matrices, summits


def plot_call_loop_grid(
    matrices: dict[tuple[str, int], np.ndarray],
    summits: dict[tuple[str, int], np.ndarray],
    output_dir: str | Path,
    *,
    style_dir: str | Path,
    dpi: int = 600,
) -> dict[str, Path]:
    """Render the approved grid with the existing Panel-B drawing function."""
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for publication output")
    _validate_panel_payloads(matrices, summits)
    style_dir = Path(style_dir).resolve()
    _add_import_path(style_dir)
    gr_stagefig = importlib.import_module("gr_stagefig")
    gr_panels = importlib.import_module("gr_panels_bcd")
    gr_stagefig.set_gr_style()

    display_matrices = {
        (METHOD_LABEL_BY_SLUG[method], count): matrices[(method, count)]
        for method in METHOD_SLUGS
        for count in CELL_COUNTS
    }
    display_summits = {
        (METHOD_LABEL_BY_SLUG[method], count): summits[(method, count)]
        for method in METHOD_SLUGS
        for count in CELL_COUNTS
    }

    fig_w_mm = 174.0
    fig_h_mm = 245.0
    figure = plt.figure(figsize=(gr_stagefig.mm(fig_w_mm), gr_stagefig.mm(fig_h_mm)))
    bottom_mm = gr_panels.draw_loop_grid(
        figure,
        matrices=display_matrices,
        loops=display_summits,
        methods=METHOD_LABELS,
        cell_counts=CELL_COUNTS,
        highlight="scHiC-Diff",
        fig_w_mm=fig_w_mm,
        fig_h_mm=fig_h_mm,
        x_mm=12.0,
        top_mm=4.0,
        width_mm=150.0,
        left_gutter_mm=24.0,
        gap_mm=1.1,
        header_mm=5.0,
        cbar_mm=4.0,
        scale="sum",
        vq=99.0,
        resolution=RESOLUTION_BP,
    )
    if bottom_mm > fig_h_mm:
        plt.close(figure)
        raise RuntimeError(f"grid height {bottom_mm:.1f} mm exceeds canvas {fig_h_mm:.1f} mm")
    figure.text(0.010, 0.994, "B", ha="left", va="top", fontsize=gr_stagefig.PT_TAG, fontweight="bold")
    figure.text(
        0.080,
        0.993,
        "early neurons · chr1 at 20 kb · seed42 · circles mark called loop summits",
        ha="left",
        va="top",
        fontsize=gr_stagefig.PT_SMALL,
        color=gr_stagefig.TEXT_MUTED,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {extension: output_dir / f"{OUTPUT_STEM}.{extension}" for extension in ("png", "pdf", "svg")}
    figure.savefig(outputs["png"], dpi=dpi, facecolor="white")
    figure.savefig(outputs["pdf"], facecolor="white")
    figure.savefig(outputs["svg"], facecolor="white")
    plt.close(figure)
    return outputs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--style-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--out-dir", type=Path, default=script_dir)
    parser.add_argument("--rebuild-source", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    source_path = data_dir / SOURCE_NPZ_NAME
    root: Path | None = None
    if args.rebuild_source or not source_path.is_file():
        root = discover_experiment_root(args.experiment_root)
        matrices, summits = build_source_data(root, data_dir)
        source_mode = "rebuilt"
    else:
        matrices, summits, _ = load_source_archive(source_path)
        source_mode = "frozen"
    print(f"source: {source_mode} ({source_path})")
    print(f"panels: {len(matrices)} matrices + {len(summits)} summit arrays; group={GROUP}; seed={SEED}")
    if args.source_only:
        return
    if root is None:
        root = discover_experiment_root(args.experiment_root)
    style_dir = discover_style_dir(root, args.style_dir)
    outputs = plot_call_loop_grid(matrices, summits, args.out_dir, style_dir=style_dir, dpi=args.dpi)
    for extension, path in outputs.items():
        print(f"{extension}: {path}")


if __name__ == "__main__":
    main()
