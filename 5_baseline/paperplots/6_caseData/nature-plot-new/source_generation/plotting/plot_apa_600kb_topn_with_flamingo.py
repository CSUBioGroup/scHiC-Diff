#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild and plot the held-out ≥600 kb Top10/20/50 APA grid with FLAMINGO."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["schicdiff", "flamingo", "scvi3d", "schicluster", "higashi_0", "higashi_5"]
METHOD_LABELS = {
    "schicdiff": "scHiC-Diff",
    "flamingo": "FLAMINGO",
    "scvi3d": "scVI-3D",
    "schicluster": "scHiCluster",
    "higashi_0": "Higashi-0",
    "higashi_5": "Higashi-5",
}
BASE_METHOD_ORDER = [method for method in METHOD_ORDER if method != "flamingo"]
TOP_N_VALUES = [10, 20, 50]
SEEDS = [42, 43, 44]
BASE_ARCHIVE_NAME = "apa_600kb_top10_top20_top50_normed_matrices.npz"
BASE_PER_SPLIT_NAME = "apa_600kb_top10_top20_top50_per_split.csv"
BASE_MANIFEST_NAME = "apa_600kb_top10_top20_top50_source_manifest.csv"
ARCHIVE_NAME = "apa_600kb_top10_top20_top50_normed_matrices_with_flamingo.npz"
PER_SPLIT_NAME = "apa_600kb_top10_top20_top50_per_split_with_flamingo.csv"
METRICS_NAME = "apa_600kb_top10_top20_top50_metrics_with_flamingo.csv"
MANIFEST_NAME = "apa_600kb_top10_top20_top50_source_manifest_with_flamingo.csv"
OUTPUT_STEM = "apa_600kb_top10_top20_top50_with_flamingo"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_key(method: str, top_n: int, seed: int) -> str:
    return f"{method}__top{int(top_n)}__seed{int(seed)}"


def discover_experiment_root(explicit: str | Path | None = None) -> Path:
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
    for candidate in candidates:
        if (candidate / "4_test_corrected_benchmark/results_diagnostics/heldout_flamingo").is_dir():
            return candidate
    raise FileNotFoundError(f"cannot locate experiment root; checked {[str(path) for path in candidates]}")


def _load_base_payload(base_data_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    archive_path = base_data_dir / BASE_ARCHIVE_NAME
    split_path = base_data_dir / BASE_PER_SPLIT_NAME
    manifest_path = base_data_dir / BASE_MANIFEST_NAME
    for required in (archive_path, split_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    payload: dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        if archive["method_ids"].astype(str).tolist() != BASE_METHOD_ORDER:
            raise ValueError("base APA archive method order is not the approved five-method source")
        if archive["top_n_values"].astype(int).tolist() != TOP_N_VALUES:
            raise ValueError("base APA archive Top-N values differ from Top10/20/50")
        if archive["seeds"].astype(int).tolist() != SEEDS:
            raise ValueError("base APA archive seeds differ from 42/43/44")
        for method in BASE_METHOD_ORDER:
            for top_n in TOP_N_VALUES:
                for seed in SEEDS:
                    key = matrix_key(method, top_n, seed)
                    payload[key] = np.asarray(archive[key], dtype=float)
    return payload, pd.read_csv(split_path), pd.read_csv(manifest_path)


def _flamingo_split_rows(experiment_root: Path) -> pd.DataFrame:
    path = (
        experiment_root
        / "4_test_corrected_benchmark/results_diagnostics/heldout_flamingo/apa_per_split.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    source = pd.read_csv(path)
    source = source.loc[
        source["method"].eq("flamingo")
        & source["comparison_kind"].eq("rank_prefix")
        & source["min_distance_bins"].astype(int).eq(30)
        & source["set_label"].isin([f"top{value}" for value in TOP_N_VALUES])
    ].copy()
    if len(source) != len(TOP_N_VALUES) * len(SEEDS):
        raise ValueError(f"expected nine FLAMINGO APA rows, found {len(source)}")
    output = pd.DataFrame(
        {
            "method": "flamingo",
            "method_name": "FLAMINGO",
            "top_n": source["set_label"].str.removeprefix("top").astype(int),
            "seed": source["seed"].astype(int),
            "p2ll": source["P2LL"].astype(float),
            "p2m": source["P2M"].astype(float),
            "zscorell": source["ZscoreLL"].astype(float),
            "effective_count": source["effective_count"].astype(int),
            "loop_count": source["loop_count"].astype(int),
            "raw_p2ll": source["raw_p2ll"].astype(float),
            "oe_p2ll": source["oe_p2ll"].astype(float),
        }
    )
    return output.sort_values(["top_n", "seed"]).reset_index(drop=True)


def _aggregate_metrics(per_split: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, method_name, top_n), values in per_split.groupby(
        ["method", "method_name", "top_n"], sort=False
    ):
        if sorted(values["seed"].astype(int).tolist()) != SEEDS:
            raise ValueError(f"{method} Top{top_n} does not contain all three seeds")
        record = {
            "method": method,
            "method_name": method_name,
            "top_n": int(top_n),
            "set_label": f"top{int(top_n)}",
            "min_distance_bins": 30,
            "resolution_bp": 20_000,
            "window_bins": 10,
            "n_splits": 3,
        }
        for source, target in [
            ("p2ll", "p2ll"),
            ("p2m", "p2m"),
            ("zscorell", "zscorell"),
            ("raw_p2ll", "raw_p2ll"),
            ("oe_p2ll", "oe_p2ll"),
            ("effective_count", "effective_count"),
            ("loop_count", "loop_count"),
        ]:
            numbers = values[source].to_numpy(dtype=float)
            record[f"{target}_mean"] = float(numbers.mean())
            record[f"{target}_sd"] = float(numbers.std(ddof=1))
        records.append(record)
    frame = pd.DataFrame(records)
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    frame["_order"] = frame["method"].map(order)
    return frame.sort_values(["_order", "top_n"]).drop(columns="_order").reset_index(drop=True)


def build_sources(experiment_root: Path, base_data_dir: Path, data_dir: Path) -> None:
    payload, base_split, base_manifest = _load_base_payload(base_data_dir)
    flamingo_split = _flamingo_split_rows(experiment_root)
    manifest_rows = []
    heldout_root = (
        experiment_root
        / "4_test_corrected_benchmark/results_diagnostics/heldout_flamingo"
    )
    for row in flamingo_split.itertuples(index=False):
        path = (
            heldout_root
            / f"seed{int(row.seed)}/apa/min30bins/flamingo/runs/top{int(row.top_n)}"
            / "20000/chr1vchr1/normedAPA.npy"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (21, 21) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid FLAMINGO normedAPA matrix: {path}")
        key = matrix_key("flamingo", int(row.top_n), int(row.seed))
        payload[key] = np.asarray(matrix, dtype=np.float64)
        manifest_rows.append(
            {
                "method": "flamingo",
                "method_name": "FLAMINGO",
                "top_n": int(row.top_n),
                "seed": int(row.seed),
                "npz_key": key,
                "source_path": str(path.resolve()),
                "source_sha256": sha256_file(path),
                "shape": "21x21",
                "matrix_kind": "Juicer normedAPA",
            }
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_payload = {
        "method_ids": np.asarray(METHOD_ORDER, dtype="U32"),
        "method_labels": np.asarray([METHOD_LABELS[method] for method in METHOD_ORDER], dtype="U32"),
        "top_n_values": np.asarray(TOP_N_VALUES, dtype=np.int32),
        "seeds": np.asarray(SEEDS, dtype=np.int32),
        "resolution_bp": np.asarray(20_000, dtype=np.int32),
        "min_distance_bins": np.asarray(30, dtype=np.int32),
        "window_bins": np.asarray(10, dtype=np.int32),
        **payload,
    }
    np.savez_compressed(data_dir / ARCHIVE_NAME, **archive_payload)
    combined_split = pd.concat([base_split, flamingo_split], ignore_index=True, sort=False)
    combined_split["_order"] = combined_split["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    combined_split = combined_split.sort_values(["_order", "top_n", "seed"]).drop(columns="_order")
    combined_split.to_csv(data_dir / PER_SPLIT_NAME, index=False)
    _aggregate_metrics(combined_split).to_csv(data_dir / METRICS_NAME, index=False)
    combined_manifest = pd.concat(
        [base_manifest, pd.DataFrame(manifest_rows)], ignore_index=True, sort=False
    )
    combined_manifest.to_csv(data_dir / MANIFEST_NAME, index=False)


def load_sources(data_dir: Path) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    matrices = {}
    with np.load(data_dir / ARCHIVE_NAME, allow_pickle=False) as archive:
        if archive["method_ids"].astype(str).tolist() != METHOD_ORDER:
            raise ValueError("combined APA archive method order mismatch")
        for method in METHOD_ORDER:
            for top_n in TOP_N_VALUES:
                arrays = []
                for seed in SEEDS:
                    matrix = np.asarray(archive[matrix_key(method, top_n, seed)], dtype=float)
                    if matrix.shape != (21, 21) or not np.isfinite(matrix).all():
                        raise ValueError(f"invalid matrix for {method}/Top{top_n}/seed{seed}")
                    arrays.append(matrix)
                matrices[(method, top_n)] = np.mean(np.stack(arrays), axis=0)
    metrics = pd.read_csv(data_dir / METRICS_NAME)
    if len(metrics) != len(METHOD_ORDER) * len(TOP_N_VALUES):
        raise ValueError("combined APA metrics table must contain 18 rows")
    return matrices, metrics


def plot_grid(
    matrices: dict[tuple[str, int], np.ndarray],
    metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> dict[str, Path]:
    stack = np.stack([matrices[(method, top_n)] for top_n in TOP_N_VALUES for method in METHOD_ORDER])
    vmax = math.ceil(float(np.nanmax(stack)) * 4.0) / 4.0
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(3, 6, figsize=(6.85, 4.45))
        image = None
        for row_index, top_n in enumerate(TOP_N_VALUES):
            for column_index, method in enumerate(METHOD_ORDER):
                axis = axes[row_index, column_index]
                image = axis.imshow(
                    matrices[(method, top_n)],
                    origin="lower",
                    interpolation="nearest",
                    cmap="Reds",
                    vmin=0,
                    vmax=vmax,
                )
                if row_index == 0:
                    axis.set_title(METHOD_LABELS[method], pad=4)
                axis.set_xticks([0, 10, 20] if row_index == 2 and column_index == 0 else [])
                axis.set_yticks([0, 10, 20] if column_index == 0 else [])
                if row_index == 2 and column_index == 0:
                    axis.set_xticklabels(["−200", "0", "+200"])
                if column_index == 0:
                    axis.set_yticklabels(["−200", "0", "+200"])
                metric = metrics.loc[
                    metrics["method"].eq(method) & metrics["top_n"].astype(int).eq(top_n)
                ].iloc[0]
                axis.text(
                    0.5,
                    0.035,
                    f"P2LL {metric.p2ll_mean:.2f} ± {metric.p2ll_sd:.2f}",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=5.3,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.8},
                )
                axis.tick_params(length=1.8, width=0.5, pad=1.2)
        if image is None:
            raise RuntimeError("no APA panels were rendered")
        color_axis = figure.add_axes([0.932, 0.205, 0.013, 0.56])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label("Normalized APA", fontsize=6.8)
        colorbar.ax.tick_params(labelsize=6, length=2, width=0.5)
        figure.text(0.025, 0.977, "A", ha="left", va="top", fontsize=11, fontweight="medium")
        figure.text(
            0.073,
            0.977,
            "Held-out normalized APA of ≥600 kb loops",
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="medium",
        )
        figure.text(
            0.073,
            0.942,
            "Mean across three mutually exclusive 238/238-cell splits",
            ha="left",
            va="top",
            fontsize=6.8,
            color="#5C6570",
        )
        figure.text(0.515, 0.018, "Anchor 2 offset from loop center (kb)", ha="center", fontsize=7)
        figure.text(0.010, 0.47, "Anchor 1 offset (kb)", va="center", rotation=90, fontsize=7)
        figure.subplots_adjust(left=0.145, right=0.91, top=0.86, bottom=0.10, wspace=0.12, hspace=0.13)
        for row_index, top_n in enumerate(TOP_N_VALUES):
            position = axes[row_index, 0].get_position()
            figure.text(
                0.067,
                (position.y0 + position.y1) / 2,
                f"Top{top_n}\n(n={top_n})",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="medium",
            )
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
    parser.add_argument("--base-data-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--out-dir", type=Path, default=script_dir)
    parser.add_argument("--rebuild-source", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300")
    data_dir = args.data_dir.resolve()
    if args.rebuild_source or not (data_dir / ARCHIVE_NAME).is_file():
        root = discover_experiment_root(args.experiment_root)
        base_dir = args.base_data_dir.resolve() if args.base_data_dir else Path(__file__).resolve().parent
        build_sources(root, base_dir, data_dir)
        print(f"source: rebuilt ({data_dir / ARCHIVE_NAME})")
    else:
        print(f"source: frozen ({data_dir / ARCHIVE_NAME})")
    matrices, metrics = load_sources(data_dir)
    outputs = plot_grid(matrices, metrics, args.out_dir.resolve(), args.dpi)
    for extension, path in outputs.items():
        print(f"{extension}: {path}")


if __name__ == "__main__":
    main()

