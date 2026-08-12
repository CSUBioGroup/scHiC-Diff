#!/usr/bin/env python3
"""
Plot a HiRES 1Mb stage-split UMAP from a 20-dimensional SVD embedding.

Input:
  - NPZ embedding with key arr_0 or X, shape: cells x dimensions.
  - cell_labels.csv with columns: cell_id, cellname, stage, celltype.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


try:
    import umap

    UMAP_IMPL = "umap-learn"
except ImportError:
    umap = None
    UMAP_IMPL = "sklearn-TSNE"


DEFAULT_EMBEDDING = Path("svd_embedding/method/final_svd_decomp.npz")
DEFAULT_LABELS = Path("cell_labels.csv")
DEFAULT_OUT_PREFIX = Path("fligures/method/method_1Mb_umap_celltype_split_stage_with_scores")


RAW_CELLTYPE_COLORS = {
    "mitosis": "#bebebe",
    "blood": "#ff9400",
    "ExE endoderm": "#fec44f",
    "ExE ectoderm": "#cab2d6",
    "epiblast and PS": "#addd8e",
    "neural ectoderm": "#AECBE6",
    "NMP": "#96B9DB",
    "neural tube": "#7EA8D0",
    "notochord": "#6696C6",
    "radial glias": "#4F85BB",
    "oligodendrocytes and progenitors": "#3773B1",
    "early neurons": "#1F62A6",
    "schwann cell precursors": "#08519C",
    "early mesoderm": "#FC9272",
    "ExE mesoderm": "#EF7F64",
    "early mesenchyme": "#E36C57",
    "intermediate mesoderm": "#D6594A",
    "myocytes": "#CA473C",
    "mix late mesenchyme": "#BD342F",
    "endoderm": "#fe9929",
    "epithelial cells": "#fa9fb5",
}


CELLTYPE_RENAME = {
    "mitosis": "Mitotic cell",
    "blood": "Blood",
    "ExE endoderm": "ExE endoderm",
    "ExE ectoderm": "ExE ectoderm",
    "epiblast and PS": "EPI",
    "neural ectoderm": "Neural ectoderm",
    "NMP": "NMP",
    "neural tube": "Neural tube",
    "notochord": "Notochord",
    "radial glias": "Radial glia",
    "oligodendrocytes and progenitors": "OPC",
    "early neurons": "Early neuron",
    "schwann cell precursors": "Schwann cell precursor",
    "early mesoderm": "Early mesoderm",
    "ExE mesoderm": "ExE mesoderm",
    "early mesenchyme": "Early mesenchyme",
    "intermediate mesoderm": "Intermediate mesoderm",
    "myocytes": "Myocyte",
    "mix late mesenchyme": "Mixed late mesenchyme",
    "endoderm": "Endoderm",
    "epithelial cells": "Epithelial cell",
}


BLUE_SERIES = {
    "Neural ectoderm",
    "NMP",
    "Neural tube",
    "Notochord",
    "Radial glia",
    "OPC",
    "Early neuron",
    "Schwann cell precursor",
}


RED_SERIES = {
    "Blood",
    "Early mesoderm",
    "ExE mesoderm",
    "Early mesenchyme",
    "Intermediate mesoderm",
    "Myocyte",
    "Mixed late mesenchyme",
}


def load_embeddings(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {npz_path}")
    with np.load(npz_path) as data:
        key = "arr_0" if "arr_0" in data.files else ("X" if "X" in data.files else data.files[0])
        embedding = data[key]
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding, got shape {embedding.shape}.")
    return embedding


def load_labels(labels_path: Path, expected_rows: int) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = pd.read_csv(labels_path)
    required = {"cell_id", "stage", "celltype"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels file is missing required columns: {sorted(missing)}")
    if len(labels) != expected_rows:
        raise ValueError(
            "Row count mismatch: "
            f"embedding has {expected_rows} rows, labels has {len(labels)} rows."
        )
    labels = labels.copy()
    labels["celltype"] = labels["celltype"].map(CELLTYPE_RENAME).fillna(labels["celltype"])
    labels["stage"] = labels["stage"].astype(str)
    labels["celltype"] = labels["celltype"].astype(str)
    return labels


def preferred_palette(categories) -> Dict[str, str]:
    renamed_colors = {CELLTYPE_RENAME.get(k, k): v for k, v in RAW_CELLTYPE_COLORS.items()}
    palette = dict(renamed_colors)
    missing = [category for category in categories if category not in palette]
    if missing:
        colors = sns.color_palette("tab20", n_colors=max(20, len(missing)))
        for category, color in zip(missing, colors):
            palette[category] = mcolors.to_hex(color)
    return palette


def compute_2d(
    embedding: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    if embedding.shape[0] < 3:
        raise ValueError("Need at least 3 cells to compute a 2D embedding.")

    if umap is not None:
        effective_neighbors = min(max(2, n_neighbors), embedding.shape[0] - 1)
        reducer = umap.UMAP(
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        return reducer.fit_transform(embedding), "UMAP"

    perplexity = min(30, max(2, min(10, embedding.shape[0] - 1)))
    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    return reducer.fit_transform(embedding), "t-SNE"


def compute_stage_score(stage_df: pd.DataFrame) -> Optional[float]:
    score_df = stage_df[stage_df["series"].isin(["Red", "Blue"])]
    n_classes = score_df["series"].nunique()
    if n_classes < 2 or n_classes >= len(score_df):
        return None
    return float(silhouette_score(score_df[["DIM1", "DIM2"]].to_numpy(), score_df["series"].to_numpy()))


def plot_stage_umap(
    coords: np.ndarray,
    labels: pd.DataFrame,
    dim_label: str,
    point_size: float,
    alpha: float,
    base_fontsize: int,
    legend_fontsize: int,
):
    df = labels.reset_index(drop=True).copy()
    df["DIM1"] = coords[:, 0]
    df["DIM2"] = coords[:, 1]
    df["series"] = df["celltype"].apply(
        lambda value: "Blue" if value in BLUE_SERIES else ("Red" if value in RED_SERIES else "Other")
    )

    stages = sorted(df["stage"].dropna().unique().tolist())
    if not stages:
        raise ValueError("No stage values found in labels.")

    palette = preferred_palette(df["celltype"].unique())
    panel_width = 4.0
    panel_height = 5.0
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(stages),
        figsize=(panel_width * len(stages), panel_height),
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, stage in enumerate(stages):
        ax = axes[idx]
        stage_df = df[df["stage"] == stage].copy()
        score = compute_stage_score(stage_df)

        for celltype, subset in stage_df.groupby("celltype", sort=True):
            ax.scatter(
                subset["DIM1"],
                subset["DIM2"],
                s=point_size,
                alpha=alpha,
                c=palette.get(str(celltype), "#888888"),
                label=str(celltype),
            )

        title = f"Stage: {stage} (n={len(stage_df)})"
        if score is not None:
            title += f"\nSilhouette: {score:.3f}"

        bg_height = 0.14 if score is not None else 0.09
        ax.add_patch(
            Rectangle(
                (0, 1.0),
                1,
                bg_height,
                transform=ax.transAxes,
                facecolor="#E0E0E0",
                alpha=0.8,
                edgecolor="none",
                clip_on=False,
            )
        )
        ax.text(
            0.5,
            1.0 + bg_height / 2,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=base_fontsize,
            clip_on=False,
            linespacing=1.4,
        )

        ax.set_xlabel(f"{dim_label} 1", fontsize=base_fontsize)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(idx == 0)
        ax.spines["bottom"].set_visible(True)
        ax.set_ylabel(f"{dim_label} 2" if idx == 0 else "", fontsize=base_fontsize)

    handles = [
        Line2D([0], [0], marker="o", color="none", label=label, markerfacecolor=color, markersize=6)
        for label, color in sorted(palette.items())
    ]
    fig.legend(
        handles=handles,
        title="celltype",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=legend_fontsize,
        ncol=14,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig, df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 1Mb stage-split UMAP from a 20d SVD embedding.")
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING, help="Input final_svd_decomp.npz.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Input cell_labels.csv.")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX, help="Output prefix without suffix.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--random-state", type=int, default=42, help="UMAP/t-SNE random seed.")
    parser.add_argument("--point-size", type=float, default=6.0, help="Scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Scatter point alpha.")
    parser.add_argument("--base-fontsize", type=int, default=12, help="Axis and strip title font size.")
    parser.add_argument("--legend-fontsize", type=int, default=12, help="Legend font size.")
    parser.add_argument("--dpi", type=int, default=400, help="PNG output DPI.")
    parser.add_argument("--no-coordinates", action="store_true", help="Do not save UMAP coordinates CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding = load_embeddings(args.embedding)
    labels = load_labels(args.labels, expected_rows=embedding.shape[0])
    coords, dim_label = compute_2d(
        embedding=embedding,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    fig, plot_df = plot_stage_umap(
        coords=coords,
        labels=labels,
        dim_label=dim_label,
        point_size=args.point_size,
        alpha=args.alpha,
        base_fontsize=args.base_fontsize,
        legend_fontsize=args.legend_fontsize,
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = args.out_prefix.with_suffix(".png")
    out_pdf = args.out_prefix.with_suffix(".pdf")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot PNG: {out_png}")
    print(f"Saved plot PDF: {out_pdf}")

    if not args.no_coordinates:
        out_csv = args.out_prefix.with_suffix(".csv")
        plot_df.to_csv(out_csv, index=False)
        print(f"Saved coordinates CSV: {out_csv}")


if __name__ == "__main__":
    main()
