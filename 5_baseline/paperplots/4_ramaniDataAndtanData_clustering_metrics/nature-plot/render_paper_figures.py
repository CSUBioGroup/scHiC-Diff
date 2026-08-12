#!/usr/bin/env python3
"""Render formal Ramani scHiCluster-style ARI and clustering figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import plot_ramani_style as style
from schicluster_office_code_ramani_plot_config import (
    METHOD_ORDER,
    METHOD_REPORTED_NDIMS,
    METHOD_SOURCE_EMBEDDING_DIMS,
    SWEEP_NDIMS,
)


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


DEFAULT_RESULTS_DIR = Path("results/schicluster_office_code_Ramani")
DEFAULT_TAN_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("figures")
SUPPORTED_EXPORT_SUFFIXES = (".svg", ".pdf", ".png", ".tiff")

HERO_METHOD = "scHiC-Diff"
ARI_RULE = "method_specific_embedding_and_prefix"
EXPECTED_COUNTS = {"HeLa": 258, "HAP1": 214, "K562": 110, "GM12878": 44}
FIGURE_CHOICES = ("main", "ari", "cluster", "sweep", "tan")
MAIN_WIDTH_MM = 174
MAIN_HEIGHT_MM = 142
MAIN_ARI_DISPLAY_LABELS = {}
TAN_SEGMENT_ORDER = ("2050", "160190")
TAN_SEGMENT_LABELS = {"2050": "20–50", "160190": "160–190"}
TAN_CELL_TYPES = ("GM12878", "PBMC")
TAN_METHOD_LABELS = {
    "Raw": "Raw",
    "scHiCluster": "scHiCluster",
    "HiCImpute": "HiCImpute",
    "Higashi-nbr0": "Higashi-0",
    "Higashi-nbr5": "Higashi-5",
    "scVI-3D": "scVI-3D",
    "Tensor-FLAMINGO": "T-FLAMINGO",
    "scHiC-Diff": "scHiC-Diff",
}
TAN_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "tan_confusion",
    ("#F4F6F8", "#B8C7D9", style.CELL_TYPE_COLORS["HeLa"]),
)


class PaperData:
    def __init__(self, results_dir):
        results_dir = Path(results_dir)
        self.metrics = pd.read_csv(
            results_dir / "schicluster_office_code_Ramani_plot_ARI_long.csv"
        )
        self.coordinates = pd.read_csv(
            results_dir
            / "schicluster_office_code_Ramani_cluster_coordinates.csv"
        )
        self._validate()

    def _validate(self):
        required_metric_columns = {
            "display_name",
            "source_embedding_dim",
            "ndim",
            "ari",
            "n_cells",
            "n_clusters",
            "n_init",
            "selected_for_main",
        }
        missing = required_metric_columns - set(self.metrics.columns)
        if missing:
            raise ValueError(f"ARI table lacks columns: {sorted(missing)}")
        expected_source_dims = self.metrics["display_name"].map(
            METHOD_SOURCE_EMBEDDING_DIMS
        )
        selected = self.metrics[
            (self.metrics["source_embedding_dim"] == expected_source_dims)
            & (self.metrics["ndim"].isin(SWEEP_NDIMS))
        ].copy()
        expected_pairs = {
            (method, ndim) for method in METHOD_ORDER for ndim in SWEEP_NDIMS
        }
        actual_pairs = set(zip(selected["display_name"], selected["ndim"]))
        if actual_pairs != expected_pairs or len(selected) != len(expected_pairs):
            raise ValueError(
                "method-specific SVD ARI table must contain every method/dimension once"
            )
        if not np.isfinite(selected["ari"]).all():
            raise ValueError("method-specific SVD ARI table contains non-finite values")
        if not (
            (selected["n_cells"] == 626).all()
            and (selected["n_clusters"] == 4).all()
            and (selected["n_init"] == 200).all()
        ):
            raise ValueError("unexpected method-specific K-means parameters")
        self.metrics = selected

        expected_main_dims = selected["display_name"].map(METHOD_REPORTED_NDIMS)
        selected_flags = selected["ndim"].eq(expected_main_dims)
        if not np.array_equal(
            selected["selected_for_main"].astype(bool).to_numpy(),
            selected_flags.to_numpy(),
        ):
            raise ValueError("selected_for_main does not match the plot configuration")
        reported = selected[selected["selected_for_main"].astype(bool)]
        if set(reported["display_name"]) != set(METHOD_ORDER) or len(reported) != 8:
            raise ValueError(f"{ARI_RULE} must contain Raw plus seven methods")

        required_coordinate_columns = {
            "method",
            "cell_id",
            "cell_type",
            "UMAP1",
            "UMAP2",
            "source_embedding_dim",
            "input_components",
        }
        missing = required_coordinate_columns - set(self.coordinates.columns)
        if missing:
            raise ValueError(f"coordinate table lacks columns: {sorted(missing)}")
        coordinate_source_dims = self.coordinates["method"].map(
            METHOD_SOURCE_EMBEDDING_DIMS
        )
        coordinate_input_dims = self.coordinates["method"].map(
            METHOD_REPORTED_NDIMS
        )
        if not (
            (self.coordinates["source_embedding_dim"] == coordinate_source_dims).all()
            and (self.coordinates["input_components"] == coordinate_input_dims).all()
        ):
            raise ValueError(
                "coordinates must use each method's configured SVD embedding"
            )
        if not np.isfinite(self.coordinates[["UMAP1", "UMAP2"]]).all().all():
            raise ValueError("coordinate table contains non-finite values")

        expected_coordinate_rows = {method: 626 for method in METHOD_ORDER}
        actual_coordinate_rows = self.coordinates.groupby("method").size().to_dict()
        if actual_coordinate_rows != expected_coordinate_rows:
            raise ValueError(f"unexpected coordinate rows: {actual_coordinate_rows}")

        counts = (
            self.coordinates[self.coordinates["method"] == "Raw"]["cell_type"]
            .value_counts()
            .to_dict()
        )
        if counts != EXPECTED_COUNTS:
            raise ValueError(f"unexpected cell-type counts: {counts}")

        reference = self.coordinates[self.coordinates["method"] == "Raw"][
            ["cell_id", "cell_type"]
        ].reset_index(drop=True)
        for method in METHOD_ORDER[1:]:
            observed = self.coordinates[self.coordinates["method"] == method][
                ["cell_id", "cell_type"]
            ].reset_index(drop=True)
            if not observed.equals(reference):
                raise ValueError(f"cell order or labels differ for {method}")

    def reported_ari(self):
        reported = (
            self.metrics[self.metrics["selected_for_main"].astype(bool)]
            .set_index("display_name")
            .loc[list(METHOD_ORDER)]
        )
        return reported["ari"].to_numpy(dtype=float), None

    def sweep(self):
        indexed = self.metrics.set_index(["display_name", "ndim"])["ari"]
        dimensions = np.asarray(SWEEP_NDIMS, dtype=int)
        curves = {
            method: indexed.loc[method].loc[dimensions].to_numpy(dtype=float)
            for method in METHOD_ORDER
        }
        return curves, dimensions

    def embedding(self, method):
        subset = self.coordinates[self.coordinates["method"] == method]
        return (
            subset[["UMAP1", "UMAP2"]].to_numpy(dtype=float),
            subset["cell_type"].to_numpy(dtype=object),
        )


class TanPaperData:
    def __init__(self, results_dir):
        self.summary = pd.read_csv(results_dir / "TanData_confusion_summary.csv")
        self.confusion = pd.read_csv(results_dir / "TanData_confusion_matrices.csv")
        self._validate()

    def _validate(self):
        expected_pairs = {
            (method, segment)
            for method in METHOD_ORDER
            for segment in TAN_SEGMENT_ORDER
        }
        summary_pairs = set(zip(self.summary["method"], self.summary["segment"].astype(str)))
        if summary_pairs != expected_pairs or len(self.summary) != len(expected_pairs):
            raise ValueError("Tan summary must contain each method/segment pair exactly once")
        expected_confusion_rows = len(expected_pairs) * len(TAN_CELL_TYPES) ** 2
        if len(self.confusion) != expected_confusion_rows:
            raise ValueError("unexpected number of Tan confusion rows")
        grouped = self.confusion.groupby(["method", "segment", "true_cell_type"])[
            "fraction"
        ].sum()
        if not np.allclose(grouped.to_numpy(dtype=float), 1.0):
            raise ValueError("Tan confusion rows must sum to one")

    def ari(self, segment, method):
        row = self.summary[
            (self.summary["segment"].astype(str) == str(segment))
            & (self.summary["method"] == method)
        ]
        return float(row.iloc[0]["ARI"])

    def matrix(self, segment, method):
        subset = self.confusion[
            (self.confusion["segment"].astype(str) == str(segment))
            & (self.confusion["method"] == method)
        ]
        matrix = subset.pivot(
            index="true_cell_type",
            columns="predicted_cell_type",
            values="fraction",
        )
        return matrix.loc[list(TAN_CELL_TYPES), list(TAN_CELL_TYPES)].to_numpy(float)


def draw_ari_panel(axis, data, **style_options):
    values, errors = data.reported_ari()
    return style.plot_ari_bars(
        METHOD_ORDER,
        values,
        errors=errors,
        hero=HERO_METHOD,
        axis=axis,
        **style_options,
    )


def draw_main_ari_panel(axis, data, **style_options):
    values, errors = data.reported_ari()
    return style.plot_ari_point_ranges(
        METHOD_ORDER,
        values,
        errors=errors,
        hero=HERO_METHOD,
        display_labels=MAIN_ARI_DISPLAY_LABELS,
        axis=axis,
        **style_options,
    )


def draw_cluster_grid(
    axes,
    data,
    dot_size=4.8,
    show_axis_indicator=True,
    axis_indicator_fraction=0.13,
    axis_indicator_fontsize=6.5,
    title_size=style.FS_TITLE,
    title_pad=2.5,
    display_limits=(-0.78, 0.60),
    show_frame=False,
):
    for index, (axis, method) in enumerate(zip(axes, METHOD_ORDER)):
        embedding, cell_types = data.embedding(method)
        style.plot_cluster_scatter(
            axis,
            embedding,
            cell_types,
            method,
            dot_size=dot_size,
            alpha=0.84,
            axis_indicator=show_axis_indicator and index == 4,
            axis_indicator_fraction=axis_indicator_fraction,
            axis_indicator_fontsize=axis_indicator_fontsize,
            title_size=title_size,
            title_pad=title_pad,
            display_limits=display_limits,
            show_frame=show_frame,
        )


def add_cell_type_legend(
    container,
    show_counts=True,
    handle_order=None,
    **kwargs,
):
    handles = style.cell_type_legend_handles(EXPECTED_COUNTS, show_counts=show_counts)
    if handle_order is not None:
        handles = [handles[index] for index in handle_order]
    defaults = {
        "handles": handles,
        "frameon": False,
        "fontsize": 8,
        "ncol": 4,
        "loc": "center",
        "handletextpad": 0.4,
        "columnspacing": 1.4,
    }
    defaults.update(kwargs)
    return container.legend(**defaults)


def label_cluster_grid_axes(axes, fontsize=7.8, x_pad=4.0, y_pad=5.0):
    axes = list(axes)
    for axis in (axes[0], axes[4]):
        axis.set_ylabel("UMAP2", fontsize=fontsize, labelpad=y_pad)
    for axis in axes[4:]:
        axis.set_xlabel("UMAP1", fontsize=fontsize, labelpad=x_pad)


def draw_tan_confusion_grid(
    figure,
    grid,
    data,
    title_size=6.2,
    annotation_size=5.7,
    tick_size=5.4,
):
    axes = []
    image = None
    for row_index, segment in enumerate(TAN_SEGMENT_ORDER):
        row_axes = []
        for column_index, method in enumerate(METHOD_ORDER):
            axis = figure.add_subplot(grid[row_index, column_index])
            matrix = data.matrix(segment, method)
            image = axis.imshow(
                matrix,
                cmap=TAN_CMAP,
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="equal",
            )
            for true_index in range(2):
                for predicted_index in range(2):
                    value = matrix[true_index, predicted_index]
                    axis.text(
                        predicted_index,
                        true_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=annotation_size,
                        color="white" if value >= 0.62 else style.INK,
                    )
            axis.set_xticks((0, 1))
            axis.set_yticks((0, 1))
            if row_index == len(TAN_SEGMENT_ORDER) - 1:
                axis.set_xticklabels(
                    TAN_CELL_TYPES,
                    fontsize=tick_size,
                    ha="center",
                )
            else:
                axis.set_xticklabels(())
            if column_index == 0:
                axis.set_yticklabels(TAN_CELL_TYPES, fontsize=tick_size)
            else:
                axis.set_yticklabels(())
            axis.tick_params(axis="x", length=0, pad=3.2)
            axis.tick_params(axis="y", length=0, pad=1.2)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#8A8F95")
                spine.set_linewidth(0.45)
            if row_index == 0:
                axis.set_title(
                    TAN_METHOD_LABELS[method],
                    fontsize=title_size,
                    color=style.INK,
                    fontweight="normal",
                    pad=13,
                    linespacing=0.9,
                )
            axis.text(
                0.5,
                1.025,
                f"ARI {data.ari(segment, method):.3f}",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=5.2,
                color="#555B61",
            )
            row_axes.append(axis)
        axes.append(row_axes)

    colorbar_axis = figure.add_subplot(grid[:, len(METHOD_ORDER)])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Fraction of cells", fontsize=6.2, labelpad=3)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    colorbar.outline.set_linewidth(0.5)
    return axes, colorbar_axis


def add_tan_axis_labels(figure, axes, panel_letter=None):
    first_column = [axes[row][0].get_position() for row in range(2)]
    for row_index, position in enumerate(first_column):
        figure.text(
            position.x0 - 0.070,
            (position.y0 + position.y1) / 2,
            f"Chr1 {TAN_SEGMENT_LABELS[TAN_SEGMENT_ORDER[row_index]]} Mb",
            ha="center",
            va="center",
            fontsize=6.2,
            rotation=90,
        )
    bottom_positions = [axis.get_position() for axis in axes[-1]]
    predicted_label_offset = 12.0 / (72.0 * figure.get_figheight())
    figure.text(
        (bottom_positions[0].x0 + bottom_positions[-1].x1) / 2,
        bottom_positions[0].y0 - predicted_label_offset,
        "Predicted cell type",
        ha="center",
        va="top",
        fontsize=6.5,
    )
    if panel_letter is not None:
        figure.text(
            0.015,
            first_column[0].y1 + 0.045,
            panel_letter,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            color=style.INK,
        )


def render_ari(data, output_dir, formats, dpi):
    figure, axis = plt.subplots(figsize=(style.COL15, style.mm(76)))
    draw_main_ari_panel(axis, data, value_label_size=7.2)
    axis.tick_params(axis="x", labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    axis.xaxis.label.set_fontsize(8.5)
    axis.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
    figure.subplots_adjust(left=0.30, right=0.985, top=0.97, bottom=0.14)
    outputs = style.save_figure(
        figure, "RamaniData_ARI_bar", output_dir, formats=formats, dpi=dpi
    )
    plt.close(figure)
    return outputs


def render_sweep(data, output_dir, formats, dpi):
    curves, dimensions = data.sweep()
    figure, axis = plt.subplots(figsize=(style.COL2, style.mm(78)))
    style.plot_ari_sweep(
        axis,
        curves,
        dimensions,
        hero=HERO_METHOD,
        mark_dim=None,
        selected_dims=METHOD_REPORTED_NDIMS,
    )
    figure.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.30)
    outputs = style.save_figure(
        figure,
        "RamaniData_ARI_dimension_sweep",
        output_dir,
        formats=formats,
        dpi=dpi,
    )
    plt.close(figure)
    return outputs


def render_cluster(data, output_dir, formats, dpi):
    figure, axes = plt.subplots(2, 4, figsize=(style.COL2, style.mm(108)))
    flat_axes = axes.ravel()
    draw_cluster_grid(
        flat_axes,
        data,
        dot_size=5.0,
        show_axis_indicator=False,
        title_size=8.2,
        title_pad=3.0,
        display_limits=(-0.55, 0.55),
        show_frame=True,
    )
    label_cluster_grid_axes(flat_axes, fontsize=8.2, x_pad=4.0, y_pad=5.0)
    add_cell_type_legend(
        figure,
        show_counts=True,
        ncol=4,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.91,
        bottom=0.18,
        wspace=0.04,
        hspace=0.08,
    )
    outputs = style.save_figure(
        figure, "RamaniData_cluster_grid", output_dir, formats=formats, dpi=dpi
    )
    plt.close(figure)
    return outputs


def render_tan(data, output_dir, formats, dpi):
    figure = plt.figure(figsize=(style.mm(MAIN_WIDTH_MM), style.mm(62)))
    grid = GridSpec(
        2,
        len(METHOD_ORDER) + 1,
        figure=figure,
        width_ratios=[1] * len(METHOD_ORDER) + [0.10],
        left=0.105,
        right=0.94,
        top=0.82,
        bottom=0.20,
        wspace=0.16,
        hspace=0.34,
    )
    axes, _ = draw_tan_confusion_grid(figure, grid, data)
    add_tan_axis_labels(figure, axes)
    outputs = style.save_figure(
        figure,
        "TanData_confusion_matrix",
        output_dir,
        formats=formats,
        dpi=dpi,
        tight=False,
    )
    plt.close(figure)
    return outputs


def render_main(data, tan_data, output_dir, formats, dpi):
    figure = plt.figure(
        figsize=(style.mm(MAIN_WIDTH_MM), style.mm(MAIN_HEIGHT_MM))
    )
    outer = GridSpec(
        1,
        2,
        figure=figure,
        width_ratios=(1.15, 3.85),
        wspace=0.18,
        left=0.13,
        right=0.995,
        top=0.93,
        bottom=0.45,
    )

    left_grid = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=(0.68, 0.32),
        hspace=0.04,
    )
    ari_axis = figure.add_subplot(left_grid[0, 0])
    draw_main_ari_panel(ari_axis, data, value_label_size=6.2)
    ari_axis.tick_params(axis="x", labelsize=6.8)
    ari_axis.tick_params(axis="y", labelsize=6.8)
    ari_axis.xaxis.label.set_fontsize(7.2)
    ari_axis.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)

    legend_axis = figure.add_subplot(left_grid[1, 0])
    legend_axis.axis("off")
    add_cell_type_legend(
        legend_axis,
        show_counts=False,
        handle_order=(0, 2, 1, 3),
        ncol=2,
        fontsize=6.8,
        loc="lower left",
        bbox_to_anchor=(-0.05, 0.218, 1.10, 0.24),
        mode="expand",
        borderaxespad=0.0,
        borderpad=0.0,
        handletextpad=0.3,
        columnspacing=0.8,
        labelspacing=0.5,
    )

    cluster_grid = outer[0, 1].subgridspec(
        2,
        4,
        wspace=0.04,
        hspace=-0.08,
    )
    cluster_axes = [
        figure.add_subplot(cluster_grid[row, column])
        for row in range(2)
        for column in range(4)
    ]
    draw_cluster_grid(
        cluster_axes,
        data,
        dot_size=4.4,
        show_axis_indicator=False,
        title_size=7.8,
        title_pad=3.0,
        display_limits=(-0.55, 0.55),
        show_frame=True,
    )

    label_cluster_grid_axes(cluster_axes, fontsize=7.8, x_pad=4.0, y_pad=5.0)

    first_cluster_position = cluster_axes[0].get_position()
    ari_position = ari_axis.get_position()
    ari_axis.set_position(
        [
            ari_position.x0,
            ari_position.y0,
            ari_position.width,
            first_cluster_position.y1 - ari_position.y0,
        ]
    )
    figure.text(
        0.015,
        0.975,
        "A",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        color=style.INK,
    )
    figure.text(
        first_cluster_position.x0 - 0.028,
        0.975,
        "B",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        color=style.INK,
    )

    tan_grid = GridSpec(
        2,
        len(METHOD_ORDER) + 1,
        figure=figure,
        width_ratios=[1] * len(METHOD_ORDER) + [0.10],
        left=0.105,
        right=0.94,
        top=0.36,
        bottom=0.075,
        wspace=0.16,
        hspace=0.34,
    )
    tan_axes, _ = draw_tan_confusion_grid(figure, tan_grid, tan_data)
    add_tan_axis_labels(figure, tan_axes, panel_letter="C")
    outputs = style.save_figure(
        figure,
        "RamaniData_main_ARI_cluster",
        output_dir,
        formats=formats,
        dpi=dpi,
        tight=False,
    )
    plt.close(figure)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--tan-results-dir", type=Path, default=DEFAULT_TAN_RESULTS_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("all",) + FIGURE_CHOICES,
        default=("all",),
    )
    parser.add_argument("--formats", nargs="+", default=("svg", "pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    requested = FIGURE_CHOICES if "all" in args.figures else tuple(args.figures)
    style.set_gr_style()
    data = PaperData(args.results_dir)
    tan_data = TanPaperData(args.tan_results_dir)
    renderers = {
        "ari": render_ari,
        "cluster": render_cluster,
        "sweep": render_sweep,
        "tan": render_tan,
    }
    outputs = []
    for figure_name in requested:
        if figure_name == "main":
            outputs.extend(
                render_main(
                    data,
                    tan_data,
                    args.output_dir,
                    tuple(args.formats),
                    args.raster_dpi,
                )
            )
        elif figure_name == "tan":
            outputs.extend(
                renderers[figure_name](
                    tan_data,
                    args.output_dir,
                    tuple(args.formats),
                    args.raster_dpi,
                )
            )
        else:
            outputs.extend(
                renderers[figure_name](
                    data,
                    args.output_dir,
                    tuple(args.formats),
                    args.raster_dpi,
                )
            )
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
