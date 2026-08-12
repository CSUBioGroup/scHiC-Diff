#!/usr/bin/env python3
"""Compose the approved formal main and supplementary paper figures."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_data import load_figure_data
from figure_panels import (
    add_axes_mm,
    draw_apa_grid,
    draw_contact_grid,
    draw_repeat_counts,
    draw_supported_counts,
    draw_support_fraction,
    draw_umap_grid,
    mm,
    panel_letter,
    set_publication_style,
)
from figure_registry import (
    ALL_CELL_COUNTS,
    ALL_METHODS,
    APA_TOP_N_VALUES,
    IMPUTED_METHODS,
    MAIN_APA_METHODS,
    MAIN_CELL_COUNTS,
    MAIN_CONTACT_METHODS,
    MAIN_UMAP_METHODS,
    STAGES,
)


MAIN_FIGURE_WIDTH_MM = 174.0
# The visible A–D composition ends at ~225.15 mm; 230 mm leaves a deliberate
# 4.85 mm print-safe lower margin without the former near-19 mm blank band.
MAIN_FIGURE_HEIGHT_MM = 230.0


def _new_figure(width_mm, height_mm):
    set_publication_style()
    return plt.figure(figsize=(mm(width_mm), mm(height_mm)))


def build_main_figure(bundle):
    figure = _new_figure(MAIN_FIGURE_WIDTH_MM, MAIN_FIGURE_HEIGHT_MM)

    panel_a_bottom = draw_umap_grid(
        figure,
        bundle.umap_points,
        bundle.umap_summary,
        methods=MAIN_UMAP_METHODS,
        stages=STAGES,
        fig_w_mm=MAIN_FIGURE_WIDTH_MM,
        fig_h_mm=MAIN_FIGURE_HEIGHT_MM,
        x_mm=0.0,
        top_mm=4.0,
        width_mm=MAIN_FIGURE_WIDTH_MM,
        show_legend=False,
        show_axis_indicator=False,
    )
    lower_top = panel_a_bottom + 1.0

    panel_b_bottom = draw_contact_grid(
        figure,
        bundle.contact_matrices,
        bundle.contact_summits,
        methods=MAIN_CONTACT_METHODS,
        cell_counts=MAIN_CELL_COUNTS,
        all_matrices=bundle.contact_matrices,
        fig_w_mm=MAIN_FIGURE_WIDTH_MM,
        fig_h_mm=MAIN_FIGURE_HEIGHT_MM,
        x_mm=0.0,
        top_mm=lower_top,
        width_mm=82.0,
        show_colorbar_label=False,
        colorbar_mm=4.5,
        colorbar_offset_mm=3.0,
        colorbar_end_labels=True,
        colorbar_width_mm=32.62,
        colorbar_right_mm=74.5,
        colorbar_end_label_right_mm=81.0,
        genomic_range_label="chr1:65–67 Mb",
        genomic_range_x_mm=24.8333333333,
        genomic_range_ha="center",
    )

    panel_c_bottom = draw_apa_grid(
        figure,
        bundle.apa_matrices,
        bundle.apa_metrics,
        methods=MAIN_APA_METHODS,
        top_n_values=APA_TOP_N_VALUES,
        all_matrices=bundle.apa_matrices,
        resolution_bp=bundle.apa_resolution_bp,
        window_bins=bundle.apa_window_bins,
        fig_w_mm=MAIN_FIGURE_WIDTH_MM,
        fig_h_mm=MAIN_FIGURE_HEIGHT_MM,
        x_mm=85.0,
        top_mm=lower_top,
        width_mm=82.0,
        left_gutter_mm=10.0,
        gap_mm=1.0,
        colorbar_mm=2.0,
        show_sd=False,
        show_colorbar_label=False,
        show_axis_titles=False,
        row_labels_with_n=False,
        # Match the x-centre of D's vertical y-axis label for one aligned
        # lower-right story block.
        row_label_rotation=90,
        row_label_x_mm=87.1,
        colorbar_orientation="vertical",
        vertical_colorbar_x_mm=170.5,
        vertical_colorbar_width_mm=1.5,
        method_header_fontsize=8.0,
        vertical_colorbar_height_fraction=0.62,
        vertical_colorbar_end_labels=True,
    )

    # Align D's x-axis with the lower border of the final scHiC-Diff contact
    # matrix, rather than with B's lower colour-bar extent.
    contact_cell_mm = (82.0 - 14.0 - 1.0 - 2.0 * 1.0) / len(MAIN_CELL_COUNTS)
    panel_b_heatmap_bottom = (
        lower_top
        + 5.0
        + len(MAIN_CONTACT_METHODS) * contact_cell_mm
        + (len(MAIN_CONTACT_METHODS) - 1) * 1.0
    )
    panel_d_top = panel_c_bottom + 6.0
    # Reserve exactly the B colour-bar overhang for D's tick labels so the
    # visual bottoms still coincide while the D axis itself meets the matrix.
    panel_d_tick_margin = panel_b_bottom - panel_b_heatmap_bottom
    panel_d_height = panel_b_heatmap_bottom - panel_d_top
    if panel_d_height <= 0:
        raise ValueError("panel D cannot fit below panel C above panel B heatmaps")
    axis_d = add_axes_mm(
        figure,
        MAIN_FIGURE_WIDTH_MM,
        MAIN_FIGURE_HEIGHT_MM,
        x_mm=96.0,
        top_mm=panel_d_top,
        width_mm=76.0,
        height_mm=panel_d_height,
    )
    draw_support_fraction(
        axis_d, bundle.support_fraction, show_legend=True, compact=True
    )

    panel_letter(figure, "A", 1.0, 1.5, MAIN_FIGURE_WIDTH_MM, MAIN_FIGURE_HEIGHT_MM)
    panel_letter(
        figure,
        "B",
        1.0,
        lower_top - 2.0,
        MAIN_FIGURE_WIDTH_MM,
        MAIN_FIGURE_HEIGHT_MM,
    )
    panel_letter(
        figure,
        "C",
        86.0,
        lower_top - 2.0,
        MAIN_FIGURE_WIDTH_MM,
        MAIN_FIGURE_HEIGHT_MM,
    )
    panel_letter(
        figure,
        "D",
        86.0,
        panel_d_top - 6.0,
        MAIN_FIGURE_WIDTH_MM,
        MAIN_FIGURE_HEIGHT_MM,
    )

    metadata = {
        "figure_width_mm": MAIN_FIGURE_WIDTH_MM,
        "figure_height_mm": MAIN_FIGURE_HEIGHT_MM,
        "panel_a_methods": list(MAIN_UMAP_METHODS),
        "panel_a_stages": list(STAGES),
        "panel_b_methods": list(MAIN_CONTACT_METHODS),
        "panel_b_cell_counts": list(MAIN_CELL_COUNTS),
        "panel_c_methods": list(MAIN_APA_METHODS),
        "panel_c_top_n": list(APA_TOP_N_VALUES),
        "panel_c_heatmaps": len(MAIN_APA_METHODS) * len(APA_TOP_N_VALUES),
        "panel_c_bottom_mm": panel_c_bottom,
        "panel_d_top_mm": panel_d_top,
        "panel_b_bottom_mm": panel_b_bottom,
        "panel_b_heatmap_bottom_mm": panel_b_heatmap_bottom,
        "panel_d_axis_bottom_mm": panel_d_top + panel_d_height,
        "panel_d_visual_bottom_mm": (
            panel_d_top + panel_d_height + panel_d_tick_margin
        ),
        "panel_d_methods": list(IMPUTED_METHODS),
        "metrics_recomputed": False,
    }
    return figure, metadata


def build_panel_a(bundle):
    width, height = 174.0, 106.0
    figure = _new_figure(width, height)
    draw_umap_grid(
        figure,
        bundle.umap_points,
        bundle.umap_summary,
        MAIN_UMAP_METHODS,
        STAGES,
        width,
        height,
        0.0,
        4.0,
        width,
        show_legend=False,
        show_axis_indicator=False,
    )
    panel_letter(figure, "A", 1.0, 1.5, width, height)
    return figure


def build_panel_b(bundle):
    width, height = 114.0, 180.0
    figure = _new_figure(width, height)
    draw_contact_grid(
        figure,
        bundle.contact_matrices,
        bundle.contact_summits,
        MAIN_CONTACT_METHODS,
        MAIN_CELL_COUNTS,
        bundle.contact_matrices,
        width,
        height,
        0.0,
        4.0,
        112.0,
        show_colorbar_label=False,
        colorbar_mm=4.5,
        colorbar_offset_mm=3.0,
        colorbar_end_labels=True,
        colorbar_width_mm=57.62,
        colorbar_right_mm=104.5,
        colorbar_end_label_right_mm=111.0,
        genomic_range_label="chr1:65–67 Mb",
        genomic_range_x_mm=29.8333333333,
        genomic_range_ha="center",
    )
    panel_letter(figure, "B", 1.0, 1.5, width, height)
    return figure


def build_panel_c(bundle):
    width, height = 150.0, 112.0
    figure = _new_figure(width, height)
    draw_apa_grid(
        figure,
        bundle.apa_matrices,
        bundle.apa_metrics,
        methods=MAIN_APA_METHODS,
        top_n_values=APA_TOP_N_VALUES,
        all_matrices=bundle.apa_matrices,
        resolution_bp=bundle.apa_resolution_bp,
        window_bins=bundle.apa_window_bins,
        fig_w_mm=width,
        fig_h_mm=height,
        x_mm=0.0,
        top_mm=4.0,
        width_mm=139.0,
        left_gutter_mm=20.0,
        show_sd=False,
        show_colorbar_label=False,
        colorbar_mm=6.0,
        colorbar_orientation="vertical",
        vertical_colorbar_x_mm=145.5,
        vertical_colorbar_width_mm=1.5,
        vertical_colorbar_height_fraction=0.62,
        vertical_colorbar_end_labels=True,
    )
    panel_letter(figure, "C", 1.0, 1.5, width, height)
    return figure


def build_panel_d(bundle):
    width, height = 114.0, 70.0
    figure = _new_figure(width, height)
    axis = add_axes_mm(figure, width, height, 16.0, 7.0, 95.0, 52.0)
    draw_support_fraction(axis, bundle.support_fraction, show_legend=True)
    panel_letter(figure, "D", 1.0, 1.5, width, height)
    return figure


def build_supplementary_1(bundle):
    width, height = 174.0, 180.0
    figure = _new_figure(width, height)
    draw_umap_grid(
        figure,
        bundle.umap_points,
        bundle.umap_summary,
        ALL_METHODS,
        STAGES,
        width,
        height,
        0.0,
        4.0,
        width,
        show_axis_indicator=False,
    )
    return figure


def build_supplementary_2(bundle):
    width, height = 174.0, 248.0
    figure = _new_figure(width, height)
    draw_contact_grid(
        figure,
        bundle.contact_matrices,
        bundle.contact_summits,
        ALL_METHODS,
        ALL_CELL_COUNTS,
        bundle.contact_matrices,
        width,
        height,
        14.0,
        4.0,
        145.0,
        show_colorbar_label=False,
        colorbar_end_labels=True,
        colorbar_width_mm=90.58,
        colorbar_right_mm=151.5,
        colorbar_end_label_right_mm=158.0,
        genomic_range_label="chr1:65–67 Mb",
        genomic_range_x_mm=43.875,
        genomic_range_ha="center",
    )
    return figure


def build_supplementary_3(bundle):
    width, height = 174.0, 120.0
    figure = _new_figure(width, height)
    draw_apa_grid(
        figure,
        bundle.apa_matrices,
        bundle.apa_metrics,
        IMPUTED_METHODS,
        APA_TOP_N_VALUES,
        bundle.apa_matrices,
        bundle.apa_resolution_bp,
        bundle.apa_window_bins,
        width,
        height,
        0.0,
        4.0,
        width,
        show_sd=False,
        show_colorbar_label=False,
    )
    return figure


def build_supplementary_4(bundle):
    width, height = 174.0, 150.0
    figure = _new_figure(width, height)
    axis_a = add_axes_mm(figure, width, height, 18.0, 8.0, 151.0, 52.0)
    axis_b = add_axes_mm(figure, width, height, 18.0, 78.0, 151.0, 52.0)
    draw_repeat_counts(axis_a, bundle.loop_counts, "summit_count", show_legend=True)
    draw_repeat_counts(axis_b, bundle.loop_counts, "loop_count", show_legend=False)
    panel_letter(figure, "A", 2.0, 2.0, width, height)
    panel_letter(figure, "B", 2.0, 72.0, width, height)
    figure.text(
        18.0 / width,
        1.0 - 146.0 / height,
        "Mean ± sample SD for seeds 42/43/44 at 10–200 cells; 476 cells: n=1.",
        fontsize=8,
        color="#5F5E5A",
        ha="left",
        va="bottom",
    )
    return figure


def build_supplementary_5(bundle):
    width, height = 174.0, 88.0
    figure = _new_figure(width, height)
    axis = add_axes_mm(figure, width, height, 18.0, 8.0, 151.0, 70.0)
    draw_supported_counts(axis, bundle.support_counts)
    return figure


MAIN_PANEL_BUILDERS = {
    "panel_A": build_panel_a,
    "panel_B": build_panel_b,
    "panel_C": build_panel_c,
    "panel_D": build_panel_d,
}

SUPPLEMENTARY_BUILDERS = {
    "Supplementary_Figure_1": build_supplementary_1,
    "Supplementary_Figure_2": build_supplementary_2,
    "Supplementary_Figure_3": build_supplementary_3,
    "Supplementary_Figure_4": build_supplementary_4,
    "Supplementary_Figure_5": build_supplementary_5,
}


def parse_args(argv=None):
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--out-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--dpi", type=int, default=600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--main-only", action="store_true")
    mode.add_argument("--supplementary-only", action="store_true")
    return parser.parse_args(argv)


def validate_dpi(dpi):
    if int(dpi) < 300:
        raise ValueError("--dpi must be at least 300")
    return int(dpi)


def selected_groups(args):
    if args.main_only:
        return True, False
    if args.supplementary_only:
        return False, True
    return True, True


def save_all_formats(figure, output_dir, stem, dpi=600):
    dpi = validate_dpi(dpi)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        extension: output_dir / "{}.{}".format(stem, extension)
        for extension in ("pdf", "png", "svg")
    }
    figure.savefig(paths["pdf"], facecolor="white")
    figure.savefig(paths["svg"], facecolor="white")
    figure.savefig(paths["png"], dpi=dpi, facecolor="white")
    return paths


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def export_main_bundle(bundle, data_dir, output_dir, dpi):
    output_dir = Path(output_dir).resolve()
    written = []
    figure, metadata = build_main_figure(bundle)
    try:
        paths = save_all_formats(figure, output_dir / "main", "main_figure", dpi)
        written.extend(paths.values())
    finally:
        plt.close(figure)

    for stem, builder in MAIN_PANEL_BUILDERS.items():
        figure = builder(bundle)
        try:
            paths = save_all_formats(figure, output_dir / "panels", stem, dpi)
            written.extend(paths.values())
        finally:
            plt.close(figure)

    metadata.update(
        {
            "created_with_python": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
            "source_manifest": str((Path(data_dir) / "copied_data_manifest.csv").resolve()),
            "source_manifest_sha256": _sha256_file(
                Path(data_dir) / "copied_data_manifest.csv"
            ),
            "output_dpi": int(dpi),
        }
    )
    metadata_path = output_dir / "main/main_figure_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    written.append(metadata_path)
    return written


def export_supplementary_bundle(bundle, output_dir, dpi):
    output_dir = Path(output_dir).resolve()
    written = []
    for stem, builder in SUPPLEMENTARY_BUILDERS.items():
        figure = builder(bundle)
        try:
            paths = save_all_formats(
                figure, output_dir / "supplementary", stem, dpi
            )
            written.extend(paths.values())
        finally:
            plt.close(figure)
    return written


def main(argv=None):
    args = parse_args(argv)
    validate_dpi(args.dpi)
    make_main, make_supplementary = selected_groups(args)
    bundle = load_figure_data(args.data_dir)
    written = []
    if make_main:
        written.extend(export_main_bundle(bundle, args.data_dir, args.out_dir, args.dpi))
    if make_supplementary:
        written.extend(export_supplementary_bundle(bundle, args.out_dir, args.dpi))
    for path in written:
        print(path)
    print("{} files written".format(len(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
