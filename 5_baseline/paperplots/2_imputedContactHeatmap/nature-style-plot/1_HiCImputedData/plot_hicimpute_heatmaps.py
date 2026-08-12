#!/usr/bin/env python3
"""Render HiCImputeData comparison grids and legacy-style single heatmaps."""

from __future__ import annotations

import argparse
import csv
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.sparse import load_npz


SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from gr_style import (  # noqa: E402
    C_GRID,
    C_HERO,
    C_INK,
    FS_ANNOT,
    FS_LABEL,
    FS_PANEL,
    FS_TICK,
    GR_DOUBLE_COL_IN,
    LW_SPINE,
    apply_gr_style,
    get_cmap,
    robust_vmax,
)


DEFAULT_MATRIX_PATHS = SCRIPT_DIR / "HiCImputeData_heatmap_matrix_paths.tsv"
DEFAULT_METRICS = SCRIPT_DIR / "HiCImputeData_PCC_MAE_metrics.tsv"
FIGURES_DIR = SCRIPT_DIR / "figures"

NBIN = 61
CELL_TYPES = ("T1", "T2", "T3")
DEPTHS = ("1k", "2k", "4k", "7k")
MAIN_DEPTHS = ("1k", "2k", "4k")
SUPPLEMENT_DEPTHS = ("7k",)
HERO = "scHiC-Diff"

# Canonical names match the matrix-path and metric tables.
METHODS = (
    "Raw",
    "True",
    "scHiCluster",
    "HiCImpute",
    "Higashi_nbr0",
    "Higashi_nbr5",
    "scVI-3D",
    "Tensor-FLAMINGO",
    HERO,
)

DISPLAY_NAME = {
    "Raw": "Input",
    "True": "GT",
    "scHiCluster": "scHiCluster",
    "HiCImpute": "HiCImpute",
    "Higashi_nbr0": "Higashi-0",
    "Higashi_nbr5": "Higashi-5",
    "scVI-3D": "scVI-3D",
    "Tensor-FLAMINGO": "T-FLAMINGO",
    HERO: HERO,
}

METHOD_ALIASES = {
    "raw": "Raw",
    "input": "Raw",
    "true": "True",
    "gt": "True",
    "schicluster": "scHiCluster",
    "hicimpute": "HiCImpute",
    "higashi_nbr0": "Higashi_nbr0",
    "higashi-nbr0": "Higashi_nbr0",
    "higashi-0": "Higashi_nbr0",
    "higashi_nbr5": "Higashi_nbr5",
    "higashi-nbr5": "Higashi_nbr5",
    "higashi-5": "Higashi_nbr5",
    "scvi-3d": "scVI-3D",
    "scvi3d": "scVI-3D",
    "tensor-flamingo": "Tensor-FLAMINGO",
    "flamingo": "Tensor-FLAMINGO",
    "schic-diff": HERO,
    "schicdiff": HERO,
}

SINGLE_OUTPUT_STEM = {
    "Raw": "Raw",
    "True": "True",
    "scVI-3D": "scVI3D",
    "HiCImpute": "hicimpute",
    "scHiCluster": "scHiCluster",
    "Higashi_nbr0": "higashi_nbr0",
    "Higashi_nbr5": "higashi_nbr5",
    "Tensor-FLAMINGO": "flamingo",
    HERO: "scHiCDiff",
}


def canonical_method(method: str) -> str:
    if method in METHODS:
        return method
    key = method.strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(
            f"Unsupported method {method!r}; choose from {', '.join(METHODS)}"
        )
    return METHOD_ALIASES[key]


@lru_cache(maxsize=None)
def load_matrix_path_index(path: str) -> dict[str, dict[str, str]]:
    matrix_paths_path = Path(path)
    with matrix_paths_path.open("r", encoding="utf-8", newline="") as handle:
        rows = {
            row["method"]: {
                "path_template": row["path_template"],
                "feature_order": row["feature_order"].strip().lower(),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        }
    missing = set(METHODS) - set(rows)
    if missing:
        raise ValueError(f"Matrix-path table is missing methods: {sorted(missing)}")
    return rows


@lru_cache(maxsize=None)
def load_pcc_index(path: str) -> dict[tuple[str, str], float]:
    metrics_path = Path(path)
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"method", "data_name", "pcc_mean"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(
            f"PCC/MAE metrics table is missing columns {sorted(required)}: "
            f"{metrics_path}"
        )
    return {
        (row["method"], row["data_name"]): float(row["pcc_mean"])
        for row in rows
        if row.get("method") and row.get("data_name") and row.get("pcc_mean")
    }


def lookup_pcc(method: str, data_name: str, metrics_path: Path) -> float | None:
    if method in {"Raw", "True"}:
        return None
    key = (method, data_name)
    index = load_pcc_index(str(metrics_path.resolve()))
    if key not in index:
        raise KeyError(f"Missing PCC for method={method}, data_name={data_name}")
    return index[key]


def resolve_input(
    method: str,
    data_name: str,
    matrix_paths_path: Path,
) -> tuple[Path, str]:
    matrix_paths = load_matrix_path_index(str(matrix_paths_path.resolve()))
    config = matrix_paths[method]
    path = Path(config["path_template"].format(data_name=data_name))
    if not path.is_absolute():
        path = matrix_paths_path.resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    feature_order = config["feature_order"]
    if feature_order not in {"tril", "triu"}:
        raise ValueError(
            f"Unsupported feature_order={feature_order!r} for method={method}"
        )
    return path, feature_order


def _read_npz(path: Path) -> np.ndarray:
    try:
        array = load_npz(path).toarray()
    except Exception as sparse_error:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "arr_0" in archive.files:
                    array = archive["arr_0"]
                elif len(archive.files) == 1:
                    array = archive[archive.files[0]]
                else:
                    raise ValueError(
                        f"Dense NPZ has multiple arrays and no arr_0: {path}"
                    )
        except Exception as dense_error:
            raise ValueError(f"Cannot read NPZ matrix: {path}") from dense_error
        if array is None:
            raise ValueError(f"Cannot read sparse NPZ matrix: {path}") from sparse_error

    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got {array.shape}: {path}")

    n_features = NBIN * (NBIN - 1) // 2
    if array.shape[1] != n_features:
        if array.shape[0] == n_features:
            array = array.T
        else:
            raise ValueError(
                f"Expected {n_features} features, got shape={array.shape}: {path}"
            )
    return array


def triu_to_tril(array: np.ndarray, nbin: int = NBIN) -> np.ndarray:
    upper = np.triu_indices(nbin, k=1)
    lower = np.tril_indices(nbin, k=-1)
    full = np.zeros((array.shape[0], nbin, nbin), dtype=np.float64)
    full[:, upper[0], upper[1]] = array
    full[:, upper[1], upper[0]] = array
    return full[:, lower[0], lower[1]]


@lru_cache(maxsize=32)
def load_feature_matrix(
    method: str,
    data_name: str,
    matrix_paths_path: str,
) -> np.ndarray:
    path, feature_order = resolve_input(method, data_name, Path(matrix_paths_path))
    array = _read_npz(path)
    if feature_order == "triu":
        array = triu_to_tril(array)
    return array


def contact_matrix(
    method: str,
    data_name: str,
    matrix_paths_path: Path,
    cell_index: int,
    *,
    symmetric: bool,
) -> np.ndarray:
    features = load_feature_matrix(
        method,
        data_name,
        str(matrix_paths_path.resolve()),
    )
    if not 0 <= cell_index < features.shape[0]:
        raise IndexError(
            f"cell_index={cell_index} outside [0, {features.shape[0]}) for "
            f"method={method}, data_name={data_name}"
        )

    lower = np.tril_indices(NBIN, k=-1)
    matrix = np.zeros((NBIN, NBIN), dtype=np.float64)
    matrix[lower] = features[cell_index]
    matrix += matrix.T
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, None)
    np.fill_diagonal(matrix, 0.0)
    if not symmetric:
        matrix = np.triu(matrix, k=1)
    return matrix


def normalize_per_10k(matrix: np.ndarray) -> np.ndarray:
    total = float(matrix.sum())
    if total <= 0:
        return matrix.copy()
    return matrix / total * 10_000.0


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    *,
    dpi: int,
    tight: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        kwargs = {"dpi": dpi}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": 0.01})
        if output_format in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def relative_output_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(STYLE_DIR.resolve())
    except ValueError:
        return path


def _grid_block_geometry(depths: tuple[str, ...], show_titles: bool) -> float:
    title_height = 0.30 if show_titles else 0.02
    panel_width = (
        GR_DOUBLE_COL_IN - 0.06 - 0.06 - 0.40 - 0.60
    ) / (len(METHODS) + (len(METHODS) - 1) * 0.055)
    return title_height + panel_width * (
        len(depths) + (len(depths) - 1) * 0.055
    )


def draw_grid_block(
    fig,
    rect: tuple[float, float, float, float],
    figure_width: float,
    figure_height: float,
    *,
    cell_type: str,
    depths: tuple[str, ...],
    cell_index: int,
    matrix_paths_path: Path,
    metrics_path: Path,
    show_titles: bool,
    show_cell_type_label: bool,
    override_ylabel: str | None = None,
) -> None:
    left, top, width, _height = rect
    title_height = 0.30 if show_titles else 0.02
    colorbar_width = 0.60
    row_label_width = 0.40
    horizontal_gap = vertical_gap = 0.055

    axes_left = left + row_label_width
    axes_width = width - row_label_width - colorbar_width
    panel = axes_width / (
        len(METHODS) + (len(METHODS) - 1) * horizontal_gap
    )
    axes_top = top + title_height
    axes_height = panel * (len(depths) + (len(depths) - 1) * vertical_gap)
    cmap = get_cmap("hic_fall")
    last_image = None

    for row_index, depth in enumerate(depths):
        data_name = f"K562_{cell_type}_{depth}"
        matrices = {
            method: normalize_per_10k(
                contact_matrix(
                    method,
                    data_name,
                    matrix_paths_path,
                    cell_index,
                    symmetric=True,
                )
            )
            for method in METHODS
        }
        norm = Normalize(
            vmin=0.0,
            vmax=robust_vmax(matrices["True"], pct=99.0),
        )

        for column_index, method in enumerate(METHODS):
            x = axes_left + column_index * panel * (1 + horizontal_gap)
            y = axes_top + row_index * panel * (1 + vertical_gap)
            ax = fig.add_axes(
                [
                    x / figure_width,
                    1 - (y + panel) / figure_height,
                    panel / figure_width,
                    panel / figure_height,
                ]
            )
            last_image = ax.imshow(
                matrices[method],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                origin="upper",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            is_hero = method == HERO
            for spine in ax.spines.values():
                spine.set_edgecolor(C_HERO if is_hero else C_GRID)
                spine.set_linewidth(LW_SPINE)
                spine.set_zorder(10)

            if show_titles and row_index == 0:
                ax.set_title(
                    DISPLAY_NAME[method],
                    fontsize=8,
                    pad=3,
                    color=C_HERO if is_hero else C_INK,
                    fontweight="bold" if is_hero else "normal",
                )
            if column_index == 0:
                ylabel = override_ylabel if override_ylabel is not None else depth.upper()
                ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=2)

            pcc = lookup_pcc(method, data_name, metrics_path)
            if pcc is not None:
                ax.text(
                    0.02,
                    0.08,
                    f"r = {pcc:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=FS_ANNOT,
                    color=C_HERO if is_hero else C_INK,
                    fontweight="bold" if is_hero else "normal",
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                    },
                    zorder=5,
                )

    if show_cell_type_label:
        fig.text(
            (left + 0.10) / figure_width,
            1 - (axes_top + axes_height / 2) / figure_height,
            cell_type,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FS_LABEL,
            color=C_INK,
        )

    if last_image is None:
        raise RuntimeError("No heatmap was rendered")
    colorbar_x = axes_left + axes_width + 0.10
    colorbar_ax = fig.add_axes(
        [
            colorbar_x / figure_width,
            1 - (axes_top + axes_height) / figure_height,
            0.085 / figure_width,
            axes_height / figure_height,
        ]
    )
    colorbar = fig.colorbar(last_image, cax=colorbar_ax)
    colorbar.ax.tick_params(labelsize=FS_TICK, width=LW_SPINE, length=2)
    colorbar.outline.set_linewidth(LW_SPINE)
    colorbar.outline.set_edgecolor(C_GRID)


def _panel_letter(fig, y_from_top: float, letter: str, height: float) -> None:
    fig.text(
        0.01 / GR_DOUBLE_COL_IN,
        1 - y_from_top / height,
        letter,
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
        ha="left",
        va="top",
    )


def plot_grid(
    mode: str,
    *,
    cell_index: int,
    matrix_paths_path: Path,
    metrics_path: Path,
    output_dir: Path | None,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    apply_gr_style()
    figure_width = GR_DOUBLE_COL_IN
    left = right = 0.06

    if mode == "main":
        depths = MAIN_DEPTHS
        block_height = _grid_block_geometry(depths, show_titles=True)
        gap = 0.20
        figure_height = block_height * 3 + gap * 2 + 0.20
        fig = plt.figure(figsize=(figure_width, figure_height))
        for index, cell_type in enumerate(CELL_TYPES):
            top = 0.02 + index * (block_height + gap)
            draw_grid_block(
                fig,
                (left, top, figure_width - left - right, block_height),
                figure_width,
                figure_height,
                cell_type=cell_type,
                depths=depths,
                cell_index=cell_index,
                matrix_paths_path=matrix_paths_path,
                metrics_path=metrics_path,
                show_titles=True,
                show_cell_type_label=True,
            )
            _panel_letter(fig, top, chr(ord("A") + index), figure_height)
        stem = "HiCImputeData_heatmap_grid_main_1k_2k_4k"
        destination = output_dir or FIGURES_DIR / "main_fig"
    elif mode == "supplement":
        depths = SUPPLEMENT_DEPTHS
        first_height = _grid_block_geometry(depths, show_titles=True)
        other_height = _grid_block_geometry(depths, show_titles=False)
        gap = 0.08
        figure_height = first_height + other_height * 2 + gap * 2 + 0.20
        fig = plt.figure(figsize=(figure_width, figure_height))
        top = 0.02
        for index, cell_type in enumerate(CELL_TYPES):
            show_titles = index == 0
            height = first_height if show_titles else other_height
            draw_grid_block(
                fig,
                (left, top, figure_width - left - right, height),
                figure_width,
                figure_height,
                cell_type=cell_type,
                depths=depths,
                cell_index=cell_index,
                matrix_paths_path=matrix_paths_path,
                metrics_path=metrics_path,
                show_titles=show_titles,
                show_cell_type_label=False,
                override_ylabel=cell_type,
            )
            top += height + gap
        stem = "HiCImputeData_heatmap_grid_supplement_7k"
        destination = output_dir or FIGURES_DIR / "supplement_fig"
    else:
        raise ValueError(f"Unsupported grid mode: {mode}")

    paths = save_figure(
        fig,
        destination,
        stem,
        formats,
        dpi=dpi,
        tight=False,
    )
    plt.close(fig)
    return paths


def plot_single(
    method: str,
    cell_type: str,
    depth: str,
    *,
    cell_index: int,
    matrix_paths_path: Path,
    metrics_path: Path,
    output_dir: Path | None,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    method = canonical_method(method)
    data_name = f"K562_{cell_type}_{depth}"

    # Preserve the legacy single-panel behavior: upper triangle, per-panel
    # linear min/max normalization, and the seaborn icefire palette.
    matrix = contact_matrix(
        method,
        data_name,
        matrix_paths_path,
        cell_index,
        symmetric=False,
    )
    normalized = colors.Normalize()(matrix)

    fig, ax = plt.subplots(figsize=(4, 4))
    show_colorbar = method == HERO
    sns.heatmap(
        normalized,
        cmap="icefire",
        square=True,
        cbar=show_colorbar,
        cbar_kws={"shrink": 0.8, "orientation": "vertical"},
        ax=ax,
    )

    pcc = lookup_pcc(method, data_name, metrics_path)
    if pcc is not None:
        ax.text(
            0.02,
            0.02,
            f"PCC={pcc:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            color="black",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
                "pad": 2,
            },
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_colorbar:
        colorbar_ax = fig.axes[-1]
        colorbar_ax.tick_params(labelsize=12)
        for spine in colorbar_ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    destination = output_dir or FIGURES_DIR / "single_fig" / "color=icefire"
    stem = f"{data_name}_{SINGLE_OUTPUT_STEM[method]}"
    paths = save_figure(
        fig,
        destination,
        stem,
        formats,
        dpi=dpi,
        tight=True,
    )
    plt.close(fig)
    return paths


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument(
        "--matrix-paths",
        type=Path,
        default=DEFAULT_MATRIX_PATHS,
        help="TSV table containing method matrix path templates and feature order.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="TSV table containing PCC and MAE evaluation metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render HiCImputeData main, supplementary, or single heatmaps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid = subparsers.add_parser("grid", help="Render a comparison grid.")
    grid.add_argument("--mode", choices=("main", "supplement"), required=True)
    grid.add_argument("--formats", nargs="+", choices=("pdf", "png", "tiff"), default=("pdf", "png"))
    grid.add_argument("--dpi", type=int, default=600)
    _add_common_arguments(grid)

    single = subparsers.add_parser("single", help="Render one legacy-style heatmap.")
    single.add_argument(
        "--method",
        "--methods",
        dest="methods",
        nargs="+",
        default=None,
        help="Methods to render; omit to render every matrix-path-table method.",
    )
    single.add_argument(
        "--ctype",
        "--ctypes",
        dest="ctypes",
        nargs="+",
        choices=CELL_TYPES,
        default=None,
        help="Cell types to render; omit to render T1, T2, and T3.",
    )
    single.add_argument(
        "--depth",
        "--depths",
        dest="depths",
        nargs="+",
        choices=DEPTHS,
        default=None,
        help="Depths to render; omit to render 1k, 2k, 4k, and 7k.",
    )
    single.add_argument("--formats", nargs="+", choices=("pdf", "png", "tiff"), default=("png",))
    single.add_argument("--dpi", type=int, default=300)
    _add_common_arguments(single)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "grid":
        paths = plot_grid(
            args.mode,
            cell_index=args.cell_index,
            matrix_paths_path=args.matrix_paths,
            metrics_path=args.metrics,
            output_dir=args.output_dir,
            formats=tuple(args.formats),
            dpi=args.dpi,
        )
    else:
        methods = tuple(canonical_method(method) for method in (args.methods or METHODS))
        cell_types = tuple(args.ctypes or CELL_TYPES)
        depths = tuple(args.depths or DEPTHS)
        paths = []
        for method in methods:
            for cell_type in cell_types:
                for depth in depths:
                    paths.extend(
                        plot_single(
                            method,
                            cell_type,
                            depth,
                            cell_index=args.cell_index,
                            matrix_paths_path=args.matrix_paths,
                            metrics_path=args.metrics,
                            output_dir=args.output_dir,
                            formats=tuple(args.formats),
                            dpi=args.dpi,
                        )
                    )
    for path in paths:
        print(relative_output_path(path))


if __name__ == "__main__":
    main()
