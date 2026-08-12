#!/usr/bin/env python3
"""Generate all imputation-runtime figures and summary tables."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

import imputation_runtime_style as style


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"

REQUIRED_COLUMNS = {
    "method",
    "dataset",
    "impute_time_seconds",
    "gpu",
    "time_source",
}
METHOD_HARDWARE_OVERRIDES = {
    "Tensor-FLAMINGO": "CPU x20",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    slug: str
    input_path: Path


@dataclass(frozen=True)
class RenderConfig:
    formats: tuple[str, ...]
    dpi: int


DATASETS = (
    DatasetSpec(
        key="tensorflamingo_500x500",
        label="Tensor-FLAMINGO simulations 500x500",
        slug="tensorflamingo_simulations_500x500",
        input_path=DATA_DIR / "TensorFLAMINGO_simulations_500x500_runtime.csv",
    ),
    DatasetSpec(
        key="hicimpute_61x61",
        label="HiCImpute simulations 61x61",
        slug="hicimpute_simulations_61x61",
        input_path=DATA_DIR / "HiCImpute_simulations_61x61_runtime.csv",
    ),
)


def hardware_tag(value: str) -> str:
    text = str(value)
    upper = text.upper()
    if "V100" in upper or "A100" in upper or "GPU" in upper:
        return "V100"
    if "20 CORES" in upper or "20 CPU" in upper:
        return "CPU x20"
    return "CPU"


def load_runtime_data(spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.input_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.input_path}: missing columns {sorted(missing)}")

    runtime = pd.to_numeric(frame["impute_time_seconds"], errors="coerce")
    invalid = runtime.isna() | (runtime <= 0)
    if invalid.any():
        rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
        raise ValueError(f"{spec.input_path}: invalid runtime values at rows {rows}")

    frame = frame.copy()
    frame["impute_time_seconds"] = runtime.astype(float)
    frame["hardware"] = frame["gpu"].map(hardware_tag)
    for method, hardware in METHOD_HARDWARE_OVERRIDES.items():
        frame.loc[frame["method"] == method, "hardware"] = hardware
    return frame


def aggregate_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        runs = group["impute_time_seconds"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "n": len(runs),
                "median": float(np.median(runs)),
                "vmin": float(runs.min()),
                "vmax": float(runs.max()),
                "runs": runs,
                "hardware": group["hardware"].mode().iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("median").reset_index(drop=True)


def batch_flags(frame: pd.DataFrame) -> dict[str, tuple[bool, int]]:
    flags = {}
    for method, group in frame.groupby("method", sort=False):
        source = " ".join(group["time_source"].astype(str)).lower()
        is_batch = any(
            marker in source
            for marker in ("parallel", "per-dataset=total", "per-dataset = total")
        )
        flags[method] = (is_batch, len(group))
    return flags


def average_per_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    flags = batch_flags(frame)
    rows = []
    for method, group in frame.groupby("method", sort=False):
        values = group["impute_time_seconds"].to_numpy(float)
        is_batch, count = flags[method]
        average = float(np.median(values)) / count if is_batch else float(values.mean())
        rows.append(
            {
                "method": method,
                "average": average,
                "amortized": is_batch,
                "hardware": group["hardware"].mode().iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("average").reset_index(drop=True)


def runtime_axes(figure, subplot_spec):
    inner = subplot_spec.subgridspec(
        1,
        2,
        width_ratios=(0.76, 0.24),
        wspace=0.035,
    )
    runtime_ax = figure.add_subplot(inner[0, 0])
    readout_ax = figure.add_subplot(inner[0, 1], sharey=runtime_ax)
    return runtime_ax, readout_ax


def panel_header(axis, letter: str | None, label: str) -> None:
    if letter:
        style.add_panel_label(axis, letter, dx=-0.20, dy=1.13)
    axis.text(
        0.0,
        1.145,
        label,
        transform=axis.transAxes,
        fontsize=style.FS_AXIS,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_runtime_panel(
    runtime_ax,
    readout_ax,
    aggregate: pd.DataFrame,
    frame: pd.DataFrame,
    plot_type: str,
) -> bool:
    names = aggregate["method"].tolist()
    medians = aggregate["median"].tolist()
    runs = aggregate["runs"].tolist()
    hardware = aggregate["hardware"].tolist()
    durations = [style.fmt_duration(value) for value in medians]

    has_batch_caveat = False
    flags = batch_flags(frame)
    for method in names:
        is_batch, _ = flags.get(method, (False, 1))
        method_row = aggregate.loc[aggregate["method"] == method].iloc[0]
        constant_runtime = np.isclose(method_row["vmin"], method_row["vmax"])
        if is_batch and constant_runtime:
            has_batch_caveat = True
            break

    plotter = (
        style.log_runtime_lollipop
        if plot_type == "lollipop"
        else style.log_runtime_bars
    )
    plotter(
        runtime_ax,
        names,
        medians,
        runs,
        hardware,
        durations,
        readout_ax=readout_ax,
    )
    return has_batch_caveat


def save_and_close(figure, stem: str, config: RenderConfig) -> list[Path]:
    outputs = style.save_figure(
        figure,
        stem,
        FIGURE_DIR,
        formats=config.formats,
        dpi=config.dpi,
    )
    plt.close(figure)
    return outputs


def render_two_panel(
    specs,
    frames,
    aggregates,
    plot_type: str,
    stem: str,
    config: RenderConfig,
) -> tuple[list[Path], bool]:
    style.apply_gr_style()
    figure = plt.figure(figsize=(style.GR_WIDTH_2COL, 132 * style.MM))
    outer = GridSpec(2, 1, figure=figure, hspace=0.68)
    caveat = False
    for index, spec in enumerate(specs):
        runtime_ax, readout_ax = runtime_axes(figure, outer[index])
        caveat = (
            draw_runtime_panel(
                runtime_ax,
                readout_ax,
                aggregates[spec.key],
                frames[spec.key],
                plot_type,
            )
            or caveat
        )
        panel_header(runtime_ax, "A" if index == 0 else "B", spec.label)
    figure.subplots_adjust(left=0.205, right=0.985, top=0.91, bottom=0.08)
    return save_and_close(figure, stem, config), caveat


def render_single_panel(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    config: RenderConfig,
) -> list[Path]:
    style.apply_gr_style()
    figure = plt.figure(figsize=(style.GR_WIDTH_2COL, 91 * style.MM))
    outer = GridSpec(1, 1, figure=figure)
    runtime_ax, readout_ax = runtime_axes(figure, outer[0])
    draw_runtime_panel(runtime_ax, readout_ax, aggregate, frame, "bars")
    panel_header(runtime_ax, None, spec.label)
    figure.subplots_adjust(left=0.205, right=0.985, top=0.82, bottom=0.15)
    return save_and_close(
        figure,
        f"imputation_runtime_{spec.slug}",
        config,
    )


def render_average_per_dataset(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    config: RenderConfig,
) -> list[Path]:
    average = average_per_dataset(frame)
    names = average["method"].tolist()
    values = average["average"].tolist()
    hardware = average["hardware"].tolist()
    durations = [style.fmt_duration(value) for value in values]

    style.apply_gr_style()
    figure = plt.figure(figsize=(style.GR_WIDTH_2COL, 91 * style.MM))
    outer = GridSpec(1, 1, figure=figure)
    runtime_ax, readout_ax = runtime_axes(figure, outer[0])
    style.log_runtime_dots(
        runtime_ax,
        names,
        values,
        hardware,
        durations,
        xlabel="Mean wall-clock time per dataset (seconds, log scale)",
        readout_ax=readout_ax,
    )
    panel_header(runtime_ax, None, spec.label)
    figure.subplots_adjust(left=0.205, right=0.985, top=0.82, bottom=0.15)
    return save_and_close(
        figure,
        f"imputation_runtime_{spec.slug}_mean_per_dataset",
        config,
    )


def render_scaling(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    config: RenderConfig,
) -> list[Path]:
    subset = frame.copy()
    subset["size"] = pd.to_numeric(
        subset["dataset"].str.extract(r"(\d+)k", expand=False),
        errors="coerce",
    ) * 1000
    subset = subset.dropna(subset=["size"])
    subset = subset[subset["dataset"].str.contains(r"_T\d_", regex=True)]
    if subset.empty:
        raise ValueError(f"{spec.label}: no T1/T2/T3 size-series rows found")

    style.apply_gr_style()
    figure, axis = plt.subplots(figsize=(style.GR_WIDTH_1P5COL, 76 * style.MM))
    sizes = sorted(subset["size"].unique())
    for method, group in subset.groupby("method", sort=False):
        median = group.groupby("size")["impute_time_seconds"].median()
        values = median.to_numpy(float)
        is_hero = method == style.HERO_METHOD
        is_constant_batch = is_hero and np.ptp(values) < 1e-6
        axis.plot(
            median.index.to_numpy(float),
            values,
            marker="o",
            ms=3.2,
            lw=1.6 if is_hero else 1.0,
            color=style.GR_COLORS.get(method, "#7F7F7F"),
            ls="--" if is_constant_batch else "-",
            zorder=5 if is_hero else 3,
            label=style.display_method_name(method)
            + (" (batch)" if is_constant_batch else ""),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(FixedLocator(sizes))
    axis.xaxis.set_major_formatter(
        FixedFormatter([f"{int(size / 1000)}k" for size in sizes])
    )
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel("Input size (cells)")
    axis.set_ylabel("Runtime (s, log scale)")
    axis.set_title(spec.label, loc="left", fontsize=style.FS_AXIS, pad=8)
    for runtime in (60, 600, 3600):
        axis.axhline(
            runtime,
            color=style.LAND_COLOR,
            lw=0.4,
            ls=(0, (3, 3)),
            zorder=0,
        )
    axis.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper left",
        handlelength=1.4,
        labelspacing=0.35,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    figure.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.17)
    return save_and_close(
        figure,
        f"imputation_runtime_{spec.slug}_scaling",
        config,
    )


def build_summary_tables(specs, frames, aggregates) -> list[pd.DataFrame]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    latex_blocks = []
    for spec in specs:
        aggregate = aggregates[spec.key]
        average = average_per_dataset(frames[spec.key]).set_index("method")
        slowest = aggregate["median"].max()
        table = pd.DataFrame(
            {
                "Method": [
                    style.display_method_name(method) for method in aggregate["method"]
                ],
                "Hardware": aggregate["hardware"],
                "n": aggregate["n"],
                "Median": [style.fmt_duration(value) for value in aggregate["median"]],
                "Range": [
                    f"{style.fmt_duration(low)}-{style.fmt_duration(high)}"
                    for low, high in zip(aggregate["vmin"], aggregate["vmax"])
                ],
                "Mean/dataset": [
                    style.fmt_duration(average.loc[method, "average"])
                    + ("*" if average.loc[method, "amortized"] else "")
                    for method in aggregate["method"]
                ],
                "x vs slowest": [
                    f"{slowest / value:.1f}x" for value in aggregate["median"]
                ],
            }
        )
        table.insert(0, "Dataset", spec.label)
        tables.append(table)
        table.to_csv(
            RESULT_DIR / f"{spec.slug}_runtime_summary.csv",
            index=False,
        )

        body = " \\\\\n".join(
            " & ".join(str(value) for value in row)
            for row in table.drop(columns="Dataset").itertuples(index=False)
        )
        latex_blocks.append(
            f"% {spec.label}\n"
            "\\begin{tabular}{llrlllr}\n"
            "\\toprule\n"
            "Method & Hardware & $n$ & Median & Range & Mean/dataset & "
            "$\\times$ vs slowest \\\\\n"
            "\\midrule\n"
            f"{body} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )

    full_table = pd.concat(tables, ignore_index=True)
    full_table.to_csv(RESULT_DIR / "imputation_runtime_all_summary.csv", index=False)
    (RESULT_DIR / "imputation_runtime_summary.tex").write_text(
        "\n\n".join(latex_blocks),
        encoding="utf-8",
    )
    return tables


def render_summary_table(
    tables: list[pd.DataFrame],
    config: RenderConfig,
) -> list[Path]:
    style.apply_gr_style()
    figure = plt.figure(figsize=(style.GR_WIDTH_2COL, 112 * style.MM))
    outer = GridSpec(len(tables), 1, figure=figure, hspace=0.32)
    columns = [
        "Method",
        "Hardware",
        "n",
        "Median",
        "Range",
        "Mean/dataset",
        "x vs slowest",
    ]
    column_widths = (0.22, 0.11, 0.055, 0.115, 0.21, 0.16, 0.13)
    for table_index, table in enumerate(tables):
        axis = figure.add_subplot(outer[table_index])
        axis.axis("off")
        axis.text(
            0.0,
            1.01,
            table["Dataset"].iloc[0],
            transform=axis.transAxes,
            fontsize=style.FS_AXIS,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        cell_text = []
        cell_colours = []
        for _, row in table.iterrows():
            cell_text.append([row[column] for column in columns])
            is_hero = row["Method"] == style.HERO_METHOD
            cell_colours.append(
                ["#FBEAEC" if is_hero else "white"] * len(columns)
            )

        rendered = axis.table(
            cellText=cell_text,
            colLabels=columns,
            cellColours=cell_colours,
            colWidths=column_widths,
            bbox=(0.0, 0.0, 1.0, 0.91),
            cellLoc="left",
        )
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(style.FS_ANNOT)
        for (row_index, _), cell in rendered.get_celld().items():
            cell.set_linewidth(0.4)
            cell.set_edgecolor("#CFD4D9")
            if row_index == 0:
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
            elif cell.get_text().get_text() == style.HERO_METHOD:
                cell.set_text_props(fontweight="bold", color=style.HERO_EDGE)

    figure.text(
        0.015,
        0.008,
        "* Batch wall time amortized by the number of datasets.",
        fontsize=7,
        ha="left",
        va="bottom",
        color="#4A4F55",
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.97, bottom=0.045)
    return save_and_close(figure, "imputation_runtime_summary_table", config)


def print_summary(specs, frames, aggregates) -> None:
    print("\n===== per-method median runtimes =====")
    for spec in specs:
        print(f"\n--- {spec.label} ---")
        for _, row in aggregates[spec.key].iterrows():
            method = style.display_method_name(row["method"])
            print(
                f"  {method:18s} {style.fmt_duration(row['median']):>8s}  "
                f"n={row['n']:2d}  "
                f"[{style.fmt_duration(row['vmin'])}-"
                f"{style.fmt_duration(row['vmax'])}]  {row['hardware']}"
            )

    print("\n===== mean runtime per dataset =====")
    for spec in specs:
        print(f"\n--- {spec.label} ---")
        for _, row in average_per_dataset(frames[spec.key]).iterrows():
            method = style.display_method_name(row["method"])
            suffix = " (amortized)" if row["amortized"] else ""
            print(
                f"  {method:18s} {style.fmt_duration(row['average']):>8s}  "
                f"{row['hardware']}{suffix}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "eps"),
        default=("png", "pdf"),
        help="figure formats (default: png pdf)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RenderConfig(formats=tuple(dict.fromkeys(args.formats)), dpi=args.dpi)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {spec.key: load_runtime_data(spec) for spec in DATASETS}
    aggregates = {
        spec.key: aggregate_runtime(frames[spec.key]) for spec in DATASETS
    }

    outputs = []
    combined, caveat = render_two_panel(
        DATASETS,
        frames,
        aggregates,
        "bars",
        "imputation_runtime_combined",
        config,
    )
    outputs.extend(combined)
    lollipop, _ = render_two_panel(
        DATASETS,
        frames,
        aggregates,
        "lollipop",
        "imputation_runtime_lollipop",
        config,
    )
    outputs.extend(lollipop)
    for spec in DATASETS:
        outputs.extend(
            render_single_panel(
                spec,
                frames[spec.key],
                aggregates[spec.key],
                config,
            )
        )

    hicimpute_spec = next(spec for spec in DATASETS if spec.key == "hicimpute_61x61")
    outputs.extend(
        render_average_per_dataset(
            hicimpute_spec,
            frames[hicimpute_spec.key],
            config,
        )
    )
    outputs.extend(
        render_scaling(
            hicimpute_spec,
            frames[hicimpute_spec.key],
            config,
        )
    )
    tables = build_summary_tables(DATASETS, frames, aggregates)
    outputs.extend(render_summary_table(tables, config))

    print_summary(DATASETS, frames, aggregates)
    print(f"\nbatch caveat applied: {caveat}")
    print("\nfiles:")
    for path in outputs:
        print(f"  {path.relative_to(BASE_DIR)}")
    for path in sorted(RESULT_DIR.iterdir()):
        print(f"  {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
