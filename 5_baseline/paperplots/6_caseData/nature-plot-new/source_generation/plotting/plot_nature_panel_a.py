#!/usr/bin/env python3

import argparse
import importlib.util
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    storage_name: str
    display_name: str


@dataclass(frozen=True)
class StageSpec:
    storage_name: str
    display_name: str


METHODS = (
    MethodSpec("Raw", "Raw"),
    MethodSpec("scHiCDiff", "scHiC-Diff"),
    MethodSpec("Higashi_nbr0", "Higashi nbr0"),
    MethodSpec("Higashi_nbr5", "Higashi nbr5"),
    MethodSpec("FLAMINGO", "FLAMINGO"),
    MethodSpec("scVI-3D", "scVI-3D"),
    MethodSpec("scHiCluster", "scHiCluster"),
)

STAGES = (
    StageSpec("E70", "E7.0"),
    StageSpec("E75", "E7.5"),
    StageSpec("E80", "E8.0"),
    StageSpec("E85", "E8.5"),
    StageSpec("E95", "E9.5"),
    StageSpec("EX05", "E10.5"),
    StageSpec("EX15", "E11.5"),
)

MEMBERSHIP_COLUMNS = ("cell_id", "stage", "celltype", "lineage")
FIGURE_WIDTH_MM = 174.0
HIGHLIGHT_METHOD = "scHiC-Diff"
PANEL_ROOT = Path(__file__).resolve().parents[1]
NATURE_PLOT_DIR = PANEL_ROOT.parent
CASE_ROOT = NATURE_PLOT_DIR.parent
DEFAULT_DATA_ROOT = CASE_ROOT / "1_cluster1mb"
DEFAULT_OUTPUT_DIR = PANEL_ROOT / "outputs"
LOGGER = logging.getLogger("nature_panel_a")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw Nature-style Panel A from formal 1 Mb HiRES outputs."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--nature-plot-dir", type=Path, default=NATURE_PLOT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--point-size", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.75)
    return parser.parse_args(argv)


def _read_table(path: Path, required: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(label, path))
    table = pd.read_csv(path, sep="\t")
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError("{} is missing required columns: {}".format(label, missing))
    return table


def _reject_duplicate_cell_ids(table: pd.DataFrame, label: str) -> None:
    duplicated = table["cell_id"].duplicated(keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, "cell_id"].astype(str).unique()[:5]
        raise ValueError(
            "duplicate {} cell_id values: {}".format(label, ", ".join(examples))
        )


def load_method_result(
    data_root: Path,
    method: MethodSpec,
    stages: Sequence[StageSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    method_root = data_root / "outputs" / method.storage_name
    umap = _read_table(
        method_root / "seurat" / "umap_coordinates.tsv",
        ("cell_id", "schUMAP_1", "schUMAP_2"),
        "{} UMAP".format(method.storage_name),
    )
    per_cell = _read_table(
        method_root / "metrics" / "red_blue_silhouette_per_cell.tsv",
        ("cell_id", "stage", "celltype", "lineage", "silhouette"),
        "{} per-cell silhouette".format(method.storage_name),
    )
    summary = _read_table(
        method_root / "metrics" / "red_blue_silhouette_by_stage.tsv",
        (
            "stage",
            "status",
            "reason",
            "n_red",
            "n_blue",
            "n_used",
            "mean_silhouette",
        ),
        "{} stage silhouette summary".format(method.storage_name),
    )

    _reject_duplicate_cell_ids(umap, "UMAP")
    _reject_duplicate_cell_ids(per_cell, "per-cell silhouette")
    if summary["stage"].duplicated(keep=False).any():
        raise ValueError("{} has duplicate stage summary rows".format(method.storage_name))

    expected_stages = [stage.storage_name for stage in stages]
    observed_summary_stages = summary["stage"].astype(str).tolist()
    if set(observed_summary_stages) != set(expected_stages):
        raise ValueError(
            "{} stage summary mismatch: expected {}, observed {}".format(
                method.storage_name, expected_stages, observed_summary_stages
            )
        )
    unexpected_per_cell = sorted(set(per_cell["stage"].astype(str)) - set(expected_stages))
    if unexpected_per_cell:
        raise ValueError(
            "{} per-cell table contains unexpected stages: {}".format(
                method.storage_name, unexpected_per_cell
            )
        )
    invalid_lineages = sorted(set(per_cell["lineage"].dropna().astype(str)) - {"Red", "Blue"})
    if invalid_lineages or per_cell["lineage"].isna().any():
        raise ValueError(
            "{} per-cell table has invalid lineage values: {}".format(
                method.storage_name, invalid_lineages
            )
        )

    points = per_cell.merge(
        umap,
        on="cell_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_umap = points["_merge"].ne("both")
    if missing_umap.any():
        examples = points.loc[missing_umap, "cell_id"].astype(str).tolist()[:5]
        raise ValueError(
            "{} has missing UMAP coordinates for: {}".format(
                method.storage_name, ", ".join(examples)
            )
        )
    points = points.drop(columns="_merge")

    numeric_columns = ("schUMAP_1", "schUMAP_2", "silhouette")
    for column in numeric_columns:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    if not np.isfinite(points.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("{} contains missing or non-finite numeric values".format(method.storage_name))

    stage_display = {stage.storage_name: stage.display_name for stage in stages}
    summary = summary.copy()
    for stage in stages:
        row_index = summary.index[summary["stage"].astype(str).eq(stage.storage_name)]
        row = summary.loc[row_index[0]]
        if str(row["status"]) != "ok":
            raise ValueError(
                "{} stage {} has status {}: {}".format(
                    method.storage_name,
                    stage.storage_name,
                    row["status"],
                    row["reason"],
                )
            )

        stage_points = points.loc[points["stage"].astype(str).eq(stage.storage_name)]
        observed_counts = {
            "n_red": int(stage_points["lineage"].eq("Red").sum()),
            "n_blue": int(stage_points["lineage"].eq("Blue").sum()),
            "n_used": int(len(stage_points)),
        }
        stored_counts = {name: int(row[name]) for name in observed_counts}
        if stored_counts != observed_counts:
            raise ValueError(
                "{} count mismatch for {}: stored {}, observed {}".format(
                    method.storage_name,
                    stage.storage_name,
                    stored_counts,
                    observed_counts,
                )
            )

        observed_mean = float(stage_points["silhouette"].mean())
        stored_mean = float(row["mean_silhouette"])
        if not np.isclose(stored_mean, observed_mean, rtol=1e-6, atol=1e-7):
            raise ValueError(
                "{} silhouette mean mismatch for {}: stored {}, observed {}".format(
                    method.storage_name,
                    stage.storage_name,
                    stored_mean,
                    observed_mean,
                )
            )

    points["method_storage"] = method.storage_name
    points["method"] = method.display_name
    points["stage_display"] = points["stage"].map(stage_display)
    summary["method_storage"] = method.storage_name
    summary["method"] = method.display_name
    summary["stage_display"] = summary["stage"].map(stage_display)

    stage_order = {stage.storage_name: index for index, stage in enumerate(stages)}
    points["_stage_order"] = points["stage"].map(stage_order)
    summary["_stage_order"] = summary["stage"].map(stage_order)
    points = points.sort_values(["_stage_order", "cell_id"], kind="stable").drop(
        columns="_stage_order"
    )
    summary = summary.sort_values("_stage_order", kind="stable").drop(
        columns="_stage_order"
    )
    return points.reset_index(drop=True), summary.reset_index(drop=True)


def validate_cross_method_membership(
    points_by_method: Mapping[str, pd.DataFrame],
    reference_method: str = "Raw",
) -> None:
    if reference_method not in points_by_method:
        raise ValueError("reference method is missing: {}".format(reference_method))

    def membership(table: pd.DataFrame, method_name: str) -> set:
        missing = sorted(set(MEMBERSHIP_COLUMNS).difference(table.columns))
        if missing:
            raise ValueError(
                "{} membership table is missing columns: {}".format(method_name, missing)
            )
        return set(table.loc[:, MEMBERSHIP_COLUMNS].itertuples(index=False, name=None))

    reference = membership(points_by_method[reference_method], reference_method)
    for method_name, table in points_by_method.items():
        if method_name == reference_method:
            continue
        observed = membership(table, method_name)
        if observed != reference:
            raise ValueError(
                "{} membership mismatch relative to {}: missing {}, extra {}".format(
                    method_name,
                    reference_method,
                    len(reference - observed),
                    len(observed - reference),
                )
            )


def load_gr_stagefig(nature_plot_dir: Path) -> ModuleType:
    module_path = Path(nature_plot_dir) / "gr_stagefig.py"
    if not module_path.is_file():
        raise FileNotFoundError("gr_stagefig.py does not exist: {}".format(module_path))

    module_name = "hires_nature_gr_stagefig"
    loaded = sys.modules.get(module_name)
    if loaded is not None and Path(loaded.__file__).resolve() == module_path.resolve():
        return loaded

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot create an import spec for {}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def make_plot_inputs(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    methods: Sequence[MethodSpec],
    stages: Sequence[StageSpec],
    stage_data_class,
) -> Tuple[Dict[str, Dict[str, object]], dict, dict]:
    data: Dict[str, Dict[str, object]] = {}
    silhouettes = {}
    counts_by_stage = {stage.storage_name: {} for stage in stages}

    for method in methods:
        data[method.display_name] = {}
        for stage in stages:
            stage_points = points.loc[
                points["method_storage"].eq(method.storage_name)
                & points["stage"].eq(stage.storage_name)
            ].sort_values("cell_id", kind="stable")
            stage_summary = summary.loc[
                summary["method_storage"].eq(method.storage_name)
                & summary["stage"].eq(stage.storage_name)
            ]
            if len(stage_summary) != 1:
                raise ValueError(
                    "expected one summary row for {} {}, found {}".format(
                        method.storage_name, stage.storage_name, len(stage_summary)
                    )
                )
            stored_n = int(stage_summary.iloc[0]["n_used"])
            if stored_n != len(stage_points):
                raise ValueError(
                    "n_used mismatch for {} {}: summary {}, points {}".format(
                        method.storage_name,
                        stage.storage_name,
                        stored_n,
                        len(stage_points),
                    )
                )
            counts_by_stage[stage.storage_name][method.storage_name] = stored_n
            data[method.display_name][stage.display_name] = stage_data_class(
                xy=stage_points.loc[:, ["schUMAP_1", "schUMAP_2"]].to_numpy(),
                group=stage_points["lineage"].to_numpy(),
            )
            silhouettes[(method.display_name, stage.display_name)] = (
                float(stage_summary.iloc[0]["mean_silhouette"]),
                np.nan,
                np.nan,
            )

    stage_n = {}
    for stage in stages:
        method_counts = counts_by_stage[stage.storage_name]
        unique_counts = set(method_counts.values())
        if len(unique_counts) != 1:
            raise ValueError(
                "n_used mismatch across methods for {}: {}".format(
                    stage.storage_name, method_counts
                )
            )
        stage_n[stage.display_name] = unique_counts.pop()
    return data, silhouettes, stage_n


def build_panel_a(
    data: dict,
    silhouettes: dict,
    stage_n: dict,
    methods: Sequence[MethodSpec],
    stages: Sequence[StageSpec],
    style: ModuleType,
    point_size: float = 1.4,
    alpha: float = 0.75,
):
    style.set_gr_style()
    method_names = [method.display_name for method in methods]
    stage_names = [stage.display_name for stage in stages]

    top_mm = 5.0
    left_mm = 17.0
    right_mm = 1.0
    gap_mm = 1.2
    header_mm = 6.5
    key_mm = 6.0
    cell_mm = (
        FIGURE_WIDTH_MM
        - left_mm
        - right_mm
        - gap_mm * (len(stage_names) - 1)
    ) / len(stage_names)
    figure_height_mm = (
        top_mm
        + header_mm
        + len(method_names) * cell_mm
        + (len(method_names) - 1) * gap_mm
        + key_mm
        + 3.0
    )
    figure = style.plt.figure(
        figsize=(style.mm(FIGURE_WIDTH_MM), style.mm(figure_height_mm))
    )
    style.plot_stage_grid(
        figure,
        data,
        silhouettes,
        method_names,
        stage_names,
        HIGHLIGHT_METHOD,
        fig_w_mm=FIGURE_WIDTH_MM,
        fig_h_mm=figure_height_mm,
        top_mm=top_mm,
        left_mm=left_mm,
        right_mm=right_mm,
        gap_mm=gap_mm,
        header_mm=header_mm,
        key_mm=key_mm,
        stage_n=stage_n,
        point_size=point_size,
        alpha=alpha,
        mode="group",
    )
    path_effects = importlib.import_module("matplotlib.patheffects")
    for axis in figure.axes:
        for text in axis.texts:
            if text.get_position() == (0.97, 0.03):
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=2.0, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
                text.set_zorder(10)
    style.panel_letter(
        figure,
        2.0,
        2.0,
        "A",
        FIGURE_WIDTH_MM,
        figure_height_mm,
    )
    return figure


def _sort_for_export(
    table: pd.DataFrame,
    methods: Sequence[MethodSpec],
    stages: Sequence[StageSpec],
    final_columns: Sequence[str],
) -> pd.DataFrame:
    method_order = {method.storage_name: index for index, method in enumerate(methods)}
    stage_order = {stage.storage_name: index for index, stage in enumerate(stages)}
    result = table.copy()
    result["_method_order"] = result["method_storage"].map(method_order)
    result["_stage_order"] = result["stage_storage"].map(stage_order)
    sort_columns = ["_method_order", "_stage_order"]
    if "cell_id" in result.columns:
        sort_columns.append("cell_id")
    result = result.sort_values(sort_columns, kind="stable")
    return result.loc[:, final_columns].reset_index(drop=True)


def export_panel_a(
    figure,
    points: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    methods: Sequence[MethodSpec],
    stages: Sequence[StageSpec],
    style: ModuleType,
    input_paths: Mapping[str, Path],
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = style.save_figure(
        figure, "nature_panel_a", outdir=output_dir
    )

    point_export = points.rename(
        columns={"stage": "stage_storage", "stage_display": "stage"}
    )
    point_columns = (
        "method",
        "method_storage",
        "stage",
        "stage_storage",
        "cell_id",
        "celltype",
        "lineage",
        "schUMAP_1",
        "schUMAP_2",
        "silhouette",
    )
    point_export = _sort_for_export(point_export, methods, stages, point_columns)
    points_path = output_dir / "source_data_panel_a.csv"
    point_export.to_csv(points_path, index=False)

    summary_export = summary.rename(
        columns={"stage": "stage_storage", "stage_display": "stage"}
    )
    summary_columns = (
        "method",
        "method_storage",
        "stage",
        "stage_storage",
        "status",
        "n_red",
        "n_blue",
        "n_used",
        "mean_silhouette",
    )
    summary_export = _sort_for_export(summary_export, methods, stages, summary_columns)
    summary_path = output_dir / "source_data_panel_a_summary.csv"
    summary_export.to_csv(summary_path, index=False)

    metadata_path = output_dir / "nature_panel_a_run_metadata.json"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": [
            {"storage_name": method.storage_name, "display_name": method.display_name}
            for method in methods
        ],
        "stages": [
            {"storage_name": stage.storage_name, "display_name": stage.display_name}
            for stage in stages
        ],
        "input_paths": {
            name: str(Path(path).resolve()) for name, path in input_paths.items()
        },
        "point_rows": int(len(point_export)),
        "summary_rows": int(len(summary_export)),
        "figure_width_mm": FIGURE_WIDTH_MM,
        "figure_height_mm": float(figure.get_size_inches()[1] * 25.4),
        "highlight_method": HIGHLIGHT_METHOD,
        "stage_n_definition": "Red plus Blue cells plotted (n_used)",
        "silhouette_space": "final SVD dimensions 1-15",
        "silhouette_metric": "euclidean",
        "silhouette_recomputed": False,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": style.mpl.__version__,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "pdf": Path(pdf_path),
        "png": Path(png_path),
        "points": points_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def load_all_results(
    data_root: Path,
    methods: Sequence[MethodSpec],
    stages: Sequence[StageSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    points_by_method = {}
    summaries = []
    input_paths: Dict[str, Path] = {}
    for method in methods:
        LOGGER.info("Loading formal results: %s", method.storage_name)
        points, summary = load_method_result(data_root, method, stages)
        points_by_method[method.storage_name] = points
        summaries.append(summary)
        method_root = Path(data_root) / "outputs" / method.storage_name
        input_paths["{}.umap".format(method.storage_name)] = (
            method_root / "seurat" / "umap_coordinates.tsv"
        )
        input_paths["{}.silhouette_per_cell".format(method.storage_name)] = (
            method_root / "metrics" / "red_blue_silhouette_per_cell.tsv"
        )
        input_paths["{}.silhouette_by_stage".format(method.storage_name)] = (
            method_root / "metrics" / "red_blue_silhouette_by_stage.tsv"
        )

    validate_cross_method_membership(points_by_method)
    all_points = pd.concat(
        [points_by_method[method.storage_name] for method in methods],
        ignore_index=True,
    )
    all_summaries = pd.concat(summaries, ignore_index=True)
    return all_points, all_summaries, input_paths


def run_panel_a(
    data_root: Path,
    nature_plot_dir: Path,
    output_dir: Path,
    methods: Sequence[MethodSpec] = METHODS,
    stages: Sequence[StageSpec] = STAGES,
    point_size: float = 1.4,
    alpha: float = 0.75,
) -> Dict[str, Path]:
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    style = load_gr_stagefig(nature_plot_dir)
    points, summary, input_paths = load_all_results(data_root, methods, stages)
    data, silhouettes, stage_n = make_plot_inputs(
        points,
        summary,
        methods,
        stages,
        style.StageData,
    )
    LOGGER.info(
        "Validated %d methods, %d stages, %d point rows, and %d summary rows",
        len(methods),
        len(stages),
        len(points),
        len(summary),
    )
    figure = build_panel_a(
        data,
        silhouettes,
        stage_n,
        methods,
        stages,
        style,
        point_size=point_size,
        alpha=alpha,
    )
    try:
        outputs = export_panel_a(
            figure,
            points,
            summary,
            output_dir,
            methods,
            stages,
            style,
            input_paths,
        )
    finally:
        style.plt.close(figure)
    return outputs


def _configure_logging(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "nature_panel_a.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)
    return log_path


def main(argv=None) -> int:
    args = parse_args(argv)
    log_path = _configure_logging(args.output_dir)
    LOGGER.info("Formal data root: %s", args.data_root.resolve())
    LOGGER.info("Nature plotting code: %s", args.nature_plot_dir.resolve())
    LOGGER.info("Output directory: %s", args.output_dir.resolve())
    outputs = run_panel_a(
        data_root=args.data_root,
        nature_plot_dir=args.nature_plot_dir,
        output_dir=args.output_dir,
        methods=METHODS,
        stages=STAGES,
        point_size=args.point_size,
        alpha=args.alpha,
    )
    for name, path in outputs.items():
        LOGGER.info("Wrote %s: %s", name, path.resolve())
    LOGGER.info("Wrote log: %s", log_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
