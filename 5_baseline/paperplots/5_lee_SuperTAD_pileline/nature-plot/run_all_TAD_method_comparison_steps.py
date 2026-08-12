"""Run and verify the three-step Lee TAD method comparison workflow."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import TAD_method_comparison_config as config
import step03_plot_TAD_method_comparison as plot_TAD
from step01_calculate_PCC_trials import calculate_official_PCC_trials
from step02_call_SuperTAD_domains import (
    call_official_SuperTAD_domains,
    validate_supertad_tsv,
)


FULL_STAGE_ORDER = (
    "validate",
    "calculate_PCC_trials",
    "call_SuperTAD_domains",
    "plot_TAD_method_comparison",
    "verification",
)


def select_stages(verify_only=False):
    return ("verification",) if verify_only else FULL_STAGE_ORDER


def require_force_for_existing(paths, force):
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise FileExistsError(
            "--force is required to replace canonical output(s): "
            + ", ".join(str(path) for path in existing)
        )


def execute_stages(stages, functions, force=False):
    for index, stage in enumerate(stages, start=1):
        if stage not in functions:
            raise KeyError(f"no function registered for pipeline stage: {stage}")
        print(f"[{index}/{len(stages)}] {stage}", flush=True)
        if force:
            functions[stage](force=True)
        else:
            functions[stage]()


def _validate_matrix(path, symmetry_tolerance=1e-8):
    matrix = load_npz(str(path)).toarray()
    if matrix.shape != (config.N_BINS, config.N_BINS):
        raise ValueError(f"{path}: expected 49x49, found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: contains non-finite values")
    if not np.allclose(matrix, matrix.T, atol=symmetry_tolerance, rtol=0):
        raise ValueError(f"{path}: matrix is not symmetric")
    return matrix


def validate_inputs(force=False):
    """Validate official inputs before calculation."""
    del force
    config.validate_project_cwd()
    for cell_type in config.CELL_TYPES:
        path = config.TARGET_ROOT / f"{cell_type}_target.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_matrix(path)

    checked = {}
    for method, input_key in config.MAIN_METHOD_INPUT_KEYS.items():
        source = config.MAIN_METHOD_SOURCES[method]
        counts = {}
        for cell_type in config.CELL_TYPES:
            paths = sorted(Path(source).glob(f"{cell_type}_cell_*.npz"))
            expected = config.EXPECTED_CELL_COUNTS[input_key][cell_type]
            if len(paths) != expected:
                raise ValueError(
                    f"{method}/{cell_type}: expected {expected} files, "
                    f"found {len(paths)}"
                )
            counts[cell_type] = len(paths)
            for sample in {paths[0], paths[-1]}:
                _validate_matrix(sample)
        checked[method] = counts
    return {"target_count": len(config.CELL_TYPES), "source_counts": checked}


def _assert_relative_serialized_paths():
    project_absolute = str(Path.cwd().resolve())
    source_files = [
        Path("LeeData_TAD_method_comparison_input_paths.csv"),
        Path("TAD_method_comparison_config.py"),
        Path("step01_calculate_PCC_trials.py"),
        Path("step02_call_SuperTAD_domains.py"),
        Path("step03_plot_TAD_method_comparison.py"),
        Path("run_all_TAD_method_comparison_steps.py"),
        Path("README.md"),
    ]
    offenders = [
        path.as_posix()
        for path in source_files
        if path.is_file() and project_absolute in path.read_text(errors="replace")
    ]
    roots = (
        config.INTERMEDIATE_ROOT,
        config.SUPERTAD_DOMAIN_ROOT,
        config.RESULTS_ROOT,
    )
    for root in roots:
        if not Path(root).exists():
            continue
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}:
                if project_absolute in path.read_text(errors="replace"):
                    offenders.append(path.as_posix())
    if offenders:
        raise ValueError(f"absolute project path found in: {sorted(set(offenders))}")


def verify_outputs(force=False):
    del force
    expected_representatives = len(config.MAIN_METHOD_SOURCES) * len(config.CELL_TYPES)
    pcc_rows = 0
    for method in config.MAIN_METHOD_SOURCES:
        path = config.PCC_RESULTS_ROOT / method / f"{method}_PCC_trials.csv"
        table = pd.read_csv(path)
        if len(table) != len(config.CELL_TYPES) * config.N_TRIALS:
            raise ValueError(f"{method}: incorrect PCC row count")
        if set(table["cell_type"]) != set(config.CELL_TYPES):
            raise ValueError(f"{method}: incomplete cell types")
        pcc_rows += len(table)

    representatives = pd.read_csv(config.REPRESENTATIVE_TRIALS_FILE)
    if len(representatives) != expected_representatives:
        raise ValueError(
            f"expected {expected_representatives} selected representative trials"
        )
    matrix_paths = [Path(path) for path in representatives["matrix_path"]]
    if any(path.is_absolute() or not path.is_file() for path in matrix_paths):
        raise ValueError("selected representative matrix path is invalid")
    if len(list(config.REPRESENTATIVE_MATRIX_ROOT.glob("*/*.npz"))) != expected_representatives:
        raise ValueError("representative matrix count is incorrect")

    target_tsvs = list((config.SUPERTAD_DOMAIN_ROOT / "target").glob("*.tsv"))
    representative_tsvs = list(
        (config.SUPERTAD_DOMAIN_ROOT / "representatives").glob("*/*.tsv")
    )
    if len(target_tsvs) != len(config.CELL_TYPES):
        raise ValueError("Target SuperTAD count is incorrect")
    if len(representative_tsvs) != expected_representatives:
        raise ValueError("representative SuperTAD count is incorrect")
    for path in target_tsvs + representative_tsvs:
        validate_supertad_tsv(path)

    exports = [Path(path) for path in plot_TAD.build_export_manifest(config.FIGURE_ROOT)]
    if any(not path.is_file() or path.stat().st_size == 0 for path in exports):
        raise ValueError("one or more TAD comparison figure exports are missing")
    direct_results = (
        config.PCC_SUMMARY_FILE,
        config.TAD_PLOT_CHECK_FILE,
        config.RUN_INFORMATION_FILE,
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in direct_results):
        raise ValueError("one or more direct TAD comparison result files are missing")

    _assert_relative_serialized_paths()
    report = {
        "schema_version": 2,
        "status": "verified",
        "paths_are_relative": True,
        "pcc_rows": pcc_rows,
        "selected_representatives": expected_representatives,
        "target_supertad": len(target_tsvs),
        "representative_supertad": len(representative_tsvs),
        "figure_exports": len(exports),
        "direct_result_files": len(direct_results),
    }
    config.VERIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
    path = config.VERIFICATION_ROOT / "pipeline_output_check.json"
    with path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def _plot_stage(force=False):
    args = plot_TAD.build_arg_parser().parse_args([])
    args.force = force
    return plot_TAD.render_TAD_method_comparison(args)


def stage_functions():
    return {
        "validate": validate_inputs,
        "calculate_PCC_trials": calculate_official_PCC_trials,
        "call_SuperTAD_domains": call_official_SuperTAD_domains,
        "plot_TAD_method_comparison": _plot_stage,
        "verification": verify_outputs,
    }


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _new_log_path(log_root):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_root / f"TAD_method_comparison_{timestamp}.log"
    if path.exists():
        raise FileExistsError(f"log already exists for this second: {path}")
    return path


def execute_with_log(stages, functions, log_root=config.LOG_ROOT, force=False):
    log_path = _new_log_path(log_root)
    with log_path.open("w") as log_handle:
        stdout = _Tee(sys.stdout, log_handle)
        stderr = _Tee(sys.stderr, log_handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print("command_cwd=.")
            print(f"python={Path(sys.executable).name}")
            try:
                execute_stages(stages, functions, force=force)
            except Exception as error:
                print(f"status=failed error={type(error).__name__}: {error}")
                raise
            print("status=completed")
    return log_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    config.validate_project_cwd()
    log_path = execute_with_log(
        select_stages(verify_only=args.verify_only),
        stage_functions(),
        force=args.force,
    )
    print(f"Pipeline stages completed. Log: {log_path}")


if __name__ == "__main__":
    main()
