"""Portable configuration for the Lee PDGFRA TAD method comparison.

Run every command with nature-plot as the current directory. All runtime and
serialized paths are relative to this project.
"""
from __future__ import annotations

import csv
from pathlib import Path


INPUT_PATHS_FILE = Path("LeeData_TAD_method_comparison_input_paths.csv")
INPUT_ROOT = Path("../input_leeData")
IMPUTED_ROOT = Path("../imputedData")
TARGET_ROOT = INPUT_ROOT / "target"
INTERMEDIATE_ROOT = Path("intermediate_data")
REPRESENTATIVE_MATRIX_ROOT = INTERMEDIATE_ROOT / "representative_matrices"
SUPERTAD_ROOT = Path("SuperTAD")
SUPERTAD_BIN = SUPERTAD_ROOT / "bin" / "SuperTAD"
SUPERTAD_DOMAIN_ROOT = SUPERTAD_ROOT / "domains"
RESULTS_ROOT = Path("results")
PCC_RESULTS_ROOT = RESULTS_ROOT / "PCC_trials_by_method"
REPRESENTATIVE_TRIALS_FILE = RESULTS_ROOT / "selected_representative_trials.csv"
PCC_SUMMARY_FILE = RESULTS_ROOT / "PCC_method_comparison_summary.csv"
TAD_PLOT_CHECK_FILE = RESULTS_ROOT / "TAD_boundary_plot_check.csv"
RUN_INFORMATION_FILE = RESULTS_ROOT / "TAD_method_comparison_run_information.json"
VERIFICATION_ROOT = RESULTS_ROOT / "reproducibility_verification"
FIGURES_ROOT = Path("figures")
FIGURE_ROOT = FIGURES_ROOT / "TAD_method_comparison"
LOG_ROOT = Path("logs")

OFFICIAL_METHODS = (
    "Raw",
    "scHiCluster",
    "Higashi-nbr0",
    "Higashi-nbr5",
    "scVI-3D",
    "HiCImpute",
    "T-FLAMINGO",
    "scHiC-Diff",
)

# Graphical order only.  Keep OFFICIAL_METHODS in the input/calculation order
# so PCC trials, representative matrices, and SuperTAD calls remain unchanged.
FIGURE_METHOD_ORDER = (
    "Raw",
    "scHiCluster",
    "HiCImpute",
    "Higashi-nbr0",
    "Higashi-nbr5",
    "scVI-3D",
    "T-FLAMINGO",
    "scHiC-Diff",
)
if (
    len(FIGURE_METHOD_ORDER) != len(OFFICIAL_METHODS)
    or set(FIGURE_METHOD_ORDER) != set(OFFICIAL_METHODS)
):
    raise ValueError("figure method order must be a permutation of official methods")

APPROVED_SOURCE_PATHS = {
    "HiCImpute": Path("../imputedData/HiCImpute_fig1_current"),
    "scVI-3D": Path("../imputedData/scVI-3D_candidate_per_cell"),
    "T-FLAMINGO": Path("../imputedData/FLAMINGO_fixed_contact"),
}


def _load_input_registry(path=INPUT_PATHS_FILE):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing input registry: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"role", "display_method", "input_key", "relative_path", "description"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"input registry columns must be {sorted(required)}")

    registry = {}
    for row in rows:
        display_method = row["display_method"].strip()
        relative_path = Path(row["relative_path"].strip())
        if relative_path.is_absolute():
            raise ValueError(f"absolute input path is forbidden: {relative_path}")
        if display_method in registry:
            raise ValueError(f"duplicate input registry method: {display_method}")
        registry[display_method] = {
            "role": row["role"].strip(),
            "input_key": row["input_key"].strip(),
            "path": relative_path,
            "description": row["description"].strip(),
        }
    return registry


INPUT_REGISTRY = _load_input_registry()
if "Target" not in INPUT_REGISTRY:
    raise ValueError("input registry must contain Target")
if tuple(name for name in INPUT_REGISTRY if name != "Target") != OFFICIAL_METHODS:
    raise ValueError("input registry methods must match the approved display order")

MAIN_METHOD_INPUT_KEYS = {
    method: INPUT_REGISTRY[method]["input_key"] for method in OFFICIAL_METHODS
}
MAIN_METHOD_SOURCES = {
    method: INPUT_REGISTRY[method]["path"] for method in OFFICIAL_METHODS
}
for method, approved_path in APPROVED_SOURCE_PATHS.items():
    if MAIN_METHOD_SOURCES[method] != approved_path:
        raise ValueError(
            f"{method} must use {approved_path}; found {MAIN_METHOD_SOURCES[method]}"
        )

CELL_TYPES = ("Astro", "Endo", "ODC", "OPC")
STANDARD_CELL_COUNTS = {"Astro": 449, "Endo": 202, "ODC": 1244, "OPC": 203}
EXPECTED_CELL_COUNTS = {
    "Raw": {"Astro": 449, "Endo": 205, "ODC": 1244, "OPC": 203},
    "scHiCluster": dict(STANDARD_CELL_COUNTS),
    "Higashi_nbr0": dict(STANDARD_CELL_COUNTS),
    "Higashi_nbr5": dict(STANDARD_CELL_COUNTS),
    "FLAMINGO_fixed_contact": dict(STANDARD_CELL_COUNTS),
    "scHiC-Diff": dict(STANDARD_CELL_COUNTS),
    "HiCImpute_fig1_current": {
        "Astro": 449,
        "Endo": 205,
        "ODC": 1245,
        "OPC": 203,
    },
    "scVI-3D_candidate_per_cell": dict(STANDARD_CELL_COUNTS),
}

N_TRIALS = 100
N_SAMPLE = 30
BASE_SEED = 42

CHROM = "chr4"
REGION_START = 54890000
RESOLUTION = 10000
N_BINS = 49
PDGFRA_START = 55090000
PDGFRA_END = 55170000
PDGFRA_SUB_BINS = (20, 28)

SUPERTAD_MODE = "multi"
SUPERTAD_HEIGHT = 3


def serialized_configuration():
    """Return the portable configuration written into run information files."""
    return {
        "roots": {
            "input": str(INPUT_ROOT),
            "imputed": str(IMPUTED_ROOT),
            "target": str(TARGET_ROOT),
            "intermediate": str(INTERMEDIATE_ROOT),
            "representative_matrices": str(REPRESENTATIVE_MATRIX_ROOT),
            "supertad": str(SUPERTAD_ROOT),
            "supertad_domains": str(SUPERTAD_DOMAIN_ROOT),
            "figures": str(FIGURES_ROOT),
            "results": str(RESULTS_ROOT),
            "logs": str(LOG_ROOT),
        },
        "input_registry": str(INPUT_PATHS_FILE),
        "main_method_sources": {
            method: str(path) for method, path in MAIN_METHOD_SOURCES.items()
        },
        "figure_method_order": list(FIGURE_METHOD_ORDER),
        "expected_cell_counts": EXPECTED_CELL_COUNTS,
        "cell_types": list(CELL_TYPES),
        "trial": {
            "n_trials": N_TRIALS,
            "n_sample": N_SAMPLE,
            "base_seed": BASE_SEED,
        },
        "region": {
            "chrom": CHROM,
            "start": REGION_START,
            "resolution": RESOLUTION,
            "n_bins": N_BINS,
            "pdgfra_sub_bins": list(PDGFRA_SUB_BINS),
        },
    }


def validate_project_cwd():
    """Fail early unless the command is launched from nature-plot."""
    expected = Path(__file__).resolve().parent
    actual = Path.cwd().resolve()
    if actual != expected:
        raise RuntimeError(
            "Run this command with nature-plot as the current directory; "
            f"received {actual.name!r}."
        )
    return True


def get_bin_start(bin_idx):
    return REGION_START + bin_idx * RESOLUTION


def get_bin_end(bin_idx):
    return REGION_START + (bin_idx + 1) * RESOLUTION
