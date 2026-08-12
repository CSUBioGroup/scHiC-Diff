"""Shared configuration for Lee PDGFRA SuperTAD pipeline (portable)."""
import os
import platform

# === Paths (all relative to pipeline dir for portability) ===
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(PIPELINE_DIR, "bin")


def select_supertad_binary(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return os.path.join(BIN_DIR, "SuperTAD_linux_x86_64")
    return os.path.join(BIN_DIR, "SuperTAD_macos_arm64")


SUPERTAD_BIN = select_supertad_binary()

INPUT_LEE_DIR = os.path.join(PIPELINE_DIR, "input_lee")
PER_CELL_NPZ_DIR = os.path.join(INPUT_LEE_DIR, "per_cell_npz")
PER_CELL_BEDPE_DIR = os.path.join(INPUT_LEE_DIR, "per_cell_bedpe")
TARGET_DIR = os.path.join(PIPELINE_DIR, "target")
IMPUTED_DIR = os.path.join(PIPELINE_DIR, "imputed")
TRIALS_DIR = os.path.join(PIPELINE_DIR, "trials")
SUPERTAD_DIR = os.path.join(PIPELINE_DIR, "supertad")
FIGURES_DIR = os.path.join(PIPELINE_DIR, "figures")

# === Genomic coordinates ===
CHROM = "chr4"
RESOLUTION = 10000  # 10kb
N_BINS = 49
REGION_START = 54890000  # bin 0; bin 20 = 55090000 = PDGFRA start
PDGFRA_START = 55090000
PDGFRA_END = 55170000
PDGFRA_SUB_BINS = (20, 28)  # 8x8 sub-region for visualization

# === Cell types ===
CELL_TYPES = ["Astro", "Endo", "ODC", "OPC"]

# === Trial parameters ===
N_TRIALS = 100
N_SAMPLE = 30
BASE_SEED = 42

# === SuperTAD parameters ===
SUPERTAD_MODE = "multi"
SUPERTAD_HEIGHT = 3


def get_bin_start(bin_idx):
    return REGION_START + bin_idx * RESOLUTION


def get_bin_end(bin_idx):
    return REGION_START + (bin_idx + 1) * RESOLUTION
