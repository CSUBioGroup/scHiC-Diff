#!/usr/bin/env bash
#SBATCH -J hic_main_fig
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=1_HiCImputedData/logs/main_fig_%j.out
#SBATCH --error=1_HiCImputedData/logs/main_fig_%j.err

set -euo pipefail

PLOT_DIR="${SLURM_SUBMIT_DIR}/1_HiCImputedData"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PLOT_DIR}"
"${PYTHON_BIN}" plot_hicimpute_heatmaps.py grid --mode main --formats pdf png
