#!/usr/bin/env bash
#SBATCH -J dropout_main_figure
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=1_HiCImputedData/logs/main_dropout_figure_%j.out
#SBATCH --error=1_HiCImputedData/logs/main_dropout_figure_%j.err

set -euo pipefail

PLOT_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PLOT_ROOT}"
"${PYTHON_BIN}" plot_main_dropout_figure.py --formats pdf png
