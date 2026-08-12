#!/usr/bin/env bash
#SBATCH -J hic_main_rocpr
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=1_HiCImputedData/logs/hicimputed_main_roc_pr_%j.out
#SBATCH --error=1_HiCImputedData/logs/hicimputed_main_roc_pr_%j.err

set -euo pipefail

PLOT_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PLOT_ROOT}"
"${PYTHON_BIN}" plot_hicimputed_pcc_mae_roc_pr_main_figure.py --formats pdf png
