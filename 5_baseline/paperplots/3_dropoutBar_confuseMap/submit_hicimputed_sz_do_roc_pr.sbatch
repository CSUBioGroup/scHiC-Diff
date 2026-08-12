#!/usr/bin/env bash
#SBATCH -J hic_szdo_rocpr
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=1_HiCImputedData/logs/hicimputed_sz_do_roc_pr_%j.out
#SBATCH --error=1_HiCImputedData/logs/hicimputed_sz_do_roc_pr_%j.err

set -euo pipefail

PLOT_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"
MODE="${MODE:-all}"

if [[ "${MODE}" != "all" && "${MODE}" != "metrics" && "${MODE}" != "plots" && "${MODE}" != "selftest" ]]; then
  echo "MODE must be one of: all, metrics, plots, selftest" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PLOT_ROOT}"

if [[ "${MODE}" == "all" || "${MODE}" == "metrics" || "${MODE}" == "selftest" ]]; then
  "${PYTHON_BIN}" calculate_sz_do_roc_pr.py --self-test
fi

if [[ "${MODE}" == "all" || "${MODE}" == "metrics" ]]; then
  "${PYTHON_BIN}" calculate_sz_do_roc_pr.py
fi

if [[ "${MODE}" == "all" || "${MODE}" == "plots" ]]; then
  "${PYTHON_BIN}" plot_sz_do_roc_pr_curves.py --formats pdf png
fi
