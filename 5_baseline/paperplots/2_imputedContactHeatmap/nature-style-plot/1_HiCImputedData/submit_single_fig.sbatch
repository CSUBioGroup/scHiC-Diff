#!/usr/bin/env bash
#SBATCH -J hic_single_fig
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=1_HiCImputedData/logs/single_fig_%j.out
#SBATCH --error=1_HiCImputedData/logs/single_fig_%j.err

set -euo pipefail

PLOT_DIR="${SLURM_SUBMIT_DIR}/1_HiCImputedData"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"
METHODS="${METHODS:-${METHOD:-}}"
CTYPES="${CTYPES:-${CTYPE:-}}"
DEPTHS="${DEPTHS:-${DEPTH:-}}"
CELL_INDEX="${CELL_INDEX:-0}"
FORMATS="${FORMATS:-png}"

read -r -a FORMAT_ARGS <<< "${FORMATS}"
ARGS=(
  plot_hicimpute_heatmaps.py single
  --cell-index "${CELL_INDEX}"
  --formats "${FORMAT_ARGS[@]}"
)
if [[ -n "${METHODS}" ]]; then
  read -r -a METHOD_ARGS <<< "${METHODS}"
  ARGS+=(--method "${METHOD_ARGS[@]}")
fi
if [[ -n "${CTYPES}" ]]; then
  read -r -a CTYPE_ARGS <<< "${CTYPES}"
  ARGS+=(--ctype "${CTYPE_ARGS[@]}")
fi
if [[ -n "${DEPTHS}" ]]; then
  read -r -a DEPTH_ARGS <<< "${DEPTHS}"
  ARGS+=(--depth "${DEPTH_ARGS[@]}")
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${PLOT_DIR}"
"${PYTHON_BIN}" "${ARGS[@]}"
