#!/usr/bin/env bash
#SBATCH -J flamingo_main_fig
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=2_FLAMINGOData/logs/main_fig_%j.out
#SBATCH --error=2_FLAMINGOData/logs/main_fig_%j.err

set -euo pipefail

PLOT_DIR="${SLURM_SUBMIT_DIR}/2_FLAMINGOData"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ARGS=(
  plot_flamingo_heatmaps.py
  --cell-index "${CELL_INDEX:-0}"
  --vmax-percentile "${VMAX_PERCENTILE:-99}"
  --formats pdf png
)
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  ARGS+=(--validate-only)
fi

cd "${PLOT_DIR}"
"${PYTHON_BIN}" "${ARGS[@]}"
