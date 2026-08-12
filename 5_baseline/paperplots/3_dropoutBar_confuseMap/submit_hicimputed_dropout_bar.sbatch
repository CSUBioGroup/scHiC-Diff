#!/usr/bin/env bash
#SBATCH -J hic_dropout_bar
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=1_HiCImputedData/logs/hicimputed_dropout_bar_%j.out
#SBATCH --error=1_HiCImputedData/logs/hicimputed_dropout_bar_%j.err

set -euo pipefail

PLOT_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"
read -r -a METRIC_ARGS <<< "${METRICS:-pcc mae}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ARGS=(
  plot_dropout_pcc_mae_bar.py
  --dataset hicimpute
  --metrics "${METRIC_ARGS[@]}"
  --formats pdf png
)
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  ARGS+=(--validate-only)
fi

cd "${PLOT_ROOT}"
"${PYTHON_BIN}" "${ARGS[@]}"
