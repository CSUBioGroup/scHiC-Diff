#!/usr/bin/env bash
#SBATCH -J imputation_runtime
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/imputation_runtime_%j.out
#SBATCH --error=logs/imputation_runtime_%j.err

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p figures results logs

export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" plot_imputation_runtime.py \
  --formats png pdf \
  --dpi 600
