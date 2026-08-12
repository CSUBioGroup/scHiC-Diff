#!/usr/bin/env bash
#SBATCH -J imputation_metrics
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/metrics_%A_%a.out
#SBATCH --error=logs/metrics_%A_%a.err

set -euo pipefail

# Slurm executes a spool copy of this file, so BASH_SOURCE is not the project path.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from paperplots/1_pccAndMae_all}"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"
SCRIPT_PATH="${SCRIPT_DIR}/calculate_imputation_metrics.py"
MANIFEST_PATH="${SCRIPT_DIR}/imputation_metric_tasks.csv"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" run-task --manifest "${MANIFEST_PATH}" --task-id "${SLURM_ARRAY_TASK_ID}"
