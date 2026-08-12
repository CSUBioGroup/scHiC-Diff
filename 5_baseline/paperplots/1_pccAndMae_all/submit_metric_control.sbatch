#!/usr/bin/env bash
#SBATCH -J metric_control
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/control_%x_%j.out
#SBATCH --error=logs/control_%x_%j.err

set -euo pipefail

MODE="${1:?Usage: submit_metric_control.sbatch prepare|aggregate [FLAMINGOData|HiCImputeData]}"
DATASET_FAMILY="${2:-}"
# Slurm executes a spool copy of this file, so BASH_SOURCE is not the project path.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from paperplots/1_pccAndMae_all}"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"
SCRIPT_PATH="${SCRIPT_DIR}/calculate_imputation_metrics.py"
MANIFEST_PATH="${SCRIPT_DIR}/imputation_metric_tasks.csv"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "${MODE}" in
  prepare)
    "${PYTHON_BIN}" "${SCRIPT_PATH}" prepare-manifest --output-csv "${MANIFEST_PATH}"
    ;;
  aggregate)
    args=(aggregate --manifest "${MANIFEST_PATH}")
    if [[ -n "${DATASET_FAMILY}" ]]; then
      args+=(--dataset-family "${DATASET_FAMILY}")
    fi
    "${PYTHON_BIN}" "${SCRIPT_PATH}" "${args[@]}"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac
