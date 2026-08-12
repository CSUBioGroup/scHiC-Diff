#!/usr/bin/env bash
#SBATCH --job-name=lee_impute
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts/logs/impute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts/logs/impute_%A_%a.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# scHiCluster imputation for Lee SuperTAD pipeline
# 4 cell types, 1 per array task, 20 CPU cores per task
#
# Submit:
#   sbatch 05_submit_impute.sbatch
#
# After imputation, run collect:
#   sbatch 06_submit_collect.sbatch

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/schicluster_input"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/result"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}"
export PATH="${ENV_DIR}/bin:${PATH}"

DATASETS=("Astro" "Endo" "ODC" "OPC")

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Task: ${TASK_ID}  Dataset: ${STEM}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=============================================="

"${PYTHON}" --version
"${PYTHON}" "${SCRIPT_DIR}/02_impute_lee.py" \
  --dataset "${STEM}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chrom chr4 \
  --n-bins 49 \
  --resolution 10000 \
  --pad 1 --std 1.0 --rp 0.5 --tol 0.01 \
  --window-size 500000 --step-size 500000 \
  --output-format npz \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="