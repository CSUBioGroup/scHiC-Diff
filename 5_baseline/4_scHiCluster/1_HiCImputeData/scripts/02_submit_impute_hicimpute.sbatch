#!/usr/bin/env bash
#SBATCH --job-name=schicluster_hicimp
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/scripts/logs/impute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/scripts/logs/impute_%A_%a.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-11

# scHiCluster imputation for HiCImputeData (K562, chr19, 61 bins, 100 cells/dataset)
# 12 datasets, 1 dataset per array task, 20 CPU cores per task
#
# Submit:
#   sbatch 02_submit_impute_hicimpute.sbatch
#
# After imputation, run collect + evaluate:
#   sbatch 07_submit_collect_eval.sbatch

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/input/1_Simu_Data"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/result/1_Simu_Data"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}"

export PATH="${ENV_DIR}/bin:${PATH}"

DATASETS=(
  "K562_T1_1k" "K562_T1_2k" "K562_T1_4k" "K562_T1_7k"
  "K562_T2_1k" "K562_T2_2k" "K562_T2_4k" "K562_T2_7k"
  "K562_T3_1k" "K562_T3_2k" "K562_T3_4k" "K562_T3_7k"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Task: ${TASK_ID}  Dataset: ${STEM}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=============================================="

"${PYTHON}" --version
"${PYTHON}" "${SCRIPT_DIR}/01_impute_hicimpute.py" \
  --dataset "${STEM}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chrom chr19 \
  --n-bins 61 \
  --resolution 1 \
  --pad 1 --std 1.0 --rp 0.5 --tol 0.01 \
  --window-size 500 --step-size 500 \
  --output-format hdf5 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
