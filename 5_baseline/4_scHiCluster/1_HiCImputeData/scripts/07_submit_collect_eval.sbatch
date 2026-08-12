#!/usr/bin/env bash
#SBATCH --job-name=simu_schicluster_collect
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/scripts/logs/collect_eval_%A.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/scripts/logs/collect_eval_%A.log
#SBATCH --account=pi_limin
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/result/1_Simu_Data"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeDate/output"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}/metrics"
export PATH="${ENV_DIR}/bin:${PATH}"

echo "Started at: $(date)"
echo "Host: $(hostname)"
"${PYTHON}" --version

cd "${SCRIPT_DIR}"
echo "===== COLLECT ====="
"${PYTHON}" 05_collect_simu_hdf5.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chrom chr19 \
  --n-bins 61 \
  --impute-bins 62 \
  --pad 1 --std 1 --rp 0.5 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "===== EVALUATE ====="
"${PYTHON}" 06_evaluate_simu.py \
  --output-root "${OUTPUT_ROOT}" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Finished at: $(date)"
