#!/usr/bin/env bash
#SBATCH --job-name=ramani_schicluster
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/scripts/logs/run_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/scripts/logs/run_%A_%a.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-22

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/scripts"
BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
HICLUSTER="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${BASE_DIR}/output/1_imputed_hdf5" "${BASE_DIR}/output/chrom_npz" "${BASE_DIR}/output/metrics"
export PATH="${ENV_DIR}/bin:${PATH}"

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
CHROM="${CHROMS[${TASK_ID}]}"

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Task: ${TASK_ID}  Chrom: ${CHROM}"
"${PYTHON}" --version

cd "${SCRIPT_DIR}"

echo "===== STEP 1: PREPARE ${CHROM} ====="
"${PYTHON}" 01_prepare_ramani.py \
  --chroms "${CHROM}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "===== STEP 2: IMPUTE ${CHROM} ====="
"${PYTHON}" 03_impute_ramani.py \
  --chroms "${CHROM}" \
  --hicluster "${HICLUSTER}" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "===== STEP 3: COLLECT ${CHROM} ====="
"${PYTHON}" 05_collect_ramani.py \
  --chroms "${CHROM}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "Finished at: $(date)"
