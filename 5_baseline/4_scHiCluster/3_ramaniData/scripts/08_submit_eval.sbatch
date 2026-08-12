#!/usr/bin/env bash
#SBATCH --job-name=ramani_eval
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/scripts/logs/eval_%A.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/scripts/logs/eval_%A.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/scripts"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/3_ramaniData/output"
PYTHON="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/hic-impute"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}/metrics"
export PATH="${ENV_DIR}/bin:${PATH}"

echo "Started at: $(date)"
echo "Host: $(hostname)"
"${PYTHON}" --version

cd "${SCRIPT_DIR}"

echo "===== MERGE EMBEDDING + EVALUATE ARI ====="
"${PYTHON}" /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/scripts/ramani_cluster_eval.py \
  --chrom-dir "${OUTPUT_ROOT}/chrom_npz" \
  --method-name "scHiCluster" \
  --output-dir "${OUTPUT_ROOT}" \
  --svd-seed 100 \
  --svd-dim 5 \
  --umap-seed 500 \
  --kmeans-n-init 100 \
  --kmeans-seed 100

echo "Finished at: $(date)"
