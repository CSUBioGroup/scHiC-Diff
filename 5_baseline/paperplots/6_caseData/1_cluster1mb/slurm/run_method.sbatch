#!/usr/bin/env bash
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --account=pi_limin_r
#SBATCH --job-name=hires1mb_method
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/method_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/method_%j.err

set -euo pipefail

: "${METHOD:?Submit with sbatch --export=ALL,METHOD=Raw run_method.sbatch}"

TEST_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/monae_env/bin/Rscript"
MANIFEST="${TEST_ROOT}/config/methods.tsv"
LABELS="${TEST_ROOT}/config/cell_metadata.csv"
OUTPUT_DIR="${TEST_ROOT}/outputs/${METHOD}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUTPUT_DIR}/logs" "${TEST_ROOT}/slurm_logs"

echo "[$(date --iso-8601=seconds)] Python SVD stage: ${METHOD}"
"${PYTHON}" "${TEST_ROOT}/src/prepare_and_svd.py" \
  --manifest "${MANIFEST}" \
  --method "${METHOD}" \
  --labels "${LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --dim 20 \
  --silhouette-dims 15 \
  --random-state none \
  >"${OUTPUT_DIR}/logs/python_prepare_and_svd.log" 2>&1

echo "[$(date --iso-8601=seconds)] Seurat stage: ${METHOD}"
"${RSCRIPT}" "${TEST_ROOT}/src/seurat_cluster_plot.R" \
  --embedding "${OUTPUT_DIR}/decomp/total_decomp.tsv.gz" \
  --labels "${LABELS}" \
  --output-dir "${OUTPUT_DIR}/seurat" \
  --method "${METHOD}" \
  >"${OUTPUT_DIR}/logs/r_seurat_cluster_plot.log" 2>&1

echo "[$(date --iso-8601=seconds)] Completed: ${METHOD}"
