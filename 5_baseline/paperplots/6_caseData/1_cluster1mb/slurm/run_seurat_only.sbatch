#!/usr/bin/env bash
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --account=pi_limin_r
#SBATCH --job-name=hires1mb_seurat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/seurat_only_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/seurat_only_%j.err

set -euo pipefail

: "${METHOD:?Submit with sbatch --export=ALL,METHOD=Raw run_seurat_only.sbatch}"

TEST_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/monae_env/bin/Rscript"
LABELS="${TEST_ROOT}/config/cell_metadata.csv"
OUTPUT_DIR="${TEST_ROOT}/outputs/${METHOD}"
EMBEDDING="${OUTPUT_DIR}/decomp/total_decomp.tsv.gz"

if [[ ! -f "${EMBEDDING}" ]]; then
  echo "Missing final SVD table: ${EMBEDDING}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs" "${TEST_ROOT}/slurm_logs"

"${RSCRIPT}" "${TEST_ROOT}/src/seurat_cluster_plot.R" \
  --embedding "${EMBEDDING}" \
  --labels "${LABELS}" \
  --output-dir "${OUTPUT_DIR}/seurat" \
  --method "${METHOD}" \
  >"${OUTPUT_DIR}/logs/r_seurat_cluster_plot.log" 2>&1

echo "[$(date --iso-8601=seconds)] Seurat-only run completed: ${METHOD}"
