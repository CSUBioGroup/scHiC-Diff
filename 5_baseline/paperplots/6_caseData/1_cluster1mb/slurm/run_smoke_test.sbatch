#!/usr/bin/env bash
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --account=pi_limin_r
#SBATCH --job-name=hires1mb_smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/smoke_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/smoke_%j.err

set -euo pipefail

TEST_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/monae_env/bin/Rscript"

echo "[$(date --iso-8601=seconds)] Python synthetic tests"
"${PYTHON}" -m unittest discover \
  -s "${TEST_ROOT}/tests" \
  -p "test_prepare_and_svd.py" \
  -v

echo "[$(date --iso-8601=seconds)] R/Seurat synthetic test"
"${RSCRIPT}" \
  "${TEST_ROOT}/tests/test_seurat_smoke.R" \
  "${TEST_ROOT}/src/seurat_cluster_plot.R"

echo "[$(date --iso-8601=seconds)] Raw real-data smoke test"
export METHOD=Raw
bash "${TEST_ROOT}/slurm/run_method.sbatch"

echo "[$(date --iso-8601=seconds)] Smoke test completed"
