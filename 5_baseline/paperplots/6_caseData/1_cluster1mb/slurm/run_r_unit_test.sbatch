#!/usr/bin/env bash
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --account=pi_limin_r
#SBATCH --job-name=hires1mb_rtest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/r_unit_test_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb/slurm_logs/r_unit_test_%j.err

set -euo pipefail

TEST_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/1_cluster1mb"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/monae_env/bin/Rscript"

"${RSCRIPT}" \
  "${TEST_ROOT}/tests/test_seurat_smoke.R" \
  "${TEST_ROOT}/src/seurat_cluster_plot.R"
