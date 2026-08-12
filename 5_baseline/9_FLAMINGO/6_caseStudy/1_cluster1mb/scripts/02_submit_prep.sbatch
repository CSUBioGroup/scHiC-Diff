#!/usr/bin/env bash
#SBATCH --job-name=flam_1mb_prep
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/1_cluster1mb/scripts/logs/prep_%j.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/1_cluster1mb/scripts/logs/prep_%j.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=06:00:00

# Prepare all chromosome h5ad files into RawCount_Cell_NNN.txt for FLAMINGO.
# This is a single job (not array) because prep is I/O bound and fast.
#
# Submit:
#   sbatch 02_submit_prep.sbatch

set -euo pipefail
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1

CASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/1_cluster1mb"
PYTHON="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/7_caseData/1_cluster1mb/inputData/6_hires_h5ad_1mb"

mkdir -p "${CASE_DIR}/scripts/logs" "${CASE_DIR}/work"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "=============================================="

"${PYTHON}" --version
"${PYTHON}" "${CASE_DIR}/scripts/01_prep_h5ad_to_flamingo.py" \
  --input-root "${INPUT_ROOT}" \
  --work-root "${CASE_DIR}/work" \
  --force

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="