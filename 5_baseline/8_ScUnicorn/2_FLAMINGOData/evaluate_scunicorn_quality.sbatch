#!/usr/bin/env bash
#SBATCH -J scuni_flam_eval
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"

"${PYTHON_BIN}" "${BASE_DIR}/scripts/evaluate_scunicorn_quality.py" \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --output-root "${BASE_DIR}/output" \
  --metrics-dir "${BASE_DIR}/output/metrics"

echo "== requested metrics =="
head -n 12 "${BASE_DIR}/output/metrics/scunicorn_flamingo_requested_pcc_spearman.csv"
