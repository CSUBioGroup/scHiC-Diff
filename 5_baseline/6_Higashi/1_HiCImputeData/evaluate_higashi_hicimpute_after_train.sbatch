#!/usr/bin/env bash
#SBATCH -J higashi_hicimp_eval
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/logs/%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"

hostname
date

"${PYTHON_BIN}" "${BASE_DIR}/scripts/hicimpute_higashi_pipeline.py" verify-outputs \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --output-root "${BASE_DIR}/output" \
  --report "${BASE_DIR}/output/higashi_output_report.tsv" \
  --neighbors 0 5

"${PYTHON_BIN}" "${BASE_DIR}/scripts/calc_higashi_all_pearson_mae.py" \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --impute-root "${BASE_DIR}/output/npz_lower_tri" \
  --output-csv "${BASE_DIR}/output/metrics/higashi_hicimpute_ALL_Pearson_MAE.csv"

date
