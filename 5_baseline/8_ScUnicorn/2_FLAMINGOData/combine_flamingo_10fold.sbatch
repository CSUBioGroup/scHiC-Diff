#!/usr/bin/env bash
#SBATCH -J scuni_flam_combine
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"

"${PYTHON_BIN}" "${BASE_DIR}/scripts/flamingo_scunicorn_pipeline.py" combine-folds \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --output-root "${BASE_DIR}/output" \
  --split-root "${BASE_DIR}/splits" \
  --n-folds 10

find "${BASE_DIR}/output/combined_10fold/npz_lower_tri" -maxdepth 1 -type f -name '*.npz' | wc -l
