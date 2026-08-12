#!/usr/bin/env bash
#SBATCH -J flmg_v3_h5ad_contact_merge
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/logs/slurm_%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/logs/slurm_%x_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
PREP_EVAL="${BASE_DIR}/scripts/v3_h5ad_contact_lrtc.py"
OUTPUT_PARENT="${BASE_DIR}/v3ContactOutput"

mkdir -p "${OUTPUT_PARENT}/logs"

"${PYTHON_BIN}" "${PREP_EVAL}" merge --output-parent "${OUTPUT_PARENT}"
