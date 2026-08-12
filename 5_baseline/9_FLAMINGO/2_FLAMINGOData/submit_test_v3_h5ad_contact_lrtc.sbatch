#!/usr/bin/env bash
#SBATCH -J test_flmg_triu
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/logs/test_triu_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/logs/test_triu_%j.err

set -euo pipefail

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"

mkdir -p "${BASE_DIR}/v3ContactOutput/logs"
cd "${BASE_DIR}/scripts"
"${PYTHON_BIN}" -m unittest -v test_v3_h5ad_contact_lrtc.py
/public/software/anaconda3/bin/pytest -q "${BASE_DIR}/tests/test_v3_h5ad_contact_lrtc.py"
