#!/usr/bin/env bash
#SBATCH -J flam_all_tests
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=scripts/test_outputs/flamingo_contact_only_tests_%j.out
#SBATCH --error=scripts/test_outputs/flamingo_contact_only_tests_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
PYTHON=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python

cd "${ROOT}"
"${PYTHON}" -m unittest discover -s scripts/tests -p 'test_*.py' -v
