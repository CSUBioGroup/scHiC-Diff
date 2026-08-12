#!/usr/bin/env bash
#SBATCH -J flam_tad_audit
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=scripts/test_outputs/flamingo_supertad_audit_%j.out
#SBATCH --error=scripts/test_outputs/flamingo_supertad_audit_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
PYTHON=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python

cd "${ROOT}"
"${PYTHON}" scripts/audit_supertad_outputs.py \
    --method FLAMINGO_fixed_contact \
    --cell-types Astro Endo ODC OPC \
    --n-trials 100
