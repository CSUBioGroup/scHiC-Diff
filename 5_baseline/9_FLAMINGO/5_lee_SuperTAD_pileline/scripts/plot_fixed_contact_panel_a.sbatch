#!/usr/bin/env bash
#SBATCH -J flam_panel_a
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=scripts/test_outputs/flamingo_fixed_panel_a_%j.out
#SBATCH --error=scripts/test_outputs/flamingo_fixed_panel_a_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
PYTHON=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python

mkdir -p "${MPLCONFIGDIR}"
cd "${ROOT}"
"${PYTHON}" nature-style-plots/plot_fixed_contact_panel_a.py \
    --method FLAMINGO_fixed_contact \
    --display-name T-FLAMINGO \
    --norm q99 \
    --dpi 600 \
    --outdir nature-style-plots/results_fixed
