#!/usr/bin/env bash
#SBATCH -J hires_eval_v2_diag
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/diagnostics/eval_v2_quality_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/diagnostics/eval_v2_quality_%j.err

set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
PYTHON="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
OUTPUT="$BASE/results/full/eval_v2/diagnostics/eval_v2_quality.json"

"$PYTHON" "$BASE/scripts/diagnose_eval_v2_outputs.py" \
    --base "$BASE" \
    --output "$OUTPUT"
