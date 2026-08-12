#!/usr/bin/env bash
#SBATCH -J hires_eval_v2_reassess
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/diagnostics/eval_v2_reassess_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/diagnostics/eval_v2_reassess_%j.err

set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
RESULT_DIR="${RESULT_DIR:-$BASE/results/corrected/eval_v2/smoke}"
PYTHON="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"

"$PYTHON" "$BASE/scripts/validate_outputs.py" \
    --result-dir "$RESULT_DIR" \
    --expected-rows 7466 \
    --expected-cols 5050

"$PYTHON" "$BASE/scripts/assess_imputation_quality.py" \
    --result-dir "$RESULT_DIR" \
    --train-log "$RESULT_DIR/train.log" \
    --expected-rows 7466 \
    --expected-cols 5050
