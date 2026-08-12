#!/usr/bin/env bash
#SBATCH -J flam_contact_gate
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=scripts/test_outputs/flamingo_contact_full_gate_%j.out
#SBATCH --error=scripts/test_outputs/flamingo_contact_full_gate_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
PYTHON=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python
EVALUATOR=${ROOT}/scripts/evaluate_flamingo_contact_maps.py
IMPUTED_DIR=${ROOT}/imputed/FLAMINGO_fixed_contact
METRIC_DIR=${ROOT}/scripts/test_outputs/contact_space_full_metrics

cd "${ROOT}"
"${PYTHON}" "${EVALUATOR}" \
    --cell-types Astro Endo ODC OPC \
    --imputed-dir "${IMPUTED_DIR}" \
    --observed-dir "${ROOT}/input_lee/per_cell_npz" \
    --target-dir "${ROOT}/target" \
    --output-dir "${METRIC_DIR}" \
    --n-sample 30 \
    --seed 42

awk -F ',' '
    NR > 1 {
        gsub(/\r/, "", $13)
        rows++
        if ($13 != "True") failed=1
    }
    END {exit (rows == 4 && !failed) ? 0 : 1}
' "${METRIC_DIR}/contact_metrics.csv"

echo "All four contact-space cell types passed the hard gate."
