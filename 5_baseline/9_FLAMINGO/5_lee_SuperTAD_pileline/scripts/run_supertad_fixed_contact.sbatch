#!/usr/bin/env bash
#SBATCH -J flam_contact_tads
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/test_outputs/flamingo_contact_supertad_%j.out
#SBATCH --error=scripts/test_outputs/flamingo_contact_supertad_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
PYTHON=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python
SUPERTAD=${ROOT}/bin/SuperTAD_linux_x86_64
TARGET_VALIDATION=${ROOT}/scripts/test_outputs/supertad_linux_all_targets_${SLURM_JOB_ID}
METHOD=FLAMINGO_fixed_contact

mkdir -p "${TARGET_VALIDATION}"

for CELL_TYPE in Astro Endo ODC OPC; do
    cp "${ROOT}/target/${CELL_TYPE}_target.txt" "${TARGET_VALIDATION}/"
    cd "${TARGET_VALIDATION}"
    "${SUPERTAD}" multi "${CELL_TYPE}_target.txt" \
        --chrom1 chr4 -r 10000 -h 3 \
        >"${CELL_TYPE}.stdout.txt" 2>"${CELL_TYPE}.stderr.txt"

    NEW_TSV=${TARGET_VALIDATION}/${CELL_TYPE}_target.txt.multi.tsv
    OLD_TSV=${ROOT}/supertad/target/${CELL_TYPE}_target.tsv
    if [[ ! -s "${NEW_TSV}" ]]; then
        echo "Missing or empty Linux target output: ${NEW_TSV}" >&2
        exit 1
    fi
    awk -F '\t' 'NF != 8 {bad=1} END {exit bad}' "${NEW_TSV}"
    awk -F '\t' '{print $2 "\t" $6}' "${NEW_TSV}" | sort -u \
        >"${TARGET_VALIDATION}/${CELL_TYPE}_linux_domains.tsv"
    awk -F '\t' '{print $2 "\t" $6}' "${OLD_TSV}" | sort -u \
        >"${TARGET_VALIDATION}/${CELL_TYPE}_macos_domains.tsv"
    diff -u "${TARGET_VALIDATION}/${CELL_TYPE}_macos_domains.tsv" \
        "${TARGET_VALIDATION}/${CELL_TYPE}_linux_domains.tsv" \
        >"${TARGET_VALIDATION}/${CELL_TYPE}_domain_comparison.diff"
done

cd "${ROOT}"
"${PYTHON}" run_supertad.py --method "${METHOD}"

SUMMARY=${ROOT}/supertad/${METHOD}/summary.json
grep -q '"total_errors": 0' "${SUMMARY}"
N_TSV=$(find "${ROOT}/supertad/${METHOD}/trials" -maxdepth 1 \
    -type f -name '*_trial*.tsv' -size +0c | wc -l)
if [[ "${N_TSV}" -ne 400 ]]; then
    echo "Expected 400 nonempty trial TSVs, found ${N_TSV}" >&2
    exit 1
fi

echo "Validated four Linux targets and ${N_TSV} corrected trial TSVs."
