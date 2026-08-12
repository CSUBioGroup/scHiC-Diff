#!/usr/bin/env bash
#SBATCH -J build_supertad
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=scripts/test_outputs/build_supertad_linux_%j.out
#SBATCH --error=scripts/test_outputs/build_supertad_linux_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
COMMIT=d217a98aca51c1490e59846dfb9daa6b61a8265e
SOURCE_DIR=${ROOT}/6_SuperTAD/SuperTAD
BIN_DIR=${ROOT}/bin
GXX=/public/software/gcc-12.2.0/bin/g++
VALIDATION_DIR=${ROOT}/scripts/test_outputs/supertad_linux_validation_${SLURM_JOB_ID}
TMP_BIN=${BIN_DIR}/.SuperTAD_linux_x86_64.${SLURM_JOB_ID}.tmp
FINAL_BIN=${BIN_DIR}/SuperTAD_linux_x86_64
PROVENANCE=${BIN_DIR}/SUPERTAD_PROVENANCE.md

mkdir -p "${BIN_DIR}" "${VALIDATION_DIR}"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    echo "Local SuperTAD source repository is missing: ${SOURCE_DIR}" >&2
    exit 1
fi

ACTUAL_COMMIT=$(git -C "${SOURCE_DIR}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${COMMIT}" ]]; then
    echo "Unexpected source commit: ${ACTUAL_COMMIT}" >&2
    exit 1
fi
if ! git -C "${SOURCE_DIR}" diff --quiet -- '*.cpp' '*.h' CMakeLists.txt; then
    echo "Tracked SuperTAD source files differ from commit ${COMMIT}" >&2
    exit 1
fi

"${GXX}" --version
cd "${SOURCE_DIR}"
"${GXX}" -std=gnu++17 -O3 ./*.cpp -o "${TMP_BIN}"

file "${TMP_BIN}" | tee "${VALIDATION_DIR}/file.txt"
if ! file "${TMP_BIN}" | grep -q 'ELF 64-bit.*x86-64'; then
    echo "Compiled binary is not ELF x86-64" >&2
    exit 1
fi

set +e
"${TMP_BIN}" >"${VALIDATION_DIR}/help.txt" 2>&1
HELP_RC=$?
set -e
echo "help_exit_code=${HELP_RC}"
if [[ "${HELP_RC}" -ne 1 ]]; then
    echo "Unexpected no-argument usage exit code: ${HELP_RC}" >&2
    exit 1
fi
grep -q '2.0' "${VALIDATION_DIR}/help.txt"
grep -q 'multi' "${VALIDATION_DIR}/help.txt"
grep -q -- '-h' "${VALIDATION_DIR}/help.txt"
grep -q -- '--chrom1' "${VALIDATION_DIR}/help.txt"
grep -q -- '-r' "${VALIDATION_DIR}/help.txt"

cp "${ROOT}/target/Endo_target.txt" "${VALIDATION_DIR}/Endo_target.txt"
cd "${VALIDATION_DIR}"
"${TMP_BIN}" multi Endo_target.txt --chrom1 chr4 -r 10000 -h 3 \
    >run_target.stdout.txt 2>run_target.stderr.txt

NEW_TSV=${VALIDATION_DIR}/Endo_target.txt.multi.tsv
OLD_TSV=${ROOT}/supertad/target/Endo_target.tsv
if [[ ! -s "${NEW_TSV}" ]]; then
    echo "SuperTAD did not produce a nonempty target TSV" >&2
    exit 1
fi
awk -F '\t' 'NF != 8 {bad=1} END {exit bad}' "${NEW_TSV}"
awk -F '\t' '{print $2 "\t" $6}' "${NEW_TSV}" | sort -u \
    >"${VALIDATION_DIR}/linux_domains.tsv"
awk -F '\t' '{print $2 "\t" $6}' "${OLD_TSV}" | sort -u \
    >"${VALIDATION_DIR}/macos_domains.tsv"
diff -u "${VALIDATION_DIR}/macos_domains.tsv" \
    "${VALIDATION_DIR}/linux_domains.tsv" \
    >"${VALIDATION_DIR}/domain_comparison.diff"

mv "${TMP_BIN}" "${FINAL_BIN}"
sha256sum "${FINAL_BIN}" >"${VALIDATION_DIR}/sha256.txt"
git -C "${SOURCE_DIR}" show -s --format=fuller HEAD \
    >"${VALIDATION_DIR}/source_commit.txt"
git -C "${SOURCE_DIR}" status --short \
    >"${VALIDATION_DIR}/source_worktree_status.txt"
"${GXX}" --version >"${VALIDATION_DIR}/compiler_version.txt"
date --iso-8601=seconds >"${VALIDATION_DIR}/build_date.txt"

REPOSITORY=$(git -C "${SOURCE_DIR}" remote get-url origin)
BUILD_DATE=$(cat "${VALIDATION_DIR}/build_date.txt")
SHA256=$(cut -d ' ' -f 1 "${VALIDATION_DIR}/sha256.txt")
{
    printf '# Linux SuperTAD Provenance\n\n'
    printf -- '- Source directory: `%s`\n' "${SOURCE_DIR}"
    printf -- '- Repository: `%s`\n' "${REPOSITORY}"
    printf -- '- Commit: `%s`\n' "${ACTUAL_COMMIT}"
    printf -- '- Compiler: `%s`\n' "${GXX}"
    printf -- '- Build date: `%s`\n' "${BUILD_DATE}"
    printf -- '- Binary: `%s`\n' "${FINAL_BIN}"
    printf -- '- SHA-256: `%s`\n' "${SHA256}"
    printf -- '- Validation directory: `%s`\n' "${VALIDATION_DIR}"
    printf -- '- Validation: ELF x86-64, v2.0 CLI, eight-column target output, and exact target domain-pair agreement with the preserved macOS result.\n'
} >"${PROVENANCE}"

echo "Built and validated ${FINAL_BIN}"
