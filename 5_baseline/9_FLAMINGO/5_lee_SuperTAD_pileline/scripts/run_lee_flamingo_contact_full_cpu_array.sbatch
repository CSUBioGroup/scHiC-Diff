#!/usr/bin/env bash
#SBATCH -J flam_contact_full
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=scripts/test_outputs/flamingo_contact_full_%A_%a.out
#SBATCH --error=scripts/test_outputs/flamingo_contact_full_%A_%a.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/5_lee_SuperTAD_pileline
PYTHON=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python
RUNNER=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
PIPELINE=${ROOT}/scripts/lee_flamingo_pipeline.py
WORK_ROOT=${ROOT}/work/FLAMINGO_fixed_contact
INPUT_ROOT=${WORK_ROOT}/input
OUTPUT_ROOT=${WORK_ROOT}/output
IMPUTED_DIR=${ROOT}/imputed/FLAMINGO_fixed_contact

CELL_TYPES=(Astro Endo ODC OPC)
TASK_ID=${SLURM_ARRAY_TASK_ID}
CELL_TYPE=${CELL_TYPES[${TASK_ID}]}

mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}" "${IMPUTED_DIR}"

echo "FLAMINGO contact-space full run"
echo "task_id=${TASK_ID}"
echo "cell_type=${CELL_TYPE}"
echo "node=$(hostname)"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"

"${PYTHON}" "${PIPELINE}" prep \
    --cell-type "${CELL_TYPE}" \
    --input-root "${INPUT_ROOT}" \
    --force

"${PYTHON}" "${RUNNER}" \
    --input-root "${INPUT_ROOT}" \
    --input-subdir contact_matrices \
    --output-root "${OUTPUT_ROOT}" \
    --datasets "${CELL_TYPE}" \
    --max-iter 500 \
    --tol 1e-4 \
    --mu 1e-4 \
    --max-mu 1e10 \
    --rho 1.1 \
    --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
    --svd-backend serial \
    --selection best \
    --patience 0 \
    --min-rel-improvement 0.0 \
    --keep-observed

"${PYTHON}" "${PIPELINE}" post \
    --cell-type "${CELL_TYPE}" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --imputed-dir "${IMPUTED_DIR}"

echo "end=$(date '+%Y-%m-%d %H:%M:%S')"
