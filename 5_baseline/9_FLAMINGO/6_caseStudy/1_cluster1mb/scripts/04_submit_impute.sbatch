#!/usr/bin/env bash
#SBATCH --job-name=flam_1mb_imp
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/1_cluster1mb/scripts/logs/impute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/1_cluster1mb/scripts/logs/impute_%A_%a.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --array=0-19

# Parallel Tensor-FLAMINGO imputation for 1Mb case-study data.
# One chromosome per array task, 20 CPUs each.
#
# Prerequisites: run 02_submit_prep.sbatch first.
#
# Submit:
#   sbatch 04_submit_impute.sbatch
#
# After imputation, collect:
#   sbatch 06_submit_collect.sbatch

set -euo pipefail
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/1_cluster1mb"
PYTHON="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
LRTC_SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py"
WORK_ROOT="${CASE_DIR}/work"

# 20 chromosomes: chr1-19, chrX, chrY
CHROMS=("chr1"  "chr2"  "chr3"  "chr4"  "chr5"  "chr6"  "chr7"  "chr8"  "chr9"  "chr10"
        "chr11" "chr12" "chr13" "chr14" "chr15" "chr16" "chr17" "chr18" "chr19" "chrX")

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
CHROM="${CHROMS[${TASK_ID}]}"

INPUT_DIR="${WORK_ROOT}/input/${CHROM}/contact_matrices"
OUTPUT_DIR="${WORK_ROOT}/output/${CHROM}"
META="${WORK_ROOT}/input/${CHROM}/metadata.json"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Task: ${TASK_ID}  Chrom: ${CHROM}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=============================================="

if [ ! -f "${WORK_ROOT}/input/${CHROM}/.complete" ]; then
    echo "ERROR: ${CHROM} not prepared. Run 02_submit_prep.sbatch first."
    exit 1
fi

"${PYTHON}" --version
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${LRTC_SCRIPT}" \
  --input-root "${WORK_ROOT}/input" \
  --input-subdir "contact_matrices" \
  --output-root "${WORK_ROOT}/output" \
  --datasets "${CHROM}" \
  --max-iter 500 \
  --tol 1e-4 \
  --mu 1e-4 \
  --max-mu 1e10 \
  --rho 1.1 \
  --n-threads "${SLURM_CPUS_PER_TASK}" \
  --svd-backend serial \
  --selection best \
  --keep-observed \
  --patience 0 \
  --min-rel-improvement 0.0

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="