#!/usr/bin/env bash
#SBATCH -J ramani_flamingo
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=96:00:00
#SBATCH --array=0-22
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData/output/logs/slurm_%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData/output/logs/slurm_%x_%A_%a.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
PREP="${BASE}/scripts/prepare_ramani_flamingo_inputs.py"
RUNNER="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py"
INPUT_PARENT="${BASE}/input"
OUTPUT_PARENT="${BASE}/output"
MANIFEST="${INPUT_PARENT}/manifest.tsv"
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
DATASET="${CHROMS[${SLURM_ARRAY_TASK_ID:-0}]}"

mkdir -p "${INPUT_PARENT}" "${OUTPUT_PARENT}/logs" "${OUTPUT_PARENT}/${DATASET}"

START_EPOCH="$(date +%s)"
START_ISO="$(date -Is)"
{
  echo "job_id=${SLURM_JOB_ID:-NA}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-NA}"
  echo "node=$(hostname)"
  echo "cpus=${SLURM_CPUS_PER_TASK:-20}"
  echo "mem=50G"
  echo "start_epoch=${START_EPOCH}"
  echo "start_iso=${START_ISO}"
  echo "dataset=${DATASET}"
  echo "scale=raw_contact"
} | tee "${OUTPUT_PARENT}/${DATASET}/process_time.tsv"

"${PYTHON}" "${PREP}" \
  --input-root "${INPUT_PARENT}" \
  --manifest "${MANIFEST}" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}"

"${PYTHON}" "${RUNNER}" \
  --input-root "${INPUT_PARENT}" \
  --input-subdir contact_matrices \
  --output-root "${OUTPUT_PARENT}" \
  --datasets "${DATASET}" \
  --max-iter "${FLAMINGO_MAX_ITER:-500}" \
  --tol "${FLAMINGO_TOL:-1e-4}" \
  --mu "${FLAMINGO_MU:-1e-4}" \
  --max-mu "${FLAMINGO_MAX_MU:-1e10}" \
  --rho "${FLAMINGO_RHO:-1.1}" \
  --n-threads "${SLURM_CPUS_PER_TASK:-20}" \
  --selection final \
  --keep-observed

END_EPOCH="$(date +%s)"
END_ISO="$(date -Is)"
{
  echo "end_epoch=${END_EPOCH}"
  echo "end_iso=${END_ISO}"
  echo "elapsed_seconds=$((END_EPOCH - START_EPOCH))"
} | tee -a "${OUTPUT_PARENT}/${DATASET}/process_time.tsv"
