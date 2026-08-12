#!/usr/bin/env bash
#SBATCH -J flmg_v3_h5ad_contact
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=96:00:00
#SBATCH --array=0-6
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/logs/slurm_%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/logs/slurm_%x_%A_%a.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
PREP_EVAL="${BASE_DIR}/scripts/v3_h5ad_contact_lrtc.py"
LRTC_SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py"

H5AD_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
INPUT_PARENT="${BASE_DIR}/v3ContactInput"
OUTPUT_PARENT="${BASE_DIR}/v3ContactOutput"
MANIFEST="${INPUT_PARENT}/manifest.tsv"

mkdir -p "${INPUT_PARENT}" "${OUTPUT_PARENT}/logs"

if [[ ! -s "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${PREP_EVAL}" write-manifest \
    --h5ad-root "${H5AD_ROOT}" \
    --manifest "${MANIFEST}" \
    --observed-layer counts \
    --gt-layer gt
fi

DATASET="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $1}' "${MANIFEST}")"
H5AD="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $3}' "${MANIFEST}")"
if [[ -z "${DATASET}" || -z "${H5AD}" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_PARENT}/${DATASET}"
mkdir -p "${OUTPUT_DIR}/logs"

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
  echo "h5ad=${H5AD}"
  echo "scale=raw_contact"
} | tee "${OUTPUT_DIR}/process_time.tsv"

"${PYTHON_BIN}" "${PREP_EVAL}" prep \
  --manifest "${MANIFEST}" \
  --input-parent "${INPUT_PARENT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"

"${PYTHON_BIN}" "${LRTC_SCRIPT}" \
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

"${PYTHON_BIN}" "${PREP_EVAL}" eval \
  --manifest "${MANIFEST}" \
  --input-parent "${INPUT_PARENT}" \
  --output-parent "${OUTPUT_PARENT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"

END_EPOCH="$(date +%s)"
END_ISO="$(date -Is)"
{
  echo "end_epoch=${END_EPOCH}"
  echo "end_iso=${END_ISO}"
  echo "elapsed_seconds=$((END_EPOCH - START_EPOCH))"
} | tee -a "${OUTPUT_DIR}/process_time.tsv"
