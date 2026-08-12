#!/usr/bin/env bash
#SBATCH -J tan_pd_lrtc
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline/flamingo_impute/output_pd_lee/logs/slurm_%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline/flamingo_impute/output_pd_lee/logs/slurm_%x_%A_%a.err

# =============================================================================
# PD-space FLAMINGO t-SVD ADMM for Tan data (Lee-style rerun)
# Array: 0-1 = segments 2050, 160190
# Each task: prep (contact->PD) -> FLAMINGO LRTC (mu=1e-4, selection=best) -> post (PD->contact)
# Based on the successful 5_lee_SuperTAD_pileline pipeline.
# =============================================================================

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline/flamingo_impute"
PIPELINE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline"
PYTHON="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
PD_SCRIPT="${BASE}/scripts/pd_flamingo_rerun.py"
LRTC_SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py"
MANIFEST="${BASE}/input/manifest.tsv"

# PD-space working dirs
PD_INPUT_ROOT="${BASE}/input_pd_lee"
PD_OUTPUT_ROOT="${BASE}/output_pd_lee"
FINAL_OUTPUT_DIR="${PIPELINE_DIR}/prepareData"

mkdir -p "${PD_OUTPUT_ROOT}/logs" "${PD_INPUT_ROOT}" "${FINAL_OUTPUT_DIR}"

SEGMENTS=("2050" "160190")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${SEGMENTS[${TASK_ID}]}"

echo "============================================================"
echo "Tan PD-space FLAMINGO (Lee-style)"
echo "  task_id=${TASK_ID}  dataset=${DATASET}"
echo "  node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "  start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- Step 1: prep (contact -> PD) ---
echo ""
echo "[Step 1/3] prep: contact -> PD"
"${PYTHON}" "${PD_SCRIPT}" prep \
    --dataset "${DATASET}" \
    --input-root "${BASE}/input" \
    --pd-input-root "${PD_INPUT_ROOT}" \
    --contact-subdir contact_matrices \
    --pd-subdir distance_matrices \
    --force

# --- Step 2: FLAMINGO t-SVD ADMM (mu=1e-4, selection=best, 500 iters) ---
echo ""
echo "[Step 2/3] FLAMINGO LRTC (mu=1e-4, selection=best, 500 iters)"
"${PYTHON}" "${LRTC_SCRIPT}" \
    --input-root "${PD_INPUT_ROOT}" \
    --input-subdir distance_matrices \
    --output-root "${PD_OUTPUT_ROOT}" \
    --datasets "${DATASET}" \
    --max-iter "${FLAMINGO_MAX_ITER:-500}" \
    --tol "${FLAMINGO_TOL:-1e-4}" \
    --mu "${FLAMINGO_MU:-1e-4}" \
    --max-mu "${FLAMINGO_MAX_MU:-1e10}" \
    --rho "${FLAMINGO_RHO:-1.1}" \
    --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
    --selection best \
    --keep-observed

# --- Step 3: post (PD -> contact -> unified NPZ for clustering) ---
echo ""
echo "[Step 3/3] post: PD -> contact -> FLAMINGO_PD_{seg}.npz"
"${PYTHON}" "${PD_SCRIPT}" post \
    --dataset "${DATASET}" \
    --pd-input-root "${PD_INPUT_ROOT}" \
    --output-root "${PD_OUTPUT_ROOT}" \
    --final-output-dir "${FINAL_OUTPUT_DIR}" \
    --output-format tan_unified_npz \
    --clip-factor 2.0

echo ""
echo "Done: ${DATASET} (task_id=${TASK_ID})"
echo "  Output: ${FINAL_OUTPUT_DIR}/FLAMINGO_PD_${DATASET}.npz"
echo "  end=$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"