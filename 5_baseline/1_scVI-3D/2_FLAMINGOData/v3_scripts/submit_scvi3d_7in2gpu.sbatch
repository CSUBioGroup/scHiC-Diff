#!/usr/bin/env bash
#SBATCH -J scvi3d_7in2gpu
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/scvi3d_7in2gpu_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/scvi3d_7in2gpu_%j.err

# ============================================================================
# scVI-3D: 7 datasets on 2 GPUs (4 on GPU0 + 3 on GPU1)
#
# Each scVI-3D task uses <1 GB VRAM. V100 has 32 GB per card.
# 20 CPUs / 7 tasks ≈ 3 CPUs per task (enough for data loading).
# 200 GB RAM / 7 tasks ≈ 28 GB per task.
#
# Submit:
#   sbatch v3_scripts/submit_scvi3d_7in2gpu.sbatch
# ============================================================================

set -uo pipefail  # NOT -e: subshell failures caught by wait loop
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_inputData"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_outputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"

IMPUTE_PY="${SCRIPT_DIR}/run_scvi3d_flamingo.py"
RESOLUTION=1000000
N_BINS=500
N_LATENT=100
BAND_MAX="whole"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}/npz_lower_tri" "${OUTPUT_ROOT}/impute_work"

DATASETS=(
    "v3_hybrid_W0p5_500cells_level0"
    "v3_hybrid_W0p6_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0_r0p01"
    "v3_hybrid_W0p7_500cells_level0_r0p05"
    "v3_hybrid_W0p8_500cells_level0"
    "v3_hybrid_W0p9_500cells_level0"
)

# GPU assignment: indices 0-3 -> GPU0, indices 4-6 -> GPU1
GPU_MAP=(0 0 0 0 1 1 1)

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "GPUs: 2 × V100 (32GB each), 7 tasks split 4+3, ~5 CPUs/task"
"${PYTHON}" --version

pids=()
names=()

for i in "${!DATASETS[@]}"; do
    STEM="${DATASETS[$i]}"
    GPU_ID="${GPU_MAP[$i]}"
    IN_DIR="${INPUT_ROOT}/${STEM}"
    WORK_DIR="${OUTPUT_ROOT}/impute_work/${STEM}"
    NPZ_OUT="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_scVI3D_imputed.npz"
    LOG_FILE="${SCRIPT_DIR}/logs/scvi3d_7in2gpu_${STEM}.log"

    if [[ -f "${NPZ_OUT}" ]]; then
        echo "[SKIP] ${STEM} — ${NPZ_OUT} exists"
        continue
    fi

    mkdir -p "${WORK_DIR}"

    echo "[LAUNCH] ${STEM} -> GPU${GPU_ID}"
    (
        set -e
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        "${PYTHON}" "${IMPUTE_PY}" \
            --input-dir "${IN_DIR}" \
            --output-dir "${WORK_DIR}" \
            --collect-dir "${OUTPUT_ROOT}/npz_lower_tri" \
            --stem "${STEM}" \
            --resolution "${RESOLUTION}" \
            --n-bins "${N_BINS}" \
            --band-max "${BAND_MAX}" \
            --n-latent "${N_LATENT}" \
            --gpu --overwrite \
            > "${LOG_FILE}" 2>&1
        touch "${WORK_DIR}/done.flag"
    ) &
    pids+=($!)
    names+=("${STEM}")
done

echo ""
echo "Launched ${#pids[@]} parallel tasks"
echo "GPU0: ${DATASETS[0]}, ${DATASETS[1]}, ${DATASETS[2]}, ${DATASETS[3]}"
echo "GPU1: ${DATASETS[4]}, ${DATASETS[5]}, ${DATASETS[6]}"
echo ""

# Wait for each PID; report failures
fail=0
for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || {
        echo "[FAIL] ${names[$j]} (exit $?)"
        fail=1
    }
    if [[ -f "${OUTPUT_ROOT}/impute_work/${names[$j]}/done.flag" ]]; then
        echo "[DONE] ${names[$j]}"
    else
        echo "[FAIL] ${names[$j]} — no done.flag"
        fail=1
    fi
done

echo ""
if [[ $fail -eq 0 ]]; then
    echo "ALL 7 TASKS COMPLETED"
else
    echo "SOME TASKS FAILED (check logs)"
fi
echo "Finished at: $(date)"
[[ $fail -eq 0 ]] || exit 1