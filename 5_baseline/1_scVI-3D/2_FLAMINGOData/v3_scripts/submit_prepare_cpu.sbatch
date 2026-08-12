#!/usr/bin/env bash
#SBATCH --job-name=scvi3d_prepare
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/prepare_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/prepare_%A_%a.err
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-0

# Submit:
#   sbatch /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/submit_prepare_cpu.sbatch
#
# This script prepares scVI-3D input from raw FLAMINGO simulation data:
#   raw 500x500 dense txt -> per-cell contact txt (ceil to int) + genome.txt + cell_summary.txt

set -euo pipefail
export LC_ALL=C
export LANG=C
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts"
RAW_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/3_fixed_flamnigoGen"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_inputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"

PREPARE_PY="${SCRIPT_DIR}/prepare_scvi3d_input.py"

mkdir -p "${SCRIPT_DIR}/logs" "${INPUT_ROOT}"

DATASETS=(
    "v3_hybrid_W0p5_500cells_level0"
    "v3_hybrid_W0p6_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0_r0p01"
    "v3_hybrid_W0p7_500cells_level0_r0p05"
    "v3_hybrid_W0p8_500cells_level0"
    "v3_hybrid_W0p9_500cells_level0"
)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"

echo "Started at: $(date)"
echo "Host: $(hostname)  Task: ${TASK_ID}  Dataset: ${STEM}"
"${PYTHON}" --version

echo "===== PREPARE scVI-3D INPUT ====="
"${PYTHON}" "${PREPARE_PY}" \
    --raw-root "${RAW_ROOT}" \
    --input-root "${INPUT_ROOT}" \
    --datasets "${STEM}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --overwrite

echo "Finished at: $(date)"