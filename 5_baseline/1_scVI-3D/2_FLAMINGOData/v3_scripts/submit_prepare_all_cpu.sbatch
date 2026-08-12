#!/usr/bin/env bash
#SBATCH -J scvi3d_prep_all
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/prepare_all_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/prepare_all_%j.err

# Submit:
#   sbatch /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/submit_prepare_all_cpu.sbatch
#
# Prepare all 7 FLAMINGO datasets for scVI-3D input in one job.
# Each dataset has 1500 cells (500x500 dense matrix per cell).
# 80 cores = ~11 cores per dataset if parallel, but we process sequentially
# with 80 workers per dataset for maximum throughput.

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

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
"${PYTHON}" --version

for STEM in "${DATASETS[@]}"; do
    echo ""
    echo "===== PREPARE: ${STEM} ====="
    "${PYTHON}" "${PREPARE_PY}" \
        --raw-root "${RAW_ROOT}" \
        --input-root "${INPUT_ROOT}" \
        --datasets "${STEM}" \
        --workers "${SLURM_CPUS_PER_TASK}" \
        --overwrite
done

echo ""
echo "All 7 datasets prepared."
echo "Finished at: $(date)"