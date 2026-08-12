#!/usr/bin/env bash
#SBATCH -J scvi3d_eval
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/eval_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/eval_%j.err

# Submit:
#   sbatch /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/submit_eval_scvi3d_cpu.sbatch

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts"
PRED_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_outputData/npz_lower_tri"
GT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
OUTPUT_CSV="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_outputData/metrics/scVI3D_FLAMINGO_v3_whole_ceil_metrics.csv"
PYTHON="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"

mkdir -p "${SCRIPT_DIR}/logs" "$(dirname "${OUTPUT_CSV}")"

echo "Started at: $(date)"
echo "Host: $(hostname)"
"${PYTHON}" --version

"${PYTHON}" "${SCRIPT_DIR}/evaluate_scvi3d_flamingo.py" \
    --pred-dir "${PRED_DIR}" \
    --gt-dir "${GT_DIR}" \
    --output-csv "${OUTPUT_CSV}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --log1p

echo "Finished at: $(date)"
