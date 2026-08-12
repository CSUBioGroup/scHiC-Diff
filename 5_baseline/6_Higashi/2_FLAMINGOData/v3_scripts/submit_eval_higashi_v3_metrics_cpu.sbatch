#!/usr/bin/env bash
#SBATCH -J higashi_v3_metrics
#SBATCH -A pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=20
#SBATCH --time=UNLIMITED
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_outputData/metrics/higashi_v3_metrics_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_outputData/metrics/higashi_v3_metrics_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_scripts/13_cal_FLAMINGO_Baseline_metrics.py \
  --higashi-v3-pred-dir /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_outputData/npz_lower_tri \
  --higashi-v3-gt-dir /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets \
  --higashi-v3-output-csv /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_outputData/metrics/higashi_FLAMINGO_v3_paramsweep_raw_gt_PCC_MAE.csv
