#!/usr/bin/env bash
#SBATCH -J ramani_hig5
#SBATCH -A pi_limin_r
#SBATCH -p gpu4Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-22
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/ramani_hig5_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/ramani_hig5_%A_%a.err

set -euo pipefail
BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData"
PREP_PYTHON="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
HIGASHI_PYTHON="/public/home/hpc254701055/micromamba/envs/6_higashi/bin/python"
RUNNER="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/scripts/run_higashi_one.py"
MANIFEST="${BASE}/input/ramani_higashi_nbr5_manifest.tsv"

mkdir -p "${BASE}/logs" "${BASE}/output"
"${PREP_PYTHON}" "${BASE}/scripts/prepare_ramani_higashi_inputs.py" --neighbor-num 5 --cpu-num "${SLURM_CPUS_PER_TASK:-10}"
ROW=$(awk -v n="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==n {print}' "${MANIFEST}")
IFS=$'\t' read -r chrom neighbor_num config dataset_root n_cells n_features n_bins <<< "${ROW}"
"${HIGASHI_PYTHON}" "${RUNNER}" --config "${config}" --training-updates "${TRAINING_UPDATES:-1000}" --eval-updates "${EVAL_UPDATES:-10}"
