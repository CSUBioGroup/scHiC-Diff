#!/usr/bin/env bash
#SBATCH -J ramani_scvi3d_collect
#SBATCH -A pi_limin_r
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/3_ramaniData/logs/ramani_scvi3d_collect_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/3_ramaniData/logs/ramani_scvi3d_collect_%j.err

set -euo pipefail
BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
mkdir -p "${BASE}/logs" "${BASE}/output"
"${PYTHON}" "${BASE}/scripts/collect_scvi3d_ramani_outputs.py" --make-embedding
