#!/usr/bin/env bash
#SBATCH -J ari_v12
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData/logs/calc_ari_v12_final_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData/logs/calc_ari_v12_final_%j.err

set -euo pipefail
/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData/scripts/calc_ari_v12_final.py
echo "ALL_DONE"
