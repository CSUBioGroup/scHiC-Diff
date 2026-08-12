#!/usr/bin/env bash
#SBATCH -J ramani_hig_u200_prep
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/u200_prep_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/u200_prep_%j.err

set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"

cd "${BASE}"
mkdir -p logs

echo "[START] prepare updates200 input host=$(hostname) time=$(date)"
"${PYTHON}" scripts/make_ramani_higashi_update_control.py \
  --training-updates 200 \
  --output-root input_updates200

"${PYTHON}" - <<'PY'
from pathlib import Path
import csv

tasks = Path('input_updates200/ramani_higashi_updates200_multi5_tasks.tsv')
rows = list(csv.DictReader(tasks.open(), delimiter='\t'))
missing = []
for row in rows:
    root = Path(row['dataset_root'])
    for rel in [
        'config.JSON',
        'data/ramani.chrom.sizes',
        'data/label_info.pickle',
        'temp/data.npy',
        'temp/weight.npy',
        'temp/chrom_start_end.npy',
        'temp/node_feats.hdf5',
        'temp/sparse_nondiag_adj_nbr_1.npy',
    ]:
        if not (root / rel).exists():
            missing.append((row['nbr'], row['chrom'], rel))

print(f'tasks={len(rows)} missing_inputs={len(missing)}')
if missing:
    for item in missing[:20]:
        print('missing', item)
    raise SystemExit(1)
PY

echo "[DONE] prepare updates200 input time=$(date)"
