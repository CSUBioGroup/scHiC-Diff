#!/usr/bin/env bash
#SBATCH -J scuni_flam_debug
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q,gpu4Q,gpu8Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
FOLD_NAME="fold_01"
CKPT="${BASE_DIR}/checkpoints_smoke/${FOLD_NAME}/scunicorn_${FOLD_NAME}_best.pytorch"
OUT_ROOT="${BASE_DIR}/output_debug"

rm -rf "${OUT_ROOT}/${FOLD_NAME}"
mkdir -p "${OUT_ROOT}"

hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

"${PYTHON_BIN}" "${BASE_DIR}/scripts/flamingo_scunicorn_pipeline.py" infer-fold \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --checkpoint "${CKPT}" \
  --output-root "${OUT_ROOT}" \
  --datasets-root "${BASE_DIR}/datasets_smoke" \
  --fold-name "${FOLD_NAME}" \
  --patch-size 40 \
  --batch-size 32 \
  --cuda 0

echo "== after infer shell =="
find "${OUT_ROOT}" -maxdepth 5 -type d -printf 'DIR %p\n' | sort
find "${OUT_ROOT}" -maxdepth 5 -type f -printf 'FILE %p %s\n' | sort

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from scipy import sparse
root = Path("/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/output_debug")
files = sorted(root.rglob("*.npz"))
print("python_file_count", len(files))
for path in files:
    print(path, path.stat().st_size)
    if path.name.endswith("_lower_tri.npz"):
        mat = sparse.load_npz(path)
        print("shape", mat.shape, "nnz", mat.nnz)
PY
