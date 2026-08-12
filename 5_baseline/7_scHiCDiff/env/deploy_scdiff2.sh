#!/usr/bin/env bash
#============================================================
# scdiff2 环境一键部署脚本 (方案 A: conda + pip 重建)
#------------------------------------------------------------
# 源环境: /public/home/hpc254701055/micromamba/envs/scdiff2
#   Python 3.9.23 | torch 1.12.1+cu116 | pytorch-lightning 1.9.0
#   cudatoolkit 11.6 | cuml 23.08 (rapidsai)
#
# 用法:
#   bash deploy_scdiff2.sh                  # 默认: 当前用户的 micromamba
#   MAMBA_ROOT_PREFIX=/opt/micromamba bash deploy_scdiff2.sh
#   ENV_NAME=myenv bash deploy_scdiff2.sh   # 自定义环境名
#
# 前置要求:
#   - Linux x86_64, glibc >= 2.17
#   - NVIDIA GPU + CUDA 11.6 兼容驱动 (cuml/cudf 需要 GPU)
#   - 若无 GPU, 设置 NO_GPU=1 跳过 rapidsai 层
#============================================================
set -euo pipefail

# ---------- 配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-scdiff2}"
MAMBA="${MAMBA:-micromamba}"
ENV_YML="${SCRIPT_DIR}/scdiff2_environment.yml"
REQ_TXT="${SCRIPT_DIR}/scdiff2_requirements.txt"
NO_GPU="${NO_GPU:-0}"

echo "================================================"
echo "  scdiff2 环境部署"
echo "  环境名:    ${ENV_NAME}"
echo "  清单目录:  ${SCRIPT_DIR}"
echo "  NO_GPU:    ${NO_GPU}"
echo "================================================"

# ---------- 1. 检查 micromamba ----------
if ! command -v "${MAMBA}" >/dev/null 2>&1; then
    echo "[1/5] micromamba 未找到, 正在安装到 \$HOME/.local/bin ..."
    mkdir -p "${HOME}/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "${HOME}/.local" bin/micromamba
    MAMBA="${HOME}/.local/bin/micromamba"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
    echo "  -> 安装完成: ${MAMBA}"
else
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname $(dirname $(${MAMBA} --prefix 2>/dev/null || echo ${MAMBA})))}"
    echo "[1/5] 使用已有 micromamba: $(${MAMBA} --version 2>/dev/null || echo ${MAMBA})"
fi

# ---------- 2. 配置国内镜像 (加速 conda) ----------
echo "[2/5] 配置 conda 镜像 (清华) ..."
CONDARC="${HOME}/.condarc"
if [ ! -f "${CONDARC}" ] || ! grep -q "mirrors.tuna" "${CONDARC}" 2>/dev/null; then
    cat > "${CONDARC}" <<'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/rapidsai
channel_priority: strict
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
EOF
    echo "  -> 已写入 ${CONDARC}"
else
    echo "  -> 已存在 .condarc, 跳过"
fi

# ---------- 3. 创建 conda 环境 ----------
echo "[3/5] 创建 conda 环境 (python 3.9 + cudatoolkit 11.6 + cuml 23.08) ..."

if [ "${NO_GPU}" = "1" ]; then
    echo "  [NO_GPU=1] 跳过 rapidsai 的 cuml/cudf, 仅装 CPU 可用部分"
    # 生成无 GPU 版的临时 environment.yml
    TMP_YML="${SCRIPT_DIR}/.scdiff2_nogpu.yml"
    grep -vE "cuml|cudf|cupy|dask-cuda|dask-cudf|raft|rmm|libcud|libcufft|libcublas|libcucu|libcurand|libcusparse|libcufile|nvcomp|nccl|nvidia" \
        "${ENV_YML}" > "${TMP_YML}" || true
    ENV_YML="${TMP_YML}"
    "${MAMBA}" env create -n "${ENV_NAME}" -f "${ENV_YML}" -y
    rm -f "${TMP_YML}"
else
    "${MAMBA}" env create -n "${ENV_NAME}" -f "${ENV_YML}" -y
fi

# ---------- 4. 激活环境并装 pip 包 ----------
echo "[4/5] 安装 pip 依赖 (含 torch 1.12.1+cu116) ..."

# shellcheck disable=SC1091
eval "$(${MAMBA} shell hook --shell bash)"
"${MAMBA}" activate "${ENV_NAME}"

# 4a. torch 系列: 需 PyTorch 官方 cu116 index (+cu116 后缀不在默认 PyPI)
echo "  [4a] torch 1.12.1+cu116 ..."
pip install --index-url https://download.pytorch.org/whl/cu116 \
    torch==1.12.1+cu116 \
    torchvision==0.13.1+cu116 \
    torchaudio==0.12.1+cu116

# 4b. 其余 pip 包: 默认 PyPI (加清华镜像加速)
echo "  [4b] 其余 pip 包 ..."
pip install -r "${REQ_TXT}" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu116

# ---------- 5. 验证 ----------
echo "[5/5] 验证关键包 ..."
python - <<'PYEOF'
import sys, importlib
print(f"Python: {sys.version.split()[0]}")
checks = [
    ("torch", "1.12.1+cu116"),
    ("torchvision", "0.13.1+cu116"),
    ("pytorch_lightning", "1.9.0"),
    ("anndata", None),
    ("scanpy", None),
    ("einops", None),
    ("omegaconf", None),
    ("wandb", None),
]
ok = 0
for mod, want in checks:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        flag = "OK" if (want is None or v == want) else f"MISMATCH(want {want})"
        print(f"  {mod:25s} {v:15s} {flag}")
        if want is None or v == want: ok += 1
    except Exception as e:
        print(f"  {mod:25s} FAIL: {e}")
print(f"\n{ok}/{len(checks)} 关键包通过")
PYEOF

echo ""
echo "================================================"
echo "  部署完成!"
echo "  激活: ${MAMBA} activate ${ENV_NAME}"
echo "  python: $(which python)"
echo "================================================"