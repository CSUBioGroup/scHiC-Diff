#!/bin/bash

# ==============================================================================
# 本地测试脚本 - 使用离线预处理数据
# 
# 用途: 在本地环境测试使用预处理数据的训练流程
# 使用方法:
#   bash run/local_run_hires_preprocessed.sh [SEED] [CHROMOSOMES...]
# 
# 示例:
#   bash run/local_run_hires_preprocessed.sh 10 chr1
#   bash run/local_run_hires_preprocessed.sh 10 chr1 chr2 chr3
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 打印作业信息
# ==============================================================================
echo "================================================="
echo "Local training started on: $(date)"
echo "Running on: $(hostname)"
echo "Using PREPROCESSED data for faster initialization"
echo "================================================="

# ==============================================================================
# 环境设置
# ==============================================================================
# 自动检测 Python 环境
if command -v python &> /dev/null; then
    PYTHON_EXEC="python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
else
    echo "ERROR: Python not found in PATH"
    exit 1
fi

echo "Using python: ${PYTHON_EXEC}"
${PYTHON_EXEC} --version
echo "-------------------------------------------------"

# ==============================================================================
# 设置项目主目录
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOMEDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$HOMEDIR"
echo "Changed directory to project home: $HOMEDIR"

# ==============================================================================
# 定义静态参数
# ==============================================================================
CONFIG_PREFIX="configs"
NAME="scHiC_v1.2_preprocessed"
LOSS_STRATEGY="recon_masked"
OFFLINE_SETTINGS="--wandb_offline t"
TEST_FLAG=${TEST_FLAG:-0}
RESOLUTION="1Mb"

# 预处理数据目录
PREPROCESSED_DIR="data/h5ad_final_1000000/preprocessed"

# ==============================================================================
# 解析命令行参数
# ==============================================================================
SEED="${1:-10}"
shift || true  # 允许没有额外参数

CHROMOSOMES=("$@") 

if [ ${#CHROMOSOMES[@]} -eq 0 ]; then
    CHROMOSOMES=("chr1")
    echo "No chromosomes provided, using default: ${CHROMOSOMES[*]}"
fi

echo "Running with SEED: $SEED"
echo "Processing CHROMOSOMES: ${CHROMOSOMES[*]}"
echo "Using preprocessed data from: ${PREPROCESSED_DIR}"
echo "-------------------------------------------------"

# ==============================================================================
# 主执行逻辑
# ==============================================================================
for chrom in "${CHROMOSOMES[@]}"; do
    echo "==================== Processing: $chrom ===================="
    save_path="${HOMEDIR}/results/0_Hires_impute_preprocessed/${RESOLUTION}/${chrom}"
    mkdir -p "$save_path"
    echo "Ensured output directory exists: $save_path"

    # --- 准备预处理数据参数 ---
    dataset_name="hires"
    data_prefix="${RESOLUTION}_${chrom}"
    
    # 预处理文件名
    preprocessed_fname="hires_${data_prefix}_preprocessed.h5ad"
    
    # 构建预处理文件的绝对路径
    absolute_data_fname="${HOMEDIR}/${PREPROCESSED_DIR}/${preprocessed_fname}"
    
    # 检查预处理文件是否存在
    if [ ! -f "${absolute_data_fname}" ]; then
        echo "ERROR: Preprocessed file not found: ${absolute_data_fname}"
        echo ""
        echo "Please run offline preprocessing first:"
        echo "  cd ${HOMEDIR}"
        echo "  bash tools/1_offline_preprocess.sh"
        echo ""
        echo "Or preprocess this specific file:"
        original_fname="hires_${data_prefix}.h5ad"
        original_path="${HOMEDIR}/data/h5ad_final_1000000/${original_fname}"
        echo "  python tools/offline_preprocess.py \\"
        echo "    --input ${original_path} \\"
        echo "    --output ${absolute_data_fname} \\"
        echo "    --seed ${SEED}"
        exit 1
    fi

    echo "Using PREPROCESSED data file: ${absolute_data_fname}"

    # 数据参数
    data_args=(
        "data.params.train.params.dataset=${dataset_name}"
        "data.params.train.params.fname=${absolute_data_fname}"
        "data.params.validation.params.dataset=${dataset_name}"
        "data.params.validation.params.fname=${absolute_data_fname}"
        "data.params.test.params.dataset=${dataset_name}"
        "data.params.test.params.fname=${absolute_data_fname}"
    )

    # 构建完整命令
    cmd=(
        "${PYTHON_EXEC}" main.py
        -b "${CONFIG_PREFIX}/${LOSS_STRATEGY}.yaml"
        --name "$NAME"
        --seed "$SEED"
        --save_path "$save_path"
        --logdir "logs"
        --postfix "seed${SEED}_preprocessed"
        ${OFFLINE_SETTINGS}
        "${data_args[@]}"
    )
    
    # 执行命令
    echo "Executing command for ${chrom}:"
    printf "  %s\n" "${cmd[@]}"
    echo "-------------------------------------------------"

    if [[ $TEST_FLAG == 0 ]]; then
        "${cmd[@]}"
    else
        echo "DRY RUN: TEST_FLAG is set to 1. Command was not executed."
    fi

    echo "Finished processing chromosome: $chrom"
done

# ==============================================================================
# 作业完成
# ==============================================================================
echo "================================================="
echo "All chromosomes processed. Job finished on: $(date)"
echo ""
echo "Performance Notes:"
echo "- Used preprocessed data for faster initialization"
echo "- Skipped normalization and log transformation"
echo "- Used consistent masks across train/valid/test"
echo "================================================="
