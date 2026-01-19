#!/bin/bash

# Mask 一致性测试脚本
# 用途：测试离线预处理生成的mask与原始代码是否一致

set -e  # 遇到错误立即退出

# ============ 配置部分 ============
#PYTHON_EXEC="/home/duxuyan/micromamba/envs/scdiff/bin/python"
PYTHON_EXEC="/data1/zhanglinna/micromamba/envs/scdiff/bin/python"

#PROJECT_ROOT="/home/duxuyan/Projects/1_scHiC/3_DiffusionModel/scHiC-Diff-master"
PROJECT_ROOT=$(dirname $(dirname $(realpath "$0")))
cd "$PROJECT_ROOT"
echo "Project root successfully set to: $PROJECT_ROOT"

# 数据目录
INPUT_DIR="${PROJECT_ROOT}/data/h5ad_final_1000000"

# ============ 检查和准备 ============
echo "============================================"
echo "Mask 一致性测试"
echo "============================================"
echo "Python: ${PYTHON_EXEC}"
echo "项目根目录: ${PROJECT_ROOT}"
echo ""

# 检查 Python 是否存在
if [ ! -f "${PYTHON_EXEC}" ]; then
    echo "错误: Python 可执行文件不存在: ${PYTHON_EXEC}"
    exit 1
fi

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# ============ 查找原始数据文件 ============
echo "查找原始数据文件..."
ORIGINAL_FILE=""

# 尝试查找第一个 .h5ad 文件（非 preprocessed）
for file in "${INPUT_DIR}"/*.h5ad; do
    if [ -f "$file" ]; then
        # 确保不是预处理文件
        if [[ ! "$file" =~ _preprocessed\.h5ad$ ]]; then
            ORIGINAL_FILE="$file"
            echo "✓ 找到原始文件: ${ORIGINAL_FILE}"
            break
        fi
    fi
done

if [ -z "${ORIGINAL_FILE}" ]; then
    echo "❌ 错误: 在 ${INPUT_DIR} 中找不到原始数据文件！"
    echo "请确保原始 .h5ad 文件存在于该目录"
    exit 1
fi

# ============ 运行测试 ============
echo ""
echo "测试说明："
echo "此测试将验证使用相同seed多次生成mask是否一致"
echo "如果一致，说明离线预处理的mask与原始代码相同"
echo ""

${PYTHON_EXEC} test_mask_consistency.py "${ORIGINAL_FILE}"

echo ""
echo "============================================"
echo "测试完成"
echo "============================================"
