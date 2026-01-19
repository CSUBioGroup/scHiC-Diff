#!/bin/bash

# 离线数据预处理脚本
# 用途：对原始 H5AD 数据进行预处理（归一化、log变换、mask生成）

set -e  # 遇到错误立即退出

# ============ 配置部分 ============
#PYTHON_EXEC="/home/duxuyan/micromamba/envs/scdiff/bin/python"
PYTHON_EXEC="/data1/zhanglinna/micromamba/envs/scdiff/bin/python"

#PROJECT_ROOT="/home/duxuyan/Projects/1_scHiC/3_DiffusionModel/scHiC-Diff-master"
PROJECT_ROOT=$(dirname $(dirname $(realpath "$0")))
cd "$PROJECT_ROOT"
echo "Project root successfully set to: $PROJECT_ROOT"

# 输入输出路径
INPUT_DIR="${PROJECT_ROOT}/data/h5ad_final_1000000"
OUTPUT_DIR="${PROJECT_ROOT}/data/h5ad_final_1000000/preprocessed"
REPORT_DIR="${PROJECT_ROOT}/data/preprocessing_reports"



# 预处理参数
VALID_SPLIT=0.2
MASK_STRATEGY="none_zero"
MASK_TYPE="mar"
SEED=10

# ============ 检查和准备 ============
echo "============================================"
echo "离线数据预处理"
echo "============================================"
echo "Python: ${PYTHON_EXEC}"
echo "项目根目录: ${PROJECT_ROOT}"
echo ""

# 检查 Python 是否存在
if [ ! -f "${PYTHON_EXEC}" ]; then
    echo "错误: Python 可执行文件不存在: ${PYTHON_EXEC}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPORT_DIR}"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# ============ 预处理数据文件 ============
echo "开始批量预处理..."
echo ""

# 统计变量
TOTAL_FILES=0
SUCCESS_FILES=0
FAILED_FILES=0

# 自动查找所有 .h5ad 文件（排除已预处理的文件）
for INPUT_FILE in "${INPUT_DIR}"/*.h5ad; do
    # 检查文件是否存在（处理通配符不匹配的情况）
    if [ ! -f "${INPUT_FILE}" ]; then
        continue
    fi
    
    # 获取文件名（不含路径）
    FILENAME=$(basename "${INPUT_FILE}")
    
    # 跳过已经预处理的文件
    if [[ "${FILENAME}" == *"_preprocessed.h5ad" ]]; then
        echo "跳过已预处理文件: ${FILENAME}"
        continue
    fi
    
    TOTAL_FILES=$((TOTAL_FILES + 1))
    
    # 生成输出文件名（添加 _preprocessed 后缀）
    BASE_NAME="${FILENAME%.h5ad}"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_NAME}_preprocessed.h5ad"
    REPORT_FILE="${REPORT_DIR}/${BASE_NAME}_report.json"
    
    echo "----------------------------------------"
    echo "处理文件 #${TOTAL_FILES}: ${FILENAME}"
    echo "----------------------------------------"
    echo "  输入: ${INPUT_FILE}"
    echo "  输出: ${OUTPUT_FILE}"
    echo "  报告: ${REPORT_FILE}"
    echo ""
    
    # 检查输出文件是否已存在
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "⚠️  输出文件已存在，跳过: ${OUTPUT_FILE}"
        echo "   如需重新处理，请先删除该文件"
        echo ""
        SUCCESS_FILES=$((SUCCESS_FILES + 1))
        continue
    fi
    
    # 执行预处理
    if ${PYTHON_EXEC} tools/offline_preprocess.py \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_FILE}" \
        --valid-split ${VALID_SPLIT} \
        --mask-strategy ${MASK_STRATEGY} \
        --mask-type ${MASK_TYPE} \
        --seed ${SEED} \
        --save-report "${REPORT_FILE}"; then
        
        SUCCESS_FILES=$((SUCCESS_FILES + 1))
        echo ""
        echo "✓ ${FILENAME} 处理完成"
        echo ""
    else
        FAILED_FILES=$((FAILED_FILES + 1))
        echo ""
        echo "✗ ${FILENAME} 处理失败"
        echo ""
    fi
done

# ============ 处理总结 ============
echo "============================================"
echo "批量预处理完成"
echo "============================================"
echo "总文件数: ${TOTAL_FILES}"
echo "成功处理: ${SUCCESS_FILES}"
echo "处理失败: ${FAILED_FILES}"
echo ""
echo "输出目录: ${OUTPUT_DIR}"
echo "报告目录: ${REPORT_DIR}"
echo ""

if [ ${TOTAL_FILES} -eq 0 ]; then
    echo "⚠️  警告: 未找到需要处理的文件"
    echo "请检查输入目录: ${INPUT_DIR}"
    exit 1
elif [ ${FAILED_FILES} -gt 0 ]; then
    echo "❌ 部分文件处理失败，请检查错误信息"
    exit 1
else
    echo "✅ 所有文件处理成功！"
    echo ""
    echo "下一步："
    echo "1. 验证预处理结果: bash tools/2_validate_preprocessing.sh"
    echo "2. 测试 mask 一致性: bash tools/3_test_mask_consistency.sh"
    echo "3. 使用预处理数据训练: bash run/local_run_hires_preprocessed.sh"
fi

echo "============================================"