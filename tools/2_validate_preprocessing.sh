#!/bin/bash

# 预处理数据验证脚本
# 用途：验证预处理后的数据是否正确

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
PREPROCESSED_DIR="${PROJECT_ROOT}/data/h5ad_final_1000000/preprocessed"
REPORT_DIR="${PROJECT_ROOT}/data/preprocessing_reports"

# ============ 检查和准备 ============
echo "============================================"
echo "预处理数据验证"
echo "============================================"
echo "Python: ${PYTHON_EXEC}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "原始数据目录: ${INPUT_DIR}"
echo "预处理数据目录: ${PREPROCESSED_DIR}"
echo "报告保存目录: ${REPORT_DIR}"
echo ""

# 创建报告目录（如果不存在）
mkdir -p "${REPORT_DIR}"

# 检查 Python 是否存在
if [ ! -f "${PYTHON_EXEC}" ]; then
    echo "错误: Python 可执行文件不存在: ${PYTHON_EXEC}"
    exit 1
fi

# 检查预处理目录是否存在
if [ ! -d "${PREPROCESSED_DIR}" ]; then
    echo "错误: 预处理数据目录不存在: ${PREPROCESSED_DIR}"
    echo "请先运行 1_offline_preprocess.sh 进行预处理"
    exit 1
fi

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# ============ 验证预处理文件 ============
echo "开始验证预处理文件..."
echo ""

# 统计变量
TOTAL_FILES=0
VALID_FILES=0
INVALID_FILES=0

# 遍历所有预处理文件
for PREPROCESSED_FILE in "${PREPROCESSED_DIR}"/*_preprocessed.h5ad; do
    if [ -f "${PREPROCESSED_FILE}" ]; then
        TOTAL_FILES=$((TOTAL_FILES + 1))
        FILENAME=$(basename "${PREPROCESSED_FILE}")
        
        echo "----------------------------------------"
        echo "验证文件 #${TOTAL_FILES}: ${FILENAME}"
        echo "----------------------------------------"
        
        # 推断原始文件名（去掉 _preprocessed 后缀）
        ORIGINAL_FILENAME="${FILENAME/_preprocessed/}"
        ORIGINAL_FILE="${INPUT_DIR}/${ORIGINAL_FILENAME}"
        
        # 检查原始文件是否存在
        if [ ! -f "${ORIGINAL_FILE}" ]; then
            echo "⚠️  警告: 找不到原始文件: ${ORIGINAL_FILE}"
            echo "跳过此文件的验证"
            INVALID_FILES=$((INVALID_FILES + 1))
            echo ""
            continue
        fi
        
        echo "原始文件: ${ORIGINAL_FILE}"
        echo "预处理文件: ${PREPROCESSED_FILE}"
        
        # 生成报告文件名
        REPORT_FILE="${REPORT_DIR}/${FILENAME/.h5ad/_validation_report.txt}"
        echo "报告文件: ${REPORT_FILE}"
        
        # 运行验证工具，传递原始文件、预处理文件和报告路径
        if ${PYTHON_EXEC} tools/validate_preprocessing.py \
            --original "${ORIGINAL_FILE}" \
            --preprocessed "${PREPROCESSED_FILE}" \
            --output "${REPORT_FILE}"; then
            VALID_FILES=$((VALID_FILES + 1))
            echo "✓ ${FILENAME} 验证通过"
        else
            INVALID_FILES=$((INVALID_FILES + 1))
            echo "✗ ${FILENAME} 验证失败"
        fi
        echo ""
    fi
done

# ============ 验证总结 ============
echo "============================================"
echo "验证总结"
echo "============================================"
echo "总文件数: ${TOTAL_FILES}"
echo "验证通过: ${VALID_FILES}"
echo "验证失败: ${INVALID_FILES}"
echo ""

if [ ${TOTAL_FILES} -eq 0 ]; then
    echo "⚠️  警告: 未找到预处理文件"
    echo "请先运行 1_offline_preprocess.sh 进行预处理"
    exit 1
elif [ ${INVALID_FILES} -gt 0 ]; then
    echo "❌ 部分文件验证失败，请检查错误信息"
    exit 1
else
    echo "✅ 所有文件验证通过！"
    echo ""
    echo "验证报告已保存到: ${REPORT_DIR}"
    echo ""
    echo "下一步："
    echo "1. 查看验证报告了解详细信息"
    echo "2. 修改配置文件使用预处理数据"
    echo "3. 运行训练脚本测试"
fi

echo "============================================"
