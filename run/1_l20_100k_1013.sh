#!/bin/bash

# ==============================================================================
# NEW SECTION: 自动日志记录设置
# ==============================================================================
# 根据脚本的输入参数，动态创建一个唯一的日志文件名
# ... (省略注释) ...
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ==============================================================================
# !! 在这里设置你想要的日志目录 !!
LOG_DIR="/root/Projects/1_scHiCDiff/scHiC-Diff-master/my_run_logs" # <--- 新增：定义日志存放目录
# ==============================================================================

# 确保日志目录存在，如果不存在则创建它
mkdir -p "${LOG_DIR}" # <--- 新增：确保目录存在

# 提取参数用于命名
SEED_FOR_LOG="${1:-10}" 
CHROMS_FOR_LOG=("${@:2}") 

if [ ${#CHROMS_FOR_LOG[@]} -eq 0 ]; then
    CHROMS_STR="chr1_default"
else
    CHROMS_STR=$(IFS=_; echo "${CHROMS_FOR_LOG[*]}")
fi

# 组合成最终的日志文件名 (包含完整路径)
LOG_FILE="${LOG_DIR}/run_log_seed${SEED_FOR_LOG}_${CHROMS_STR}.log" # <--- 修改：在文件名前加上目录路径


# !! 关键步骤 !!
# 在重定向之前，先在屏幕上告诉用户日志文件在哪里
echo "✅ 脚本启动，所有输出将被自动保存到: ${LOG_FILE}"
echo "   您可以使用 'tail -f ${LOG_FILE}' 来实时监控进度。"

# 使用 exec 将此脚本后续的所有 stdout 和 stderr 都重定向到日志文件
exec > "${LOG_FILE}" 2>&1

# ==============================================================================
# 原始脚本内容开始 (无需任何修改)
# ==============================================================================
set -euo pipefail

# 打印作业信息
echo "================================================="
# 这个 'echo' 以及之后的所有输出，现在都会进入上面的 LOG_FILE 文件中
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "================================================="

# 环境设置
PYTHON_EXEC="/root/micromamba/envs/scdiff/bin/python"
echo "Using python executable directly at: ${PYTHON_EXEC}"

# 验证 Python 版本
echo "Python command: $(which ${PYTHON_EXEC} || echo ${PYTHON_EXEC})"
${PYTHON_EXEC} --version
echo "-------------------------------------------------"

# 设置项目主目录
HOMEDIR="$(pwd)"
cd "$HOMEDIR"
echo "Project home directory set to: $HOMEDIR"

# 定义静态参数
CONFIG_PREFIX="configs"
NAME="scHiC_v1.2"
LOSS_STRATEGY="recon_masked"
OFFLINE_SETTINGS="--wandb_offline t"
TEST_FLAG=${TEST_FLAG:-0} 
RESOLUTION="100k"

# 解析命令行参数
SEED="${1:-10}"
shift 

CHROMOSOMES=("$@") 

if [ ${#CHROMOSOMES[@]} -eq 0 ]; then
    CHROMOSOMES=("chr1")
    echo "No chromosomes provided, using default: ${CHROMOSOMES[*]}"
fi

echo "Running with SEED: $SEED"
echo "Processing CHROMOSOMES: ${CHROMOSOMES[*]}"
echo "-------------------------------------------------"

# 主执行逻辑
for chrom in "${CHROMOSOMES[@]}"; do
    echo "==================== Processing: $chrom ===================="
    save_path="${HOMEDIR}/results/0_Hires_impute/${RESOLUTION}/${chrom}"
    mkdir -p "$save_path"
    echo "Ensured output directory exists: $save_path"

    # 准备数据参数
    dataset_name="hires"
    data_prefix="100k_${chrom}"
    base_fname="${dataset_name}_${data_prefix}.h5ad"
    # data_subdir="data/2_hires_schicdiff/h5ad_final_100000"
    # absolute_data_fname="${HOMEDIR}/${data_subdir}/${base_fname}"
    absolute_data_fname="/root/private_data/liuwuhao/h5ad_final_100000/${base_fname}"

    echo "Using ABSOLUTE data file path: ${absolute_data_fname}"

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
        --postfix "seed${SEED}"
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

# 作业完成
echo "================================================="
echo "All chromosomes processed. Job finished on: $(date)"
echo "================================================="