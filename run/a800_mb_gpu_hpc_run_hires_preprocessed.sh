#!/bin/bash

# ==============================================================================
# 使用离线预处理数据的训练脚本
# 
# 主要改进:
# 1. 使用预处理后的数据文件 (*_preprocessed.h5ad)
# 2. 跳过数据加载时的归一化和 log 变换步骤
# 3. 加快训练初始化速度
# 4. 确保训练/验证/测试使用一致的 mask
# ==============================================================================

# ==============================================================================
# Slurm SBATCH Directives
# ==============================================================================
#SBATCH --job-name=mb_hires_impute_prep  # 作业名称 (添加 _prep 标识使用预处理数据)
#SBATCH --output=./run/logs/mb_imputejob_prep_%j.log  # 标准输出日志
#SBATCH --error=./run/logs/mb_imputejob_prep_%j.log   # 标准错误日志
#SBATCH --account=pi_limin                # 您的账户
#SBATCH --partition=gpu2Q                 # 计算分区/队列
#SBATCH --mem=100G 
#SBATCH --qos=gpuq                        # 服务质量
#SBATCH --nodes=1                         # 节点数
#SBATCH --ntasks-per-node=1               # 每节点任务数
#SBATCH --cpus-per-task=20                # CPU 核心数
#SBATCH --gres=gpu:1                      # 每个作业占用的GPU数量

# ==============================================================================
# 脚本执行严格模式
# ==============================================================================
set -euo pipefail

# ==============================================================================
# 打印作业信息
# ==============================================================================
echo "================================================="
echo "Job started on: $(date)"
# echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
# echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Using PREPROCESSED data for faster initialization"
echo "================================================="

# ==============================================================================
# 环境设置
# ==============================================================================
PYTHON_EXEC="/data1/zhanglinna/micromamba/envs/scdiff/bin/python"
echo "Using python executable directly at: ${PYTHON_EXEC}"

# 验证 Python 版本
echo "Python command: $(which ${PYTHON_EXEC})"
${PYTHON_EXEC} --version
echo "-------------------------------------------------"

# ==============================================================================
# 设置项目主目录
# ==============================================================================
HOMEDIR=$(dirname $(dirname $(realpath "$0")))
cd "$HOMEDIR"
echo "Project root successfully set to: $HOMEDIR"


# ==============================================================================
# 定义静态参数
# ==============================================================================
CONFIG_PREFIX="configs"
NAME="scHiC_v1.2_preprocessed"  # 添加标识以区分使用预处理数据
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
shift

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
    
    # 预处理文件名 (添加 _preprocessed 后缀)
    preprocessed_fname="hires_${data_prefix}_preprocessed.h5ad"
    
    # 构建预处理文件的绝对路径
    absolute_data_fname="${HOMEDIR}/${PREPROCESSED_DIR}/${preprocessed_fname}"
    
    # 检查预处理文件是否存在
    if [ ! -f "${absolute_data_fname}" ]; then
        echo "ERROR: Preprocessed file not found: ${absolute_data_fname}"
        echo "Please run offline preprocessing first:"
        echo "  bash tools/1_offline_preprocess.sh"
        exit 1
    fi

    echo "Using PREPROCESSED data file: ${absolute_data_fname}"

    # 将所有数据相关参数放入一个数组
    data_args=(
        "data.params.train.params.dataset=${dataset_name}"
        "data.params.train.params.fname=${absolute_data_fname}"
        "data.params.validation.params.dataset=${dataset_name}"
        "data.params.validation.params.fname=${absolute_data_fname}"
        "data.params.test.params.dataset=${dataset_name}"
        "data.params.test.params.fname=${absolute_data_fname}"
    )

    # --- 构建完整命令 ---
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
    
    # --- 执行命令 ---
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
