#!/bin/bash

# ==============================================================================
# Slurm SBATCH Directives

# ==============================================================================
#SBATCH --job-name=gpu_hires_impute       # 作业名称
#SBATCH --output=./run/logs/gpu_hires_imputejob_%j.log          # 标准输出日志
#SBATCH --error=./run/logs/gpu_hires_imputejob_%j.log           # 标准错误日志 (与输出合并，强烈推荐)
#SBATCH --account=pi_limin_r                    # 您的账户
#SBATCH --partition=gpu8Q                 # 计算分区/队列
#SBATCH --mem=80G 
#SBATCH --qos=gpuq                        # 服务质量
#SBATCH --nodes=1                         # 节点数
#SBATCH --ntasks-per-node=1               # 每节点任务数
#SBATCH --cpus-per-task=10                # CPU 核心数 (请根据数据加载等需求调整)
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *

# ==============================================================================
# 脚本执行严格模式 (推荐)
# ==============================================================================
set -euo pipefail

# ==============================================================================
# 打印作业信息
# ==============================================================================
echo "================================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "================================================="

# ==============================================================================
# 环境设置 (推荐方案 A，备用方案 B)
# ==============================================================================

# --- 方案 A (推荐): 稳健的环境激活 ---
# 这是在脚本中激活环境最可靠的方式，能确保所有依赖库路径都设置正确
#eval "$(micromamba shell hook -s bash)"
#micromamba activate scdiff
#echo "Successfully activated micromamba environment: scdiff"
#PYTHON_EXEC="python" # 激活后，直接使用 python 命令即可

# --- 方案 B (备用): 直接使用 Python 路径 ---
# 如果方案 A 出现问题，可以注释掉方案 A，并取消下面两行的注释
PYTHON_EXEC="/public/home/hpc254701055/micromamba/envs/scdiff/bin/python"
echo "Using python executable directly at: ${PYTHON_EXEC}"

# 验证 Python 版本
echo "Python command: $(which ${PYTHON_EXEC})"
${PYTHON_EXEC} --version
echo "-------------------------------------------------"

# ==============================================================================
# 设置项目主目录
# ==============================================================================
# 使用 Slurm 提供的 $SLURM_SUBMIT_DIR 环境变量，这是最稳健、可移植性最好的方法
HOMEDIR="$SLURM_SUBMIT_DIR"
cd "$HOMEDIR"
echo "Changed directory to project home: $HOMEDIR"

# ==============================================================================
# 定义静态参数
# ==============================================================================
CONFIG_PREFIX="configs"
NAME="scHiC_v1.2"
LOSS_STRATEGY="recon_masked"
OFFLINE_SETTINGS="--wandb_offline t"
TEST_FLAG=${TEST_FLAG:-0} # 设置为 1 可进行“空跑”，只打印命令不执行
RESOLUTION="1Mb"
# ==============================================================================
# 解析命令行参数
# ==============================================================================
# 第一个参数为随机种子 (SEED)，默认为 10
SEED="${1:-10}"
shift # 移除第一个参数，剩下的是染色体列表

# 将所有剩余参数存入 CHROMOSOMES 数组
CHROMOSOMES=("$@") 

# 如果未提供任何染色体，则使用默认值 "chr1"
if [ ${#CHROMOSOMES[@]} -eq 0 ]; then
    CHROMOSOMES=("chr1")
    echo "No chromosomes provided, using default: ${CHROMOSOMES[*]}"
fi

echo "Running with SEED: $SEED"
echo "Processing CHROMOSOMES: ${CHROMOSOMES[*]}"
echo "-------------------------------------------------"

# ==============================================================================
# 获取当前作业的索引
# ==============================================================================
chrom_index=$SLURM_ARRAY_TASK_ID  # 获取作业数组的索引

# 使用索引选择当前的染色体
chrom="${CHROMOSOMES[$chrom_index]}"

# 处理当前染色体
echo "==================== Processing: $chrom ===================="
save_path="${HOMEDIR}/results/0_Hires_impute/${RESOLUTION}/${chrom}"
mkdir -p "$save_path"
echo "Ensured output directory exists: $save_path"

# --- 准备数据参数 ---
dataset_name="hires"
data_prefix="1Mb_${chrom}"

# 基础文件名
base_fname="${dataset_name}_${data_prefix}.h5ad"

# 文件实际存放的子目录
data_subdir="data/2_hires_schicdiff/h5ad_final_1000000"

# 构建文件的【绝对路径】
absolute_data_fname="${HOMEDIR}/${data_subdir}/${base_fname}"

echo "Using ABSOLUTE data file path: ${absolute_data_fname}"

# 将所有数据相关参数放入一个数组
data_args=(
    "data.params.train.params.dataset=${dataset_name}"
    "data.params.train.params.fname=${absolute_data_fname}"
    "data.params.validation.params.dataset=${dataset_name}"
    "data.params.validation.params.fname=${absolute_data_fname}"
    "data.params.test.params.dataset=${dataset_name}"
    "data.params.test.params.fname=${absolute_data_fname}"
)

# --- 使用数组构建完整命令 ---
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