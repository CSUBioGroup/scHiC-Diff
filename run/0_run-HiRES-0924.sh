#!/bin/bash

# ==============================================================================
# 脚本执行严格模式 (推荐的最佳实践)
# -e: 命令失败时立即退出
# -u: 使用未定义的变量时报错
# -o pipefail: 管道中的任何命令失败，整个管道都算失败
# ==============================================================================
set -euo pipefail

# ==============================================================================
# 静态配置
# ==============================================================================
CONFIG_PREFIX="configs"
NAME="scHiC_v1.2"
LOSS_STRATEGY="recon_masked"
OFFLINE_SETTINGS="--wandb_offline t"
LOGDIR="logs"
TEST_FLAG=${TEST_FLAG:-0} # 设置为 1 可进行“空跑”，只打印命令不执行

# ==============================================================================
# Python 执行路径 (按您的要求)
# ==============================================================================
# 直接指定 scdiff 环境中 Python 解释器的绝对路径
PYTHON_EXEC="/home/duxuyan/anaconda3/envs/scdiff/bin/python"

# 验证指定的 Python 路径是否存在且可执行
if [ ! -x "$PYTHON_EXEC" ]; then
    echo "错误: Python 解释器路径不存在或不可执行: $PYTHON_EXEC"
    exit 1
fi
echo "Using Python executable at: $PYTHON_EXEC"
$PYTHON_EXEC --version # 打印版本以确认

# ==============================================================================
# 动态设置项目主目录
# ==============================================================================
# 假设此脚本位于项目的子目录 (如 'run/') 中，此命令会自动找到项目根目录
HOMEDIR=$(dirname $(dirname $(realpath "$0")))
cd "$HOMEDIR"
echo "Project root successfully set to: $HOMEDIR"

# ==============================================================================
# 解析命令行参数 (更清晰的逻辑)
# ==============================================================================
# 第一个参数为随机种子 (SEED)，如果没有提供，则默认为 10
SEED="${1:-10}"
shift # 移除第一个参数 (SEED)，剩下的都是染色体列表

# 将所有剩余的参数存入 CHROMOSOMES 数组
CHROMOSOMES=("$@")

# 如果未在命令行提供任何染色体，则使用默认值 "chr1"
if [ ${#CHROMOSOMES[@]} -eq 0 ]; then
    CHROMOSOMES=("chr1")
    echo "No chromosomes provided on command line, using default: ${CHROMOSOMES[*]}"
fi

# ==============================================================================
# 主函数定义
# ==============================================================================
main() {
    local seed=$1
    shift
    local chrom_list=("$@")

    echo "-------------------------------------------------"
    echo "Starting job with SEED: $seed"
    echo "Processing CHROMOSOMES: ${chrom_list[*]}"
    echo "-------------------------------------------------"

    for chrom in "${chrom_list[@]}"; do
        echo "==================== Processing: $chrom ===================="
        local save_path="${HOMEDIR}/results/0_Hires_impute/${chrom}"
        mkdir -p "$save_path"
        echo "Ensured output directory exists: $save_path"

        # --- 构建数据文件的绝对路径 ---
        local dataset_name="hires"
        local data_prefix="100k_${chrom}"
        local base_fname="${dataset_name}_${data_prefix}.h5ad"
        local data_subdir="data"
        local absolute_data_fname="${HOMEDIR}/${data_subdir}/${base_fname}"

        local data_args=(
            "data.params.train.params.dataset=${dataset_name}"
            "data.params.train.params.fname=${absolute_data_fname}"
            "data.params.validation.params.dataset=${dataset_name}"
            "data.params.validation.params.fname=${absolute_data_fname}"
            "data.params.test.params.dataset=${dataset_name}"
            "data.params.test.params.fname=${absolute_data_fname}"
        )

        # --- 使用数组构建完整命令 (安全且清晰) ---
        local cmd=(
            "${PYTHON_EXEC}" main.py # <--- 使用我们定义的绝对路径变量
            -b "${CONFIG_PREFIX}/${LOSS_STRATEGY}.yaml"
            --name "$NAME"
            --seed "$seed"
            --save_path "$save_path"
            --logdir "$LOGDIR"
            --postfix "seed${seed}"
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
}

# ==============================================================================
# 脚本入口: 调用主函数
# ==============================================================================
main "$SEED" "${CHROMOSOMES[@]}"

echo "================================================="
echo "All chromosomes processed. Job finished successfully."
echo "================================================="