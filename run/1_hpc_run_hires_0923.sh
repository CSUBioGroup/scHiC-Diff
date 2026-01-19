#!/bin/bash

# ==============================================================================
# SBATCH Directives - More generic and descriptive
# ==============================================================================
#SBATCH -J gpu_hires_impute     # Generic Job name, easier to identify
#SBATCH -o ./logs/gpu_hires_impute.%j.out # Generic log name with Job ID
#SBATCH -A free
#SBATCH -p gpu2Q
#SBATCH -q gpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38

# ==============================================================================
# Script Configuration & Best Practices
# ==============================================================================
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines return the exit status of the last command to fail, not the last command.
set -o pipefail

# --- Print Job Information ---
echo "================================================="
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "================================================="

# ==============================================================================
# Environment Setup - Using Direct Python Path
# ==============================================================================
# 直接定义 Python 解释器的绝对路径
# 这样做可以避免在 Slurm 的非交互式 Shell 中激活 conda/mamba 环境时可能遇到的问题。
PYTHON_EXEC="/public/home/hpc254701055/micromamba/envs/scdiff/bin/python"

echo "================================================="
echo "Using python executable directly at: ${PYTHON_EXEC}"
# 验证 Python 版本
${PYTHON_EXEC} --version
echo "================================================="


# ==============================================================================
# Script Parameters 1.1
# ==============================================================================
# 直接指定项目主目录的绝对路径，避免因 Slurm 复制脚本导致路径错误
# !!! 请务必将下面的路径替换为您真实的、存放 main.py 和 configs 的项目根目录 !!!
HOMEDIR="/public/home/hpc254701055/Projects/1_scHiCDiff/3_DiffusionModel/scHiC-Diff-master"

cd "$HOMEDIR"
echo "Changed directory to project home: $HOMEDIR"

# # ==============================================================================
# # Script Parameters 1.2
# # ==============================================================================
# # 使用 Slurm 提供的环境变量 $SLURM_SUBMIT_DIR 来确定项目主目录
# # $SLURM_SUBMIT_DIR 是您执行 sbatch 命令时所在的目录，这通常是正确的项目路径
# HOMEDIR="$SLURM_SUBMIT_DIR"

# cd "$HOMEDIR"
# echo "Changed directory to project home: $HOMEDIR"


# --- Static configuration ---
LOGDIR="logs"
CONFIG_PREFIX="configs"
NAME="scHiC_v1.2"
LOSS_STRATEGY="recon_masked"
OFFLINE_SETTINGS="--wandb_offline t"
TEST_FLAG=${TEST_FLAG:-0} # Set to 1 for a dry-run

# ==============================================================================
# Command-line Argument Parsing - CLEARER LOGIC
# ==============================================================================
# Priority: Command-line > Environment Variable > Default
# Use a more readable way to set defaults.
SEED="${1:-${SEED:-10}}" # Use 1st arg if present, else use env var SEED, else default to 10
shift # Remove the seed argument, the rest are chromosomes

CHROMOSOMES=("$@") # Assign the remaining arguments to the CHROMOSOMES array

# If no chromosomes were provided on the command line, use a default value.
if [ ${#CHROMOSOMES[@]} -eq 0 ]; then
    CHROMOSOMES=("chr1")
    echo "No chromosomes provided, using default: ${CHROMOSOMES[*]}"
fi

echo "Running with SEED: $SEED"
echo "Processing CHROMOSOMES: ${CHROMOSOMES[*]}"
echo "-------------------------------------------------"

# ==============================================================================
# Main Execution Logic
# ==============================================================================
# The main loop is now outside a function, or you can define a main() and call it.

for chrom in "${CHROMOSOMES[@]}"; do
    echo "======================================================================"
    echo "Processing chromosome: $chrom"
    echo "======================================================================"

    save_path="${HOMEDIR}/results/0_Hires_impute/${chrom}"

    # Create directory if it doesn't exist
    if mkdir -p "$save_path"; then
        echo "Ensured directory exists: $save_path"
    else
        echo "ERROR: Could not create directory: $save_path"
        exit 1
    fi

    # --- Prepare data settings ---
    dataset_name="hires"
    data_prefix="100k_${chrom}"
    
    # These are repeated, so let's define them once.
    dataset_arg="data.params.train.params.dataset=${dataset_name}"
    fname_arg="data.params.train.params.fname=${dataset_name}_${data_prefix}.h5ad"
    # Assuming validation and test use the same data, as in the original script
    val_dataset_arg="data.params.validation.params.dataset=${dataset_name}"
    val_fname_arg="data.params.validation.params.fname=${dataset_name}_${data_prefix}.h5ad"
    test_dataset_arg="data.params.test.params.dataset=${dataset_name}"
    test_fname_arg="data.params.test.params.fname=${dataset_name}_${data_prefix}.h5ad"

    # --- Build the command using a Bash array (SAFER THAN EVAL) ---
    cmd=(
        "${PYTHON_EXEC}" main.py
        -b "${CONFIG_PREFIX}/${LOSS_STRATEGY}.yaml"
        --name "$NAME"
        --seed "$SEED"
        --save_path "$save_path"
        --logdir "$LOGDIR"
        --postfix "seed${SEED}"
        ${OFFLINE_SETTINGS} # This is a single string, so it's fine
        "$dataset_arg"
        "$fname_arg"
        "$val_dataset_arg"
        "$val_fname_arg"
        "$test_dataset_arg"
        "$test_fname_arg"
    )
    
    # --- Execute the command ---
    echo "Executing command:"
    # Use printf to safely print each argument on a new line for clarity
    printf "  %s\n" "${cmd[@]}"
    echo "-------------------------------------------------"

    if [[ $TEST_FLAG == 0 ]]; then
        # Execute the command. Quotes around "${cmd[@]}" are crucial.
        "${cmd[@]}"
    else
        echo "DRY RUN: TEST_FLAG is set to 1. Command was not executed."
    fi

    echo "Finished processing chromosome: $chrom"
done

# ==============================================================================
# Job Completion
# ==============================================================================
echo "================================================="
echo "All chromosomes processed."
echo "Job finished on: $(date)"
echo "================================================="