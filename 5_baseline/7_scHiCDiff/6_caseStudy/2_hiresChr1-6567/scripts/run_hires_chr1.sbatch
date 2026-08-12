#!/usr/bin/env bash
#SBATCH -J hires_chr1_scdiff
#SBATCH -A pi_limin_r
#SBATCH -p gpu4RQ
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/slurm_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/slurm_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
INPUT_H5AD="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/2_callLoop_apa/inputData/6_hires_chr1/hires_20kb_chr1.h5ad"
PROJECT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master"
PYTHON="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
VALIDATOR="$BASE/scripts/validate_outputs.py"
RUN_MODE="${RUN_MODE:-smoke}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"

case "$TASK_ID" in
    0)
        CONFIG_TAG="eval_v2"
        CONFIG_PATH="$BASE/configs/eval_denoising_v2.yaml"
        FULL_EPOCHS=1500
        BATCH_SIZE=8000
        NUM_WORKERS=20
        CHECKPOINT_ARGS=()
        ;;
    1)
        CONFIG_TAG="ramani"
        CONFIG_PATH="$BASE/configs/recon_masked_ramani_3000.yaml"
        FULL_EPOCHS=3000
        BATCH_SIZE=1024
        NUM_WORKERS=10
        CHECKPOINT_ARGS=(
            "lightning.modelcheckpoint.params.every_n_epochs=500"
            "lightning.modelcheckpoint.params.save_top_k=-1"
        )
        ;;
    *)
        printf 'Invalid SLURM_ARRAY_TASK_ID: %s\n' "$TASK_ID" >&2
        exit 2
        ;;
esac

case "$RUN_MODE" in
    smoke) MAX_EPOCHS=1 ;;
    full) MAX_EPOCHS="$FULL_EPOCHS" ;;
    *)
        printf 'Invalid RUN_MODE: %s\n' "$RUN_MODE" >&2
        exit 2
        ;;
esac

RESULT_DIR="$BASE/results/$RUN_MODE/$CONFIG_TAG"
MODEL_LOG_ROOT="$BASE/model_logs/$RUN_MODE"

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'config_tag=%s run_mode=%s max_epochs=%s batch_size=%s num_workers=%s result_dir=%s\n' \
        "$CONFIG_TAG" "$RUN_MODE" "$MAX_EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$RESULT_DIR"
    exit 0
fi

for required in "$INPUT_H5AD" "$PROJECT_DIR/main.py" "$PYTHON" "$CONFIG_PATH" "$VALIDATOR"; do
    [[ -e "$required" ]] || {
        printf 'Missing required path: %s\n' "$required" >&2
        exit 1
    }
done

if [[ -d "$RESULT_DIR" ]] && find "$RESULT_DIR" -mindepth 1 -print -quit | grep -q .; then
    printf 'Refusing to overwrite nonempty result directory: %s\n' "$RESULT_DIR" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR" "$MODEL_LOG_ROOT"

on_exit() {
    local status=$?
    if (( status != 0 )); then
        printf 'exit_code=%s\n' "$status" > "$RESULT_DIR/failed.flag"
    fi
}
trap on_exit EXIT

hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
printf '[INFO] config_tag=%s run_mode=%s max_epochs=%s batch_size=%s num_workers=%s\n' \
    "$CONFIG_TAG" "$RUN_MODE" "$MAX_EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS"
printf '[INFO] input=%s\n[INFO] result_dir=%s\n' "$INPUT_H5AD" "$RESULT_DIR"
date

cd "$PROJECT_DIR"

COMMAND=(
    "$PYTHON" main.py
    -b "$CONFIG_PATH"
    --train True
    --seed 10
    --save_path "$RESULT_DIR"
    --logdir "$MODEL_LOG_ROOT"
    --postfix "hires_chr1_${RUN_MODE}_${CONFIG_TAG}.seed10"
    "lightning.trainer.devices=[0]"
    "lightning.trainer.max_epochs=$MAX_EPOCHS"
    "data.params.batch_size=$BATCH_SIZE"
    "data.params.num_workers=$NUM_WORKERS"
    "data.params.train.params.dataset=HiresChr1"
    "data.params.train.params.fname=$INPUT_H5AD"
    "data.params.validation.params.dataset=HiresChr1"
    "data.params.validation.params.fname=$INPUT_H5AD"
    "data.params.test.params.dataset=HiresChr1"
    "data.params.test.params.fname=$INPUT_H5AD"
    "${CHECKPOINT_ARGS[@]}"
)

"${COMMAND[@]}" 2>&1 | tee "$RESULT_DIR/train.log"

"$PYTHON" "$VALIDATOR" \
    --result-dir "$RESULT_DIR" \
    --expected-rows 7466 \
    --expected-cols 5050

rm -f "$RESULT_DIR/failed.flag"
trap - EXIT
printf '[DONE] %s %s\n' "$RUN_MODE" "$CONFIG_TAG"
date
