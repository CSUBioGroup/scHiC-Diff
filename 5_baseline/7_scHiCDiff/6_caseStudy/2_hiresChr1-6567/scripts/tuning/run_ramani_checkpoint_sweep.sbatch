#!/usr/bin/env bash
#SBATCH -J ramani_ckpt_sweep
#SBATCH -A pi_limin_r
#SBATCH -p gpu4RQ
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=19
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/results/full/ramani/checkpoint_sweep/logs/%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/results/full/ramani/checkpoint_sweep/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
PROJECT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master"
PYTHON="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
RUN_DIR="$BASE/model_logs/full/recon_masked_ramani_3000_hires_chr1_full_ramani.seed10_2026-07-17T17-41-55"
SAVED_CONFIG="$RUN_DIR/configs/2026-07-17T17-41-55-project.yaml"
VALIDATOR="$BASE/scripts/validate_outputs.py"
COLLECTOR="$BASE/scripts/tuning/collect_inference_metrics.py"
CHECKPOINT_EPOCHS=(499 999 1499 1999 2499)
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
DENOISE_T=750
SEED=10

if [[ ! "$TASK_ID" =~ ^[0-4]$ ]]; then
    printf 'invalid task id: %s\n' "$TASK_ID" >&2
    exit 2
fi
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCHS[$TASK_ID]}"
printf -v PADDED_EPOCH '%06d' "$CHECKPOINT_EPOCH"
CHECKPOINT="$RUN_DIR/checkpoints/epoch=$PADDED_EPOCH.ckpt"

SWEEP_ROOT="$BASE/results/full/ramani/checkpoint_sweep"
RESULT_DIR="$SWEEP_ROOT/t750/seed10/epoch$CHECKPOINT_EPOCH"
MODEL_LOG_ROOT="$BASE/model_logs/tuning/ramani_checkpoint_sweep"

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'task_id=%s checkpoint_epoch=%s denoise_t_sample=%s checkpoint=%s result_dir=%s\n' \
        "$TASK_ID" "$CHECKPOINT_EPOCH" "$DENOISE_T" \
        "$CHECKPOINT" "$RESULT_DIR"
    exit 0
fi

for required in \
    "$PROJECT/main.py" \
    "$PYTHON" \
    "$CHECKPOINT" \
    "$SAVED_CONFIG" \
    "$VALIDATOR" \
    "$COLLECTOR"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required path: %s\n' "$required" >&2
        exit 1
    fi
done

if [[ -d "$RESULT_DIR" ]] && \
        find "$RESULT_DIR" -mindepth 1 -print -quit | rg -q .; then
    printf 'refusing to overwrite nonempty result directory: %s\n' \
        "$RESULT_DIR" >&2
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
printf 'checkpoint_epoch=%s denoise_t_sample=%s seed=%s checkpoint=%s\n' \
    "$CHECKPOINT_EPOCH" "$DENOISE_T" "$SEED" "$CHECKPOINT"
date

cd "$PROJECT"
COMMAND=(
    "$PYTHON" main.py
    -b "$SAVED_CONFIG"
    --train False
    --seed "$SEED"
    --save_path "$RESULT_DIR"
    --logdir "$MODEL_LOG_ROOT"
    --postfix "ramani_ep${CHECKPOINT_EPOCH}_t${DENOISE_T}.seed${SEED}"
    "model.params.ckpt_path=$CHECKPOINT"
    "model.params.denoise_t_sample=$DENOISE_T"
    "lightning.trainer.devices=[0]"
)
"${COMMAND[@]}" 2>&1 | tee "$RESULT_DIR/inference.log"

"$PYTHON" "$VALIDATOR" \
    --result-dir "$RESULT_DIR" \
    --expected-rows 7466 \
    --expected-cols 5050
"$PYTHON" "$COLLECTOR" \
    --result-dir "$RESULT_DIR" \
    --denoise-t "$DENOISE_T" \
    --checkpoint "$CHECKPOINT" \
    --seed "$SEED"
cp "$SAVED_CONFIG" "$RESULT_DIR/effective_training_config.yaml"
printf 'checkpoint=%s\ncheckpoint_epoch=%s\ndenoise_t_sample=%s\nseed=%s\n' \
    "$CHECKPOINT" "$CHECKPOINT_EPOCH" "$DENOISE_T" "$SEED" > \
    "$RESULT_DIR/run_metadata.txt"
printf 'completed\n' > "$RESULT_DIR/success.flag"
rm -f "$RESULT_DIR/failed.flag"
trap - EXIT
date
