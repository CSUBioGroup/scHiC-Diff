#!/usr/bin/env bash
#SBATCH -J hires_eval_v2_compat
#SBATCH -A pi_limin_r
#SBATCH -p gpu4RQ
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/corrected/%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567/logs/corrected/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
INPUT_H5AD="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/2_callLoop_apa/inputData/6_hires_chr1/hires_20kb_chr1.h5ad"
PROJECT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master"
PYTHON="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
CONFIG="$BASE/configs/eval_denoising_v2_compat.yaml"
VALIDATOR="$BASE/scripts/validate_outputs.py"
ASSESSOR="$BASE/scripts/assess_imputation_quality.py"
RUN_MODE="${RUN_MODE:-smoke}"

case "$RUN_MODE" in
    smoke) MAX_EPOCHS=50 ;;
    full) MAX_EPOCHS=1500 ;;
    *)
        printf 'Invalid RUN_MODE: %s\n' "$RUN_MODE" >&2
        exit 2
        ;;
esac

RESULT_DIR="$BASE/results/corrected/eval_v2/$RUN_MODE"
MODEL_LOG_ROOT="$BASE/model_logs/corrected/eval_v2/$RUN_MODE"
COMPAT_PYTHONPATH="$BASE/compat:$PROJECT_DIR"

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'run_mode=%s max_epochs=%s result_dir=%s pythonpath=%s\n' \
        "$RUN_MODE" "$MAX_EPOCHS" "$RESULT_DIR" "$COMPAT_PYTHONPATH"
    exit 0
fi

for required in \
    "$INPUT_H5AD" \
    "$PROJECT_DIR/main.py" \
    "$PYTHON" \
    "$CONFIG" \
    "$VALIDATOR" \
    "$ASSESSOR" \
    "$BASE/compat/eval_v2_model.py"; do
    [[ -e "$required" ]] || {
        printf 'Missing required path: %s\n' "$required" >&2
        exit 1
    }
done

if [[ -d "$RESULT_DIR" ]] && find "$RESULT_DIR" -mindepth 1 -print -quit | rg -q .; then
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

export PYTHONPATH="$COMPAT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
hostname
printf '[INFO] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
printf '[INFO] config_tag=eval_v2_compat run_mode=%s max_epochs=%s lr=8e-5 v2_cell_mask_ratio=0.75\n' \
    "$RUN_MODE" "$MAX_EPOCHS"
printf '[INFO] input=%s\n[INFO] result_dir=%s\n' "$INPUT_H5AD" "$RESULT_DIR"
date

cd "$PROJECT_DIR"

COMMAND=(
    "$PYTHON" main.py
    -b "$CONFIG"
    --train True
    --seed 10
    --save_path "$RESULT_DIR"
    --logdir "$MODEL_LOG_ROOT"
    --postfix "hires_chr1_corrected_${RUN_MODE}_eval_v2.seed10"
    "lightning.trainer.devices=[0]"
    "lightning.trainer.max_epochs=$MAX_EPOCHS"
    "data.params.batch_size=8000"
    "data.params.num_workers=20"
    "data.params.train.params.dataset=HiresChr1"
    "data.params.train.params.fname=$INPUT_H5AD"
    "data.params.validation.params.dataset=HiresChr1"
    "data.params.validation.params.fname=$INPUT_H5AD"
    "data.params.test.params.dataset=HiresChr1"
    "data.params.test.params.fname=$INPUT_H5AD"
)

"${COMMAND[@]}" 2>&1 | tee "$RESULT_DIR/train.log"

"$PYTHON" "$VALIDATOR" \
    --result-dir "$RESULT_DIR" \
    --expected-rows 7466 \
    --expected-cols 5050

rm -f "$RESULT_DIR/done.flag"
"$PYTHON" "$ASSESSOR" \
    --result-dir "$RESULT_DIR" \
    --train-log "$RESULT_DIR/train.log" \
    --expected-rows 7466 \
    --expected-cols 5050

printf 'structural and quality validation passed\n' > "$RESULT_DIR/done.flag"
rm -f "$RESULT_DIR/failed.flag"
trap - EXIT
printf '[DONE] corrected %s eval_v2\n' "$RUN_MODE"
date
