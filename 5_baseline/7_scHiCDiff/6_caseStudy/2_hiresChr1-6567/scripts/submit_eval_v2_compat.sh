#!/usr/bin/env bash
set -euo pipefail

ASSET_BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
STATE_BASE="${BASE_OVERRIDE:-$ASSET_BASE}"
RUNNER="$ASSET_BASE/scripts/run_eval_v2_compat.sbatch"
MODE="${1:-}"
PARTITION="${2:-gpu4RQ}"

case "$MODE" in
    smoke) TIME_LIMIT="02:00:00" ;;
    full) TIME_LIMIT="08:00:00" ;;
    *)
        printf 'Usage: %s {smoke|full} [gpu2Q|gpu4Q|gpu8Q|gpu4RQ]\n' "$0" >&2
        exit 2
        ;;
esac

case "$PARTITION" in
    gpu2Q|gpu4Q|gpu8Q|gpu4RQ) ;;
    *)
        printf 'Unsupported GPU partition: %s\n' "$PARTITION" >&2
        exit 2
        ;;
esac

if [[ "$MODE" == "full" ]]; then
    marker="$STATE_BASE/results/corrected/eval_v2/smoke/quality_passed.flag"
    [[ -s "$marker" ]] || {
        printf 'Corrected full phase blocked; missing smoke quality marker: %s\n' \
            "$marker" >&2
        exit 1
    }
fi

SBATCH_ARGS=(
    --parsable
    --partition="$PARTITION"
    --qos=gpuq
    --gres=gpu:1
    --cpus-per-task=20
    --mem=40G
    --time="$TIME_LIMIT"
    --job-name="hires_eval_v2_compat_$MODE"
    --export="ALL,RUN_MODE=$MODE"
    --output="$STATE_BASE/logs/corrected/%x_%j.out"
    --error="$STATE_BASE/logs/corrected/%x_%j.err"
)

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}" "$RUNNER"
    printf '\n'
    exit 0
fi

mkdir -p "$STATE_BASE/logs/corrected" "$STATE_BASE/model_logs/corrected"
submission="$(sbatch "${SBATCH_ARGS[@]}" "$RUNNER")"
job_id="${submission%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected sbatch response: %s\n' "$submission" >&2
    exit 1
}
printf '%s\n' "$job_id" | tee "$STATE_BASE/logs/corrected/${MODE}_job_id.txt"
