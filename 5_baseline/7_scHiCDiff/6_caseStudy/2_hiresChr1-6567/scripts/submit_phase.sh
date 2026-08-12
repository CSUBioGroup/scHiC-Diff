#!/usr/bin/env bash
set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
RUNNER="$BASE/scripts/run_hires_chr1.sbatch"
MODE="${1:-}"
PARTITION="${2:-gpu4RQ}"

case "$MODE" in
    smoke|full) ;;
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
    for tag in eval_v2 ramani; do
        marker="$BASE/results/smoke/$tag/done.flag"
        [[ -s "$marker" ]] || {
            printf 'Full phase blocked; missing smoke marker: %s\n' "$marker" >&2
            exit 1
        }
    done
fi

mkdir -p "$BASE/logs/$MODE" "$BASE/model_logs/$MODE"

SBATCH_ARGS=(
    --parsable
    --partition="$PARTITION"
    --qos=gpuq
    --gres=gpu:1
    --array=0-1%2
    --job-name="hires_chr1_$MODE"
    --export="ALL,RUN_MODE=$MODE"
    --output="$BASE/logs/$MODE/%x_%A_%a.out"
    --error="$BASE/logs/$MODE/%x_%A_%a.err"
)

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}" "$RUNNER"
    printf '\n'
    exit 0
fi

free_nodes="$(
    scontrol show nodes -o | awk -v target_partition="$PARTITION" '
        function gpu_count(field, value) {
            value = field
            if (value !~ /gres\/gpu/) {
                return 0
            }
            sub(/^.*gres\/gpu(:[^=,]+)?=/, "", value)
            sub(/,.*/, "", value)
            return value + 0
        }
        {
            node = state = partitions = cfg_field = alloc_field = ""
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^NodeName=/) {
                    split($i, pair, "=")
                    node = pair[2]
                } else if ($i ~ /^State=/) {
                    split($i, pair, "=")
                    state = pair[2]
                } else if ($i ~ /^Partitions=/) {
                    split($i, pair, "=")
                    partitions = pair[2]
                } else if ($i ~ /^CfgTRES=/) {
                    cfg_field = $i
                } else if ($i ~ /^AllocTRES=/) {
                    alloc_field = $i
                }
            }
            configured = gpu_count(cfg_field)
            allocated = gpu_count(alloc_field)
            free_gpu = configured - allocated
            if (partitions == target_partition &&
                    state !~ /(DRAIN|DOWN|FAIL|INVAL)/ &&
                    free_gpu > 0) {
                printf "%s|%s|free_gpu=%d\n", node, state, free_gpu
            }
        }
    '
)"

[[ -n "$free_nodes" ]] || {
    printf 'No free GPU is currently visible on a schedulable %s node.\n' "$PARTITION" >&2
    exit 1
}
printf '[INFO] %s nodes with a free GPU at submission time:\n%s\n' "$PARTITION" "$free_nodes"

submission="$(sbatch "${SBATCH_ARGS[@]}" "$RUNNER")"
job_id="${submission%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'Unexpected sbatch response: %s\n' "$submission" >&2
    exit 1
}
printf '%s\n' "$job_id" | tee "$BASE/logs/$MODE/job_id.txt"
