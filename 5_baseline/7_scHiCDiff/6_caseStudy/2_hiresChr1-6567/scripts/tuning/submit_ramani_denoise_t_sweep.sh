#!/usr/bin/env bash
set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/6_caseStudy/2_hiresChr1-6567"
RUNNER="$BASE/scripts/tuning/run_ramani_denoise_t_sweep.sbatch"
SWEEP_ROOT="$BASE/results/full/ramani/denoise_t_sweep"
PARTITION="${1:-gpu4RQ}"
CONCURRENCY="${2:-2}"

case "$PARTITION" in
    gpu2Q|gpu4Q|gpu8Q|gpu4RQ) ;;
    *)
        printf 'unsupported GPU partition: %s\n' "$PARTITION" >&2
        exit 2
        ;;
esac
if [[ ! "$CONCURRENCY" =~ ^[1-6]$ ]]; then
    printf 'concurrency must be an integer from 1 through 6\n' >&2
    exit 2
fi

SBATCH_ARGS=(
    --parsable
    --partition="$PARTITION"
    --qos=gpuq
    --gres=gpu:1
    --cpus-per-task=19
    --mem=40G
    --time=01:00:00
    --array="0-5%$CONCURRENCY"
    --job-name=ramani_t_sweep
    --output="$SWEEP_ROOT/logs/%x_%A_%a.out"
    --error="$SWEEP_ROOT/logs/%x_%A_%a.err"
)

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}" "$RUNNER"
    printf '\n'
    exit 0
fi

if [[ ! -s "$RUNNER" ]]; then
    printf 'missing runner: %s\n' "$RUNNER" >&2
    exit 1
fi
mkdir -p "$SWEEP_ROOT/logs"

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
            node = state = partitions = cfg = alloc = ""
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
                    cfg = $i
                } else if ($i ~ /^AllocTRES=/) {
                    alloc = $i
                }
            }
            free_gpu = gpu_count(cfg) - gpu_count(alloc)
            if (("," partitions ",") ~ ("," target_partition ",") &&
                    state !~ /(DRAIN|DOWN|FAIL|INVAL)/ && free_gpu > 0) {
                printf "%s|%s|free_gpu=%d\n", node, state, free_gpu
            }
        }
    '
)"
if [[ -z "$free_nodes" ]]; then
    printf 'no free GPU visible on %s\n' "$PARTITION" >&2
    exit 1
fi
printf '[INFO] free GPU nodes on %s:\n%s\n' "$PARTITION" "$free_nodes"

submission="$(sbatch "${SBATCH_ARGS[@]}" "$RUNNER")"
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'unexpected sbatch response: %s\n' "$submission" >&2
    exit 1
fi
printf '%s\n' "$job_id" > "$SWEEP_ROOT/job_id.txt"
printf '%s\n' "$job_id"
