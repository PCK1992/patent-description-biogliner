#!/usr/bin/env bash
# Submit from the shared checkout. No Stanford account or partition is assumed.
set -euo pipefail
test -f code/score.py
test -f input/zenodo_15062212.json
test -f original_roster_audit.json
export PYTHON="$(command -v python)"
[[ "$PYTHON" = /* ]]
mkdir -p logs
base=(--parsable --chdir="$PWD" --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err --kill-on-invalid-dep=yes)
if [[ -n ${ACCOUNT:-} ]]; then base+=(--account="$ACCOUNT"); fi
if [[ -n ${MAIL_TO:-} ]]; then base+=(--mail-user="$MAIL_TO" --mail-type=BEGIN,END,FAIL); fi
cpu=(--cpus-per-task=2 --mem=16G)
gpu=(--gres="${GRES:-gpu:1}" --mem=32G)
if [[ -n ${CPU_PARTITION:-} ]]; then cpu+=(--partition="$CPU_PARTITION"); fi
if [[ -n ${GPU_PARTITION:-} ]]; then gpu+=(--partition="$GPU_PARTITION"); fi
job() {
    local response
    response=$(sbatch "${base[@]}" "$@")
    printf '%s\n' "${response%%;*}"
}
case "${1:-}" in
    pilot)
        prep=$(job "${cpu[@]}" -t 04:00:00 -J desc76_text scripts/worker.sh year 1976)
        pilot=$(job "${gpu[@]}" --cpus-per-task=2 -t 00:15:00 -J desc76_pilot --dependency="afterok:$prep" scripts/worker.sh pilot)
        echo "1976 text: $prep; GPU pilot: $pilot. Wait for pilot/1976/PASS.json before production."
        ;;
    production|all)
        if [[ $1 == production && ! -s pilot/1976/PASS.json ]]; then
            echo "Run a local pilot or use the 'all' dependency chain." >&2; exit 1
        fi
        if [[ -n ${START_YEAR:-} ]]; then
            [[ -z ${SHARDS:-} && $START_YEAR =~ ^(19|20)[0-9]{2}$ ]] && (( START_YEAR >= 1976 && START_YEAR <= 2024 )) || { echo "Use either START_YEAR (1976-2024) or SHARDS." >&2; exit 2; }
            SHARDS="$(((START_YEAR-1976)*8))-391"
        fi
        SHARDS=${SHARDS:-0-391}
        [[ $SHARDS =~ ^([0-9]+)-([0-9]+)$ ]] || { echo "Invalid inclusive retry range." >&2; exit 2; }
        first=${BASH_REMATCH[1]}; last=${BASH_REMATCH[2]}
        (( 0 <= first && first <= last && last <= 391 ))
        cap=${MAX_GPUS:-4}
        [[ $cap =~ ^[1-9][0-9]*$ ]]
        y0=$((first/8)); y1=$((last/8))
        years_array=$y0
        if (( y0 != y1 )); then years_array="$y0-$y1"; fi
        if [[ $1 == all && $y0 != 0 ]]; then years_array="0,$years_array"; fi
        years=$(job "${cpu[@]}" -t 07:00:00 -J desc_years --array="${years_array}%3" scripts/worker.sh year)
        texts=$(job "${cpu[@]}" -t 02:00:00 -J desc_text_audit --dependency="afterok:$years" scripts/worker.sh audit texts "$SHARDS")
        dependencies="afterok:$texts"
        if [[ $1 == all ]]; then
            pilot=$(job "${gpu[@]}" --cpus-per-task=2 -t 00:15:00 -J desc76_pilot --dependency="afterok:${years}_0" scripts/worker.sh pilot)
            dependencies+=":$pilot"
            echo "1976 GPU pilot: $pilot"
        fi
        score=$(job "${gpu[@]}" --cpus-per-task=4 -t "${GPU_TIME:-24:00:00}" -J desc_score --array="${SHARDS}%$cap" --dependency="$dependencies" scripts/worker.sh full "$SHARDS")
        audit=$(job "${cpu[@]}" -t "${AUDIT_TIME:-04:00:00}" -J desc_span_audit --dependency="afterok:$score" scripts/worker.sh audit scores "$SHARDS")
        echo "Years: $years; text audit: $texts; production: $score; span audit: $audit. Range: $SHARDS."
        ;;
    *) echo "Usage: bash scripts/submit.sh all|pilot|production" >&2; exit 2 ;;
esac
