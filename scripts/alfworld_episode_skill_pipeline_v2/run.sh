#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-copd}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/alfworld_episode_skill_pipeline_v2_qwen25_3b}"
BASELINE_ROLLOUTS="${BASELINE_ROLLOUTS:-$PROJECT_ROOT/outputs/alfworld_episode_skill_pipeline_qwen25_3b/baseline_rollouts.jsonl}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

SKILL_BASE_URL="${SKILL_BASE_URL:-${OPENAI_BASE_URL:?Please set OPENAI_BASE_URL in .env or SKILL_BASE_URL.}}"
SKILL_API_KEY="${SKILL_API_KEY:-${OPENAI_API_KEY:?Please set OPENAI_API_KEY in .env or SKILL_API_KEY.}}"
SKILL_MODEL="${SKILL_MODEL:-${OPENAI_MODEL:?Please set OPENAI_MODEL in .env or SKILL_MODEL.}}"
SKILL_TEMPERATURE="${SKILL_TEMPERATURE:-0.0}"
SKILL_MAX_COMPLETION_TOKENS="${SKILL_MAX_COMPLETION_TOKENS:-1024}"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-120}"
SKILL_RETRIES="${SKILL_RETRIES:-5}"
SKILL_RETRY_DELAY="${SKILL_RETRY_DELAY:-1.0}"
SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-128}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SEED="${SEED:-2026}"
RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"
REGENERATE_CANDIDATES="${REGENERATE_CANDIDATES:-true}"
PROGRESS_MONITOR="${PROGRESS_MONITOR:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

args=(
    "$SCRIPT_DIR/pipeline_v2.py"
    --env-file "$ENV_FILE"
    --output-dir "$OUTPUT_DIR"
    --baseline-rollouts "$BASELINE_ROLLOUTS"
    --skill-base-url "$SKILL_BASE_URL"
    --skill-model "$SKILL_MODEL"
    --skill-temperature "$SKILL_TEMPERATURE"
    --skill-max-completion-tokens "$SKILL_MAX_COMPLETION_TOKENS"
    --skill-timeout "$SKILL_TIMEOUT"
    --skill-retries "$SKILL_RETRIES"
    --skill-retry-delay "$SKILL_RETRY_DELAY"
    --skill-gen-workers "$SKILL_GEN_WORKERS"
    --sft-val-ratio "$SFT_VAL_RATIO"
    --seed "$SEED"
)

export SKILL_OPENAI_API_KEY="$SKILL_API_KEY"
export SKILL_OPENAI_BASE_URL="$SKILL_BASE_URL"
export SKILL_OPENAI_MODEL="$SKILL_MODEL"

if [[ -n "${MAX_CANDIDATES:-}" ]]; then
    args+=(--max-candidates "$MAX_CANDIDATES")
fi

if [[ -n "${SKILL_EXTRA_BODY_JSON:-}" ]]; then
    args+=(--skill-extra-body-json "$SKILL_EXTRA_BODY_JSON")
fi

if [[ "$OVERWRITE" == "true" ]]; then
    args+=(--overwrite)
elif [[ "$REGENERATE_CANDIDATES" == "true" ]]; then
    args+=(--regenerate-candidates)
elif [[ "$RESUME" == "true" ]]; then
    args+=(--resume)
fi

cd "$PROJECT_ROOT"

if [[ ! -f "$BASELINE_ROLLOUTS" ]]; then
    echo "Baseline rollouts not found: $BASELINE_ROLLOUTS" >&2
    exit 1
fi

progress_monitor_pid=""

print_progress() {
    local progress_file="$OUTPUT_DIR/progress.json"
    if [[ ! -f "$progress_file" ]]; then
        echo "[progress] waiting for $progress_file"
        return
    fi

    python3 - "$progress_file" <<'PY' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    progress = json.load(f)

parts = [
    f"[progress] {progress.get('updated_at', '')}",
    f"stage={progress.get('stage', 'unknown')}",
    f"status={progress.get('status', 'unknown')}",
]
for key in (
    "baseline_rollouts",
    "used_for_generation",
    "completed_skills",
    "expected_skills",
    "completed_in_current_run",
    "pending_in_current_run",
    "parse_ok_skills",
    "sft_records",
    "last_skill_id",
    "last_parse_ok",
):
    if key in progress:
        parts.append(f"{key}={progress[key]}")
print(" ".join(parts), flush=True)
PY
}

cleanup() {
    if [[ -n "$progress_monitor_pid" ]]; then
        kill "$progress_monitor_pid" >/dev/null 2>&1 || true
        wait "$progress_monitor_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "$PROGRESS_MONITOR" == "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
    print_progress
    (
        while true; do
            sleep "$PROGRESS_INTERVAL"
            print_progress
        done
    ) &
    progress_monitor_pid=$!
fi

echo "Running ALFWorld episode-skill pipeline v2"
echo "  conda env:          $CONDA_ENV"
echo "  output dir:         $OUTPUT_DIR"
echo "  baseline rollouts:  $BASELINE_ROLLOUTS"
echo "  skill model:        $SKILL_MODEL @ $SKILL_BASE_URL"
echo "  skill gen workers:  $SKILL_GEN_WORKERS"
echo "  regenerate:         $REGENERATE_CANDIDATES"

set +e
conda run -n "$CONDA_ENV" python "${args[@]}"
run_status=$?
set -e

print_progress
exit "$run_status"
