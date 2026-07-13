#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${MODEL_PATH:-}" && -z "${MODELS_ROOT:-}" ]]; then
    echo "Please set MODELS_ROOT in $ENV_FILE, or set MODEL_PATH explicitly." >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-7B-Instruct}"
export MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
export SKILL_MODEL="${SKILL_MODEL:-${OPENAI_MODEL:?Please set OPENAI_MODEL in $ENV_FILE.}}"
export SFT_TEACHER_SHORT="${SFT_TEACHER_SHORT:-gpt56}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_7b_gpt56_self}"
export BASELINE_ROLLOUTS="${BASELINE_ROLLOUTS:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_7b_glm_self/baseline_rollouts.jsonl}"
export BASELINE_HISTORY_LENGTH="${BASELINE_HISTORY_LENGTH:-5}"
export START_VLLM="${START_VLLM:-0}"
export OVERWRITE="${OVERWRITE:-true}"

exec "$SCRIPT_DIR/run.sh" "$@"
