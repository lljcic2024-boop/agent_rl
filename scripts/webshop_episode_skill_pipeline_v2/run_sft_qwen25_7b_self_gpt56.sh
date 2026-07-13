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

export SFT_TEACHER_SHORT="${SFT_TEACHER_SHORT:-gpt56}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_7b_gpt56_self}"
export EXPORT_MODEL_NAME="${EXPORT_MODEL_NAME:-Qwen2.5-7B-Instruct-webshop-episode-skill-sft-gpt56-self}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-7b-webshop-episode-skill-sft-gpt56-self-ep${TOTAL_EPOCHS:-3}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MODELS_ROOT:?Please set MODELS_ROOT}/outputs/sft/webshop_episode_skill_analyzer_v2_qwen25_7b_gpt56_self_ep${TOTAL_EPOCHS:-3}_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-$MODELS_ROOT/logs/sft/webshop_episode_skill_analyzer_v2_qwen25_7b_gpt56_self_ep${TOTAL_EPOCHS:-3}_${RUN_ID}.log}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"

exec bash "$SCRIPT_DIR/run_sft_qwen25_7b.sh" "$@"
