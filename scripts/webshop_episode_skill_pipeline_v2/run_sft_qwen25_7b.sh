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
# shellcheck source=../sft_teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft_teacher_naming.sh"

if [[ -z "${MODELS_ROOT:-}" && -z "${MODEL_PATH:-}" ]]; then
    echo "Please set MODELS_ROOT in $ENV_FILE, or set MODEL_PATH explicitly." >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-7B-Instruct}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
export EXPORT_MODEL_NAME="${EXPORT_MODEL_NAME:-Qwen2.5-7B-Instruct-webshop-episode-skill-sft-${SFT_SELF_SUFFIX}}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-7b-webshop-episode-skill-sft-v2-ep${TOTAL_EPOCHS:-3}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_ROOT/outputs/sft/webshop_episode_skill_analyzer_v2_qwen25_7b_ep${TOTAL_EPOCHS:-3}_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-$MODELS_ROOT/logs/sft/webshop_episode_skill_analyzer_v2_qwen25_7b_ep${TOTAL_EPOCHS:-3}_${RUN_ID}.log}"

exec "$SCRIPT_DIR/run_sft.sh" "$@"
