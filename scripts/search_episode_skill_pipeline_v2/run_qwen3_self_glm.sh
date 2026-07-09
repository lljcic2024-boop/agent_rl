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

# Generate baseline rollouts with the unsupervised Qwen3-1.7B base model, then
# ask the GLM analyzer to produce episode_summary + episode_skill labels.
if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$MODELS_ROOT/Qwen3-1.7B"
fi
export MODEL_PATH
export MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/search_episode_skill_pipeline_v2_qwen3_1.7b_glm}"
export SKILL_PROMPT_VERSION="${SKILL_PROMPT_VERSION:-opid}"
export SKILL_MODEL="${SKILL_MODEL:-glm-5.2}"

# Qwen3 should act as an instruction-following search policy here, not as a
# thinking-model data generator.
export POLICY_EXTRA_BODY_JSON="${POLICY_EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"

export VLLM_LOG_FILE="${VLLM_LOG_FILE:-$PROJECT_ROOT/logs/vllm/search_policy_${MODEL_NAME}_self_glm_$(date +%Y%m%d_%H%M%S).log}"

exec "$SCRIPT_DIR/run.sh" "$@"
