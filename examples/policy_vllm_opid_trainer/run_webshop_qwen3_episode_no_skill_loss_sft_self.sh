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

if [[ -z "${MODELS_ROOT:-}" && -z "${HF_MODEL_PATH:-}" ]]; then
    echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

export HF_MODEL_PATH="${HF_MODEL_PATH:-$MODELS_ROOT/Qwen3-1.7B-webshop-episode-skill-sft-glm-self}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen3_1.7b_webshop_episode_no_skill_loss_sft_glm_self_policy-vllm}"

exec "$SCRIPT_DIR/run_webshop_qwen3_episode_no_skill_loss_sft.sh" "$@"
