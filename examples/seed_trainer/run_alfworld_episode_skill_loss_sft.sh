#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set +x
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    set -x
fi

if [[ -z "${HF_MODEL_PATH:-}" ]]; then
    if [[ -z "${MODELS_ROOT:-}" ]]; then
        echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
        exit 1
    fi
    HF_MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct-alfworld-episode-skill-sft"
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    exit 1
fi

# Initialize both policy and policy-vLLM analyzer from the SFT episode-skill model.
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"

# Follow the episode-skill-gen policy-vLLM pipeline and keep the auxiliary
# skill-generation LM loss enabled during RL.
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-True}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_FAILED_REWARD_MODE="${SEED_SKILL_GEN_FAILED_REWARD_MODE:-negate}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_3b_alfworld_episode_skill_loss_sft}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/$EXPERIMENT_NAME}"

exec "$SCRIPT_DIR/run_alfworld_both_no_skill_loss_base.sh" "$@"
