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
    HF_MODEL_PATH="$MODELS_ROOT/Qwen3-1.7B-webshop-episode-skill-sft"
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    echo "Run scripts/webshop_episode_skill_pipeline_v2/run_sft_qwen3.sh first, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

# Initialize both policy and policy-vLLM analyzer from the WebShop Qwen3 SFT
# episode-summary + episode-skill model.
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"

# Use the SFT model as an episode-only analyzer during SEED RL, while disabling
# the auxiliary skill-generation LM loss.
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-False}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.0}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"
export SEED_ANALYSIS_PROMPT_VERSION="${SEED_ANALYSIS_PROMPT_VERSION:-seed}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen3_1.7b_webshop_sft}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

thinking_arg_seen=false
for arg in "$@"; do
    if [[ "$arg" == data.apply_chat_template_kwargs.enable_thinking=* ]] \
        || [[ "$arg" == +data.apply_chat_template_kwargs.enable_thinking=* ]]; then
        thinking_arg_seen=true
        break
    fi
done

extra_args=()
if [[ "$thinking_arg_seen" == "false" ]]; then
    extra_args+=(+data.apply_chat_template_kwargs.enable_thinking=False)
fi

exec bash "$SCRIPT_DIR/run_webshop_both_no_skill_loss_base.sh" \
    "${extra_args[@]}" \
    algorithm.seed.skill_gen.enable="$SEED_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$SEED_SKILL_GEN_LOSS_COEF" \
    "$@"
