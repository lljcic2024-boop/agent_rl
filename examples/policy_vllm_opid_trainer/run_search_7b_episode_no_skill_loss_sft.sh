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
    HF_MODEL_PATH="$MODELS_ROOT/Qwen2.5-7B-Instruct-search-episode-skill-sft"
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    exit 1
fi

# Initialize both policy and policy-vLLM analyzer from the Search QA SFT
# episode-skill model.
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"

# Use the SFT model as an episode-only analyzer during OPID RL, while disabling
# the auxiliary skill-generation LM loss.
export OPID_SKILL_MODE="${OPID_SKILL_MODE:-episode_only}"
export OPID_SDAR_LOSS_COEF="${OPID_SDAR_LOSS_COEF:-0.01}"
export OPID_SKILL_GEN_LOSS_ENABLE="${OPID_SKILL_GEN_LOSS_ENABLE:-False}"
export OPID_SKILL_GEN_LOSS_COEF="${OPID_SKILL_GEN_LOSS_COEF:-0.0}"
export OPID_ANALYSIS_BACKEND="${OPID_ANALYSIS_BACKEND:-policy_vllm}"
if [[ -z "${OPID_ANALYSIS_PROMPT_VERSION:-}" ]]; then
    if [[ "$(basename "$HF_MODEL_PATH")" == *"skill_only"* ]]; then
        export OPID_ANALYSIS_PROMPT_VERSION="search_skill_only"
    elif [[ "$(basename "$HF_MODEL_PATH")" == *"strategy_bank"* ]]; then
        export OPID_ANALYSIS_PROMPT_VERSION="search_strategy_bank"
    else
        export OPID_ANALYSIS_PROMPT_VERSION="opid"
    fi
elif [[ "$OPID_ANALYSIS_PROMPT_VERSION" == "strategy_bank" ]]; then
    export OPID_ANALYSIS_PROMPT_VERSION="search_strategy_bank"
elif [[ "$OPID_ANALYSIS_PROMPT_VERSION" == "skill_only" ]]; then
    export OPID_ANALYSIS_PROMPT_VERSION="search_skill_only"
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opid-grpo_qwen2.5_7b_search_episode_no_skill_loss_sft_policy-vllm}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

exec "$SCRIPT_DIR/run_search_both_no_skill_loss_base.sh" \
    algorithm.opid.skill_gen.enable="$OPID_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$OPID_SKILL_GEN_LOSS_COEF" \
    "$@"
