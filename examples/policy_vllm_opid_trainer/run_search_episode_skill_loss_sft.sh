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
    HF_MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct-search-episode-skill-sft"
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    exit 1
fi

# Initialize both policy and policy-vLLM analyzer from the Search QA SFT
# episode-skill model.
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"

# Use the SFT model as an episode-only analyzer during OPID RL, while keeping
# the auxiliary skill-generation LM loss enabled.
export OPID_SKILL_MODE="${OPID_SKILL_MODE:-episode_only}"
export OPID_SDAR_LOSS_COEF="${OPID_SDAR_LOSS_COEF:-0.01}"
export OPID_SKILL_GEN_LOSS_ENABLE="${OPID_SKILL_GEN_LOSS_ENABLE:-True}"
export OPID_SKILL_GEN_LOSS_COEF="${OPID_SKILL_GEN_LOSS_COEF:-0.01}"
export OPID_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU="${OPID_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU:-${OPID_SKILL_GEN_MICRO_BATCH_SIZE:-1}}"
export OPID_SKILL_GEN_MAX_SAMPLES="${OPID_SKILL_GEN_MAX_SAMPLES:-all}"
export OPID_SKILL_GEN_VALID_JSON_BONUS="${OPID_SKILL_GEN_VALID_JSON_BONUS:-0.0}"
export OPID_SKILL_GEN_NON_EMPTY_SKILL_BONUS="${OPID_SKILL_GEN_NON_EMPTY_SKILL_BONUS:-0.0}"
export OPID_SKILL_GEN_TOO_LONG_PENALTY="${OPID_SKILL_GEN_TOO_LONG_PENALTY:-0.0}"
export OPID_SKILL_GEN_MAX_OUTPUT_CHARS="${OPID_SKILL_GEN_MAX_OUTPUT_CHARS:-1200}"
export OPID_SKILL_GEN_REWARD_CLIP="${OPID_SKILL_GEN_REWARD_CLIP:-2.0}"
export OPID_SKILL_GEN_FAILED_REWARD_MODE="${OPID_SKILL_GEN_FAILED_REWARD_MODE:-negate}"
export OPID_ANALYSIS_BACKEND="${OPID_ANALYSIS_BACKEND:-policy_vllm}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opid-grpo_qwen2.5_3b_search_episode_skill_loss_sft_policy-vllm}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

exec "$SCRIPT_DIR/run_search_both_no_skill_loss_base.sh" \
    algorithm.opid.skill_gen.enable="$OPID_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$OPID_SKILL_GEN_LOSS_COEF" \
    actor_rollout_ref.actor.skill_gen_micro_batch_size_per_gpu="$OPID_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU" \
    algorithm.opid.skill_gen.max_samples="$OPID_SKILL_GEN_MAX_SAMPLES" \
    algorithm.opid.skill_gen.valid_json_bonus="$OPID_SKILL_GEN_VALID_JSON_BONUS" \
    algorithm.opid.skill_gen.non_empty_skill_bonus="$OPID_SKILL_GEN_NON_EMPTY_SKILL_BONUS" \
    algorithm.opid.skill_gen.too_long_penalty="$OPID_SKILL_GEN_TOO_LONG_PENALTY" \
    algorithm.opid.skill_gen.max_output_chars="$OPID_SKILL_GEN_MAX_OUTPUT_CHARS" \
    algorithm.opid.skill_gen.reward_clip="$OPID_SKILL_GEN_REWARD_CLIP" \
    algorithm.opid.skill_gen.failed_reward_mode="$OPID_SKILL_GEN_FAILED_REWARD_MODE" \
    "$@"
