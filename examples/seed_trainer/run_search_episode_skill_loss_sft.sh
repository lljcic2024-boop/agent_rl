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

# Use the SFT model as an episode-only analyzer during SEED RL, while keeping
# the auxiliary skill-generation LM loss enabled.
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-True}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU="${SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU:-${SEED_SKILL_GEN_MICRO_BATCH_SIZE:-1}}"
export SEED_SKILL_GEN_MAX_SAMPLES="${SEED_SKILL_GEN_MAX_SAMPLES:-all}"
export SEED_SKILL_GEN_VALID_JSON_BONUS="${SEED_SKILL_GEN_VALID_JSON_BONUS:-0.0}"
export SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS="${SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS:-0.0}"
export SEED_SKILL_GEN_TOO_LONG_PENALTY="${SEED_SKILL_GEN_TOO_LONG_PENALTY:-0.0}"
export SEED_SKILL_GEN_MAX_OUTPUT_CHARS="${SEED_SKILL_GEN_MAX_OUTPUT_CHARS:-1200}"
export SEED_SKILL_GEN_REWARD_CLIP="${SEED_SKILL_GEN_REWARD_CLIP:-2.0}"
export SEED_SKILL_GEN_FAILED_REWARD_MODE="${SEED_SKILL_GEN_FAILED_REWARD_MODE:-negate}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_3b_search_episode_skill_loss_sft}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

exec "$SCRIPT_DIR/run_search_both_no_skill_loss_base.sh" \
    algorithm.seed.skill_gen.enable="$SEED_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$SEED_SKILL_GEN_LOSS_COEF" \
    actor_rollout_ref.actor.skill_gen_micro_batch_size_per_gpu="$SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU" \
    algorithm.seed.skill_gen.max_samples="$SEED_SKILL_GEN_MAX_SAMPLES" \
    algorithm.seed.skill_gen.valid_json_bonus="$SEED_SKILL_GEN_VALID_JSON_BONUS" \
    algorithm.seed.skill_gen.non_empty_skill_bonus="$SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS" \
    algorithm.seed.skill_gen.too_long_penalty="$SEED_SKILL_GEN_TOO_LONG_PENALTY" \
    algorithm.seed.skill_gen.max_output_chars="$SEED_SKILL_GEN_MAX_OUTPUT_CHARS" \
    algorithm.seed.skill_gen.reward_clip="$SEED_SKILL_GEN_REWARD_CLIP" \
    algorithm.seed.skill_gen.failed_reward_mode="$SEED_SKILL_GEN_FAILED_REWARD_MODE" \
    "$@"
