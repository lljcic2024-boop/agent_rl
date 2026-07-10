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
    HF_MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct-webshop-episode-skill-sft"
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"

# Use the SFT model as an episode-only analyzer, and feed the
# generated episode skill through SEED teacher advantage instead of SDAR loss.
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_STEP_ADV_W="${SEED_STEP_ADV_W:-0.0}"
export SEED_EPISODE_SKILL_TEACHER_ADV_W="${SEED_EPISODE_SKILL_TEACHER_ADV_W:-0.001}"
export SEED_STEP_SKILL_TEACHER_ADV_W="${SEED_STEP_SKILL_TEACHER_ADV_W:-0.0}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.0}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-False}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.0}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_3b_webshop_episode_advantage_sft}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

exec "$SCRIPT_DIR/run_webshop_both_no_skill_loss_base.sh" \
    algorithm.seed.analysis_include_episode_summary=False \
    algorithm.seed.skill_gen.enable="$SEED_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$SEED_SKILL_GEN_LOSS_COEF" \
    "$@"
