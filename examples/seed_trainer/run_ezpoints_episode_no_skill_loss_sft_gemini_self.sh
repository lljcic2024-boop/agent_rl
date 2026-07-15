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

CONDA_ENV="${CONDA_ENV:-copd}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

if [[ -z "${HF_MODEL_PATH:-}" ]]; then
    if [[ -z "${MODELS_ROOT:-}" ]]; then
        echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
        exit 1
    fi
    HF_MODEL_PATH="$MODELS_ROOT/Qwen2.5-VL-3B-Instruct-ezpoints-episode-skill-sft-gemini-self"
fi

if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    echo "Run scripts/ezpoints_visual_seed_pipeline/run_sft.sh first, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

# Initialize both the EZPoints visual policy and the policy-vLLM analyzer from
# the SFT episode-skill model. Keep the SDAR signal, but disable the auxiliary
# skill-generation language-model loss during RL.
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-False}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.0}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"
export SEED_ANALYSIS_PROMPT_VERSION="${SEED_ANALYSIS_PROMPT_VERSION:-seed_visual}"
export SEED_MODE="${SEED_MODE:-mean_std_norm}"

export PROJECT_NAME="${PROJECT_NAME:-agentic_ezpoints}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_vl_3b_ezpoints_episode_no_skill_loss_sft_gemini_self_mean-std-norm}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

exec "$SCRIPT_DIR/run_ezpoints.sh" \
    algorithm.seed.skill_gen.enable="$SEED_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$SEED_SKILL_GEN_LOSS_COEF" \
    "$@"
