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

CONDA_ENV="${CONDA_ENV:-sgop-webshop}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

if [[ -z "${MODELS_ROOT:-}" && -z "${HF_MODEL_PATH:-}" ]]; then
    echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

export HF_MODEL_PATH="${HF_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct-webshop-episode-skill-sft-glm-self}"
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    echo "Run SFT first, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-False}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.0}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"
export SEED_ANALYSIS_PROMPT_VERSION="${SEED_ANALYSIS_PROMPT_VERSION:-seed}"
export SEED_MODE="${SEED_MODE:-mean_norm}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_3b_webshop_sft_glm_self_mean-norm_exp2}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

exec bash "$SCRIPT_DIR/run_webshop_both_no_skill_loss_base.sh" \
    algorithm.seed.analysis_include_episode_summary=True \
    algorithm.seed.skill_gen.enable="$SEED_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$SEED_SKILL_GEN_LOSS_COEF" \
    "$@"
