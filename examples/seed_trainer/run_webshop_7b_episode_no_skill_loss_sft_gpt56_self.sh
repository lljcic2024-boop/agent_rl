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

CONDA_ENV="${CONDA_ENV:-skillrl-webshop}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

if [[ -z "${MODELS_ROOT:-}" && -z "${HF_MODEL_PATH:-}" ]]; then
    echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

export HF_MODEL_PATH="${HF_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-7B-Instruct-webshop-episode-skill-sft-gpt56-self}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_7b_webshop_sft_gpt56_self_mean-norm_exp1}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"
# Eight concurrent vLLM workers can exceed this host's aggregate pinned-memory
# limit while sleeping. Pageable backups are slower but preserve the same data.
export VERL_VLLM_SLEEP_PIN_MEMORY="${VERL_VLLM_SLEEP_PIN_MEMORY:-0}"

exec bash "$SCRIPT_DIR/run_webshop_7b_episode_no_skill_loss_sft_self.sh" "$@"
