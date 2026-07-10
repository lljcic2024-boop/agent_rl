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
SFT_TEACHER_SHORT="${SFT_TEACHER_SHORT:-glm}"
# shellcheck source=../sft_teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft_teacher_naming.sh"

stop_rollout_vllm() {
    local pattern="${1:-}"
    if [[ -z "$pattern" ]]; then
        return 0
    fi
    local pids
    pids="$(pgrep -f "$pattern" || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi
    echo "Stopping rollout vLLM process(es) before SFT: $pids"
    # shellcheck disable=SC2086
    kill $pids >/dev/null 2>&1 || true
    sleep 5
    pids="$(pgrep -f "$pattern" || true)"
    if [[ -n "$pids" ]]; then
        echo "Force stopping rollout vLLM process(es): $pids"
        # shellcheck disable=SC2086
        kill -9 $pids >/dev/null 2>&1 || true
    fi
}

# The data-generation script normally cleans up its own vLLM server. This extra
# guard prevents the rollout server from occupying GPUs during SFT if a previous
# run was interrupted.
stop_rollout_vllm "vllm serve .*Qwen2.5-7B-Instruct.*--port 60002"
stop_rollout_vllm "vllm.entrypoints.openai.*Qwen2.5-7B-Instruct.*60002"

export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_7b_${SFT_SELF_DIR_SUFFIX}}"
export EXPORT_MODEL_NAME="${EXPORT_MODEL_NAME:-Qwen2.5-7B-Instruct-webshop-episode-skill-sft-${SFT_SELF_SUFFIX}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-7b-webshop-episode-skill-sft-glm-self-ep${TOTAL_EPOCHS:-3}}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MODELS_ROOT:-/data/models/Qwen}/outputs/sft/webshop_episode_skill_analyzer_v2_qwen25_7b_glm_self_ep${TOTAL_EPOCHS:-3}_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-${MODELS_ROOT:-/data/models/Qwen}/logs/sft/webshop_episode_skill_analyzer_v2_qwen25_7b_glm_self_ep${TOTAL_EPOCHS:-3}_${RUN_ID}.log}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"

exec bash "$SCRIPT_DIR/run_sft_qwen25_7b.sh" "$@"
