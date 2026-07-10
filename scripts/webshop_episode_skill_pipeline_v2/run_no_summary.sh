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
# shellcheck source=../sft_teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft_teacher_naming.sh"

export INCLUDE_EPISODE_SUMMARY="${INCLUDE_EPISODE_SUMMARY:-false}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
export BASELINE_HISTORY_LENGTH="${BASELINE_HISTORY_LENGTH:-5}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export POLICY_MAX_COMPLETION_TOKENS="${POLICY_MAX_COMPLETION_TOKENS:-512}"

exec "$SCRIPT_DIR/run.sh" "$@"
