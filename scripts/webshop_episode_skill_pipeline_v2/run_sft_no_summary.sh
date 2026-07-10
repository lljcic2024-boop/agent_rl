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

export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
export EXPORT_MODEL_NAME="${EXPORT_MODEL_NAME:-Qwen2.5-3B-Instruct-webshop-episode-skill-sft-${SFT_SELF_SUFFIX}}"

exec "$SCRIPT_DIR/run_sft.sh" "$@"
