#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKILL_PROMPT_VERSION="${SKILL_PROMPT_VERSION:-self_glm}"
export SFT_TEACHER_SHORT="${SFT_TEACHER_SHORT:-glm}"
export MAX_LENGTH="${MAX_LENGTH:-8192}"

exec "$SCRIPT_DIR/run_sft_qwen3.sh" "$@"
