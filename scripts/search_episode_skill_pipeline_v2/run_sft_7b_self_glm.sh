#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SKILL_PROMPT_VERSION="${SKILL_PROMPT_VERSION:-self_glm}"

exec "$SCRIPT_DIR/run_sft_7b.sh" "$@"
