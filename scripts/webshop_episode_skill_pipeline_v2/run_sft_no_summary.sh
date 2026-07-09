#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_3b}"
export EXPORT_MODEL_NAME="${EXPORT_MODEL_NAME:-Qwen2.5-3B-Instruct-webshop-episode-skill-sft}"

exec "$SCRIPT_DIR/run_sft.sh" "$@"
