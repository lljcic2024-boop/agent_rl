#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opid-grpo_qwen3_1.7b_webshop_no_summary}"

exec "$SCRIPT_DIR/run_webshop_opid_guide_qwen3.sh" \
    algorithm.opid.analysis_include_episode_summary=False \
    "$@"
