#!/usr/bin/env bash

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPID_SKILL_MODE=${OPID_SKILL_MODE:-episode_only}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opid-sdar-episode-skill-gen-grpo_qwen2.5_3b_alfworld_policy-vllm}

exec "$SCRIPT_DIR/run_alfworld_opid_skill_gen_policy_vllm.sh" "$@"
