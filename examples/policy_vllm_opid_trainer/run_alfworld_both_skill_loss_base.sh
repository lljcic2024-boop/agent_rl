#!/usr/bin/env bash

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Skill-generation auxiliary loss. This trainer keeps PPO RL loss and SDAR OPD
# loss enabled, then adds reward-weighted optimization for the policy-vllm skill
# JSON that produced the OPD signal.
export OPID_SKILL_GEN_LOSS_ENABLE=${OPID_SKILL_GEN_LOSS_ENABLE:-True}
export OPID_SKILL_GEN_LOSS_COEF=${OPID_SKILL_GEN_LOSS_COEF:-0.01}
export OPID_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU=${OPID_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU:-${OPID_SKILL_GEN_MICRO_BATCH_SIZE:-1}}
export OPID_SKILL_MODE=${OPID_SKILL_MODE:-episode_step}
export OPID_SKILL_GEN_MAX_SAMPLES=${OPID_SKILL_GEN_MAX_SAMPLES:-all}
export OPID_SKILL_GEN_VALID_JSON_BONUS=${OPID_SKILL_GEN_VALID_JSON_BONUS:-0.0}
export OPID_SKILL_GEN_NON_EMPTY_SKILL_BONUS=${OPID_SKILL_GEN_NON_EMPTY_SKILL_BONUS:-0.0}
export OPID_SKILL_GEN_TOO_LONG_PENALTY=${OPID_SKILL_GEN_TOO_LONG_PENALTY:-0.0}
export OPID_SKILL_GEN_MAX_OUTPUT_CHARS=${OPID_SKILL_GEN_MAX_OUTPUT_CHARS:-1200}
export OPID_SKILL_GEN_REWARD_CLIP=${OPID_SKILL_GEN_REWARD_CLIP:-2.0}
export OPID_SKILL_GEN_FAILED_REWARD_MODE=${OPID_SKILL_GEN_FAILED_REWARD_MODE:-negate}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opid-grpo_qwen2.5_3b_alfworld_both_skill_loss_base_policy-vllm}

exec "$SCRIPT_DIR/run_alfworld_both_no_skill_loss_base.sh" "$@"
