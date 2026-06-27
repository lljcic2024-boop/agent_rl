#!/usr/bin/env bash

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT, e.g. /path/to/models}"

ALFWORLD_CKPT="${ALFWORLD_CKPT:-${MODELS_ROOT}/ckpt/copd-grpo_qwen3_1.7b_alfworld_llm-5_episode-step-hint-plus-v3_opd-adv-0.001_exp1/global_step_150/actor}"
ALFWORLD_TARGET="${ALFWORLD_TARGET:-${MODELS_ROOT}/release/OPID-ALFWorld-1.7B}"

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$ALFWORLD_CKPT" \
    --target_dir "$ALFWORLD_TARGET"
