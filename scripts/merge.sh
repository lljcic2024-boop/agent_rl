#!/usr/bin/env bash

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT, e.g. /path/to/models}"

ALFWORLD_CKPT=/data/models/Qwen/Qwen2.5-3B-Instruct-alfworld-episode-skill-sft/opid-rl-sft-analyzer-qwen25-3b-alfworld-no-skill-loss/global_step_150/actor
ALFWORLD_TARGET="${ALFWORLD_TARGET:-${MODELS_ROOT}/release/SEED-ALFWorld-3B}"

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$ALFWORLD_CKPT" \
    --target_dir "$ALFWORLD_TARGET"
