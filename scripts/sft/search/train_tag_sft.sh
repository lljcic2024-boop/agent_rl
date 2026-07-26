#!/usr/bin/env bash
# Tag-SFT training entry (Phase 1a): fine-tune the student on function-tagged
# rewrites produced by tag_rewrite.sh.
#
# Student default = Qwen3-8B (decided 2026-07: same tokenizer family as the
# Qwen3-30B-A3B teacher, required for token-level external log-prob scoring).
# Exports to $MODELS_ROOT/Qwen3-8B-search-tag-sft, which is the default
# student checkpoint of examples/seed_trainer/run_search_tag_distill_qwen3_8b.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET_NAME="search"
DATASET_LABEL="Search QA (function-tag SFT)"
MODEL_BASENAME="${MODEL_BASENAME:-Qwen3-8B}"
MODEL_TAG="${MODEL_TAG:-qwen3_8b}"
DEFAULT_SFT_CONDA_ENV="copd"
DEFAULT_LR="5e-6"
DEFAULT_MAX_LENGTH="12288"
MULTIMODAL="false"

# Tag-SFT data from tag_rewrite.sh (not the episode-skill pipeline).
TAG_DATA_DIR="${TAG_DATA_DIR:-$PROJECT_ROOT/outputs/search_tag_sft}"
TRAIN_DATA="${TRAIN_DATA:-$TAG_DATA_DIR/tag_sft_train.parquet}"
VAL_DATA="${VAL_DATA:-$TAG_DATA_DIR/tag_sft_val.parquet}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
PROJECT_NAME="${PROJECT_NAME:-search-tag-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_TAG}-search-tag-sft-ep${TOTAL_EPOCHS}}"
OUTPUT_STEM="${OUTPUT_STEM:-search_tag_sft_${MODEL_TAG}_ep${TOTAL_EPOCHS}_${RUN_ID}}"
if [[ -n "${MODELS_ROOT:-}" ]]; then
    EXPORT_MODEL_DIR="${EXPORT_MODEL_DIR:-$MODELS_ROOT/$MODEL_BASENAME-search-tag-sft}"
fi
export TRAIN_DATA VAL_DATA RUN_ID TOTAL_EPOCHS PROJECT_NAME EXPERIMENT_NAME OUTPUT_STEM
if [[ -n "${EXPORT_MODEL_DIR:-}" ]]; then
    export EXPORT_MODEL_DIR
fi

# shellcheck source=../_common/trainer.sh
source "$PROJECT_ROOT/scripts/sft/_common/trainer.sh"
