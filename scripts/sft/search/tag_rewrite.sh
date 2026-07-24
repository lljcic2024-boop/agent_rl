#!/usr/bin/env bash
# Tag-SFT data construction: rewrite baseline rollouts into function-typed
# tagged thinking with an external teacher, apply QC, export SFT parquet.
#
# Prerequisite: run scripts/sft/search/prepare_data.sh (or at least its
# baseline-rollout stage) so that $ROLLOUTS_FILE exists, and have the teacher
# model served behind an OpenAI-compatible endpoint (e.g. vLLM serve
# Qwen3-30B-A3B).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_ENV="${CONDA_ENV:-copd}"

ROLLOUTS_FILE="${ROLLOUTS_FILE:?Set ROLLOUTS_FILE to a baseline_rollouts.jsonl produced by prepare_data.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/search_tag_sft}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:?Set TEACHER_BASE_URL (OpenAI-compatible endpoint of the teacher, e.g. http://127.0.0.1:60010/v1)}"
TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
TEACHER_MODEL="${TEACHER_MODEL:?Set TEACHER_MODEL (served model name, e.g. Qwen3-30B-A3B)}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.3}"
TEACHER_MAX_COMPLETION_TOKENS="${TEACHER_MAX_COMPLETION_TOKENS:-2048}"
REWRITE_WORKERS="${REWRITE_WORKERS:-32}"

TOKENIZER_PATH="${TOKENIZER_PATH:-}"   # e.g. $MODELS_ROOT/Qwen3-8B; empty -> whitespace token count
MIN_SEGMENT_TOKENS="${MIN_SEGMENT_TOKENS:-20}"
MAX_SEGMENT_TOKENS="${MAX_SEGMENT_TOKENS:-300}"
FAILURE_OVERSAMPLE="${FAILURE_OVERSAMPLE:-2.0}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SEED="${SEED:-2026}"
RESUME="${RESUME:-true}"

args=(
    "$SCRIPT_DIR/tag_rewrite_pipeline.py"
    --rollouts "$ROLLOUTS_FILE"
    --output-dir "$OUTPUT_DIR"
    --teacher-base-url "$TEACHER_BASE_URL"
    --teacher-api-key "$TEACHER_API_KEY"
    --teacher-model "$TEACHER_MODEL"
    --teacher-temperature "$TEACHER_TEMPERATURE"
    --teacher-max-completion-tokens "$TEACHER_MAX_COMPLETION_TOKENS"
    --rewrite-workers "$REWRITE_WORKERS"
    --min-segment-tokens "$MIN_SEGMENT_TOKENS"
    --max-segment-tokens "$MAX_SEGMENT_TOKENS"
    --failure-oversample "$FAILURE_OVERSAMPLE"
    --sft-val-ratio "$SFT_VAL_RATIO"
    --seed "$SEED"
)
if [[ -n "$TOKENIZER_PATH" ]]; then
    args+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ "$RESUME" == "true" || "$RESUME" == "1" ]]; then
    args+=(--resume)
fi

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

echo "Running Search QA tag-SFT rewrite pipeline"
echo "  rollouts:     $ROLLOUTS_FILE"
echo "  output dir:   $OUTPUT_DIR"
echo "  teacher:      $TEACHER_MODEL @ $TEACHER_BASE_URL"
echo "  workers:      $REWRITE_WORKERS"
echo "  oversample:   $FAILURE_OVERSAMPLE (failed trajectories)"

python "${args[@]}"

echo "Done. Train with scripts/sft/search/train_sft.sh:"
echo "  TRAIN_DATA=$OUTPUT_DIR/tag_sft_train.parquet VAL_DATA=$OUTPUT_DIR/tag_sft_val.parquet bash scripts/sft/search/train_sft.sh"
