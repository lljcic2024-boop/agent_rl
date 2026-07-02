#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-sgop-webshop}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-copd}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/webshop_episode_skill_pipeline_v2_qwen25_3b}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE.}"
MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-60002}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://${HOST}:${PORT}/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL="${POLICY_MODEL:-$MODEL_NAME}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.4}"
POLICY_MAX_COMPLETION_TOKENS="${POLICY_MAX_COMPLETION_TOKENS:-512}"
POLICY_TIMEOUT="${POLICY_TIMEOUT:-120}"
POLICY_RETRIES="${POLICY_RETRIES:-2}"
POLICY_RETRY_DELAY="${POLICY_RETRY_DELAY:-1.0}"

SKILL_BASE_URL="${SKILL_BASE_URL:-${OPENAI_BASE_URL:?Please set OPENAI_BASE_URL in .env or SKILL_BASE_URL.}}"
SKILL_API_KEY="${SKILL_API_KEY:-${OPENAI_API_KEY:?Please set OPENAI_API_KEY in .env or SKILL_API_KEY.}}"
SKILL_MODEL="${SKILL_MODEL:-${OPENAI_MODEL:?Please set OPENAI_MODEL in .env or SKILL_MODEL.}}"
SKILL_TEMPERATURE="${SKILL_TEMPERATURE:-0.0}"
SKILL_MAX_COMPLETION_TOKENS="${SKILL_MAX_COMPLETION_TOKENS:-1024}"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-120}"
SKILL_RETRIES="${SKILL_RETRIES:-5}"
SKILL_RETRY_DELAY="${SKILL_RETRY_DELAY:-1.0}"
SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-128}"

START_VLLM="${START_VLLM:-1}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$PROJECT_ROOT/logs/vllm}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$VLLM_LOG_DIR/webshop_policy_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"

NUM_TASKS="${NUM_TASKS:-180}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-15}"
HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
REQUEST_WORKERS="${REQUEST_WORKERS:-128}"
WEBSHOP_HUMAN_GOALS="${WEBSHOP_HUMAN_GOALS:-0}"
WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-1}"
WEBSHOP_TRAIN_START="${WEBSHOP_TRAIN_START:-500}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.05}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SFT_MIN_SOURCE_SCORE="${SFT_MIN_SOURCE_SCORE:-0.0}"
SFT_INCLUDE_SUCCESS="${SFT_INCLUDE_SUCCESS:-true}"
SFT_MAX_ZERO_SCORE_FAILURES="${SFT_MAX_ZERO_SCORE_FAILURES:-}"
SFT_MAX_RECORDS="${SFT_MAX_RECORDS:-}"
SEED="${SEED:-2026}"
RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"
REGENERATE_CANDIDATES="${REGENERATE_CANDIDATES:-false}"
PROGRESS_MONITOR="${PROGRESS_MONITOR:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "Policy model path not found: $MODEL_PATH" >&2
    exit 1
fi

args=(
    "$SCRIPT_DIR/pipeline_v2.py"
    --env-file "$ENV_FILE"
    --output-dir "$OUTPUT_DIR"
    --num-tasks "$NUM_TASKS"
    --rollouts-per-task "$ROLLOUTS_PER_TASK"
    --task-batch-size "$TASK_BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --history-length "$HISTORY_LENGTH"
    --request-workers "$REQUEST_WORKERS"
    --webshop-human-goals "$WEBSHOP_HUMAN_GOALS"
    --webshop-train-start "$WEBSHOP_TRAIN_START"
    --num-cpus-per-env-worker "$NUM_CPUS_PER_ENV_WORKER"
    --policy-base-url "$POLICY_BASE_URL"
    --policy-api-key "$POLICY_API_KEY"
    --policy-model "$POLICY_MODEL"
    --policy-temperature "$POLICY_TEMPERATURE"
    --policy-max-completion-tokens "$POLICY_MAX_COMPLETION_TOKENS"
    --policy-timeout "$POLICY_TIMEOUT"
    --policy-retries "$POLICY_RETRIES"
    --policy-retry-delay "$POLICY_RETRY_DELAY"
    --skill-base-url "$SKILL_BASE_URL"
    --skill-api-key "$SKILL_API_KEY"
    --skill-model "$SKILL_MODEL"
    --skill-temperature "$SKILL_TEMPERATURE"
    --skill-max-completion-tokens "$SKILL_MAX_COMPLETION_TOKENS"
    --skill-timeout "$SKILL_TIMEOUT"
    --skill-retries "$SKILL_RETRIES"
    --skill-retry-delay "$SKILL_RETRY_DELAY"
    --skill-gen-workers "$SKILL_GEN_WORKERS"
    --sft-val-ratio "$SFT_VAL_RATIO"
    --sft-min-source-score "$SFT_MIN_SOURCE_SCORE"
    --sft-include-success "$SFT_INCLUDE_SUCCESS"
    --seed "$SEED"
)

if [[ -n "${WEBSHOP_DATA:-}" ]]; then
    args+=(--webshop-data-dir "$WEBSHOP_DATA")
elif [[ -n "${WEBSHOP_DATA_DIR:-}" ]]; then
    args+=(--webshop-data-dir "$WEBSHOP_DATA_DIR")
fi

if [[ "$WEBSHOP_USE_SMALL" == "1" || "$WEBSHOP_USE_SMALL" == "true" || "$WEBSHOP_USE_SMALL" == "True" ]]; then
    args+=(--webshop-use-small)
fi

if [[ -n "${WEBSHOP_TRAIN_END:-}" ]]; then
    args+=(--webshop-train-end "$WEBSHOP_TRAIN_END")
fi

if [[ -n "${MAX_TASKS:-}" ]]; then
    args+=(--max-tasks "$MAX_TASKS")
fi

if [[ -n "${MAX_CANDIDATES:-}" ]]; then
    args+=(--max-candidates "$MAX_CANDIDATES")
fi

if [[ -n "$SFT_MAX_ZERO_SCORE_FAILURES" ]]; then
    args+=(--sft-max-zero-score-failures "$SFT_MAX_ZERO_SCORE_FAILURES")
fi

if [[ -n "$SFT_MAX_RECORDS" ]]; then
    args+=(--sft-max-records "$SFT_MAX_RECORDS")
fi

if [[ -n "${POLICY_EXTRA_BODY_JSON:-}" ]]; then
    args+=(--policy-extra-body-json "$POLICY_EXTRA_BODY_JSON")
fi

if [[ -n "${SKILL_EXTRA_BODY_JSON:-}" ]]; then
    args+=(--skill-extra-body-json "$SKILL_EXTRA_BODY_JSON")
fi

if [[ "$OVERWRITE" == "true" ]]; then
    args+=(--overwrite)
elif [[ "$REGENERATE_CANDIDATES" == "true" ]]; then
    args+=(--regenerate-candidates)
elif [[ "$RESUME" == "true" ]]; then
    args+=(--resume)
fi

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

server_pid=""
progress_monitor_pid=""

is_server_ready() {
    curl -fsS "${POLICY_BASE_URL}/models" >/dev/null 2>&1
}

print_progress() {
    local progress_file="$OUTPUT_DIR/progress.json"
    if [[ ! -f "$progress_file" ]]; then
        echo "[progress] waiting for $progress_file"
        return
    fi

    python3 - "$progress_file" <<'PY' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    progress = json.load(f)

parts = [
    f"[progress] {progress.get('updated_at', '')}",
    f"stage={progress.get('stage', 'unknown')}",
    f"status={progress.get('status', 'unknown')}",
]
for key in (
    "sampled_tasks",
    "completed_rollouts",
    "expected_rollouts",
    "wave",
    "total_waves",
    "completed_skills",
    "expected_skills",
    "completed_in_current_run",
    "pending_in_current_run",
    "parse_ok_skills",
    "sft_records",
    "last_skill_id",
    "last_parse_ok",
):
    if key in progress:
        parts.append(f"{key}={progress[key]}")
print(" ".join(parts), flush=True)
PY
}

cleanup() {
    if [[ -n "$progress_monitor_pid" ]]; then
        kill "$progress_monitor_pid" >/dev/null 2>&1 || true
        wait "$progress_monitor_pid" >/dev/null 2>&1 || true
    fi
    if [[ -n "$server_pid" && "$KEEP_VLLM_ALIVE" != "1" ]]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
    if is_server_ready; then
        echo "Using existing policy vLLM server at $POLICY_BASE_URL"
    else
        mkdir -p "$VLLM_LOG_DIR"
        read -r -a VLLM_EXTRA_ARGS_ARRAY <<< "${VLLM_EXTRA_ARGS:-}"
        VLLM_SERVE_ARGS=(
            "$MODEL_PATH"
            --host "$HOST" \
            --port "$PORT" \
            --served-model-name "$POLICY_MODEL" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --dtype "$DTYPE"
        )
        if [[ -n "$DATA_PARALLEL_SIZE" ]]; then
            VLLM_SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
        fi

        conda run -n "$VLLM_CONDA_ENV" --no-capture-output \
            "$VLLM_BIN" serve "${VLLM_SERVE_ARGS[@]}" \
            "${VLLM_EXTRA_ARGS_ARRAY[@]}" \
            >"$VLLM_LOG_FILE" 2>&1 &
        server_pid=$!

        deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
        until is_server_ready; do
            if ! kill -0 "$server_pid" >/dev/null 2>&1; then
                echo "vLLM server exited before becoming ready. See $VLLM_LOG_FILE" >&2
                exit 1
            fi
            if (( SECONDS >= deadline )); then
                echo "Timed out waiting for vLLM at $POLICY_BASE_URL. See $VLLM_LOG_FILE" >&2
                exit 1
            fi
            sleep 2
        done
    fi
fi

if [[ "$PROGRESS_MONITOR" == "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
    print_progress
    (
        while true; do
            sleep "$PROGRESS_INTERVAL"
            print_progress
        done
    ) &
    progress_monitor_pid=$!
fi

echo "Running WebShop episode-skill pipeline v2"
echo "  conda env:          $CONDA_ENV"
echo "  vLLM conda env:     $VLLM_CONDA_ENV"
echo "  output dir:         $OUTPUT_DIR"
echo "  policy model:       $MODEL_PATH"
echo "  policy endpoint:    $POLICY_BASE_URL"
echo "  skill model:        $SKILL_MODEL @ $SKILL_BASE_URL"
echo "  use small data:     $WEBSHOP_USE_SMALL"
echo "  sampled tasks:      $NUM_TASKS"
echo "  rollouts per task:  $ROLLOUTS_PER_TASK"
echo "  task batch size:    $TASK_BATCH_SIZE"
echo "  skill gen workers:  $SKILL_GEN_WORKERS"
echo "  sft min score:      $SFT_MIN_SOURCE_SCORE"
echo "  sft include success:$SFT_INCLUDE_SUCCESS"
echo "  sft zero cap:       ${SFT_MAX_ZERO_SCORE_FAILURES:-unset}"
echo "  sft max records:    ${SFT_MAX_RECORDS:-unset}"

set +e
python "${args[@]}"
run_status=$?
set -e

print_progress
exit "$run_status"
