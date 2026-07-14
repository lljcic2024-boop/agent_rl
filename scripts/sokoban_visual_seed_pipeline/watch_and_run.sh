#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"

# shellcheck source=../sft_teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft_teacher_naming.sh"
SOKOBAN_SELF_DIR_SUFFIX="${SOKOBAN_SELF_DIR_SUFFIX:-$SFT_SELF_DIR_SUFFIX}"
if [[ "$SOKOBAN_SELF_DIR_SUFFIX" == "teacher_self" ]]; then
    SOKOBAN_SELF_DIR_SUFFIX="qwen_self"
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_BASE_NAME="${PIPELINE_BASE_NAME:-sokoban_episode_skill_pipeline_qwen25_vl_3b_${SOKOBAN_SELF_DIR_SUFFIX}}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$PROJECT_ROOT/outputs/$PIPELINE_BASE_NAME}"
LOG_DIR="${LOG_DIR:-$PIPELINE_ROOT/logs}"
DATA_DIR="${DATA_DIR:-$PIPELINE_ROOT}"
mkdir -p "$PIPELINE_ROOT" "$LOG_DIR" "$DATA_DIR"

PIPELINE_CONDA_ENV="${PIPELINE_CONDA_ENV:-${CONDA_ENV:-skillrl}}"
if [[ -n "$PIPELINE_CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        set +u
        eval "$(conda shell.bash hook)"
        conda activate "$PIPELINE_CONDA_ENV"
        set -u
    elif [[ -x "/raid3/data/GTPO/conda_envs/$PIPELINE_CONDA_ENV/bin/python3" ]]; then
        export PATH="/raid3/data/GTPO/conda_envs/$PIPELINE_CONDA_ENV/bin:$PATH"
    else
        echo "Cannot activate conda environment: $PIPELINE_CONDA_ENV" >&2
        exit 1
    fi
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

STOP_RAY_BETWEEN_STAGES="${STOP_RAY_BETWEEN_STAGES:-true}"
WAIT_FOR_TRAINING_DRIVERS_CLEAR="${WAIT_FOR_TRAINING_DRIVERS_CLEAR:-true}"

has_other_training_driver() {
    ps -eo pid=,cmd= | awk -v self_pid="$$" '
        $1 == self_pid { next }
        /verl\.trainer\.main_ppo|verl\.trainer\.fsdp_sft_trainer|torchrun/ &&
        $0 !~ /watch_and_run\.sh/ &&
        $0 !~ /awk/ {
            found = 1
        }
        END { exit found ? 0 : 1 }
    '
}

stop_ray_if_requested() {
    if [[ "$STOP_RAY_BETWEEN_STAGES" != "true" ]]; then
        return
    fi
    if has_other_training_driver; then
        log "Skipping Ray cleanup because another training driver is still running."
        return
    fi
    if command -v ray >/dev/null 2>&1; then
        log "Stopping any leftover local Ray processes before the next stage."
        ray stop --force || true
    fi
}

wait_for_training_drivers_clear() {
    if [[ "$WAIT_FOR_TRAINING_DRIVERS_CLEAR" != "true" ]]; then
        return
    fi
    while has_other_training_driver; do
        log "Waiting for existing training driver processes to clear before starting the next stage."
        sleep "$WAIT_POLL_SECONDS"
    done
}

detect_running_sokoban_rl_pid() {
    ps -eo pid=,ppid=,cmd= | awk '
        /examples\/grpo_trainer\/run_sokoban\.sh/ && $0 !~ /watch_and_run\.sh/ && $0 !~ /awk/ {
            print $1
            exit
        }
    '
}

WAIT_FOR_PID="${WAIT_FOR_PID:-auto}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
POST_WAIT_SECONDS="${POST_WAIT_SECONDS:-60}"

if [[ "$WAIT_FOR_PID" == "auto" ]]; then
    WAIT_FOR_PID="$(detect_running_sokoban_rl_pid || true)"
fi

if [[ -n "$WAIT_FOR_PID" && "$WAIT_FOR_PID" != "0" && "$WAIT_FOR_PID" != "none" ]]; then
    log "Waiting for current Sokoban RL PID $WAIT_FOR_PID to finish."
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
        sleep "$WAIT_POLL_SECONDS"
    done
    log "Current RL PID $WAIT_FOR_PID finished. Waiting ${POST_WAIT_SECONDS}s before starting the pipeline."
    sleep "$POST_WAIT_SECONDS"
    wait_for_training_drivers_clear
    stop_ray_if_requested
else
    log "No current RL PID configured; starting the pipeline now."
fi

cd "$PROJECT_ROOT"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct}"
BASELINE_ROLLOUT_DIR="${BASELINE_ROLLOUT_DIR:-}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-}"
BASELINE_ROLLOUTS_JSONL="${BASELINE_ROLLOUTS_JSONL:-$PIPELINE_ROOT/baseline_rollouts.jsonl}"
BASELINE_ROLLOUTS_READY=false

if [[ -z "$BASELINE_ROLLOUT_DIR" ]]; then
    log "Stage 1/5: collecting 180-task visual Sokoban baseline rollouts in waves."
    BASELINE_LOG="${BASELINE_LOG:-$LOG_DIR/baseline_rollouts.log}"
    env \
        MODELS_ROOT="$MODELS_ROOT" \
        MODEL_PATH="$BASE_MODEL_PATH" \
        OUTPUT_DIR="$PIPELINE_ROOT" \
        BASELINE_ROLLOUTS_JSONL="$BASELINE_ROLLOUTS_JSONL" \
        BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-$PIPELINE_CONDA_ENV}" \
        NUM_TASKS="${BASELINE_NUM_TASKS:-180}" \
        ROLLOUTS_PER_TASK="${BASELINE_ROLLOUTS_PER_TASK:-8}" \
        TASK_BATCH_SIZE="${BASELINE_TASK_BATCH_SIZE:-16}" \
        REQUEST_WORKERS="${BASELINE_REQUEST_WORKERS:-128}" \
        MAX_STEPS="${BASELINE_MAX_STEPS:-15}" \
        HISTORY_LENGTH="${BASELINE_HISTORY_LENGTH:-2}" \
        SEED="${BASELINE_SEED:-2026}" \
        POLICY_TEMPERATURE="${BASELINE_POLICY_TEMPERATURE:-1.0}" \
        POLICY_MAX_COMPLETION_TOKENS="${BASELINE_POLICY_MAX_COMPLETION_TOKENS:-512}" \
        DATA_PARALLEL_SIZE="${BASELINE_DATA_PARALLEL_SIZE:-8}" \
        GPU_MEMORY_UTILIZATION="${BASELINE_GPU_MEMORY_UTILIZATION:-0.5}" \
        PORT="${BASELINE_POLICY_PORT:-60003}" \
        RESUME="${BASELINE_RESUME:-true}" \
        OVERWRITE="${BASELINE_OVERWRITE:-false}" \
        bash "$PROJECT_ROOT/scripts/sokoban_visual_seed_pipeline/run_baseline_rollouts.sh" \
        2>&1 | tee "$BASELINE_LOG"
    BASELINE_ROLLOUT_DIR="$PIPELINE_ROOT"
    BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-$PIPELINE_ROOT/sokoban_images}"
    BASELINE_ROLLOUTS_READY=true
fi

if [[ "$BASELINE_ROLLOUTS_READY" != "true" && -z "$BASELINE_IMAGE_ROOT" ]]; then
    if [[ -d "$PIPELINE_ROOT/sokoban_images" ]]; then
        BASELINE_IMAGE_ROOT="$PIPELINE_ROOT/sokoban_images"
    elif [[ -d "$BASELINE_ROLLOUT_DIR/sokoban_images" ]]; then
        BASELINE_IMAGE_ROOT="$BASELINE_ROLLOUT_DIR/sokoban_images"
    else
        BASELINE_IMAGE_ROOT="$BASELINE_ROLLOUT_DIR"
    fi
fi

if [[ "$BASELINE_ROLLOUTS_READY" != "true" ]]; then
    if ! compgen -G "$BASELINE_ROLLOUT_DIR/*.jsonl" >/dev/null; then
        log "No rollout JSONL found in $BASELINE_ROLLOUT_DIR"
        exit 1
    fi

    log "Exporting visual Sokoban baseline rollouts to $BASELINE_ROLLOUTS_JSONL."
    python3 "$PROJECT_ROOT/scripts/sokoban_visual_seed_pipeline/export_baseline_rollouts.py" \
        --rollout-dir "$BASELINE_ROLLOUT_DIR" \
        --image-root "$BASELINE_IMAGE_ROOT" \
        --output-path "$BASELINE_ROLLOUTS_JSONL" \
        2>&1 | tee "$LOG_DIR/export_baseline_rollouts.log"
fi

SKILL_GEN_LOG="${SKILL_GEN_LOG:-$LOG_DIR/generate_candidate_skills.log}"
CANDIDATE_SKILLS_JSONL="${CANDIDATE_SKILLS_JSONL:-$PIPELINE_ROOT/candidate_skills.jsonl}"
SOKOBAN_SKILL_GEN_WORKERS="${SOKOBAN_SKILL_GEN_WORKERS:-64}"
if [[ "$SFT_TEACHER_SHORT" == "gemini" ]]; then
    DEFAULT_SKILL_MAX_COMPLETION_TOKENS=8192
    DEFAULT_SKILL_PARSE_ATTEMPTS=3
else
    DEFAULT_SKILL_MAX_COMPLETION_TOKENS=2048
    DEFAULT_SKILL_PARSE_ATTEMPTS=2
fi
SOKOBAN_SKILL_MAX_COMPLETION_TOKENS="${SOKOBAN_SKILL_MAX_COMPLETION_TOKENS:-$DEFAULT_SKILL_MAX_COMPLETION_TOKENS}"
SOKOBAN_SKILL_PARSE_ATTEMPTS="${SOKOBAN_SKILL_PARSE_ATTEMPTS:-$DEFAULT_SKILL_PARSE_ATTEMPTS}"
SOKOBAN_INCLUDE_EPISODE_SUMMARY="${SOKOBAN_INCLUDE_EPISODE_SUMMARY:-true}"
SOKOBAN_SKILL_RESUME="${SOKOBAN_SKILL_RESUME:-true}"
SOKOBAN_REGENERATE_CANDIDATES="${SOKOBAN_REGENERATE_CANDIDATES:-false}"
SOKOBAN_MAX_CANDIDATES="${SOKOBAN_MAX_CANDIDATES:-null}"
SOKOBAN_SKILL_MODEL="${SOKOBAN_SKILL_MODEL:-${SKILL_MODEL:-${OPENAI_MODEL:-}}}"
SOKOBAN_SKILL_BASE_URL="${SOKOBAN_SKILL_BASE_URL:-${OPENAI_BASE_URL:-}}"
SOKOBAN_SKILL_API_KEY="${SOKOBAN_SKILL_API_KEY:-${OPENAI_API_KEY:-}}"
log "Stage 2/5: generating visual Sokoban episode skills with ${SOKOBAN_SKILL_MODEL:-$SFT_TEACHER_SHORT}."

skill_gen_args=(
    --baseline-rollouts "$BASELINE_ROLLOUTS_JSONL"
    --output-dir "$PIPELINE_ROOT"
    --skill-gen-workers "$SOKOBAN_SKILL_GEN_WORKERS"
    --skill-max-completion-tokens "$SOKOBAN_SKILL_MAX_COMPLETION_TOKENS"
    --skill-parse-attempts "$SOKOBAN_SKILL_PARSE_ATTEMPTS"
)
if [[ "$SOKOBAN_INCLUDE_EPISODE_SUMMARY" != "true" ]]; then
    skill_gen_args+=(--no-include-episode-summary)
fi
if [[ "$SOKOBAN_SKILL_RESUME" == "true" ]]; then
    skill_gen_args+=(--resume)
fi
if [[ "$SOKOBAN_REGENERATE_CANDIDATES" == "true" ]]; then
    skill_gen_args+=(--regenerate-candidates)
fi
if [[ "$SOKOBAN_MAX_CANDIDATES" != "null" ]]; then
    skill_gen_args+=(--max-candidates "$SOKOBAN_MAX_CANDIDATES")
fi
if [[ -n "$SOKOBAN_SKILL_MODEL" ]]; then
    skill_gen_args+=(--skill-model "$SOKOBAN_SKILL_MODEL")
fi
if [[ -n "$SOKOBAN_SKILL_BASE_URL" ]]; then
    skill_gen_args+=(--skill-base-url "$SOKOBAN_SKILL_BASE_URL")
fi
if [[ -n "$SOKOBAN_SKILL_API_KEY" ]]; then
    skill_gen_args+=(--skill-api-key "$SOKOBAN_SKILL_API_KEY")
fi

python3 "$PROJECT_ROOT/scripts/sokoban_visual_seed_pipeline/generate_candidate_skills.py" \
    "${skill_gen_args[@]}" \
    2>&1 | tee "$SKILL_GEN_LOG"

if [[ ! -f "$CANDIDATE_SKILLS_JSONL" ]]; then
    log "Candidate skills JSONL not found: $CANDIDATE_SKILLS_JSONL"
    exit 1
fi

log "Stage 3/5: building visual Sokoban skill SFT parquet data."
BUILD_SFT_LOG="${BUILD_SFT_LOG:-$LOG_DIR/build_sft.log}"
build_sft_args=(
    --rollout-dir "$(dirname "$BASELINE_ROLLOUTS_JSONL")"
    --image-root "$BASELINE_IMAGE_ROOT"
    --output-dir "$DATA_DIR"
    --candidate-skills "$CANDIDATE_SKILLS_JSONL"
    --val-ratio "${SFT_VAL_RATIO:-0.1}"
    --seed "${SFT_DATA_SEED:-2026}"
    --max-records "${SFT_MAX_RECORDS:-0}"
)
if [[ "$SOKOBAN_INCLUDE_EPISODE_SUMMARY" != "true" ]]; then
    build_sft_args+=(--no-include-episode-summary)
fi
python3 "$PROJECT_ROOT/scripts/sokoban_visual_seed_pipeline/build_sft_from_rollouts.py" \
    "${build_sft_args[@]}" \
    2>&1 | tee "$BUILD_SFT_LOG"

SFT_EXPORT_MODEL_DIR="${SFT_EXPORT_MODEL_DIR:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct-sokoban-episode-skill-sft-${SFT_SELF_SUFFIX}}"

log "Stage 4/5: running visual SFT."
SFT_LOG="${SFT_LOG:-$LOG_DIR/sft.log}"
env \
    MODELS_ROOT="$MODELS_ROOT" \
    MODEL_PATH="${SFT_BASE_MODEL_PATH:-$BASE_MODEL_PATH}" \
    DATA_DIR="$DATA_DIR" \
    EXPORT_MODEL_DIR="$SFT_EXPORT_MODEL_DIR" \
    SFT_CONDA_ENV="${SFT_CONDA_ENV:-$PIPELINE_CONDA_ENV}" \
    TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}" \
    bash "$PROJECT_ROOT/scripts/sokoban_visual_seed_pipeline/run_sft.sh" \
        2>&1 | tee "$SFT_LOG"

if [[ ! -d "$SFT_EXPORT_MODEL_DIR" ]]; then
    log "SFT export model directory not found: $SFT_EXPORT_MODEL_DIR"
    exit 1
fi
stop_ray_if_requested

log "Stage 5/5: running visual Sokoban seed RL from the SFT model."
SEED_RL_EXPERIMENT_NAME="${SEED_RL_EXPERIMENT_NAME:-seed_qwen2.5_vl_3b_sokoban_episode_skill_sft_${SOKOBAN_SELF_DIR_SUFFIX}_$RUN_ID}"
SEED_RL_DEFAULT_LOCAL_DIR="${SEED_RL_DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$SEED_RL_EXPERIMENT_NAME}"
SEED_RL_LOG="${SEED_RL_LOG:-$LOG_DIR/seed_rl.log}"
SEED_RL_TOTAL_EPOCHS="${SEED_RL_TOTAL_EPOCHS:-150}"
SEED_RL_TOTAL_TRAINING_STEPS="${SEED_RL_TOTAL_TRAINING_STEPS:-null}"

seed_rl_args=(
    trainer.total_epochs="$SEED_RL_TOTAL_EPOCHS"
)
if [[ "$SEED_RL_TOTAL_TRAINING_STEPS" != "null" ]]; then
    seed_rl_args+=(trainer.total_training_steps="$SEED_RL_TOTAL_TRAINING_STEPS")
fi

env \
    MODELS_ROOT="$MODELS_ROOT" \
    MODEL_PATH="$SFT_EXPORT_MODEL_DIR" \
    EXPERIMENT_NAME="$SEED_RL_EXPERIMENT_NAME" \
    DEFAULT_LOCAL_DIR="$SEED_RL_DEFAULT_LOCAL_DIR" \
    SOKOBAN_SAVE_IMAGES=True \
    VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}" \
    bash "$PROJECT_ROOT/examples/seed_trainer/run_sokoban_visual.sh" \
        "${seed_rl_args[@]}" \
        2>&1 | tee "$SEED_RL_LOG"

SUMMARY_FILE="$PIPELINE_ROOT/pipeline_summary.env"
{
    printf 'RUN_ID=%s\n' "$RUN_ID"
    printf 'PIPELINE_ROOT=%s\n' "$PIPELINE_ROOT"
    printf 'BASELINE_ROLLOUT_DIR=%s\n' "$BASELINE_ROLLOUT_DIR"
    printf 'BASELINE_ROLLOUTS_JSONL=%s\n' "$BASELINE_ROLLOUTS_JSONL"
    printf 'BASELINE_IMAGE_ROOT=%s\n' "$BASELINE_IMAGE_ROOT"
    printf 'CANDIDATE_SKILLS_JSONL=%s\n' "$CANDIDATE_SKILLS_JSONL"
    printf 'DATA_DIR=%s\n' "$DATA_DIR"
    printf 'SFT_EXPORT_MODEL_DIR=%s\n' "$SFT_EXPORT_MODEL_DIR"
    printf 'SEED_RL_DEFAULT_LOCAL_DIR=%s\n' "$SEED_RL_DEFAULT_LOCAL_DIR"
} > "$SUMMARY_FILE"

log "Pipeline finished. Summary written to $SUMMARY_FILE"
