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
    SOKOBAN_SELF_DIR_SUFFIX="self"
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_BASE_NAME="${PIPELINE_BASE_NAME:-sokoban_visual_seed_pipeline_qwen25_vl_3b_${SOKOBAN_SELF_DIR_SUFFIX}}"
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
BASELINE_EXPERIMENT_NAME="${BASELINE_EXPERIMENT_NAME:-seed_sokoban_visual_baseline_rollouts_$RUN_ID}"
BASELINE_LOCAL_DIR="${BASELINE_LOCAL_DIR:-$PIPELINE_ROOT/baseline_rollout_raw}"
BASELINE_ROLLOUTS_JSONL="${BASELINE_ROLLOUTS_JSONL:-$PIPELINE_ROOT/baseline_rollouts.jsonl}"

if [[ -z "$BASELINE_ROLLOUT_DIR" ]]; then
    log "Stage 1/5: running short visual Sokoban baseline rollouts."
    BASELINE_LOG="${BASELINE_LOG:-$LOG_DIR/baseline_rollouts.log}"
    BASELINE_TRAIN_DATA_SIZE="${BASELINE_TRAIN_DATA_SIZE:-32}"
    BASELINE_VAL_DATA_SIZE="${BASELINE_VAL_DATA_SIZE:-128}"
    BASELINE_GROUP_SIZE="${BASELINE_GROUP_SIZE:-8}"
    BASELINE_TOTAL_EPOCHS="${BASELINE_TOTAL_EPOCHS:-1}"
    BASELINE_TOTAL_TRAINING_STEPS="${BASELINE_TOTAL_TRAINING_STEPS:-1}"
    BASELINE_ENGINE="${BASELINE_ENGINE:-vllm}"
    BASELINE_PROJECT_NAME="${BASELINE_PROJECT_NAME:-agentic_sokoban}"
    BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-$PIPELINE_ROOT/sokoban_images}"

    env \
        MODELS_ROOT="$MODELS_ROOT" \
        MODEL_PATH="$BASE_MODEL_PATH" \
        ENGINE="$BASELINE_ENGINE" \
        PROJECT_NAME="$BASELINE_PROJECT_NAME" \
        EXPERIMENT_NAME="$BASELINE_EXPERIMENT_NAME" \
        DEFAULT_LOCAL_DIR="$BASELINE_LOCAL_DIR" \
        TRAIN_DATA_SIZE="$BASELINE_TRAIN_DATA_SIZE" \
        VAL_DATA_SIZE="$BASELINE_VAL_DATA_SIZE" \
        GROUP_SIZE="$BASELINE_GROUP_SIZE" \
        SOKOBAN_SAVE_IMAGES=True \
        SOKOBAN_IMAGE_SAVE_DIR="$BASELINE_IMAGE_ROOT" \
        VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}" \
        bash "$PROJECT_ROOT/examples/grpo_trainer/run_sokoban.sh" \
            trainer.total_epochs="$BASELINE_TOTAL_EPOCHS" \
            trainer.total_training_steps="$BASELINE_TOTAL_TRAINING_STEPS" \
            trainer.save_freq=-1 \
            trainer.test_freq=-1 \
            trainer.val_before_train=False \
            2>&1 | tee "$BASELINE_LOG"

    BASELINE_ROLLOUT_DIR="$BASELINE_LOCAL_DIR"
    stop_ray_if_requested
fi

if [[ -z "$BASELINE_IMAGE_ROOT" ]]; then
    if [[ -d "$PIPELINE_ROOT/sokoban_images" ]]; then
        BASELINE_IMAGE_ROOT="$PIPELINE_ROOT/sokoban_images"
    elif [[ -d "$BASELINE_ROLLOUT_DIR/sokoban_images" ]]; then
        BASELINE_IMAGE_ROOT="$BASELINE_ROLLOUT_DIR/sokoban_images"
    else
        BASELINE_IMAGE_ROOT="$BASELINE_ROLLOUT_DIR"
    fi
fi

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

log "Stage 2/5: generating visual Sokoban episode skills with OpenAI."
SKILL_GEN_LOG="${SKILL_GEN_LOG:-$LOG_DIR/generate_candidate_skills.log}"
CANDIDATE_SKILLS_JSONL="${CANDIDATE_SKILLS_JSONL:-$PIPELINE_ROOT/candidate_skills.jsonl}"
SOKOBAN_SKILL_GEN_WORKERS="${SOKOBAN_SKILL_GEN_WORKERS:-4}"
SOKOBAN_SKILL_MAX_COMPLETION_TOKENS="${SOKOBAN_SKILL_MAX_COMPLETION_TOKENS:-2048}"
SOKOBAN_SKILL_PARSE_ATTEMPTS="${SOKOBAN_SKILL_PARSE_ATTEMPTS:-2}"
SOKOBAN_INCLUDE_EPISODE_SUMMARY="${SOKOBAN_INCLUDE_EPISODE_SUMMARY:-true}"
SOKOBAN_SKILL_RESUME="${SOKOBAN_SKILL_RESUME:-true}"
SOKOBAN_REGENERATE_CANDIDATES="${SOKOBAN_REGENERATE_CANDIDATES:-false}"
SOKOBAN_MAX_CANDIDATES="${SOKOBAN_MAX_CANDIDATES:-null}"
SOKOBAN_SKILL_MODEL="${SOKOBAN_SKILL_MODEL:-${SKILL_MODEL:-${OPENAI_MODEL:-}}}"
SOKOBAN_SKILL_BASE_URL="${SOKOBAN_SKILL_BASE_URL:-${OPENAI_BASE_URL:-}}"
SOKOBAN_SKILL_API_KEY="${SOKOBAN_SKILL_API_KEY:-${OPENAI_API_KEY:-}}"

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

SFT_EXPORT_MODEL_DIR="${SFT_EXPORT_MODEL_DIR:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct-sokoban-visual-sft}"

log "Stage 4/5: running visual SFT."
SFT_LOG="${SFT_LOG:-$LOG_DIR/sft.log}"
env \
    MODELS_ROOT="$MODELS_ROOT" \
    MODEL_PATH="${SFT_BASE_MODEL_PATH:-$BASE_MODEL_PATH}" \
    DATA_DIR="$DATA_DIR" \
    EXPORT_MODEL_DIR="$SFT_EXPORT_MODEL_DIR" \
    SFT_CONDA_ENV="${SFT_CONDA_ENV:-$PIPELINE_CONDA_ENV}" \
    bash "$PROJECT_ROOT/scripts/sokoban_visual_seed_pipeline/run_sft.sh" \
        2>&1 | tee "$SFT_LOG"

if [[ ! -d "$SFT_EXPORT_MODEL_DIR" ]]; then
    log "SFT export model directory not found: $SFT_EXPORT_MODEL_DIR"
    exit 1
fi
stop_ray_if_requested

log "Stage 5/5: running visual Sokoban seed RL from the SFT model."
SEED_RL_EXPERIMENT_NAME="${SEED_RL_EXPERIMENT_NAME:-seed_qwen2.5_vl_3b_sokoban_visual_sft_$RUN_ID}"
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
