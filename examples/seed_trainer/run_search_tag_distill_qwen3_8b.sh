#!/usr/bin/env bash
# Main experiment launcher: function-typed distillation on Search QA.
#
# Student  = Qwen3-8B (tag-SFT checkpoint; same tokenizer family as teacher,
#            required for token-level log-prob alignment in external scoring).
# Teacher  = Qwen3-30B-A3B behind one vLLM OpenAI-compatible server, used both
#            by the analyzer (writes a_T) and by external log-prob scoring.
# Switches = tag menu prompts + rkl OPD loss + error-signal step selection.
#
# Prerequisites on the cluster:
#   1. Teacher server:  vllm serve Qwen3-30B-A3B --port 8100 ... (TP as planned)
#   2. Search retriever at $SEARCH_URL (see scripts/sft/search/prepare_data.sh)
#   3. Tag-SFT checkpoint from scripts/sft/search/{tag_rewrite.sh,train_sft.sh}
#      exported to $MODELS_ROOT/Qwen3-8B-search-tag-sft (or set HF_MODEL_PATH).

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

# `-` not `:-`: an explicitly empty CONDA_ENV means "use the current interpreter"
# (the cluster pods have no conda at all), while unset still defaults to copd.
CONDA_ENV="${CONDA_ENV-copd}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required to activate environment: $CONDA_ENV" >&2
        exit 1
    fi
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

# Student model: tag-SFT'd Qwen3-8B. Falls back to the base model so a
# no-SFT ablation can run without editing this script.
if [[ -z "${HF_MODEL_PATH:-}" ]]; then
    if [[ -n "${MODEL_PATH:-}" ]]; then
        HF_MODEL_PATH="$MODEL_PATH"
    elif [[ -n "${MODELS_ROOT:-}" ]]; then
        if [[ -f "$MODELS_ROOT/Qwen3-8B-search-tag-sft/config.json" ]]; then
            HF_MODEL_PATH="$MODELS_ROOT/Qwen3-8B-search-tag-sft"
        else
            HF_MODEL_PATH="$MODELS_ROOT/Qwen3-8B"
            echo "WARNING: tag-SFT checkpoint not found; using base $HF_MODEL_PATH" >&2
        fi
    else
        echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
        exit 1
    fi
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    exit 1
fi
export HF_MODEL_PATH
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"

# Teacher server (single endpoint for analyzer + external log-prob scoring).
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8100/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-30B-A3B}"
TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"

# Analyzer -> external 30B teacher (writes thinking segments / a_T).
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-openai}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$TEACHER_BASE_URL}"
export OPENAI_MODEL="${OPENAI_MODEL:-$TEACHER_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$TEACHER_API_KEY}"

# Scoring -> same server, token-id prompts with prompt_logprobs.
export SEED_EXTERNAL_TEACHER_ENABLE="${SEED_EXTERNAL_TEACHER_ENABLE:-True}"
export SEED_EXTERNAL_TEACHER_BASE_URL="${SEED_EXTERNAL_TEACHER_BASE_URL:-$TEACHER_BASE_URL}"
export SEED_EXTERNAL_TEACHER_MODEL="${SEED_EXTERNAL_TEACHER_MODEL:-$TEACHER_MODEL}"

# Function-tag menu prompts (modification 1).
export SEARCH_USE_FUNCTION_TAGS="${SEARCH_USE_FUNCTION_TAGS:-True}"

# OPD loss: standard signed per-token advantage (teacher_lp - student_lp).
export SEED_OPD_LOSS_MODE="${SEED_OPD_LOSS_MODE:-rkl}"
export SEED_OPD_LOSS_COEF="${SEED_OPD_LOSS_COEF:-0.01}"
export SEED_OPD_RKL_ADV_CLIP="${SEED_OPD_RKL_ADV_CLIP:-5.0}"
# FKL on teacher-generated a_T tokens, produced by the teacher-prefix branch
# rollout (modification 4). Both must be on together: with the branch off there
# are no a_T tokens to learn from, and with the coefficient at 0 the branch rows
# would carry tokens that are in no loss at all (the trainer self-disables in
# that case).
export SEED_OPD_FKL_LOSS_COEF="${SEED_OPD_FKL_LOSS_COEF:-0.05}"
export SEED_TEACHER_BRANCH_ENABLE="${SEED_TEACHER_BRANCH_ENABLE:-True}"
export SEED_TEACHER_BRANCH_MAX_PER_TRAJ="${SEED_TEACHER_BRANCH_MAX_PER_TRAJ:-1}"
export SEED_TEACHER_BRANCH_REQUIRE_ERROR_SIGNAL="${SEED_TEACHER_BRANCH_REQUIRE_ERROR_SIGNAL:-True}"
export SEED_TEACHER_BRANCH_PREFIX_CONCURRENCY="${SEED_TEACHER_BRANCH_PREFIX_CONCURRENCY:-8}"

# Step selection: error-signal steps only (modification 3, Phase 2a).
export SEED_STEP_SELECTOR="${SEED_STEP_SELECTOR:-error_signal}"

export SEED_MODE="${SEED_MODE:-mean_std_norm}"
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_step}"

# Environment-specific runtime scale.
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SAVE_FREQ="${SAVE_FREQ:-150}"
TEST_FREQ="${TEST_FREQ:-150}"

export PROJECT_NAME="${PROJECT_NAME:-agentic_search}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-tag_distill_qwen3_8b_search_rkl_errsel}"
if [[ -n "${MODELS_ROOT:-}" ]]; then
    default_local_dir="$MODELS_ROOT/ckpt/$EXPERIMENT_NAME"
else
    default_local_dir="$PROJECT_ROOT/outputs/rl/$EXPERIMENT_NAME"
fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$default_local_dir}"

cd "$PROJECT_ROOT"

echo "Running Search QA function-typed distillation (Qwen3-8B student)"
echo "  conda env:        ${CONDA_ENV:-<current shell>}"
echo "  student model:    $MODEL_PATH"
echo "  teacher:          $SEED_EXTERNAL_TEACHER_MODEL @ $SEED_EXTERNAL_TEACHER_BASE_URL"
echo "  analysis backend: $SEED_ANALYSIS_BACKEND ($OPENAI_MODEL @ $OPENAI_BASE_URL)"
echo "  function tags:    $SEARCH_USE_FUNCTION_TAGS"
echo "  opd loss:         mode=$SEED_OPD_LOSS_MODE coef=$SEED_OPD_LOSS_COEF clip=$SEED_OPD_RKL_ADV_CLIP fkl=$SEED_OPD_FKL_LOSS_COEF"
echo "  step selector:    $SEED_STEP_SELECTOR"
echo "  teacher branch:   $SEED_TEACHER_BRANCH_ENABLE (per-traj=$SEED_TEACHER_BRANCH_MAX_PER_TRAJ err-signal=$SEED_TEACHER_BRANCH_REQUIRE_ERROR_SIGNAL)"
echo "  experiment:       $EXPERIMENT_NAME"
echo "  output dir:       $DEFAULT_LOCAL_DIR"
echo "  training steps:   $TOTAL_TRAINING_STEPS"
echo "  GPUs:             $N_GPUS_PER_NODE"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "Dry run complete; RL training was not started."
    exit 0
fi

launch_args=(
    "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
)
if [[ -n "$HISTORY_LENGTH" ]]; then
    export history_length="$HISTORY_LENGTH"
    launch_args+=("env.history_length=$HISTORY_LENGTH")
fi

exec bash "$SCRIPT_DIR/_common/search.sh" "${launch_args[@]}" "$@"
