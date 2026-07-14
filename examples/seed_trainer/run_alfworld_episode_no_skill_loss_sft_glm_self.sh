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

CONDA_ENV="${CONDA_ENV:-skillrl}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

if [[ -z "${MODELS_ROOT:-}" && -z "${HF_MODEL_PATH:-}" ]]; then
    echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

HF_MODEL_PATH="${HF_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct-alfworld-episode-skill-sft-glm-self}"
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    echo "Run ALFWorld GLM-self SFT first, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

# Initialize both the policy and its on-policy analyzer from the SFT model.
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_SDAR_LOSS_COEF="${SEED_SDAR_LOSS_COEF:-0.01}"
export SEED_SKILL_GEN_LOSS_ENABLE="${SEED_SKILL_GEN_LOSS_ENABLE:-False}"
export SEED_SKILL_GEN_LOSS_COEF="${SEED_SKILL_GEN_LOSS_COEF:-0.0}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"
export SEED_MODE="${SEED_MODE:-mean_std_norm}"

HISTORY_LENGTH="${HISTORY_LENGTH:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-160}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

export history_length="$HISTORY_LENGTH"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_3b_alfworld_sft_glm_self_mean-std-norm_exp2}"
if [[ -n "${MODELS_ROOT:-}" ]]; then
    default_local_dir="$MODELS_ROOT/ckpt/$EXPERIMENT_NAME"
else
    default_local_dir="$PROJECT_ROOT/outputs/rl/$EXPERIMENT_NAME"
fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$default_local_dir}"

cd "$PROJECT_ROOT"

echo "Running ALFWorld RL from GLM-self SFT model"
echo "  conda env:       $CONDA_ENV"
echo "  model:           $MODEL_PATH"
echo "  experiment:      $EXPERIMENT_NAME"
echo "  output dir:      $DEFAULT_LOCAL_DIR"
echo "  history length:  $HISTORY_LENGTH"
echo "  train/val size:  ${TRAIN_DATA_SIZE:-16}/${VAL_DATA_SIZE:-128}"
echo "  group size:      ${GROUP_SIZE:-8}"
echo "  epochs:          $TOTAL_EPOCHS"
echo "  GPUs:            $N_GPUS_PER_NODE"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "Dry run complete; RL training was not started."
    exit 0
fi

exec bash "$SCRIPT_DIR/run_alfworld_both_no_skill_loss_base.sh" \
    algorithm.seed.analysis_include_episode_summary=True \
    algorithm.seed.skill_gen.enable="$SEED_SKILL_GEN_LOSS_ENABLE" \
    actor_rollout_ref.actor.skill_gen_loss_coef="$SEED_SKILL_GEN_LOSS_COEF" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    "$@"
