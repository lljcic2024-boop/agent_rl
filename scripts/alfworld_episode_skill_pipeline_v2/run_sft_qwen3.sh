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
# shellcheck source=../sft_teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft_teacher_naming.sh"

CONDA_ENV="${CONDA_ENV:-skillrl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
LR="${LR:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
EXPORT_MODEL_AFTER_TRAIN="${EXPORT_MODEL_AFTER_TRAIN:-true}"
EXPORT_MODEL_NAME="${EXPORT_MODEL_NAME:-Qwen3-1.7B-alfworld-episode-skill-sft-${SFT_SELF_SUFFIX}}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/alfworld_episode_skill_pipeline_v2_qwen3_1_7b_${SFT_SELF_DIR_SUFFIX}}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/sft_episode_skill_train.parquet}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/sft_episode_skill_val.parquet}"

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -z "${MODELS_ROOT:-}" ]]; then
        echo "Please set MODELS_ROOT in $ENV_FILE, or set MODEL_PATH explicitly." >&2
        exit 1
    fi
    MODEL_PATH="$MODELS_ROOT/Qwen3-1.7B"
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-alfworld-episode-skill-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-1.7b-alfworld-episode-skill-sft-v2-ep${TOTAL_EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_ROOT/outputs/sft/alfworld_episode_skill_analyzer_v2_qwen3_1_7b_ep${TOTAL_EPOCHS}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$MODELS_ROOT/logs/sft}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/alfworld_episode_skill_analyzer_v2_qwen3_1_7b_ep${TOTAL_EPOCHS}_${RUN_ID}.log}"
if [[ "$EXPORT_MODEL_AFTER_TRAIN" == "true" ]]; then
    if [[ -z "${MODELS_ROOT:-}" && -z "${EXPORT_MODEL_DIR:-}" ]]; then
        echo "Please set MODELS_ROOT in $ENV_FILE, or set EXPORT_MODEL_DIR explicitly." >&2
        exit 1
    fi
    EXPORT_MODEL_DIR="${EXPORT_MODEL_DIR:-$MODELS_ROOT/$EXPORT_MODEL_NAME}"
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Train parquet not found: $TRAIN_DATA" >&2
    exit 1
fi
if [[ ! -f "$VAL_DATA" ]]; then
    echo "Val parquet not found: $VAL_DATA" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "Running ALFWorld episode-skill SFT v2 for Qwen3"
echo "  conda env:      $CONDA_ENV"
echo "  model:          $MODEL_PATH"
echo "  train data:     $TRAIN_DATA"
echo "  val data:       $VAL_DATA"
echo "  output dir:     $OUTPUT_DIR"
echo "  log file:       $LOG_FILE"
echo "  epochs:         $TOTAL_EPOCHS"
echo "  max length:     $MAX_LENGTH"
echo "  nproc:          $NPROC_PER_NODE"
echo "  sp size:        $ULYSSES_SEQUENCE_PARALLEL_SIZE"
echo "  logger:         $TRAINER_LOGGER"
echo "  wandb mode:     ${WANDB_MODE:-unset}"
echo "  qwen3 thinking: disabled"
echo "  export model:   $EXPORT_MODEL_AFTER_TRAIN"
if [[ "$EXPORT_MODEL_AFTER_TRAIN" == "true" ]]; then
    echo "  export dir:     $EXPORT_MODEL_DIR"
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=[] \
    data.response_dict_keys=[] \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=True \
    optim.lr="$LR" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    use_remove_padding=true \
    2>&1 | tee "$LOG_FILE"

if [[ "$EXPORT_MODEL_AFTER_TRAIN" == "true" ]]; then
    latest_checkpoint="$(
        find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' \
            | sort -V \
            | tail -n 1
    )"
    if [[ -z "$latest_checkpoint" ]]; then
        echo "No global_step_* checkpoint found in $OUTPUT_DIR" >&2
        exit 1
    fi

    latest_checkpoint="$OUTPUT_DIR/$latest_checkpoint"
    if [[ ! -f "$latest_checkpoint/config.json" ]]; then
        echo "Latest checkpoint does not look like an HF model: $latest_checkpoint" >&2
        exit 1
    fi

    export_parent="$(dirname "$EXPORT_MODEL_DIR")"
    export_tmp="${EXPORT_MODEL_DIR}.tmp.${RUN_ID}"
    export_backup="${EXPORT_MODEL_DIR}.bak.${RUN_ID}"
    mkdir -p "$export_parent"

    if [[ -e "$export_tmp" ]]; then
        echo "Temporary export path already exists: $export_tmp" >&2
        exit 1
    fi
    if [[ -e "$EXPORT_MODEL_DIR" && -e "$export_backup" ]]; then
        echo "Backup export path already exists: $export_backup" >&2
        exit 1
    fi

    echo "Exporting latest SFT checkpoint"
    echo "  source: $latest_checkpoint"
    echo "  target: $EXPORT_MODEL_DIR"

    mkdir -p "$export_tmp"
    if ! cp -al "$latest_checkpoint"/. "$export_tmp"/ 2>/dev/null; then
        cp -a "$latest_checkpoint"/. "$export_tmp"/
    fi

    if [[ ! -f "$export_tmp/config.json" ]]; then
        echo "Exported temporary directory does not look like an HF model: $export_tmp" >&2
        exit 1
    fi

    if [[ -e "$EXPORT_MODEL_DIR" ]]; then
        mv "$EXPORT_MODEL_DIR" "$export_backup"
        echo "  previous target moved to: $export_backup"
    fi
    mv "$export_tmp" "$EXPORT_MODEL_DIR"
    echo "Export complete: $EXPORT_MODEL_DIR"
fi
