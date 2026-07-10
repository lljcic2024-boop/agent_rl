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

SFT_CONDA_ENV="${SFT_CONDA_ENV:-${CONDA_ENV:-}}"
SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
LR="${LR:-2e-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRUNCATION="${TRUNCATION:-error}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
EXPORT_MODEL_AFTER_TRAIN="${EXPORT_MODEL_AFTER_TRAIN:-true}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"
MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct}"
# shellcheck source=../sft_teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft_teacher_naming.sh"
SOKOBAN_SELF_DIR_SUFFIX="${SOKOBAN_SELF_DIR_SUFFIX:-$SFT_SELF_DIR_SUFFIX}"
if [[ "$SOKOBAN_SELF_DIR_SUFFIX" == "teacher_self" ]]; then
    SOKOBAN_SELF_DIR_SUFFIX="self"
fi
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/sokoban_visual_seed_pipeline_qwen25_vl_3b_${SOKOBAN_SELF_DIR_SUFFIX}}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/sft_sokoban_visual_train.parquet}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/sft_sokoban_visual_val.parquet}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-sokoban-visual-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-vl-3b-sokoban-visual-sft-ep${TOTAL_EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_ROOT/outputs/sft/sokoban_visual_seed_ep${TOTAL_EPOCHS}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$MODELS_ROOT/logs/sft}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sokoban_visual_sft_ep${TOTAL_EPOCHS}_${RUN_ID}.log}"
EXPORT_MODEL_DIR="${EXPORT_MODEL_DIR:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct-sokoban-visual-sft}"

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

if [[ -n "$SFT_CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$SFT_CUDA_VISIBLE_DEVICES"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

if [[ -n "$SFT_CONDA_ENV" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$SFT_CONDA_ENV"
    set -u
fi

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
    data.image_key=images \
    data.max_length="$MAX_LENGTH" \
    data.truncation="$TRUNCATION" \
    model.partial_pretrain="$MODEL_PATH" \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    optim.lr="$LR" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    "$@" \
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

    mkdir -p "$export_tmp"
    if ! cp -al "$latest_checkpoint"/. "$export_tmp"/ 2>/dev/null; then
        cp -a "$latest_checkpoint"/. "$export_tmp"/
    fi

    if [[ -e "$EXPORT_MODEL_DIR" ]]; then
        mv "$EXPORT_MODEL_DIR" "$export_backup"
    fi
    mv "$export_tmp" "$EXPORT_MODEL_DIR"
    echo "Exported visual Sokoban SFT model to $EXPORT_MODEL_DIR"
fi
