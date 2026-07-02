#!/usr/bin/env bash

set -euo pipefail
set -x

# Runtime backend.
ENGINE=${ENGINE:-vllm}

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

# Model, data, and rollout scale.
MODELS_ROOT=${MODELS_ROOT:?Please set MODELS_ROOT}
MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}

# LHOP: student samples with env.history_length; teacher scores the same
# response with a longer-history prompt from the env manager.
STUDENT_HISTORY_LENGTH=${STUDENT_HISTORY_LENGTH:-${history_length:-5}}
LHOP_TEACHER_HISTORY_LENGTH=${LHOP_TEACHER_HISTORY_LENGTH:-10}
LHOP_MAX_PROMPT_LENGTH=${LHOP_MAX_PROMPT_LENGTH:-4096}
LHOP_MAX_RESPONSE_LENGTH=${LHOP_MAX_RESPONSE_LENGTH:-512}
LHOP_MAX_MODEL_LEN=${LHOP_MAX_MODEL_LEN:-8192}
LHOP_OPD_LOSS_COEF=${LHOP_OPD_LOSS_COEF:-0.01}
LHOP_SDAR_GATE_BETA=${LHOP_SDAR_GATE_BETA:-5.0}
LHOP_START_AFTER_STEPS=${LHOP_START_AFTER_STEPS:-null}
LHOP_STOP_AFTER_STEPS=${LHOP_STOP_AFTER_STEPS:-null}
LHOP_SKIP_IF_SAME_PROMPT=${LHOP_SKIP_IF_SAME_PROMPT:-True}

# OPID estimator is reused for episode-level RL advantage; OPID LLM analysis is
# disabled because LHOP supplies teacher_log_prob directly.
OPID_MODE=${OPID_MODE:-mean_std_norm}
OPID_STEP_ADV_W=${OPID_STEP_ADV_W:-0.0}

# Experiment naming and output location.
PROJECT_NAME=${PROJECT_NAME:-agentic_alfworld}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-lhop-grpo_qwen2.5_3b_alfworld_h${STUDENT_HISTORY_LENGTH}_th${LHOP_TEACHER_HISTORY_LENGTH}}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opid \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=$LHOP_MAX_PROMPT_LENGTH \
    data.max_response_length=$LHOP_MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.sdar_loss_coef=$LHOP_OPD_LOSS_COEF \
    actor_rollout_ref.actor.sdar_gate_beta=$LHOP_SDAR_GATE_BETA \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=$LHOP_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$LHOP_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.lhop.enable=True \
    algorithm.lhop.teacher_history_length=$LHOP_TEACHER_HISTORY_LENGTH \
    algorithm.lhop.max_prompt_length=$LHOP_MAX_PROMPT_LENGTH \
    algorithm.lhop.skip_if_same_prompt=$LHOP_SKIP_IF_SAME_PROMPT \
    algorithm.opid.step_advantage_w=$OPID_STEP_ADV_W \
    algorithm.opid.episode_skill_teacher_advantage_w=0.0 \
    algorithm.opid.step_skill_teacher_advantage_w=0.0 \
    algorithm.opid.opd_start_after_steps=$LHOP_START_AFTER_STEPS \
    algorithm.opid.opd_stop_after_steps=$LHOP_STOP_AFTER_STEPS \
    algorithm.opid.mode=$OPID_MODE \
    algorithm.opid.enable_analysis=False \
    algorithm.opid.selector=none \
    algorithm.opid.normalize_teacher_adv=False \
    env.history_length=$STUDENT_HISTORY_LENGTH \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    "$@"
