set -x

# Runtime backend.
ENGINE=vllm

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Model, data, and rollout scale.
MODELS_ROOT=${MODELS_ROOT:?Please set MODELS_ROOT}
MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen3-1.7B}
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
NUM_CPUS_PER_ENV_WORKER=0.1

# SEED advantage and teacher/OPD signal schedule.
SEED_MODE=mean_norm
SEED_STEP_ADV_W=0.0
SEED_EPISODE_SKILL_TEACHER_ADV_W=${SEED_EPISODE_SKILL_TEACHER_ADV_W:-0.001}
SEED_STEP_SKILL_TEACHER_ADV_W=${SEED_STEP_SKILL_TEACHER_ADV_W:-0.001}
SEED_OPD_START_AFTER_STEPS=${SEED_OPD_START_AFTER_STEPS:-null}
SEED_OPD_STOP_AFTER_STEPS=${SEED_OPD_STOP_AFTER_STEPS:-null}

# SEED episode filtering and teacher prompt construction.
SEED_FAILED_ONLY=${SEED_FAILED_ONLY:-False}
SEED_FAILED_ONLY_AFTER_STEPS=${SEED_FAILED_ONLY_AFTER_STEPS:-null}
SEED_FAILURE_SUCCESS_THRESHOLD=${SEED_FAILURE_SUCCESS_THRESHOLD:-1.0}

# SEED episode + critical-step skill analysis.
SEED_ENABLE_ANALYSIS=${SEED_ENABLE_ANALYSIS:-True}
SEED_SELECTOR=${SEED_SELECTOR:-llm}
SEED_ANALYSIS_BACKEND=openai
SEED_ANALYSIS_NUM_WORKERS=128
SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ=${SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ:-5}

# Experiment naming and output location.
PROJECT_NAME=agentic_webshop
EXPERIMENT_NAME=${EXPERIMENT_NAME:-seed-grpo_qwen3_1.7b_webshop}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}

# Prompt observation history.
history_length=2

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=seed \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.seed.step_advantage_w=$SEED_STEP_ADV_W \
    algorithm.seed.episode_skill_teacher_advantage_w=$SEED_EPISODE_SKILL_TEACHER_ADV_W \
    algorithm.seed.step_skill_teacher_advantage_w=$SEED_STEP_SKILL_TEACHER_ADV_W \
    algorithm.seed.opd_start_after_steps=$SEED_OPD_START_AFTER_STEPS \
    algorithm.seed.opd_stop_after_steps=$SEED_OPD_STOP_AFTER_STEPS \
    algorithm.seed.failed_only=$SEED_FAILED_ONLY \
    algorithm.seed.failed_only_after_steps=$SEED_FAILED_ONLY_AFTER_STEPS \
    algorithm.seed.failure_success_threshold=$SEED_FAILURE_SUCCESS_THRESHOLD \
    algorithm.seed.mode=$SEED_MODE \
    algorithm.seed.enable_analysis=$SEED_ENABLE_ANALYSIS \
    algorithm.seed.selector=$SEED_SELECTOR \
    algorithm.seed.analysis_backend=$SEED_ANALYSIS_BACKEND \
    algorithm.seed.analysis_num_workers=$SEED_ANALYSIS_NUM_WORKERS \
    algorithm.seed.analysis_max_completion_tokens=4096 \
    algorithm.seed.analysis_max_step_skills_per_traj=$SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ \
    algorithm.seed.normalize_teacher_adv=False \
    env.history_length=$history_length \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    $@
