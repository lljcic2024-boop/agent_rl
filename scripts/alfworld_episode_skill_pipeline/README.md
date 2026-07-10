# ALFWorld Episode-Skill SFT Data Pipeline

This directory contains a standalone offline pipeline for building verified
episode-level skill data on ALFWorld. It does not modify existing trainer,
SEED, or environment code.

## What It Does

1. Sample ALFWorld train tasks: 6 task types, 30 tasks per type by default.
2. Run the policy without skill on each task for 8 rollouts.
3. Ask an OpenAI-compatible LLM to generate one `episode_skill` per rollout,
   using the same episode-only analyzer style as `seed.analysis`.
4. Replay each candidate skill on the same task for 8 rollouts.
5. Keep skills that improve success count/rate over the no-skill baseline.
6. Export accepted examples as SFT parquet files.

The default scale is:

```text
6 task types * 30 tasks/type * 8 rollouts/task = 1440 candidate skills
```

Validation is much heavier because each candidate skill is evaluated with
8 additional rollouts.

## Run Script

`run.sh` loads `.env`, starts a local vLLM server for the policy model at
`$MODELS_ROOT/Qwen2.5-3B-Instruct`, and uses `.env` `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, and `OPENAI_MODEL` for the skill-generation/analyzer model.

Minimal smoke test:

```bash
RUN_MODE=smoke ./scripts/alfworld_episode_skill_pipeline/run.sh
```

Full run:

```bash
./scripts/alfworld_episode_skill_pipeline/run.sh
```

Useful overrides for the local vLLM server:

```bash
POLICY_PORT=60001 \
TENSOR_PARALLEL_SIZE=1 \
DATA_PARALLEL_SIZE=8 \
GPU_MEMORY_UTILIZATION=0.6 \
KEEP_VLLM_ALIVE=1 \
./scripts/alfworld_episode_skill_pipeline/run.sh
```

If 1024 parallel envs is too heavy, lower the wave size:

```bash
TASK_BATCH_SIZE=4 \
SKILL_BATCH_SIZE=4 \
REQUEST_WORKERS=32 \
./scripts/alfworld_episode_skill_pipeline/run.sh
```

Set `START_VLLM=0` to use an already running local policy server.

`TASK_BATCH_SIZE` controls how many different ALFWorld tasks are rolled out in
the same baseline wave. With the default full settings, `TASK_BATCH_SIZE=128`
and `ROLLOUTS_PER_TASK=8` means one baseline wave runs up to 1024 environments.

`SKILL_BATCH_SIZE` controls how many candidate skills are validated in the same
wave. With `SKILL_BATCH_SIZE=128` and `VALIDATION_ROLLOUTS=8`, one validation
wave runs up to 1024 environments.

`SKILL_GEN_WORKERS` controls concurrent requests to the skill-generation model.
It defaults to 128.

## Outputs

The pipeline writes JSONL files as it progresses:

```text
sampled_tasks.jsonl
baseline_rollouts.jsonl
candidate_skills.jsonl
skill_validations.jsonl
accepted_skills.jsonl
rejected_skills.jsonl
metrics.json
sft_episode_skill_train.parquet
sft_episode_skill_val.parquet
```

Use `--resume` to continue a partially completed run. Use `--overwrite` only
when you want to replace prior outputs in the selected output directory.
