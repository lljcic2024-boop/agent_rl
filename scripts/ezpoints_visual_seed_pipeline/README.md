# EZPoints visual episode-skill SFT data

This pipeline creates 1,440 visual episode-skill examples by default:

1. Run 8 baseline policy rollouts for each of 180 deterministic EZPoints tasks.
2. Generate one episode skill per trajectory with the configured OpenAI-compatible teacher.
3. Build JSONL plus train/validation parquet files for visual SFT.

Run the full data pipeline from the repository root:

```bash
bash scripts/ezpoints_visual_seed_pipeline/run.sh
```

The default output directory is:

```text
outputs/ezpoints_episode_skill_pipeline_qwen25_vl_3b_gemini_self_1440/
```

Important defaults are `BASELINE_REQUEST_WORKERS=64`,
`EZPOINTS_SKILL_GEN_WORKERS=64`, and resume enabled for both stages. Override
them through environment variables when needed.

After the parquet files are built, run visual SFT:

```bash
bash scripts/ezpoints_visual_seed_pipeline/run_sft.sh
```

Useful overrides include `MODEL_PATH`, `DATA_DIR`, `NPROC_PER_NODE`,
`TOTAL_EPOCHS`, `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE_PER_GPU`,
`MAX_LENGTH`, and `EXPORT_MODEL_DIR`.

Then run SEED RL from the exported SFT model:

```bash
bash examples/seed_trainer/run_ezpoints_episode_no_skill_loss_sft_gemini_self.sh
```
