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
them through environment variables when needed. The script only creates SFT
data; it does not start SFT or RL training.
