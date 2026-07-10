#!/usr/bin/env bash

set -u

PROJECT_ROOT="${PROJECT_ROOT:-/root/works/SGOP}"
SEARCH_MAIN_PID="${SEARCH_MAIN_PID:-2571955}"
SEARCH_TASK_PID="${SEARCH_TASK_PID:-2580579}"
SEARCH_CKPT_DIR="${SEARCH_CKPT_DIR:-/data/models/Qwen/ckpt/seed-grpo_qwen2.5_7b_search_episode_no_skill_loss_sft_glm_self_policy-vllm}"
WEBSHOP_SFT_MODEL="${WEBSHOP_SFT_MODEL:-/data/models/Qwen/Qwen3-1.7B-webshop-episode-skill-sft}"
CONDA_ENV="${CONDA_ENV:-copd}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-300}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/webshop_qwen3/watch_and_run_qwen3_tmux_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

prefix="[webshop-qwen3-watch]"

cd "$PROJECT_ROOT" || exit 1

echo "$prefix started at $(date)"
echo "$prefix log file: $LOG_FILE"
echo "$prefix waiting for search RL pids: $SEARCH_MAIN_PID $SEARCH_TASK_PID"
echo "$prefix search ckpt dir: $SEARCH_CKPT_DIR"

while true; do
    ps -p "$SEARCH_MAIN_PID" >/dev/null 2>&1
    main_alive=$?
    ps -p "$SEARCH_TASK_PID" >/dev/null 2>&1
    task_alive=$?
    if [[ "$main_alive" -ne 0 && "$task_alive" -ne 0 ]]; then
        break
    fi

    latest_rollout="$(
        find "$SEARCH_CKPT_DIR" -maxdepth 1 -name '*.jsonl' -printf '%f\n' 2>/dev/null \
            | sort -V \
            | tail -n 1
    )"
    latest_analysis="$(
        find "$SEARCH_CKPT_DIR/seed_analysis" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null \
            | sort -V \
            | tail -n 1
    )"
    echo "$prefix $(date) still running; latest_rollout=${latest_rollout:-none}; latest_analysis=${latest_analysis:-none}"
    sleep "$POLL_INTERVAL_SECONDS"
done

echo "$prefix search RL pids exited at $(date)"
if [[ ! -f "$SEARCH_CKPT_DIR/150.jsonl" && ! -d "$SEARCH_CKPT_DIR/global_step_150" ]]; then
    latest_rollout="$(
        find "$SEARCH_CKPT_DIR" -maxdepth 1 -name '*.jsonl' -printf '%f\n' 2>/dev/null \
            | sort -V \
            | tail -n 1
    )"
    echo "$prefix search RL does not look complete; latest_rollout=${latest_rollout:-none}. Not starting WebShop." >&2
    exit 1
fi

echo "$prefix search RL completion marker found."
sleep 60

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

if [[ ! -f "$WEBSHOP_SFT_MODEL/config.json" ]]; then
    echo "$prefix WebShop Qwen3 SFT checkpoint missing; starting SFT at $(date)"
    bash scripts/webshop_episode_skill_pipeline_v2/run_sft_qwen3.sh
    sft_status=$?
    if [[ "$sft_status" -ne 0 ]]; then
        echo "$prefix WebShop Qwen3 SFT failed with status $sft_status" >&2
        exit "$sft_status"
    fi
else
    echo "$prefix WebShop Qwen3 SFT checkpoint exists: $WEBSHOP_SFT_MODEL"
fi

echo "$prefix starting WebShop Qwen3 RL at $(date)"
bash examples/seed/run_webshop_qwen3_episode_no_skill_loss_sft.sh
rl_status=$?
echo "$prefix WebShop Qwen3 RL exited with status $rl_status at $(date)"
exit "$rl_status"
