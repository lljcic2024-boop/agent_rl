#!/usr/bin/env bash

set -u

OLD_PID="${OLD_PID:-4157602}"
ROOT_DIR="/raid1/HOME/szhang/jywu/code/Planning/SkillRL"
LOG_DIR="${LOG_DIR:-/raid3/data/GTPO/MODELS/ckpt/seed-sdar-skill-gen-grpo_qwen2.5_3b_alfworld_policy-vllm}"
WATCH_LOG="$LOG_DIR/skill_gen_after_train_watcher.log"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
exec > >(tee -a "$WATCH_LOG") 2>&1

echo "[$(date)] Waiting for existing train PID $OLD_PID to finish."
while kill -0 "$OLD_PID" 2>/dev/null; do
  progress=""
  if tmux has-session -t train 2>/dev/null; then
    progress=$(tmux capture-pane -pt train:0 -S -30 | grep -E "Training Progress|step:" | tail -n 2 | tr "\n" " | " || true)
  fi
  echo "[$(date)] train still running. $progress"
  sleep 120
done

echo "[$(date)] Existing train PID $OLD_PID exited. Stopping stale Ray workers if any."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate skillrl
ray stop --force || true

echo "[$(date)] Waiting briefly for GPU memory to settle."
for _ in $(seq 1 15); do
  max_mem=$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | python -c 'import sys; vals=[int(x.strip()) for x in sys.stdin if x.strip()]; print(max(vals) if vals else 0)'
  )
  echo "[$(date)] max GPU memory used: ${max_mem} MiB"
  if [ "$max_mem" -lt 20000 ]; then
    break
  fi
  sleep 60
done

source .env
unset PYTORCH_CUDA_ALLOC_CONF
echo "[$(date)] Starting skill_gen training. conda=${CONDA_DEFAULT_ENV:-unknown}, cwd=$(pwd)"
bash examples/seed/run_alfworld_both_skill_loss_base.sh
status=$?
echo "[$(date)] skill_gen training exited with status $status"
exit "$status"
