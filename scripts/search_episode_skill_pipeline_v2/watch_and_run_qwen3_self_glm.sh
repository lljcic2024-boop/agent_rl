#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CURRENT_EXPERIMENT_NAME="${CURRENT_EXPERIMENT_NAME:-seed_qwen3_1.7b_search_episode_no_skill_loss_sft_policy-vllm}"
CURRENT_RL_PID="${CURRENT_RL_PID:-}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/search_episode_skill_pipeline_v2}"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/qwen3_self_glm_after_rl_${RUN_ID}.log}"

mkdir -p "$LOG_DIR" || exit 1

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*" | tee -a "$MASTER_LOG"
}

run_logged() {
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    local status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        log "Command failed with status $status: $*"
        exit "$status"
    fi
}

log "Watcher started."
log "Current RL experiment: $CURRENT_EXPERIMENT_NAME"
if [[ -n "$CURRENT_RL_PID" ]]; then
    log "Current RL PID: $CURRENT_RL_PID"
else
    log "Current RL PID not provided; falling back to process-table discovery."
    CURRENT_RL_PID="$(
        ps -eo pid=,args= -ww \
            | awk -v exp="trainer.experiment_name=$CURRENT_EXPERIMENT_NAME" \
                'index($0, "verl.trainer.main_ppo") && index($0, exp) {print $1; exit}'
    )"
    log "Discovered current RL PID: ${CURRENT_RL_PID:-none}"
fi
log "Check interval: ${CHECK_INTERVAL}s"

while true; do
    if [[ -n "$CURRENT_RL_PID" ]] && kill -0 "$CURRENT_RL_PID" >/dev/null 2>&1; then
        log "Current Qwen3 RL status: running"
        log "Current Qwen3 RL is still running; waiting."
        sleep "$CHECK_INTERVAL"
    else
        log "Current Qwen3 RL status: stopped"
        break
    fi
done

log "Current Qwen3 RL no longer appears in the process table."
log "Waiting 60s for Ray/vLLM cleanup before launching the next experiment."
sleep 60

log "Starting Qwen3 self-baseline + GLM data generation."
run_logged bash "$SCRIPT_DIR/run_qwen3_self_glm.sh"

log "Starting Qwen3 self-baseline SFT."
run_logged bash "$SCRIPT_DIR/run_sft_qwen3_self_glm.sh"

log "Starting Qwen3 self-baseline RL."
run_logged bash "$PROJECT_ROOT/examples/policy_vllm_opid_trainer/run_search_qwen3_episode_no_skill_loss_sft_self.sh"

log "Qwen3 self-baseline experiment chain finished."
