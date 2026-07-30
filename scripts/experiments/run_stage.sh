#!/usr/bin/env bash
# 实验矩阵入口:每个阶段一条命令,配置全部内聚在这里。
#
#   bash scripts/experiments/run_stage.sh <stage> <mode>
#
#   stage: e0_grpo_baseline | e1_opd | e2_opd_branch | e3a_selector_klgap | e3b_fullstep
#   mode:  smoke      冲烟一步(小 batch,strict 开,漏斗当场喊话)
#          calibrate  正式 batch 规模跑一步,拿真实 timing(第一次跑该阶段前必做)
#          train      正式 150 步,setsid nohup 脱离连接
#          dry        只打印本阶段生效的开关,不跑
#
# 设计原则(见 docs/seed_main_experiment/training-plan.md 第三节):
# - 一次只动一个变量:每个阶段相对上一阶段只改一组开关;
# - 阶段之间产出物隔离:EXPERIMENT_NAME = 阶段名,checkpoint / step_rm 数据 /
#   日志各归各目录,事后曲线可以直接对比;
# - step RM 数据在所有阶段都开着落盘(便宜,且 E0 的纯学生数据同样有用)。
#
# 前置条件:teacher 副本 + retriever 已起(seedctl teacher / services),
# .env 里 SEED_EXTERNAL_TEACHER_BASE_URL 已填全部副本。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCHER="$PROJECT_ROOT/examples/seed_trainer/run_search_tag_distill_qwen3_8b.sh"

STAGE="${1:-}"
MODE="${2:-dry}"

usage() { sed -n '2,15p' "$0"; exit 2; }
[[ -z "$STAGE" ]] && usage

# ---------------------------------------------------------------- 阶段定义
# 公共基线:tag 菜单开、RM 数据落盘开。各阶段在此之上做增量。
export SEARCH_USE_FUNCTION_TAGS="${SEARCH_USE_FUNCTION_TAGS:-True}"
export SEED_STEP_RM_DUMP="${SEED_STEP_RM_DUMP:-True}"

case "$STAGE" in
    e0_grpo_baseline)
        # 纯 GRPO:teacher 完全不介入(打分/OPD/分支全关)
        export SEED_EXTERNAL_TEACHER_ENABLE=False
        export SEED_ENABLE_ANALYSIS=False
        export SEED_OPD_LOSS_COEF=0.0
        export SEED_OPD_FKL_LOSS_COEF=0.0
        export SEED_TEACHER_BRANCH_ENABLE=False
        ;;
    e1_opd)
        # + OPD dense 过程信号(error-signal 选点);分支仍关
        export SEED_EXTERNAL_TEACHER_ENABLE=True
        export SEED_OPD_LOSS_MODE="${SEED_OPD_LOSS_MODE:-rkl}"
        export SEED_OPD_LOSS_COEF="${SEED_OPD_LOSS_COEF:-0.01}"
        export SEED_STEP_SELECTOR="${SEED_STEP_SELECTOR:-error_signal}"
        export SEED_OPD_FKL_LOSS_COEF=0.0
        export SEED_TEACHER_BRANCH_ENABLE=False
        ;;
    e2_opd_branch)
        # + teacher 前缀分支(think_prefix,error-signal 选点,每轨迹≤1)
        export SEED_EXTERNAL_TEACHER_ENABLE=True
        export SEED_OPD_LOSS_MODE="${SEED_OPD_LOSS_MODE:-rkl}"
        export SEED_OPD_LOSS_COEF="${SEED_OPD_LOSS_COEF:-0.01}"
        export SEED_STEP_SELECTOR="${SEED_STEP_SELECTOR:-error_signal}"
        export SEED_OPD_FKL_LOSS_COEF="${SEED_OPD_FKL_LOSS_COEF:-0.05}"
        export SEED_TEACHER_BRANCH_ENABLE=True
        export SEED_TEACHER_BRANCH_PREFIX_MODE=think_prefix
        export SEED_TEACHER_BRANCH_SELECTOR=null   # 沿用 require_error_signal
        ;;
    e3a_selector_klgap)
        # E2 基础上只换选点:失败轨迹里挑学生-teacher 分歧最大的步
        export SEED_EXTERNAL_TEACHER_ENABLE=True
        export SEED_OPD_LOSS_MODE="${SEED_OPD_LOSS_MODE:-rkl}"
        export SEED_OPD_LOSS_COEF="${SEED_OPD_LOSS_COEF:-0.01}"
        export SEED_STEP_SELECTOR="${SEED_STEP_SELECTOR:-error_signal}"
        export SEED_OPD_FKL_LOSS_COEF="${SEED_OPD_FKL_LOSS_COEF:-0.05}"
        export SEED_TEACHER_BRANCH_ENABLE=True
        export SEED_TEACHER_BRANCH_PREFIX_MODE=think_prefix
        export SEED_TEACHER_BRANCH_SELECTOR='low_reward&kl_gap'
        ;;
    e3b_fullstep)
        # E2 基础上只换前缀模式:teacher 写完整一步,学生下一步接管
        export SEED_EXTERNAL_TEACHER_ENABLE=True
        export SEED_OPD_LOSS_MODE="${SEED_OPD_LOSS_MODE:-rkl}"
        export SEED_OPD_LOSS_COEF="${SEED_OPD_LOSS_COEF:-0.01}"
        export SEED_STEP_SELECTOR="${SEED_STEP_SELECTOR:-error_signal}"
        export SEED_OPD_FKL_LOSS_COEF="${SEED_OPD_FKL_LOSS_COEF:-0.05}"
        export SEED_TEACHER_BRANCH_ENABLE=True
        export SEED_TEACHER_BRANCH_PREFIX_MODE=full_step
        export SEED_TEACHER_BRANCH_SELECTOR=null
        ;;
    *)
        echo "未知阶段: $STAGE" >&2
        usage
        ;;
esac

export EXPERIMENT_NAME="$STAGE"

# ---------------------------------------------------------------- 模式定义
banner() {
    echo "=== stage=$STAGE mode=$MODE ==="
    env | grep -E '^(SEED_|SEARCH_USE|EXPERIMENT_NAME|TRAIN_DATA_SIZE|GROUP_SIZE|TOTAL_TRAINING_STEPS)' | sort | sed 's/^/  /'
}

case "$MODE" in
    dry)
        DRY_RUN=true banner
        DRY_RUN=true bash "$LAUNCHER"
        ;;
    smoke)
        # 小 batch + strict:任何静默退化当场抛出来
        export SEED_TEACHER_BRANCH_STRICT=True
        export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-8}"
        export GROUP_SIZE="${GROUP_SIZE:-4}"
        export VAL_DATA_SIZE="${VAL_DATA_SIZE:-8}"
        export TOTAL_TRAINING_STEPS=1
        export EXPERIMENT_NAME="smoke_$STAGE"
        banner
        bash "$LAUNCHER"
        ;;
    calibrate)
        # 正式 batch 规模,只跑 1 步:拿真实 timing 分解,校准总时长估算
        export SEED_TEACHER_BRANCH_STRICT=True
        export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-128}"
        export GROUP_SIZE="${GROUP_SIZE:-8}"
        export TOTAL_TRAINING_STEPS=1
        export EXPERIMENT_NAME="calibrate_$STAGE"
        banner
        bash "$LAUNCHER"
        echo "查 timing:grep -oE \"timing_s/[a-z_]+':? [0-9.]+\" 日志尾部"
        ;;
    train)
        # 正式 150 步。strict 关:一个坏步骤不该杀掉整个任务。
        # setsid nohup 脱离连接(VPN 黑洞见 open-issues.md)。
        export SEED_TEACHER_BRANCH_STRICT=False
        export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
        LOG_ROOT="${LOG_ROOT:-${WORKSPACE:-$PROJECT_ROOT}/logs}"
        mkdir -p "$LOG_ROOT"
        TRAIN_LOG="$LOG_ROOT/train_$STAGE.log"
        banner
        setsid nohup bash "$LAUNCHER" > "$TRAIN_LOG" 2>&1 < /dev/null &
        echo "已起飞:$STAGE  日志 -> $TRAIN_LOG"
        echo "进度:bash ops/cluster/seedctl.sh status  (TRAIN_LOG=$TRAIN_LOG)"
        ;;
    *)
        echo "未知模式: $MODE" >&2
        usage
        ;;
esac
