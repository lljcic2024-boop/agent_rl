#!/usr/bin/env bash
# seedctl — SEED 主实验的集群侧唯一入口。
#
# 设计目标：一次远程往返 60-90 秒且随时可能断线，所以每个子命令都是
# 「自包含、幂等、把该看的信息一次全打出来」，而不是让调用方分多次问。
#
#   bash seedctl.sh doctor     # 开跑前体检：模型/数据/服务/GPU/环境变量
#   bash seedctl.sh services   # 本机起 teacher 副本 + retriever，等到健康为止
#   bash seedctl.sh teacher    # 本机只起 teacher 副本（纯 teacher 节点用）
#   bash seedctl.sh smoke      # 冲烟一步（strict 开），跑完直接打漏斗
#   bash seedctl.sh train      # 正式 150 步，setsid nohup 脱离连接
#   bash seedctl.sh status     # 一次性体检报告：进程/步数/漏斗/耗时/报错
#   bash seedctl.sh funnel     # 只看改造点 4 漏斗
#   bash seedctl.sh stop       # 安全停训练（绝不用 pkill -f）
#   bash seedctl.sh logs [n]   # 训练日志尾部 n 行（已去 ANSI）
#
# 多节点 teacher 部署（例：20 卡 = 5 副本 TP=4）：
#   混合节点（teacher 4 卡 + retriever 4 卡）:  bash seedctl.sh services
#   每个纯 teacher 节点（8 卡 = 2 副本）:
#     TEACHER_REPLICAS=2 TEACHER_GPUS=0,1,2,3,4,5,6,7 bash seedctl.sh teacher
#   训练节点 .env 里把所有副本写进一个逗号列表：
#     SEED_EXTERNAL_TEACHER_BASE_URL=http://ip1:8100/v1,http://ip2:8100/v1,http://ip2:8101/v1,...
#   打分客户端会把请求轮询到所有副本，坏一个自动跳过。
#   doctor/status 里设 TEACHER_URLS=<同上列表> 可一次体检全部副本。
#
# 所有路径都可用环境变量覆盖，默认值对应当前千帆 CFS 布局。
set -uo pipefail

SEED_ROOT="${SEED_ROOT:-/mnt/cfs_algo_bj/workspace/lijiachun/SEED}"
WORKSPACE="${WORKSPACE:-/mnt/cfs_algo_bj/workspace/lijiachun}"
MODELS_ROOT="${MODELS_ROOT:-/mnt/cfs_algo_bj/models/opensource_model/Qwen}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/searchr1_data}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE/logs}"
PYLIBS="${PYLIBS:-$WORKSPACE/pylibs}"

TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-8100}"          # 第一个副本的端口，副本 i 用 PORT+i
TEACHER_REPLICAS="${TEACHER_REPLICAS:-1}"     # 本机起几个 teacher 副本
TEACHER_TP="${TEACHER_TP:-4}"                 # 每副本张数
TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3}"       # 本机分给 teacher 的 GPU 列表（按副本切片）
TEACHER_URLS="${TEACHER_URLS:-}"              # 逗号分隔的全部副本 URL；设了则 doctor/status 逐个体检
RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-$MODELS_ROOT/Qwen3-30B-A3B-Instruct-2507}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"

TRAIN_LOG="${TRAIN_LOG:-$LOG_ROOT/train.log}"
SMOKE_LOG="${SMOKE_LOG:-$LOG_ROOT/smoke.log}"
TEACHER_LOG="${TEACHER_LOG:-$LOG_ROOT/teacher.log}"
RETRIEVER_LOG="${RETRIEVER_LOG:-$LOG_ROOT/retriever.log}"

# 训练环境：这五个缺一个就炸，见 docs/seed_main_experiment/cluster-runbook.md
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$SEED_ROOT:$PYLIBS"
export PATH="$PYLIBS/bin:$PATH"
export WANDB_MODE="${WANDB_MODE:-offline}"        # pod 连不上 wandb
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # pod 上不了 HuggingFace
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-$WORKSPACE/tmp}"         # pod 本地盘小，必须挪到 CFS

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'
ok()   { printf '  %sOK%s   %s\n'   "$GRN" "$RST" "$*"; }
bad()  { printf '  %sFAIL%s %s\n'   "$RED" "$RST" "$*"; FAILED=$((FAILED+1)); }
warn() { printf '  %sWARN%s %s\n'   "$YEL" "$RST" "$*"; }
hdr()  { printf '\n=== %s ===\n' "$*"; }
FAILED=0

# 日志里的 ANSI 色码会干扰 grep，统一先剥掉
strip_ansi() { sed -e 's/\x1b\[[0-9;]*m//g'; }

http_code() { curl -s -o /dev/null -m "${2:-8}" -w '%{http_code}' "$1" 2>/dev/null || echo 000; }

# 单副本健康检查（本机端口号），以及"全部副本"检查：
# TEACHER_URLS 设了就逐个查列表，否则查本机 TEACHER_PORT 起的 TEACHER_REPLICAS 个端口。
teacher_port_up() { [[ "$(http_code "http://$TEACHER_HOST:${1:-$TEACHER_PORT}/v1/models")" == 200 ]]; }
teacher_url_up()  { [[ "$(http_code "${1%/}/models")" == 200 ]]; }
teacher_up() {
    if [[ -n "$TEACHER_URLS" ]]; then
        local url
        for url in ${TEACHER_URLS//,/ }; do teacher_url_up "$url" || return 1; done
        return 0
    fi
    local i
    for ((i = 0; i < TEACHER_REPLICAS; i++)); do teacher_port_up $((TEACHER_PORT + i)) || return 1; done
}
teacher_report() {
    # 每个副本一行 OK/FAIL，doctor/status 共用
    if [[ -n "$TEACHER_URLS" ]]; then
        local url
        for url in ${TEACHER_URLS//,/ }; do
            teacher_url_up "$url" && ok "teacher $url" || bad "teacher 挂了：$url"
        done
    else
        local i port
        for ((i = 0; i < TEACHER_REPLICAS; i++)); do
            port=$((TEACHER_PORT + i))
            teacher_port_up "$port" && ok "teacher   :$port" || bad "teacher 不在 :$port（跑 seedctl services / teacher）"
        done
    fi
}
retriever_up() {
    # retriever 没有 health 端点，只能发一个真查询。注意 payload 是单数 query
    # 且值必须是 str —— 传 queries 或 list 都会被 pydantic 拒掉。
    local code
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' \
        -X POST "http://$RETRIEVER_HOST:$RETRIEVER_PORT/retrieve" \
        -H 'Content-Type: application/json' \
        -d '{"query":"who wrote hamlet","topk":1,"return_scores":true}' 2>/dev/null || echo 000)
    [[ "$code" == 200 ]]
}

train_pids() { pgrep -f 'verl.trainer.main_ppo' 2>/dev/null || true; }

cmd_doctor() {
    hdr "路径"
    for p in "$SEED_ROOT" "$MODELS_ROOT" "$DATA_ROOT"; do
        [[ -d "$p" ]] && ok "$p" || bad "缺目录 $p"
    done
    mkdir -p "$LOG_ROOT" "$TMPDIR" 2>/dev/null
    [[ -d "$TMPDIR" ]] && ok "TMPDIR=$TMPDIR" || bad "建不了 TMPDIR=$TMPDIR（pod 本地盘会被塞满）"

    hdr "模型"
    if [[ -f "$MODELS_ROOT/Qwen3-8B-search-tag-sft/config.json" ]]; then
        ok "student = Qwen3-8B-search-tag-sft（tag-SFT 版）"
    elif [[ -f "$MODELS_ROOT/Qwen3-8B/config.json" ]]; then
        warn "tag-SFT checkpoint 不存在，launcher 会回落到 base Qwen3-8B"
    else
        bad "student 模型找不到：$MODELS_ROOT/Qwen3-8B"
    fi
    [[ -f "$TEACHER_MODEL_DIR/config.json" ]] && ok "teacher = $TEACHER_MODEL_NAME" \
        || bad "teacher 模型找不到：$TEACHER_MODEL_DIR"
    [[ -f "$DATA_ROOT/models/e5-base-v2/config.json" ]] \
        && ok "检索编码器 e5-base-v2（本地路径）" \
        || bad "缺 $DATA_ROOT/models/e5-base-v2 —— pod 上不了 HF，必须预先下到 CFS"

    hdr "数据"
    for f in nq_search_seed/train.parquet nq_search_seed/test.parquet e5_Flat.index wiki-18.jsonl; do
        [[ -f "$DATA_ROOT/$f" ]] && ok "$f" || bad "缺 $DATA_ROOT/$f"
    done
    # env_kwargs 列必须存在：rollout_loop.py 直接 pop 它喂进 envs.reset()，
    # 缺列就是 TypeError: object of type 'NoneType' has no len()
    if [[ -f "$DATA_ROOT/nq_search_seed/train.parquet" ]]; then
        python3 - "$DATA_ROOT/nq_search_seed/train.parquet" <<'PY'
import sys
try:
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(sys.argv[1]).schema.names
    n = pq.ParquetFile(sys.argv[1]).metadata.num_rows
    if "env_kwargs" in cols:
        print(f"  \033[32mOK\033[0m   train.parquet 有 env_kwargs 列（{n} 行）")
    else:
        print(f"  \033[31mFAIL\033[0m env_kwargs 列缺失 -> envs.reset 会收到 None 当场炸；"
              f"跑 {__import__('os').environ.get('DATA_ROOT','$DATA_ROOT')}/../mk_envkwargs.py 补列")
        sys.exit(1)
except ImportError:
    print("  \033[33mWARN\033[0m 没装 pyarrow，跳过 env_kwargs 列检查")
PY
        [[ $? -ne 0 ]] && FAILED=$((FAILED+1))
    fi
    cmd_doctor_part2
}

cmd_doctor_part2() {
    hdr "代码"
    [[ -f "$SEED_ROOT/ops/cluster/seedctl.sh" ]] && ok "SEED checkout 在位" \
        || warn "$SEED_ROOT 下没有 ops/cluster/seedctl.sh —— CFS 上的代码可能是旧版"
    # 两个本地修的补丁必须已同步到 CFS，否则 metric 会炸 / 选点会静默退化
    grep -q "_episode_stat" "$SEED_ROOT/verl/trainer/ppo/metric_utils.py" 2>/dev/null \
        && ok "metric_utils object-dtype 修复已在" \
        || bad "metric_utils 缺 _episode_stat —— 分支行产出后 .item() 会炸"
    grep -q "tag_error_signal" "$SEED_ROOT/verl/trainer/ppo/ray_trainer.py" 2>/dev/null \
        && ok "ray_trainer snapshot 带 tag_error_signal" \
        || bad "ray_trainer 缺 tag_error_signal —— 选点器会静默退化成轨迹级"

    hdr "服务"
    teacher_report
    retriever_up && ok "retriever :$RETRIEVER_PORT" || bad "retriever 不在 :$RETRIEVER_PORT（跑 seedctl services）"

    hdr "GPU"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader | sed 's/^/  /'
    else
        warn "没有 nvidia-smi"
    fi

    hdr "运行中的训练"
    local pids; pids=$(train_pids)
    [[ -n "$pids" ]] && warn "已有训练在跑：$pids（要重跑先 seedctl stop）" || ok "没有残留训练进程"

    printf '\n'
    if [[ $FAILED -gt 0 ]]; then
        printf '%s体检失败 %d 项，先修完再开跑。%s\n' "$RED" "$FAILED" "$RST"; return 1
    fi
    printf '%s体检全过，可以 seedctl smoke。%s\n' "$GRN" "$RST"
}

# 起本机的第 $1 个 teacher 副本（0 起数）。GPU 从 TEACHER_GPUS 里按 TP 切片。
start_teacher_replica() {
    local idx="$1"
    local port=$((TEACHER_PORT + idx))
    local log="${TEACHER_LOG%.log}_$port.log"
    local gpus
    gpus=$(echo "$TEACHER_GPUS" | tr ',' '\n' | sed -n "$((idx * TEACHER_TP + 1)),$(((idx + 1) * TEACHER_TP))p" | paste -sd,)
    if [[ -z "$gpus" || $(echo "$gpus" | tr ',' '\n' | wc -l) -lt $TEACHER_TP ]]; then
        bad "副本 $idx 分不到 $TEACHER_TP 张卡（TEACHER_GPUS=$TEACHER_GPUS）"
        return 1
    fi
    # 显存参数说明（都踩过 OOM，别随手改回去）：
    # - prompt_logprobs 要求对每个 prompt 位置物化全词表 logits。vLLM V1 的
    #   chunked prefill 下，这个瞬时峰值 ≈ max-num-batched-tokens × 词表(15万) ×
    #   ~6B/位置，且集中在 TP rank0 —— 峰值只由 batched-tokens 决定，与并发无关。
    # - 所以治 OOM 的旋钮是 --max-num-batched-tokens（4096 ≈ 4GB 峰值），
    #   而 --max-num-seqs 和客户端并发可以放开（多的只是很小的 GQA KV cache）。
    # - 0.85 util 时代的 OOM -> EngineDeadError -> server 永久挂掉，训练侧只看到
    #   Connection refused；现在多副本 + 客户端 failover，坏一个不再是全局事故。
    CUDA_VISIBLE_DEVICES="$gpus" \
    setsid nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$TEACHER_MODEL_DIR" \
        --served-model-name "$TEACHER_MODEL_NAME" \
        --port "$port" \
        --tensor-parallel-size "$TEACHER_TP" \
        --gpu-memory-utilization "${TEACHER_GPU_UTIL:-0.72}" \
        --max-model-len 16384 \
        --max-logprobs 40 \
        --max-num-batched-tokens "${TEACHER_MAX_BATCHED_TOKENS:-4096}" \
        --max-num-seqs "${TEACHER_MAX_SEQS:-64}" \
        > "$log" 2>&1 < /dev/null &
    echo "  副本 $idx: 端口 $port GPU $gpus 日志 -> $log"
}

start_teacher_replicas() {
    local i port started=0
    for ((i = 0; i < TEACHER_REPLICAS; i++)); do
        port=$((TEACHER_PORT + i))
        if teacher_port_up "$port"; then
            ok "teacher 已在 :$port，跳过"
        else
            [[ $started -eq 0 ]] && hdr "起 teacher vLLM（$TEACHER_REPLICAS 副本 × TP=$TEACHER_TP）"
            start_teacher_replica "$i" || return 1
            started=1
        fi
    done
}

wait_local_teachers() {
    hdr "等 teacher 副本就绪（最多 ${SERVICE_WAIT:-600} 秒；30B 加载要几分钟）"
    local waited=0 step=10 i port up total
    while [[ $waited -lt ${SERVICE_WAIT:-600} ]]; do
        up=0; total=$TEACHER_REPLICAS
        for ((i = 0; i < TEACHER_REPLICAS; i++)); do
            teacher_port_up $((TEACHER_PORT + i)) && up=$((up + 1))
        done
        printf '\r  %3ds  teacher %d/%d' "$waited" "$up" "$total"
        [[ $up -eq $total ]] && { printf '\n'; ok "全部 teacher 副本就绪"; return 0; }
        sleep $step; waited=$((waited + step))
    done
    printf '\n'
    bad "等超时。第一个未就绪副本的日志尾部："
    for ((i = 0; i < TEACHER_REPLICAS; i++)); do
        port=$((TEACHER_PORT + i))
        if ! teacher_port_up "$port"; then
            tail -20 "${TEACHER_LOG%.log}_$port.log" 2>/dev/null | strip_ansi | sed 's/^/    /'
            break
        fi
    done
    return 1
}

# 纯 teacher 节点入口：只起副本，不管 retriever。
# 例：8 卡节点起 2 副本 -> TEACHER_REPLICAS=2 TEACHER_GPUS=0,1,2,3,4,5,6,7 bash seedctl.sh teacher
cmd_teacher() {
    mkdir -p "$LOG_ROOT"
    start_teacher_replicas || return 1
    wait_local_teachers || return 1
    echo
    echo "  把这台机器的副本加进训练节点 .env 的 SEED_EXTERNAL_TEACHER_BASE_URL："
    local i self_ip
    self_ip=$(hostname -i 2>/dev/null | awk '{print $1}')
    for ((i = 0; i < TEACHER_REPLICAS; i++)); do
        echo "    http://${self_ip:-<本机IP>}:$((TEACHER_PORT + i))/v1"
    done
}

cmd_services() {
    mkdir -p "$LOG_ROOT"
    start_teacher_replicas || return 1

    if retriever_up; then
        ok "retriever 已在 :$RETRIEVER_PORT，跳过"
    else
        hdr "起 retriever"
        CUDA_VISIBLE_DEVICES="${RETRIEVER_GPUS:-4,5,6,7}" \
        setsid nohup python3 "$SEED_ROOT/examples/search/retriever/retrieval_server.py" \
            --index_path      "$DATA_ROOT/e5_Flat.index" \
            --corpus_path     "$DATA_ROOT/wiki-18.jsonl" \
            --retriever_model "$DATA_ROOT/models/e5-base-v2" \
            --faiss_gpu --port "$RETRIEVER_PORT" \
            > "$RETRIEVER_LOG" 2>&1 < /dev/null &
        echo "  日志 -> $RETRIEVER_LOG"
    fi

    # 2100 万条语料建索引要 1-2 分钟，30B 加载也要几分钟，别以为卡死了
    hdr "等服务就绪（最多 ${SERVICE_WAIT:-600} 秒）"
    local waited=0 step=10 i t_up
    while [[ $waited -lt ${SERVICE_WAIT:-600} ]]; do
        local r=no
        t_up=0
        for ((i = 0; i < TEACHER_REPLICAS; i++)); do
            teacher_port_up $((TEACHER_PORT + i)) && t_up=$((t_up + 1))
        done
        retriever_up && r=yes
        printf '\r  %3ds  teacher=%d/%d retriever=%-3s' "$waited" "$t_up" "$TEACHER_REPLICAS" "$r"
        [[ $t_up -eq $TEACHER_REPLICAS && "$r" == yes ]] && { printf '\n'; ok "全部服务就绪"; return 0; }
        sleep $step; waited=$((waited+step))
    done
    printf '\n'
    bad "等超时。第一个未就绪 teacher 副本的日志尾部："
    for ((i = 0; i < TEACHER_REPLICAS; i++)); do
        if ! teacher_port_up $((TEACHER_PORT + i)); then
            tail -20 "${TEACHER_LOG%.log}_$((TEACHER_PORT + i)).log" 2>/dev/null | strip_ansi | sed 's/^/    /'
            break
        fi
    done
    return 1
}

LAUNCHER="examples/seed_trainer/run_search_tag_distill_qwen3_8b.sh"

# 冲烟：一步、小 batch、strict 开着。strict 的意义是让「改造点 4 产出 0 个分支」
# 当场抛异常，而不是静默退化成普通 PPO 跑完 150 步才发现 FKL 项全程是空的。
cmd_smoke() {
    [[ -n "$(train_pids)" ]] && { bad "已有训练在跑，先 seedctl stop"; return 1; }
    mkdir -p "$LOG_ROOT"
    hdr "冲烟一步（strict=True）"
    cd "$SEED_ROOT" || return 1
    SEED_TEACHER_BRANCH_STRICT=True \
    TRAIN_DATA_SIZE="${SMOKE_TRAIN_SIZE:-8}" \
    GROUP_SIZE="${SMOKE_GROUP_SIZE:-4}" \
    VAL_DATA_SIZE="${SMOKE_VAL_SIZE:-8}" \
    TOTAL_TRAINING_STEPS=1 \
    EXPERIMENT_NAME="${SMOKE_EXPERIMENT_NAME:-smoke_tag_distill}" \
    DEFAULT_LOCAL_DIR="${SMOKE_LOCAL_DIR:-$WORKSPACE/ckpt/smoke_tag_distill}" \
        setsid nohup bash "$LAUNCHER" > "$SMOKE_LOG" 2>&1 < /dev/null &
    local pid=$!
    echo "  pid=$pid  日志 -> $SMOKE_LOG"
    echo "  一步约 80-90 秒（teacher 打分占 54%），等它跑完……"
    # 冲烟必须等：等不到结果这一趟远程往返就白费了
    local waited=0
    while [[ $waited -lt ${SMOKE_WAIT:-1800} ]]; do
        sleep 20; waited=$((waited+20))
        if [[ -z "$(train_pids)" ]]; then
            echo "  训练进程已退出（${waited}s）"
            break
        fi
        grep -qE "step:1 |'training/global_step': 1" "$SMOKE_LOG" 2>/dev/null && {
            echo "  第 1 步完成（${waited}s）"; break; }
        printf '\r  等待中 %ds' "$waited"
    done
    printf '\n'
    cmd_funnel
    hdr "报错扫描"
    # hydra 把报错埋在几百行 config dump 之后，tail 看不到，必须 grep
    grep -nE "Error|Traceback|RayTaskError|BranchProducedNothing|ModuleNotFound|OutOfMemory" \
        "$SMOKE_LOG" 2>/dev/null | strip_ansi | head -20 | sed 's/^/  /' \
        || ok "没有报错关键字"
}

# 正式训练：必须 setsid nohup —— 本地到集群的连接会周期性断开（VPN 隧道黑洞，
# 心跳救不了），任务必须脱离这条连接独立存活。strict 关掉，一个坏步骤不该杀掉 150 步。
cmd_train() {
    [[ -n "$(train_pids)" ]] && { bad "已有训练在跑：$(train_pids)"; return 1; }
    teacher_up   || { bad "teacher 不在，先 seedctl services";   return 1; }
    retriever_up || { bad "retriever 不在，先 seedctl services"; return 1; }
    mkdir -p "$LOG_ROOT"
    hdr "正式训练 ${TOTAL_TRAINING_STEPS:-150} 步"
    cd "$SEED_ROOT" || return 1
    SEED_TEACHER_BRANCH_STRICT=False \
        setsid nohup bash "$LAUNCHER" > "$TRAIN_LOG" 2>&1 < /dev/null &
    sleep 25
    local pids; pids=$(train_pids)
    if [[ -n "$pids" ]]; then
        ok "已起飞：$pids"
        echo "  日志 -> $TRAIN_LOG"
        echo "  断线不影响它；重连后 seedctl status 看进度。"
    else
        bad "起飞失败，日志尾部："
        tail -30 "$TRAIN_LOG" 2>/dev/null | strip_ansi | sed 's/^/    /'
        return 1
    fi
}

pick_log() { [[ -f "$TRAIN_LOG" && -s "$TRAIN_LOG" ]] && echo "$TRAIN_LOG" || echo "$SMOKE_LOG"; }

# 改造点 4 漏斗。candidate 数不等于 branch 数 —— 上一轮就是看到 num_candidates=21
# 就宣布改造点 4 已上线，结果 FKL 项全程是空的。这里必须把整条链一起打出来。
cmd_funnel() {
    local log; log=$(pick_log)
    hdr "改造点 4 漏斗（$log）"
    [[ -f "$log" ]] || { bad "日志不存在"; return 1; }
    python3 - "$log" <<'PY'
import re, sys
keys = ["num_candidates", "num_specs_with_env_kwargs", "num_prefixes",
        "num_trajectories", "num_rows"]
exits = ["disabled", "schedule", "no_rows", "no_env_kwargs", "no_prefix", "no_tensors", "ok"]
last = {}
for line in open(sys.argv[1], errors="ignore"):
    for k in keys + [f"exit_{e}" for e in exits]:
        m = re.search(rf"seed/teacher_branch/{k}'?\s*:\s*([0-9.]+)", line)
        if m:
            last[k] = float(m.group(1))
if not last:
    print("  漏斗指标一个都没出现 —— 改造点 4 的 run() 可能根本没被调用，"
          "或者这一步还没跑到 teacher_branch 阶段")
    sys.exit(0)
print("  " + " -> ".join(f"{k.replace('num_','')}={last.get(k,0):g}" for k in keys))
flagged = [e for e in exits if last.get(f"exit_{e}", 0) == 1.0]
print(f"  退出原因: {flagged or '未标记'}")
# 把「看起来健康但其实没产出」的两种情况直接翻译成下一步查什么
c, s, p, r = (last.get(k, 0) for k in ("num_candidates", "num_specs_with_env_kwargs",
                                       "num_prefixes", "num_rows"))
if c > 0 and s == 0:
    print("  >> 根因=env 载荷解析失败。查 trainer 是否真把 env_kwargs 传进来了 / sample_id 是否越界")
elif s > 0 and p == 0:
    print("  >> 根因=teacher 拒绝给 a_T。查质检门是否过严 / teacher server 状态")
elif r > 0:
    print(f"  >> 改造点 4 正常工作，产出 {r:g} 行分支")
PY
}

cmd_status() {
    local log; log=$(pick_log)
    hdr "进程"
    local pids; pids=$(train_pids)
    [[ -n "$pids" ]] && ok "训练在跑：$pids" || warn "没有训练进程（跑完了 or 死了）"
    teacher_report
    retriever_up && ok "retriever :$RETRIEVER_PORT" || bad "retriever 挂了"

    hdr "进度（$log）"
    if [[ -f "$log" ]]; then
        echo "  日志行数: $(wc -l < "$log")"
        grep -oE "step:[0-9]+" "$log" 2>/dev/null | tail -1 | sed 's/^/  最新 /' || echo "  还没跑完第一步"
    else
        bad "日志不存在：$log"
    fi

    cmd_funnel

    hdr "一步耗时分解"
    grep -oE "timing_s/[a-z_]+':? [0-9.]+" "$log" 2>/dev/null | tail -12 | sed 's/^/  /' \
        || echo "  还没有 timing 数据"

    hdr "teacher 打分吞吐（最近一步）"
    # 客户端每次打分都报 elapsed_s / prefill tok/s / 重试 / 副本故障数；
    # endpoint_failures > 0 说明有副本在丢请求（failover 兜住了，但要去修）
    grep -oE "seed/external_teacher/[a-z_]+/(elapsed_s|prefill_tokens_per_s|rows_scored|num_batches|retries|endpoint_failures)':? [0-9.]+" \
        "$log" 2>/dev/null | tail -12 | sed 's/^/  /' \
        || echo "  还没有打分统计（外接 teacher 未启用或第一步未完成）"
    if grep -q "seed/teacher_signal_failed':? 1" "$log" 2>/dev/null; then
        bad "出现过 teacher_signal_failed=1 —— 有训练步的 OPD 项是空的，查 teacher 副本"
    fi

    hdr "报错扫描"
    grep -nE "Error|Traceback|RayTaskError|OutOfMemory|EngineDead|Connection refused" \
        "$log" 2>/dev/null | strip_ansi | tail -15 | sed 's/^/  /' || ok "没有报错关键字"
}

cmd_logs() { local log; log=$(pick_log); tail -"${1:-80}" "$log" 2>/dev/null | strip_ansi; }

# 绝对不能用 pkill -f main_ppo：pattern 会匹配到承载它的 bash -c 自己的命令行，
# 当场自杀 —— 表现为 exit code 143 加零输出。必须先取 pid 再逐个 kill。
cmd_stop() {
    local pids; pids=$(train_pids)
    [[ -z "$pids" ]] && { ok "没有训练进程"; return 0; }
    echo "  停止：$pids"
    for p in $pids; do kill "$p" 2>/dev/null || true; done
    sleep 15
    pids=$(train_pids)
    if [[ -n "$pids" ]]; then
        warn "还在，SIGKILL：$pids"
        for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    fi
    sleep 3
    [[ -z "$(train_pids)" ]] && ok "已停" || bad "停不掉：$(train_pids)"
}

case "${1:-}" in
    doctor)   cmd_doctor ;;
    services) cmd_services ;;
    teacher)  cmd_teacher ;;
    smoke)    cmd_smoke ;;
    train)    cmd_train ;;
    status)   cmd_status ;;
    funnel)   cmd_funnel ;;
    logs)     cmd_logs "${2:-80}" ;;
    stop)     cmd_stop ;;
    all)      cmd_doctor && cmd_services && cmd_smoke ;;
    *)        sed -n '2,20p' "$0"; exit 2 ;;
esac






