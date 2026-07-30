#!/usr/bin/env bash
# push_to_cluster.sh — 把本地改动同步到集群 CFS 的 SEED checkout。
#
# 为什么不是 scp/rsync/git pull：
#   * 堡垒机是受限 shell，不支持 scp / sftp / 端口转发 / ProxyJump；
#   * GPU pod 上不了外网（pip 三个源全挂、HF 也不通），所以集群侧无法 git pull。
# 剩下唯一可用的通道是交互式 pty，逐行喂 base64 —— 这就是 qf-put.exp 干的事。
#
# 用法：
#   bash ops/local/push_to_cluster.sh                  # 同步默认清单
#   bash ops/local/push_to_cluster.sh path/a.py path/b.sh   # 只同步指定文件
#   FILES_FROM=list.txt bash ops/local/push_to_cluster.sh   # 从文件读清单
#
# 依赖本机 ~/bin/qf-put.exp（把 base64 灌到远端并解码）和已登录的堡垒机常驻连接。
# 每个文件一次会话，大文件慢（约 900 字节/往返），所以只同步源码不同步数据。
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="${REMOTE_SEED_ROOT:-/mnt/cfs_algo_bj/workspace/lijiachun/SEED}"
PUT="${QF_PUT:-$HOME/bin/qf-put.exp}"

# 默认清单：改造点 2/3/4 的实现 + 配置 + launcher + 运维脚本。
# 刻意不含 tests/ 和 docs/ —— 集群上不跑单测，文档在 GitHub 上看。
DEFAULT_FILES=(
    verl/trainer/ppo/seed_external_teacher.py
    verl/trainer/ppo/metric_utils.py
    verl/trainer/ppo/ray_trainer.py
    verl/trainer/config/ppo_trainer.yaml
    agent_system/multi_turn_rollout/teacher_branch.py
    agent_system/multi_turn_rollout/teacher_prefix.py
    agent_system/multi_turn_rollout/branch_runner.py
    agent_system/multi_turn_rollout/rollout_loop.py
    examples/seed_trainer/_common/search.sh
    examples/seed_trainer/run_search_tag_distill_qwen3_8b.sh
    ops/cluster/seedctl.sh
)

if [[ ! -x "$PUT" ]]; then
    echo "找不到上传器：$PUT" >&2
    echo "它需要能把 <本地base64文件> 灌到 <远端路径> 并解码。" >&2
    exit 1
fi
if ! ssh -O check relay >/dev/null 2>&1; then
    echo "堡垒机常驻连接不在，先跑 qf-login（令牌只能手输一次）。" >&2
    exit 1
fi

if [[ -n "${FILES_FROM:-}" ]]; then
    # 不用 mapfile：macOS 自带 bash 是 3.2，没有这个内建
    FILES=()
    while IFS= read -r line; do FILES+=("$line"); done < "$FILES_FROM"
elif [[ $# -gt 0 ]]; then
    FILES=("$@")
else
    FILES=("${DEFAULT_FILES[@]}")
fi

cd "$PROJECT_ROOT" || exit 1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 远端目录可能还不存在（新增的 ops/ 就是这种情况）。一次往返把所有目录建齐，
# 而不是每个文件建一次 —— 一次远程往返 60-90 秒，往返次数越多越容易撞上断连。
dirs=""
for rel in "${FILES[@]}"; do
    [[ -z "$rel" || "$rel" == \#* ]] && continue
    dirs="$dirs $REMOTE_ROOT/$(dirname "$rel")"
done
if [[ -n "$dirs" ]]; then
    echo "==> 建远端目录"
    "$HOME/bin/qf" -c "mkdir -p$dirs && echo mkdir_ok" 2>&1 | tail -2
fi

failed=0
for rel in "${FILES[@]}"; do
    [[ -z "$rel" || "$rel" == \#* ]] && continue
    if [[ ! -f "$rel" ]]; then
        echo "跳过（本地不存在）：$rel" >&2
        continue
    fi
    b64="$TMP/$(echo "$rel" | tr '/' '_').b64"
    base64 < "$rel" | tr -d '\n' > "$b64"
    bytes=$(wc -c < "$rel" | tr -d ' ')
    echo "==> $rel ($bytes 字节)"
    "$PUT" "$b64" "$REMOTE_ROOT/$rel" || {
        echo "  上传失败：$rel" >&2; failed=$((failed+1)); }
done


if [[ $failed -gt 0 ]]; then
    echo "有 $failed 个文件没传上去。" >&2
    exit 1
fi
echo "全部同步完成 -> $REMOTE_ROOT"
echo "下一步：qf -c 'bash $REMOTE_ROOT/ops/cluster/seedctl.sh doctor'"
