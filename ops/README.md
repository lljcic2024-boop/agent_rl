# 运维脚本：一键开跑

两个脚本，分工是「本地把代码送上去」和「集群上把实验跑起来」：

| 脚本 | 跑在哪 | 干什么 |
|---|---|---|
| [`local/push_to_cluster.sh`](local/push_to_cluster.sh) | 本地 Mac | 把改动同步到 CFS 的 SEED checkout |
| [`cluster/seedctl.sh`](cluster/seedctl.sh) | 集群 pod 内 | 体检 / 起服务 / 冲烟 / 训练 / 看状态 / 停 |

## 完整流程

```bash
# 0. 本机一次性：堡垒机登录（令牌手输一次，之后免令牌）
qf-login

# 1. 本地 -> CFS 同步代码
bash ops/local/push_to_cluster.sh

# 2. 开跑前体检：模型/数据/服务/GPU/补丁 一次全查
qf -c 'bash /mnt/cfs_algo_bj/workspace/lijiachun/SEED/ops/cluster/seedctl.sh doctor'

# 3. 起 teacher + retriever（幂等，已在就跳过；等到健康为止）
qf -c 'bash .../seedctl.sh services'

# 4. 冲烟一步，strict 开着，跑完直接打漏斗
qf -c 'bash .../seedctl.sh smoke'

# 5. 漏斗健康就起正式 150 步
qf -c 'bash .../seedctl.sh train'

# 6. 之后任何时候看进度（断线重连后也是这一条）
qf -c 'bash .../seedctl.sh status'
```

`doctor` → `services` → `smoke` 可以合成一条 `seedctl.sh all`。

## 为什么这样设计

**一次远程往返 60-90 秒，且随时可能断线。** 所以每个子命令都是自包含、幂等、把该看的信息一次全打出来，而不是让调用方分多次问。`status` 一条命令同时给出进程、步数、漏斗、耗时分解和报错扫描，就是为了把往返次数压到 1。

**同步走 base64 灌 pty，不是 scp。** 堡垒机是受限 shell，不支持 scp / sftp / 端口转发 / ProxyJump；而 GPU pod 上不了外网（pip 三个源全挂，HF 也不通），所以集群侧也没法 `git pull`。剩下唯一可用的通道就是交互式 pty 逐行喂 base64。约 900 字节一个往返，所以**只同步源码，不同步数据和模型**。

**`train` 强制 `setsid nohup`。** 本地到集群的连接会周期性断开（VPN 隧道包黑洞，心跳救不了，详见 [open-issues.md](../docs/seed_main_experiment/open-issues.md)）。任务必须脱离这条连接独立存活，断线后重连只是重新去读日志。

## 几个不显然的实现细节

**`stop` 绝不用 `pkill -f`。** pattern 会匹配到承载它的 `bash -c` 自己的命令行，当场自杀——表现为 exit code 143 加零输出。必须先 `pgrep` 取 pid 再逐个 `kill`。

**`smoke` 开 `strict=True`，`train` 关。** strict 让「改造点 4 产出 0 个分支」当场抛 `BranchProducedNothing`，而不是静默退化成普通 PPO。冲烟时要这个；正式跑时不要——一个坏步骤不该杀掉 150 步的进度。

**`funnel` 把漏斗数字直接翻译成下一步查什么。** 因为 `num_candidates` 不等于 branch 数：

| 观测 | 根因 | 查什么 |
|---|---|---|
| `candidates>0, specs_with_env_kwargs=0` | env 载荷解析失败 | trainer 是否真传了 `env_kwargs` / `sample_id` 越界 |
| `specs>0, prefixes=0` | teacher 拒绝给 a_T | 质检门是否过严 / teacher server 状态 |

上一轮就是看到 `num_candidates=21` 就宣布改造点 4 已上线，而 FKL 项全程是空的。

**报错用 `grep` 不用 `tail`。** hydra 把报错埋在几百行 config dump 之后，`tail` 看不到。日志里的 ANSI 色码也会干扰 grep，所以统一先 `sed` 剥掉。

**`doctor` 会验 `env_kwargs` 列。** 原始 Search-R1 parquet 没有这列，而 `rollout_loop.py` 直接 `pop` 它喂进 `envs.reset()`，缺列就是当场 `TypeError: object of type 'NoneType' has no len()`。

**`doctor` 会验两个已知补丁在不在 CFS 上**（`metric_utils` 的 `_episode_stat`、`ray_trainer` snapshot 里的 `tag_error_signal`）。这两个缺了都不会立刻报错，而是等到改造点 4 真产出分支行时才炸，或者让选点器静默退化成轨迹级。

## 环境变量

两个脚本的路径全部可覆盖，默认值对应当前千帆 CFS 布局。集群侧常用的：

```bash
SEED_ROOT      # 默认 /mnt/cfs_algo_bj/workspace/lijiachun/SEED
MODELS_ROOT    # 模型根目录
DATA_ROOT      # searchr1_data
LOG_ROOT       # 日志落盘位置
TEACHER_GPUS   # 默认 0,1,2,3
RETRIEVER_GPUS # 默认 4,5,6,7
TEACHER_TP     # 默认 4
TEACHER_GPU_UTIL  # 默认 0.72 —— 别往上调到 0.85，会 OOM 打死整个 engine
```

`seedctl` 自己会设好那五个缺一个就炸的训练环境变量（`PYTHONPATH` / `WANDB_MODE=offline` /
`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `TMPDIR`），不用手动 export。
