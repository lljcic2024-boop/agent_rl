# SEED 功能类型化蒸馏：主实验文档

Student Qwen3-8B 在 Search QA 上向外接 Qwen3-30B-A3B teacher 学「工具调用的思考过程」。
基于 SEED（内嵌 verl 0.3.1.dev）的多轮 agent RL，`algorithm.adv_estimator=seed`。

文档分三份：

| 文档 | 内容 |
|---|---|
| 本文 | 四个改造点是什么、代码在哪、当前状态、怎么跑 |
| [training-plan.md](training-plan.md) | **训练思路总纲**:科学问题、损失构成、实验矩阵 E0-E3、决策门、指标清单;配套一键脚本 [`scripts/experiments/run_stage.sh`](../../scripts/experiments/run_stage.sh) |
| [cluster-runbook.md](cluster-runbook.md) | 集群拓扑、模型/数据/产出的存放位置、服务启动、踩过的接口坑 |
| [open-issues.md](open-issues.md) | 尚未解决的问题、已定位但未修的 bug、性能瓶颈的实测数据 |
| [../../ops/README.md](../../ops/README.md) | 一键开跑的运维脚本：同步、体检、起服务、冲烟、训练 |


---

## 核心思路

普通 PPO 里学生只从自己的采样中学。这套改造让 teacher 在**学生卡住的那一步**介入，
提供两种监督：

1. **OPD（打分）** —— teacher 对学生已经写出的 token 逐个打分，学生朝 teacher 的分布靠。
   反向 KL（rkl），带 gate。
2. **FKL（示范）** —— teacher 亲自写一段思考开头 `a_T`，学生从这个开头往下续写。
   `a_T` 的 token 走 FKL（模仿），学生续写的部分走 PG（试错）。

第 2 条是这套方案区别于普通蒸馏的地方：不是让学生模仿整条 teacher 轨迹，而是
**只借用一个思路开头，剩下的路仍由学生自己走、自己承担环境反馈**。

## 轨迹如何切块

一条多轮轨迹进 batch 时**已经按「一次 action + 一次 feedback」切成独立行**，
不需要额外的切块逻辑。`gather_rollout_data()`（`agent_system/multi_turn_rollout/rollout_loop.py`）
给每行打上：

```python
data['step_id'] = f"{sample_id}_{rollout_id}_{step_num}"
```

所以一条 4 轮轨迹 = 4 行训练样本。这解释了一个容易误读的数字：batch 只有 8 道题，
teacher 打分请求却有 149 个 —— **149 是「步」数，不是「题」数**。

关键问题不在切块，而在**这 149 步里只有一部分值得学**，这就是改造点 3 的职责。

---

## 四个改造点

### 改造点 1：function-tag 能力菜单

给 prompt 加一份显式的「你可以做这些动作」清单（`<plan>` / `<verify>` / `<reflect>` /
`<backtrack>` / `<search>` / `<answer>`），把隐式的思考过程变成可计数、可选点的标签。

| 位置 | 作用 |
|---|---|
| `agent_system/environments/prompts/` | tag 菜单 prompt 模板 |
| `scripts/sft/search/tag_rewrite.sh` | 用 teacher 把原始 SFT 数据重写成带 tag 的形式 |
| `scripts/sft/search/train_sft.sh` | 拿重写后的数据做 tag-SFT，产出 student 起点 |
| 开关 | `SEARCH_USE_FUNCTION_TAGS=True` |

产出的 tag 计数落到 `tag_plan_count` / `tag_verify_count` / `tag_reflect_count` /
`tag_backtrack_count` 列，既是行为指标也是改造点 3 的选点输入。

**状态**：代码完成，36 项单测。**但 tag-SFT checkpoint 还没产出** ——
`Qwen3-8B-search-tag-sft` 不存在，launcher 打 WARNING 后回落到 base `Qwen3-8B`。

### 改造点 2：外接 30B teacher 打分 + OPD

`verl/trainer/ppo/seed_external_teacher.py`（198 行）

把「策略在增广 prompt 下给自己打分」换成真的 teacher 模型打分。**以 token id 而非文本
发送**，所以只要 student / teacher 同 tokenizer 家族（Qwen3-8B vs Qwen3-30B-A3B）
token 对齐就是精确的。

```python
payload = {
    "model": self.model,
    "prompt": token_ids,      # token id 数组，不是文本
    "max_tokens": 1,          # 只生成 1 个 token，而且丢掉不用
    "temperature": 0.0,
    "prompt_logprobs": 0,     # 真正要的是这个
}
```

**这里有个关键概念区分**：`max_tokens=1` 意味着 teacher **完全没有在生成/rollout**。
要的是 `prompt_logprobs` —— 对学生已经写好的 token 逐位置打分。生成要一个 token 一个
token 挤，打分整段一次前向算完。所以「等 teacher rollout」这项耗时**不存在**。

损失侧：`SEED_OPD_LOSS_MODE=gate`、`SEED_OPD_LOSS_COEF=0.01`、`SEED_OPD_GATE_BETA=5.0`。

**吞吐设计（2026-07-30）**：客户端支持**多副本 + 合批 + 观测**——
`base_url` 接受逗号分隔的副本列表（轮询分发、重试自动换端点），每个 HTTP 请求打包
`batch_size=16` 条 prompt，并发默认 16。每次打分报
`seed/external_teacher/<label>/*` 吞吐指标（elapsed_s / prefill_tokens_per_s /
retries / endpoint_failures），打分异步失败时 `seed/teacher_signal_failed=1` +
ERROR 级 traceback，不再静默退化。

**状态**：跑通并实测有输出。合批 + 多副本已实现（146 项单测），
集群实测吞吐待验，见 [open-issues.md](open-issues.md)。

### 改造点 3：选点

只对「学生这一步犯了错」的步做 OPD，而不是全部 149 步。

- **Phase 2a（error-signal）已实现**：从环境反馈里识别异常信号（检索空结果、
  格式错误、重复动作等），落成 `tag_error_signal` 列。`SEED_STEP_SELECTOR=error_signal`。
- **Phase 2b（critic-gap）已延后**：按 critic 预测与实际回报的落差选点，尚未实现。

**踩过的坑**：打分异步执行时，选点器读的是 snapshot 而不是原 batch，
`tag_error_signal` 没进 snapshot 的 `non_tensor_keys` 白名单 → 选点器打一行
「no tag_error_signal key」然后**静默退化成轨迹级选点**。已修（`ray_trainer.py`）。

### 改造点 4：teacher 前缀分支（a_T）

`agent_system/multi_turn_rollout/` 三个文件分层：

| 文件 | 行数 | 职责 |
|---|---|---|
| `teacher_branch.py` | ~660 | 文本级编排，钩子注入，可脱离集群测试 |
| `teacher_prefix.py` | ~180 | token 簿记，两个 mask 的构造 |
| `branch_runner.py` | 776 | 唯一接触 verl 运行时对象（DataProto / worker group / env）的层 |

流程：

```
main rollout rows
  -> select_branch_specs        挑出 error-signal 步
  -> generate_teacher_prefixes  请 teacher 写思考开头 a_T
  -> run_branch_rollout_batched 环境重放到该步 + 学生从 a_T 续写
  -> DataProto rows             带 teacher_token_mask / loss_mask
```

**mask 契约**（这是整个改造点的核心，写错了训练会静默学错东西）：

- `a_T` 的 token 进 `teacher_token_mask` → 走 FKL/CE（模仿 teacher）
- `a_T` 的 token **必须从 `loss_mask` 中排除** → 不走 PG，因为它们不是学生采样出来的
- 两个 mask 严格互不重叠，并集 = 有效响应区间

分支行**继承父轨迹的 `uid`**（这样 GRPO 在同一个任务组内归一化）但有**自己新的
`traj_uid`**。

**注意**：`loss_mask` 只在 `meta_info["multi_turn"]` 为真时被尊重
（`ray_trainer.py` 的 `_use_loss_mask()`，消费点在 `dp_actor.py`）。

**状态**：代码完成，31 项单测全绿。**但集群上实际产出 0 个分支** ——
已定位到症状但根因未确认，见 [open-issues.md](open-issues.md)。

#### 选点条件框架（2026-07-30，「什么时候调用老师」可插拔）

`agent_system/multi_turn_rollout/branch_selectors.py`。选点不再硬编码
error-signal，而是一个 spec 字符串（`&` 连接为 AND），选出的点带 priority
（组内排名 + 全局 cap 都按它排）：

| spec | 含义 | priority |
|---|---|---|
| `error_signal` | 观测带异常信号（原 Phase 2a） | 0 |
| `low_reward` | 该轨迹 episode return ≤ 阈值（默认 0 = 失败轨迹） | 阈值差 |
| `kl_gap` | 学生与 teacher 的逐 token 归一化 KL 最大的步 | KL 值 |
| `any` | 全部活跃步 | 0 |

`kl_gap` 由 runner 在选点前调外接 teacher 打分算出（有 `rollout_log_probs`
时是真 KL 单样本估计 `mean(student_lp - teacher_lp)`，否则退化为交叉熵
`-mean(teacher_lp)`，`selector_kl_true_kl` 指标区分）；先用便宜条件预过滤、
`selector_kl_max_scored_rows`（默认 512）封顶打分行数。打分失败 → 该条件选不出
任何点 → 漏斗照常报告（strict 下直接抛），绝不静默乱选。

配置：`SEED_TEACHER_BRANCH_SELECTOR="low_reward&kl_gap"`（默认 null = 沿用
`require_error_signal`）。指标：`seed/teacher_branch/selector_kl_*`、
`selected_priority_mean`。

#### 前缀模式（2026-07-30，`SEED_TEACHER_BRANCH_PREFIX_MODE`）

- `think_prefix`（默认，原行为）：teacher 写一段**未闭合**的思考开头，学生在
  同一步内续写并选 action；
- `full_step`：teacher **写完整个步**（闭合的 `<think>` + 恰好一个
  `<search>/<answer>` action，质检门 `parse_teacher_full_step_response`），
  action 直接进 env 执行，**学生从下一步接管**。该步不调 policy，
  `_encode_teacher_step` 直接编码 teacher 文本 —— mask 契约是普通行的镜像：
  整段响应 `teacher_token_mask=1`（FKL）、`loss_mask=0`（不进 PG）。

### 创新点 1 基础设施：step 级 reward model

以「节点」（一个轨迹步）为单位的 RM，为 step-wise PPO 做准备。三件套：

| 组件 | 位置 | 说明 |
|---|---|---|
| 数据 | `seed/step_rm.py` + trainer hook | `SEED_STEP_RM_DUMP=True`（主实验 launcher 默认开）时每个训练步落一个 parquet 分片到 `$DEFAULT_LOCAL_DIR/step_rm_dataset/`，行 = (obs, response, step_reward, episode_return, return_to_go, episode_success, tag 列)，分支行也在内 —— 正常 RL 跑一遍就顺带把 RM 数据攒了 |
| 训练 | `scripts/step_rm/train_step_rm.py` | `AutoModelForSequenceClassification(num_labels=1)` 回归 `return_to_go`（或 `--loss bt` 组内 Bradley-Terry：成功轨迹的步 > 失败轨迹的步）；按 uid 切 train/val 防泄漏；报 MSE/Spearman/成功 AUC |
| 服务 | `scripts/step_rm/serve_step_rm.py` + `StepRewardModelClient` | FastAPI `/score`，客户端与外接 teacher 同款（合批/重试/多副本 failover），之后 step-wise PPO 直接调 |

**状态**：三件套 + 选点框架 + full_step 共 32 项新单测，全绿（总 178）。
数据攒够后（~几千步 × 每步几千行）即可第一次训 RM。

---

## 可观测性：exit 漏斗

改造点 4 原本有七条静默 `return None`，一步产出 0 个分支和一步正常工作在指标上
**完全无法区分**。加了三样东西：

**1. 漏斗，每步必报全部字段**（即使在最早的退出路径也输出 0）

```
seed/teacher_branch/num_candidates             候选点
seed/teacher_branch/num_specs_with_env_kwargs  拿到 env 载荷的
seed/teacher_branch/num_prefixes               teacher 给出可用 a_T 的
seed/teacher_branch/num_trajectories           真的 rollout 出来的
seed/teacher_branch/num_rows                   最终进 batch 的
```

「缺 key」和「值为 0」在看板上长得一样（都是断点），所以必须全部预置为 0。
这正是之前误判的原因：看到 `num_candidates=21` 就以为分支在跑，
而 **candidate 不等于 branch**。

**2. 七个 `exit_*` 一热标记**：`disabled` / `schedule` / `no_rows` / `no_env_kwargs` /
`no_prefix` / `no_tensors` / `ok`。除前两个（正常关闭）外每条都往 stderr 打带上下文的
warning。

**3. `strict` 模式**（`SEED_TEACHER_BRANCH_STRICT=True`）：非计划性的空结果直接抛
`BranchProducedNothing`，异常信息带完整漏斗数字。冲烟时开，正式 150 步跑时关 ——
不该让一个坏步骤杀掉整个任务。

被 schedule 关闭**不算失败**，strict 模式下也不报错，否则 `start_after_steps` 一配就没法跑。

---

## 怎么跑

集群上用 [`ops/`](../../ops/) 里的两个脚本，一条命令一个阶段：

```bash
# 本地 -> CFS 同步代码（堡垒机不支持 scp，走 base64 灌 pty）
bash ops/local/push_to_cluster.sh

# 每个纯 teacher 节点（8 卡 = 2 副本 TP=4）各起一次；20 卡 = 5 副本布局见 cluster-runbook
qf -c 'TEACHER_REPLICAS=2 TEACHER_GPUS=0,1,2,3,4,5,6,7 bash $SEED_ROOT/ops/cluster/seedctl.sh teacher'

# 混合节点（teacher 副本 0 + retriever）+ 训练节点体检 -> 冲烟一步（strict 开，漏斗直接打出来）
# 前提：.env 里 SEED_EXTERNAL_TEACHER_BASE_URL 已写全副本逗号列表
qf -c 'bash $SEED_ROOT/ops/cluster/seedctl.sh all'

# 漏斗健康就起正式 150 步（setsid nohup，脱离连接独立存活）
qf -c 'bash $SEED_ROOT/ops/cluster/seedctl.sh train'

# 断线重连后看进度：进程/步数/漏斗/耗时/打分吞吐/报错一次全给
qf -c 'bash $SEED_ROOT/ops/cluster/seedctl.sh status'
```

细节和设计理由见 [ops/README.md](../../ops/README.md)。

底层 launcher 也可以直接调：

```bash
cp examples/seed_trainer/env.cfs.example .env   # 按集群实际情况填
SEED_TEACHER_BRANCH_STRICT=True TRAIN_DATA_SIZE=8 GROUP_SIZE=4 TOTAL_TRAINING_STEPS=1 \
  bash examples/seed_trainer/run_search_tag_distill_qwen3_8b.sh
```

**必须 `setsid nohup`**：本地到集群的连接会周期性断开（见
[open-issues.md](open-issues.md) 的 VPN 黑洞一节），任务必须脱离连接独立存活。
`seedctl train` 已经替你做了这件事。


## 测试

全部可在笔记本上跑，不需要 GPU / ray / vllm —— tokenizer、rollout worker group、
环境都是 stub。

```bash
python3 -m pytest tests/agent_system/ tests/trainer/ppo/ -q
```

当前 **146 passed**（含外接 teacher 客户端的合批/多副本轮询/failover/统计 9 项）。
分层设计（文本级编排与 verl 运行时对象分离）就是为了让绝大部分逻辑能脱离集群验证。
