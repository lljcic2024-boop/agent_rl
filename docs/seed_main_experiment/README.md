# SEED 功能类型化蒸馏：主实验文档

Student Qwen3-8B 在 Search QA 上向外接 Qwen3-30B-A3B teacher 学「工具调用的思考过程」。
基于 SEED（内嵌 verl 0.3.1.dev）的多轮 agent RL，`algorithm.adv_estimator=seed`。

文档分三份：

| 文档 | 内容 |
|---|---|
| 本文 | 四个改造点是什么、代码在哪、当前状态、怎么跑 |
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

**状态**：跑通并实测有输出。**性能是主要瓶颈**，见 [open-issues.md](open-issues.md)。

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

# 集群侧：体检 -> 起服务 -> 冲烟一步（strict 开，漏斗直接打出来）
qf -c 'bash $SEED_ROOT/ops/cluster/seedctl.sh all'

# 漏斗健康就起正式 150 步（setsid nohup，脱离连接独立存活）
qf -c 'bash $SEED_ROOT/ops/cluster/seedctl.sh train'

# 断线重连后看进度：进程/步数/漏斗/耗时/报错一次全给
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
python3 -m pytest tests/agent_system/test_branch_runner.py \
                  tests/agent_system/test_teacher_branch.py \
                  tests/trainer/ppo/test_teacher_branch_gating.py \
                  tests/trainer/ppo/test_metric_utils.py -q
```

当前 **70 passed**。分层设计（文本级编排与 verl 运行时对象分离）就是为了让绝大部分
逻辑能脱离集群验证。
