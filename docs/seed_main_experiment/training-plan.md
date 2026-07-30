# 训练思路与实验计划(主实验总纲)

> 配套脚本:[`scripts/experiments/run_stage.sh`](../../scripts/experiments/run_stage.sh) —— 每个实验阶段一条命令。
> 代码与集群细节见 [README.md](README.md) / [cluster-runbook.md](cluster-runbook.md) / [open-issues.md](open-issues.md)。

---

## 一、要回答的科学问题

学生模型(Qwen3-8B)在多轮工具调用任务(Search QA)上做 RL 时,只从**自己的采样**里学,
存在两个结构性缺陷:

1. **稀疏结果奖励**:一条 4-5 步的轨迹只有终点一个 reward,中间"思考走歪了"没有信号;
2. **探索天花板**:学生卡住的地方,重采样 8 次往往还是同一类错误,组内归一化后梯度≈0。

本实验用一个**冻结的 30B teacher**(Qwen3-30B-A3B)从两个方向补:

- **OPD(dense 过程信号)**:teacher 对学生已写出的每个 token 打分,
  "学生认为好但 teacher 认为差"的 token 收到逐 token 的负信号 —— 把稀疏 reward 变密;
- **teacher 前缀分支(定向探索)**:在学生卡住的那一步,teacher 给一个思路开头(或直接完成
  这一步),学生从那里继续走 —— 相当于把探索预算集中投到失败点上。

**创新点归位**:
- 创新点 2(思考链路 + OPD dense reward)= 改造点 1(tag 菜单)+ 2(外接打分)+ 3(选点);
- 创新点 1(节点级 reward model + step-wise PPO)= 本轮先攒数据(step_rm_dataset),
  RM 训好后作为下一轮的 step 级信号与选点条件。

---

## 二、算法:一步训练在优化什么

### 2.1 采样与切块

每个训练步:128 道题 × 每题 G=8 条轨迹。轨迹按「一次 action + 一次 feedback」切成步,
每步一行训练样本(step_id = sample_rollout_step)。分支行(见 2.4)追加进同一 batch,
**继承父轨迹的 uid**,参与同一组归一化。

### 2.2 基础优势:GRPO(critic-free)

对同一道题的 G 条轨迹,取 episode 终点回报 R_i,组内归一化:

```
A_i = (R_i - mean(R_1..G)) / (std(R_1..G) + ε)        # SEED_MODE=mean_std_norm
```

A_i 广播到该轨迹所有行的所有 response token,走 PPO-clip 更新,
加 ref policy 的 KL 正则(coef 0.001, low_var_kl)。没有 critic。

### 2.3 OPD 项:dense 的 teacher 过程信号(改造点 2+3)

对被选中的步(改造点 3:error-signal 步;之后可换 RM-gap),teacher 在**增广 prompt**下
对学生响应逐 token 打分,得 log π_T(y_t|·)。损失侧(rkl 模式,带 clip):

```
adv_opd(t) = clip(log π_T(y_t) - log π_θ(y_t), ±5.0)     # 逐 token,SEED_OPD_RKL_ADV_CLIP
L_OPD      = -λ_opd · Σ_t adv_opd(t) · log π_θ(y_t)      # λ_opd = SEED_OPD_LOSS_COEF = 0.01
```

直觉:teacher 觉得比学生自信的 token 被拉高,反之压低 —— 单样本反向 KL 蒸馏,
但只在"值得学"的步上做。

### 2.4 分支项:a_T 前缀 + FKL(改造点 4)

选点条件(可插拔,`SEED_TEACHER_BRANCH_SELECTOR`)挑出分支点,teacher 生成 a_T:

- `think_prefix` 模式:a_T 是未闭合的思考开头,学生续写该步并走完剩余步;
- `full_step` 模式:a_T 是完整一步(思考+action),action 直接执行,学生从下一步接管。

mask 契约(写错会静默学错,已有 31 项单测锁死):

```
a_T token:   teacher_token_mask=1 → FKL/CE(模仿),λ_fkl = SEED_OPD_FKL_LOSS_COEF = 0.05
             loss_mask=0          → 不进 PG(不是学生采样的)
学生续写:    loss_mask=1          → 正常 PG(自己的选择自己承担环境反馈)
```

这是与普通蒸馏的本质区别:**只借一个思路开头,路仍由学生自己走**。

### 2.5 总损失

```
L = L_PPO-clip(GRPO 优势) + λ_opd·L_OPD + λ_fkl·L_FKL(a_T) + 0.001·KL(π_θ‖π_ref)
```

### 2.6 顺带产出:step RM 数据(创新点 1 地基)

每个训练步把全部行(含分支行)落成 parquet:
(obs, response, step_reward, episode_return, return_to_go, episode_success, tag 计数)。
150 步 ≈ 70 万行。RM 训练目标:回归 return_to_go 或组内 Bradley-Terry
(成功轨迹的步 > 同题失败轨迹的步)。训好后:
① 作为 step-wise PPO 的过程 reward;② 作为新的选点条件(RM 预测低的步 = 该分支的点)。

---

## 三、实验矩阵:一次只动一个变量

| 阶段 | 名字 | 相对上一阶段动了什么 | 回答什么 |
|---|---|---|---|
| **E0** | `e0_grpo_baseline` | 纯 GRPO(OPD/分支全关),只留 tag 菜单与 RM 数据落盘 | 基线曲线;tag 行为统计 |
| **E1** | `e1_opd` | + OPD(error-signal 选点,rkl+gate) | dense 过程信号有没有用 |
| **E2** | `e2_opd_branch` | + teacher 分支(think_prefix,error-signal,每轨迹≤1) | 定向探索有没有用 |
| **E3a** | `e3a_selector_klgap` | 选点换 `low_reward&kl_gap` | 更聪明的选点是否更好 |
| **E3b** | `e3b_fullstep` | 前缀模式换 `full_step` | 借"完整一步"vs 借"思路开头" |
| (下轮) | step-wise PPO | E0-E2 数据训 RM,接 step 级优势 | 创新点 1 |

**执行顺序**:E0 → E1 → E2 串行(每个都以上一个的结论为前提);E3a/E3b 是 E2 的平行变体,
按 E2 结果选一个先跑。每阶段 150 步,~15-25 h(视 teacher 副本数),
估算 **约 320-400 A800·时/阶段**(32 卡)。

**评估**:`TEST_FREQ` 触发的验证集(nq_search test 3610 题)success rate 为主判据;
过程指标见第五节。

### 决策门(避免无脑跑完全部矩阵)

- E1 vs E0:验证 success rate **+2pp 以上** → 继续 E2;打平 → 先查 OPD 项统计
  (gate 触发率、adv_opd 分布)再决定调 λ_opd 还是换选点;
- E2 vs E1:分支行的 done_ratio / episode_reward 显著高于父轨迹失败率 → E3;
  分支没收益 → 检查 a_T 质量(人工抽 20 条)与 λ_fkl;
- 任何阶段训练崩(entropy 塌缩、KL 爆炸)→ 停,查 `seed/` 指标组,不要带病跑完。

---

## 四、每阶段的操作流程(固定不变)

```bash
# 0. 本地:代码同步上 CFS
bash ops/local/push_to_cluster.sh

# 1. 服务(worker-2/3 各 2 副本,worker-1 副本0+retriever;URL 填 .env)
TEACHER_REPLICAS=2 TEACHER_GPUS=0,1,2,3,4,5,6,7 bash $SEED_ROOT/ops/cluster/seedctl.sh teacher
bash $SEED_ROOT/ops/cluster/seedctl.sh services

# 2. 冲烟(strict 开,一步,验证四件事:漏斗/打分耗时/RM落盘/无报错)
bash $SEED_ROOT/scripts/experiments/run_stage.sh e1_opd smoke

# 3. 正式规模单步校准 timing(第一次跑该阶段时)
bash $SEED_ROOT/scripts/experiments/run_stage.sh e1_opd calibrate

# 4. 正式 150 步(setsid nohup,断线不影响)
bash $SEED_ROOT/scripts/experiments/run_stage.sh e1_opd train

# 5. 随时看进度
bash $SEED_ROOT/ops/cluster/seedctl.sh status
```

每个阶段的 checkpoint / RM 数据 / 日志分目录存放
(`$MODELS_ROOT/ckpt/<阶段名>`),互不覆盖,便于事后对比曲线。

---

## 五、要盯的指标(wandb offline / seedctl status)

| 组 | 指标 | 健康形态 |
|---|---|---|
| 结果 | val success rate;`episode_rewards` 均值 | E1≥E0,E2≥E1 |
| OPD | `seed/teacher_log_prob_mean`;gate 触发率;adv_opd 分布 | 不全零、不发散 |
| 分支漏斗 | `num_candidates→…→num_rows`;`exit_*` | num_rows>0,exit_ok=1 |
| 分支质量 | `done_ratio`、`mean_episode_reward`(分支 vs 全体) | 分支高于失败父轨迹 |
| 选点 | `selector_kl_*`、`selected_priority_mean` | kl_gap 用时 scored_rows>0 |
| 行为 | `tag_*_count`(plan/verify/reflect/backtrack) | reflect/backtrack 在错误后上升 |
| 稳定 | actor entropy、KL(θ‖ref)、grad norm | 无塌缩/爆炸 |
| 性能 | `timing_s/*`、`seed/external_teacher/*/prefill_tokens_per_s` | seed_teacher 占比 <20% |
| 数据 | `seed/step_rm/rows_dumped` | ≈ 每步行数,dump_failed=0 |
| 兜底 | `seed/teacher_signal_failed` | 恒 0 |

---

## 六、已知风险与对策

| 风险 | 征兆 | 对策 |
|---|---|---|
| 改造点 4 集群上产 0 分支(未解决,最高优先) | 漏斗 num_rows=0 | strict 冲烟按漏斗二分根因(open-issues #1) |
| OPD 温度失配 | 日志 WARNING temperature≠1 | 学生采样温度按 launcher 默认;偏了先记账不阻塞 |
| a_T 质量差 | prefix_failure_ratio 高 / 分支 reward 低 | 抽样人工看;调质检门;必要时换 full_step |
| teacher 副本挂 | `endpoint_failures`>0 | failover 已兜住;顺手重启副本 |
| tag-SFT checkpoint 缺失 | launcher WARNING 回落 base | 接受(E0-E2 用 base 一致即可比较);tag-SFT 列 backlog |
| 熵塌缩 | entropy 持续下坠 | 参考 RLVR 熵塌缩文献(2505.22617);先降 λ_opd |
