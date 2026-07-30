# 未解决的问题与实测数据

按优先级排。每条都写清楚**证据是什么**、**已排除了什么**、**下一步查什么** ——
避免重复走已经走过的错路。

---

## 1. 改造点 4 产出 0 个分支（正确性问题，最高优先级）

**症状**：集群上 `timing_s/teacher_branch` 只有 **0.001 秒**，而
`seed/teacher_branch/num_candidates` 报告 **21-28 个候选点**。

**为什么这个比性能问题严重**：150 步正式训练可以在**指标看起来健康**的情况下
跑完，而 FKL 项全程是空的 —— 改造点 4 等于没上。

**已排除**：

- 不是 `enable` 没开：`_is_teacher_branch_enabled()` 要求
  `opd_fkl_loss_coef > 0`，launcher 设的是 0.05，满足。
- 不是 schedule 关掉：`start_after_steps` / `stop_after_steps` 都是 null。
- **env_kwargs 传递链追过一遍，看起来是完好的**：`repeat()` 返回新对象，
  `pop` 只改副本，trainer 在 pop 之前留了引用。

**矛盾点（未解决）**：把七条退出路径逐条对照「耗时 + 已发出的指标」，
只有 `spec.env_kwargs is not None` 那条过滤同时满足「~0 秒」和
「已报出 num_candidates」。但上面第三条说这条链是完好的。**两份证据互相矛盾，
根因尚未钉死。**

**为此做的事**：加了 exit 漏斗和 `strict` 模式（见 README）。
下一步只需要在集群上跑一步、看 `num_specs_with_env_kwargs` 是 0 还是 21 —— 
这个数字直接把两种可能分开：

| 观测 | 结论 | 修法方向 |
|---|---|---|
| `num_candidates=21, num_specs_with_env_kwargs=0` | env 载荷解析失败 | 查 `sample_id` 是否越界 / trainer 是否真传进来了 |
| `num_specs_with_env_kwargs=21, num_prefixes=0` | teacher 拒绝给 a_T | 查质检门是否过严 / teacher server 状态 |

**教训（写下来防止再犯）**：candidate 数不等于 branch 数。上一轮就是看到
`num_candidates=21` 就宣布改造点 4 已上线，这是错的。

---

## 2. teacher 打分占一步耗时 54%（性能问题）

**实测分解**（一步 82 秒）：

| 阶段 | 秒 | 占比 |
|---|---|---|
| `seed_teacher`（打分） | 44.5 | 54% |
| `gen`（学生 rollout） | 20.2 | 25% |
| `update_actor` | 12.9 | 16% |
| `ref` / `old_log_prob` | 2.2 / 2.1 | 5% |
| `reward` / `adv` / `teacher_branch` | ~0.07 | ~0 |

各项加起来 80.4 秒 vs 一步 82 秒 —— **没有隐藏开销**，瓶颈就是打分。

### 44 秒花在哪：既不是网络，也不是 rollout

```
一步 149 个请求（seed/teacher_batch_size）
客户端并发 = 2
-> 149 ÷ 2 ≈ 75 轮串行
-> 44 秒 ÷ 75 ≈ 每请求 0.59 秒
```

- **不是网络**：两台机器同机房同 namespace，RTT 十几毫秒，占 0.59 秒的 ~3%。
- **不是 rollout**：`max_tokens=1`，teacher 只吐一个 token 且被丢掉（见 README
  改造点 2）。
- **是 prefill 计算**，但**根因是「喂得太少」而非「算得慢」**：并发 2 意味着
  4 张 A800 任何时刻手上只有 2 个请求，绝大部分时间在空转等数据。0.59 秒是
  **单请求延迟**，不是吞吐上限。

**理论下限估算**：149 条 × ~3000 token ≈ 45 万 token 前向，30B-A3B 只激活 3B
参数，4 张 A800 上大概 **8-10 秒**。现在 44 秒，约 4 倍水分，几乎全在并发这一个数字上。

**A800 显存不是约束**：训练卡占 41651-43085 MiB / 81920 MiB，每卡还剩约 39 GB。
压力全在 teacher server 侧。

### 2026-07-30 更新：机制修正 + 改法 (2)(3) 已实现

**机制修正（推翻了当初压并发的依据）**：vLLM 0.11 只有 V1 引擎，chunked prefill
默认开。`prompt_logprobs` 的 logits 物化是**按 chunk 增量做**的，瞬时显存峰值 ≈
`max-num-batched-tokens × 词表(15万) × ~6B/位置`（集中在 TP rank0），
**只由 chunk 预算决定，与在跑的请求数无关**。并发多了只多 GQA 的 KV cache（很小）。
所以当初 OOM 后压并发到 2、max-num-seqs 到 16 是拧错了旋钮 —— 该压的是
`--max-num-batched-tokens`（8192→4096，峰值 7.5GB→4GB）。

**已实现**（本地 146 项单测全绿，待集群实测）：

- 客户端合批：`seed_external_teacher.py` 每个 HTTP 请求打包
  `batch_size`（默认 16）条 token-id prompt，vLLM 服务端连续批处理。
- 多副本轮询：`base_url` 接受逗号分隔的副本列表，batch 轮询分发，
  重试自动跳到下一个副本（单副本挂掉降速不断训）。
- 并发默认 2 → 16（`SEED_EXTERNAL_TEACHER_CONCURRENCY`），
  服务端 `--max-num-batched-tokens 4096`、`--max-num-seqs 64`
  （`seedctl services` / `seedctl teacher` 已带上）。
- 每次打分报 `seed/external_teacher/<label>/{elapsed_s, prefill_tokens_per_s,
  rows_scored, num_batches, retries, endpoint_failures}`，`seedctl status` 直接打印。
- 打分异步失败不再静默：`seed/teacher_signal_failed=1` + ERROR 级完整 traceback。

**20 卡 teacher 部署**（5 副本 × TP=4）见 cluster-runbook「多副本 teacher」一节。

### 三条对症的改法（原始分析，保留供参考），按性价比排

**（1）别再要用不到的东西 —— 最该改**

`prompt_logprobs=0` 让 teacher 对**整段 prompt 每个位置**都算并返回，但代码只用尾巴：

```python
tail_entries = prompt_logprobs[-num_response_tokens:]
```

prompt 最长 4096，真正要的响应最长 512。**要了 4096 个位置，用了 512 个，
其余 87% 算完就扔。** 单请求约 2.5 GB 的中间结果里，有 2.2 GB 是为丢掉的部分付的
（vocab 15 万 × 4096 token × 4 字节）。

**（2）把 149 个请求合批**

现在**一行一个 HTTP 请求**。vLLM 的 `/completions` 接受一次传多个 prompt，
服务端本来就是为连续批处理设计的。合批后 149 次往返塌缩成几次，且让 GPU 真正吃饱。

**（3）并发从 2 往上抬 —— 最便宜，先做**

当初 OOM 是在 `gpu-memory-utilization=0.85`，现在已降到 0.72（60 GB / 80 GB），
**有余量**。`--max-num-seqs 16` 也是当初保命压低的。这两个数字各调一格、实测不 OOM，
是零代码改动的收益。

**顺序很重要**：先做 (3) 试探边界，再做 (1)(2)。因为 (1)(2) 会显著降低单请求内存占用，
做完之后 (3) 能开得更大 —— 反过来做等于用旧的内存约束去定新的并发上限。

### 为什么异步重叠救不了

打分提交点到 join 点之间只有约 4.3 秒的 GPU 工作，而打分要 48 秒。
重叠最多省 4.3 秒。

### 关于「加节点全部部署 teacher」

方向对，但**现在加没用**：并发 2 的时候现有 4 张卡都没吃满，多出来的卡会一样闲着。
顺序应该是**先抬并发确认单节点吃满 → 再横向起第二个 teacher 副本 + 客户端轮询**。

**不建议把单个 teacher 摊到 TP=8**：
- TP 切的是**权重**，不是 `prompt_logprobs` 物化的 logits —— 切不动真正 OOM 的那块。
- TP=8 要挤掉 retriever（占 worker-1 的另外 4 张卡），而 retriever 扛着 61 GB 索引，
  搬家成本远高于收益。

### ZeRO-3 用不上

ZeRO 省的是优化器状态、梯度、参数副本，都是**训练**时的东西。teacher 是**推理**
服务，没有优化器、没有梯度、不做反向传播 —— 要省的三样一样都不存在。
而且 vLLM 不基于 DeepSpeed，没这个开关。这不是效果不好，是工具和场景不匹配。

---

## 3. 本地到集群的连接周期性断开（基础设施，绕不修）

**结论先说：心跳解决不了，唯一有效的对策是让集群侧任务不依赖这条连接。**

**证据**（10 秒一采的取证监控，抓到完整断连过程）：

```
时间       ssh  ping  隧道  VPN进程
23:53:03    1    通    up   60779     正常
23:53:13    1   不通   up   60779     ping 先断，ssh 还活着
23:53:30    0   不通   up   60779     然后 ssh 才死
23:53:47    0    通    up   60779     网络自己回来，ssh 回不来了
```

**关键：`ping` 不通比 ssh 死早 17 秒**，而隧道接口全程 up、VPN 进程 pid 全程没变。
所以不是 VPN 崩溃重连，也不是接口掉了，而是**隧道还在但包过不去** ——
一次 20-30 秒的丢包黑洞。

**推论（会误导排查方向，所以写下来）**：

- **加密的心跳完全无效**。「一直有通信就不会断」这个前提不成立 ——
  黑洞期间根本发不出包。心跳只能让「多久发现死了」变快，不能让连接活下来。
  实测把心跳从 25 秒调到 10 秒，照样在第 10 分钟死。
- 因此**不要再往 keepalive 间隔 / `ServerAliveInterval` 上投入时间**。
- 唯一有效的对策：**训练必须 `setsid nohup` 起在远端、日志落盘**，
  断开后重连只是重新去读日志，而不是重头再跑。

**已排除**：Mac 睡眠、WiFi 掉线、当前网络不可达、VPN 进程崩溃重连。

**附带修掉的一个真 bug**：本地 keepalive 守护脚本在断线后自己崩了，导致**再也没人
重挂心跳**。根因是 macOS 自带 bash 3.2 在 UTF-8 locale 下解析变量名不认多字节边界，
`"$rc）"` 里全角括号的字节被吃进变量名，`set -u` 下当场 `unbound variable`：

```
LC_ALL=C            -> 正常
LC_ALL=en_US.UTF-8  -> bash: rc?: unbound variable
```

改成 `"${rc}"` 后修复。这解释了「一断就彻底断」而不是几秒后自动恢复。

**另一个认知修正**：`ssh -O check` 只查本地 unix socket，**不产生任何跨网流量**，
所以黑洞过后它照样报「Master running」—— 连接早就是空壳了。
这就是「以前好几个小时都不断」的错觉来源。

---

## 4. 已知的其他坑

### verl object-dtype 陷阱

`collate_fn` 把所有 non-tensor 列存成 `dtype=object`，所以对它做 reduction
返回的标量类型**取决于行里存的是什么**：主 rollout 行给 `np.float32`（有 `.item()`），
而**分支行给普通 Python float（没有 `.item()`）**。

```python
# 炸
value = batch.non_tensor_batch[key][idx].mean().item()

# 对
values = np.asarray(batch.non_tensor_batch[key][idx], dtype=np.float32)
value = float(values.mean())
```

已在 `metric_utils.py` 修（抽出 `_episode_stat` helper）。
**这个坑只在改造点 4 真的产出分支行之后才会触发** —— 所以它和问题 1 是连着的。

### DataProto.concat 的 key 集合取自第一个 proto

`verl/protocol.py` 的 `list_of_dict_to_dict_of_list` 拿
`keys = list_of_dict[0].keys()` 然后断言后面每个 dict 都有这些 key。
**第一个 proto 必须声明所有列。** 这就是 `attach_default_masks()` 要给主 rollout 行
补 `is_teacher_branch` / `loss_mask` / `teacher_token_mask` 的原因 ——
即使主行根本不需要这些语义。

### 上一次训练已经死了，死因未查

`pgrep main_ppo` 空，`smoke.log` 停在 1823 行，尾部是 Ray 的 teardown
（`OSError: [Errno 16] Device or resource busy: '.nfs...'`，这是 NFS 上删临时目录的
常见噪音，**不一定是真死因**）。真正的失败原因要往上翻日志。

---

## 待办清单

| 优先级 | 事项 | 需要集群 |
|---|---|---|
| 高 | 开 `strict=True` 跑一步，读漏斗定位问题 1 根因 | 是 |
| 高 | 查上次训练的真实死因 | 是 |
| 高 | 按新参数重启 teacher（多副本），实测新的 `seed_teacher` 秒数与 `prefill_tokens_per_s` | 是 |
| 高 | 正式规模（128×8）跑一步拿完整 timing 分解，校准总时长估算 | 是 |
| ~~中~~ | ~~抬并发~~ ~~teacher 请求合批~~ → 2026-07-30 已实现（见上），待集群验证 | 是 |
| 中 | 缩 `prompt_logprobs` 作用范围（省 87% 浪费；先试缩短 teacher prompt） | 验证需要 |
| 中 | 跨步滞后 OPD：teacher 冻结 + token 已定 → step t 的打分可与 t 的 update、t+1 的 gen 全重叠 | 否（改完再验） |
| 低 | ~~改造点 3 Phase 2b critic-gap 选点~~ → 2026-07-30 已由选点条件框架覆盖（`low_reward`/`kl_gap`/组合，见 README 改造点 4 一节）；critic-gap 变体等 step RM 训出来后作为新条件加入 | 否 |
| 低 | 攒够 step_rm_dataset 后第一次训 step RM（`scripts/step_rm/train_step_rm.py`），验证成功 AUC | 是 |
| 低 | 产出真的 `Qwen3-8B-search-tag-sft` student | 是 |
