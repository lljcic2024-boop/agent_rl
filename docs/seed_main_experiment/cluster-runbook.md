# 集群运行手册：拓扑、存放位置、服务

千帆集群（CFS 共享盘）上跑 SEED 主实验的完整信息。**所有内网 IP / 主机名在本文中
用占位符**，实际值在本地 `.env`（不入库）。

## 拓扑

### 32 卡目标拓扑（2026-07-30 起，20 卡 teacher）

| 角色 | GPU | 说明 |
|---|---|---|
| 训练 | worker-0，8× A800 | student Qwen3-8B |
| retriever `:8000` | worker-1，GPU 4-7 | faiss GPU 分片，61 GB 索引不搬家 |
| teacher 副本 0 `:8100` | worker-1，GPU 0-3，TP=4 | `seedctl services` 顺带起 |
| teacher 副本 1-2 `:8100/:8101` | worker-2，GPU 0-7，2 副本 | `TEACHER_REPLICAS=2 TEACHER_GPUS=0,1,2,3,4,5,6,7 bash seedctl.sh teacher` |
| teacher 副本 3-4 `:8100/:8101` | worker-3，GPU 0-7，2 副本 | 同上 |

共 5 个 teacher 副本。训练节点 `.env` 把全部副本写成逗号列表：

```bash
SEED_EXTERNAL_TEACHER_BASE_URL="http://<w1>:8100/v1,http://<w2>:8100/v1,http://<w2>:8101/v1,http://<w3>:8100/v1,http://<w3>:8101/v1"
```

打分客户端把 batch 轮询到所有副本，某个副本挂掉时重试自动跳下一个（降速不断训）。
analyzer（写 a_T 的 chat 客户端）只用**列表第一个**；建议想清楚哪个副本最闲再用
`OPENAI_BASE_URL` 单独指过去 —— 打分是 prefill 型、写 a_T 是 decode 型，
分开互不拖慢。

### 旧 12 卡拓扑（保留参考）

| 角色 | GPU | 说明 |
|---|---|---|
| 训练 | worker-0，8× A800 全部 | student Qwen3-8B |
| teacher vLLM `:8100` | worker-1，`CUDA_VISIBLE_DEVICES=0,1,2,3`，TP=4 | analyzer + 打分共用一个 server |
| retriever `:8000` | worker-1，`CUDA_VISIBLE_DEVICES=4,5,6,7` | faiss GPU 分片 |

worker pod 同 namespace，**ping 通、TCP 通，直接用 pod IP，不用建 Service**。

---

## 存放位置

### 模型

| 内容 | 路径 | 大小 / 状态 |
|---|---|---|
| Student 基座 | `$MODELS_ROOT/Qwen3-8B` | 在用 |
| Student tag-SFT | `$MODELS_ROOT/Qwen3-8B-search-tag-sft` | **不存在**，launcher 打 WARNING 回落基座 |
| Teacher | `$MODELS_ROOT/Qwen3-30B-A3B-Instruct-2507` | 在用，MoE，激活 3B |
| 检索编码器 | `$DATA_ROOT/searchr1_data/models/e5-base-v2` | **必须本地路径** |

`e5-base-v2` 不能写成 HF 的 `intfloat/e5-base-v2` —— **pod 上不了 HuggingFace**，
必须预先下到 CFS 用本地路径引用。同理训练环境要设 `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`。

### 数据

| 内容 | 路径 | 规模 |
|---|---|---|
| 训练/验证集（已补列） | `$DATA_ROOT/searchr1_data/nq_search_seed/{train,test}.parquet` | 79168 / 3610 行 |
| faiss 索引 | `$DATA_ROOT/searchr1_data/e5_Flat.index` | 61 GB |
| 语料 | `$DATA_ROOT/searchr1_data/wiki-18.jsonl` | 14 GB，2100 万条 |
| 补列脚本 | `$DATA_ROOT/mk_envkwargs.py` | — |

**必须自己补 `env_kwargs` 列**：原始 Search-R1 parquet 没有这一列，而
`rollout_loop.py` 是

```python
obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
```

缺列就把 `None` 喂进 `envs.reset`，当场
`TypeError: object of type 'NoneType' has no len()`。列内容照抄
`examples/data_preprocess/preprocess_search_r1_dataset.py`：
`{"ground_truth":…, "question":…, "data_source":…}`。

### 产出

| 内容 | 路径 |
|---|---|
| checkpoint | `$DEFAULT_LOCAL_DIR`（CFS，**不要放 pod 本地盘**） |
| rollout dump / 分析结果 | `$DEFAULT_LOCAL_DIR` 下 |
| 训练日志 | `$LOG_ROOT/smoke.log`、`teacher.log`、`retriever.log` |
| ray/torch 临时文件 | `TMPDIR=$DATA_ROOT/tmp`（**pod 本地盘小，必须挪到 CFS**） |

wandb 用 `WANDB_MODE=offline`，pod 连不上外网。

---

## 服务启动

统一用 `setsid nohup bash <script> > <log> 2>&1 < /dev/null &`。

### teacher vLLM（用 `seedctl services` / `seedctl teacher` 起，参数已内置）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model      $MODELS_ROOT/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.72 \
  --max-model-len 16384 \
  --max-logprobs 40 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64
```

参数是踩出来的，不是抄的（2026-07-30 修正了旋钮归因）：

- **`--max-logprobs 40`**：外接打分要 `prompt_logprobs`，默认上限太小直接拒绝请求。
- **`--max-num-batched-tokens 4096`**：**这才是治 OOM 的旋钮**。vLLM 0.11（V1 引擎、
  chunked prefill）下 `prompt_logprobs` 的 logits 物化按 chunk 增量做，瞬时峰值 ≈
  chunk 预算 × 词表 15 万 × ~6B/位置 ≈ 8192 时 7.5GB、4096 时 4GB，集中在 TP rank0。
  **峰值与并发无关** —— 当初 0.85 util OOM → `EngineDeadError` → server 永久挂掉
  （之后所有请求 500，训练侧表现为 `Connection refused`，极易误判成网络问题）之后
  压并发到 2、max-num-seqs 到 16，是拧错了旋钮。
- **`--max-num-seqs 64`**：并发请求只多占 GQA 的 KV cache（4k token 序列约几十 MB），
  可以放开让调度器凑满 batch。
- **`--gpu-memory-utilization 0.72`** 先不动，多副本稳定跑通后可试 0.80。

客户端并发同步放开：`SEED_EXTERNAL_TEACHER_CONCURRENCY=16`（每副本 4-8 路在途合适），
合批 `SEED_EXTERNAL_TEACHER_BATCH_SIZE=16`。若报 prompt_logprobs 与 prefix caching
冲突，给 server 加 `--no-enable-prefix-caching`。

### retriever

```bash
python examples/search/retriever/retrieval_server.py \
  --index_path      $DATA_ROOT/searchr1_data/e5_Flat.index \
  --corpus_path     $DATA_ROOT/searchr1_data/wiki-18.jsonl \
  --retriever_model $DATA_ROOT/searchr1_data/models/e5-base-v2 \
  --faiss_gpu --port 8000
```

2100 万条语料，加载 1-2 分钟，别以为卡死了。

---

## 接口 payload（都踩过）

**retriever 是单数 `query`，而且值必须是 `str` 不能是 `list`**：

```json
{"query": "who wrote hamlet", "topk": 3, "return_scores": true}
```

传 `queries` → `missing field query`；传 `{"query": ["..."]}` →
`Input should be a valid string`。见 `retrieval_server.py` 的 `QueryRequest`
和客户端 `search/third_party/skyrl_gym/tools/search.py` 的 `payload`。

**teacher 两条路都实测 OK**：打分走 `/v1/completions` + `prompt_logprobs`
（传 token id 数组当 prompt），返回形如
`prompt_logprobs: [null, {"6722": {"logprob": ..., "rank": ...}}, ...]`
（**第 0 位恒为 null**）；analyzer 走 `/v1/chat/completions`。

---

## 训练环境变量（缺一个就炸）

```bash
PYTHONPATH=<SEED_ROOT>:<pylibs>
PATH=<pylibs>/bin:$PATH
WANDB_MODE=offline                          # pod 连不上 wandb
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # pod 上不了 HF
TMPDIR=$DATA_ROOT/tmp                       # pod 本地盘小
```

pod 里 pip 三个源全挂，装不了新包。预装 vllm 0.11.0 / transformers 4.57.1，
比 SEED 要求的新，实测大概率不用降级。

---

## `.env` 写法：必须用 `${VAR:-default}`

**每一项都不能裸赋值**。裸赋值会**反过来覆盖调用者已经导出的变量**：

```bash
# 错 —— 会把冲烟脚本设的 TOTAL_TRAINING_STEPS=1 覆盖成 150
TOTAL_TRAINING_STEPS=150

# 对
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
```

2026-07-29 实测踩到：冲烟脚本设的 `TOTAL_TRAINING_STEPS=1` /
`DEFAULT_LOCAL_DIR=<冲烟目录>` 被 `.env` 里的裸赋值吃掉，跑出一个
「batch 是冲烟规模、步数和输出目录是正式实验」的四不像，**冲烟白跑并污染了正式
ckpt 目录**。

模板见 [`examples/seed_trainer/env.cfs.example`](../../examples/seed_trainer/env.cfs.example)。

---

## 远程操作的坑

- **payload 里绝对不要 `pkill -f <pattern>`**：pattern 会匹配到承载它的 `bash -c`
  自己的命令行，当场自杀 —— 表现为 `exit code 143` + 零输出。
  用 `for p in $(pgrep -f main_ppo); do kill $p; done`。
- **hydra 报错埋在几百行 config dump 之后**，用
  `grep -nE "ModuleNotFoundError|Traceback|RayTaskError"` 抓，别 tail。
- 一次远程往返 60-90 秒，**尽量把多条命令合成一次调用** —— 往返次数越多越容易撞上断连。
- 复杂的内联 payload（嵌套引号多）有时**静默返回 0 字节**，紧接着一个 `echo` 又正常。
  遇到就把逻辑写成脚本上传，改用 `bash <script>` 调。
- 日志里的 ANSI 色码会干扰 grep，先 `sed -e 's/\x1b\[[0-9;]*m//g'`。
