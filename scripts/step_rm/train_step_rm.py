# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Train a step-level reward model on parquet shards dumped by StepDatasetDumper.

以「节点」(一个轨迹步) 为单位: 输入 = 该步观测 + 该步完整响应, 输出 = 标量分数。
默认回归 return_to_go (该步起的折扣回报); `--loss bt` 切换成组内 Bradley-Terry
配对损失 (同一 uid 组里 成功轨迹的步 > 失败轨迹的步)。

The checkpoint is a standard `AutoModelForSequenceClassification(num_labels=1)`
save, served by `serve_step_rm.py` and consumed later as the step-wise PPO
reward/critic signal.

Example (cluster, one node):

    python scripts/step_rm/train_step_rm.py \
        --data-glob "$CKPT_DIR/step_rm_dataset/step_rows_*.parquet" \
        --model-path $MODELS_ROOT/Qwen3-1.7B \
        --output-dir $CKPT_DIR/step_rm \
        --target return_to_go --epochs 1 --batch-size 8

Plain torch training loop on purpose: no accelerate/Trainer dependency, easy
to read, runs on one GPU (or CPU for smoke tests).
"""

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from seed.step_rm import build_step_rm_text  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob", required=True, help="glob of step_rows_*.parquet shards")
    parser.add_argument("--model-path", required=True, help="HF backbone (e.g. Qwen3-1.7B)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target",
        default="return_to_go",
        choices=["return_to_go", "episode_return", "step_reward", "episode_success"],
        help="regression target column (ignored by --loss bt, which uses episode_success)",
    )
    parser.add_argument("--loss", default="mse", choices=["mse", "bt"], help="mse regression or Bradley-Terry pairs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1, help="validation fraction, split by uid (no leakage)")
    parser.add_argument("--group-normalize", action="store_true", help="z-normalize targets within each uid group")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda / cpu; default auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None, help="cap rows (smoke tests)")
    return parser.parse_args()


def load_records(data_glob: str, max_rows=None) -> List[Dict]:
    import pandas as pd

    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"no parquet shards match {data_glob!r}")
    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        df = df.iloc[: int(max_rows)]
    records = df.to_dict("records")
    print(f"loaded {len(records)} step rows from {len(paths)} shard(s)")
    return records


def split_by_uid(records: List[Dict], val_frac: float, seed: int):
    """Split train/val by uid (task group) so no task leaks across the split."""
    uids = sorted({str(r["uid"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(uids)
    n_val = max(1, int(len(uids) * val_frac)) if val_frac > 0 else 0
    val_uids = set(uids[:n_val])
    train = [r for r in records if str(r["uid"]) not in val_uids]
    val = [r for r in records if str(r["uid"]) in val_uids]
    return train, val


def group_normalize(records: List[Dict], target: str) -> None:
    """z-normalize `target` within each uid group (GRPO-style), in place."""
    groups: Dict[str, List[float]] = {}
    for r in records:
        groups.setdefault(str(r["uid"]), []).append(float(r[target]))
    stats = {}
    for uid, values in groups.items():
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        stats[uid] = (mean, math.sqrt(var) + 1e-6)
    for r in records:
        mean, std = stats[str(r["uid"])]
        r[target] = (float(r[target]) - mean) / std


class StepRMDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, target: str, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.target = target
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        text = build_step_rm_text(record["obs_text"], record["response_text"])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "target": torch.tensor(float(record[self.target]), dtype=torch.float32),
            "uid": str(record["uid"]),
            "episode_success": int(record.get("episode_success", 0)),
        }


def collate(batch, pad_token_id: int):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = item["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target": torch.stack([item["target"] for item in batch]),
        "uid": [item["uid"] for item in batch],
        "episode_success": torch.tensor([item["episode_success"] for item in batch]),
    }


def bt_pairs_loss(scores: torch.Tensor, uids: List[str], success: torch.Tensor) -> torch.Tensor:
    """In-batch Bradley-Terry: within a uid group, success steps beat failures."""
    losses = []
    for i in range(len(uids)):
        for j in range(len(uids)):
            if uids[i] == uids[j] and success[i] > success[j]:
                losses.append(torch.nn.functional.softplus(scores[j] - scores[i]))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    preds, targets, successes = [], [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits.squeeze(-1)
        preds.extend(logits.float().cpu().tolist())
        targets.extend(batch["target"].tolist())
        successes.extend(batch["episode_success"].tolist())
    model.train()
    n = len(preds)
    if n == 0:
        return {"val_mse": float("nan"), "val_spearman": float("nan"), "val_success_auc": float("nan")}
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / n

    def _ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank, idx in enumerate(order):
            ranks[idx] = float(rank)
        return ranks
    rp, rt = _ranks(preds), _ranks(targets)
    mean_rp, mean_rt = sum(rp) / n, sum(rt) / n
    cov = sum((a - mean_rp) * (b - mean_rt) for a, b in zip(rp, rt))
    var_p = math.sqrt(sum((a - mean_rp) ** 2 for a in rp))
    var_t = math.sqrt(sum((b - mean_rt) ** 2 for b in rt))
    spearman = cov / (var_p * var_t) if var_p > 0 and var_t > 0 else float("nan")

    pos = [p for p, s in zip(preds, successes) if s == 1]
    neg = [p for p, s in zip(preds, successes) if s == 0]
    if pos and neg:
        wins = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a in pos for b in neg)
        auc = wins / (len(pos) * len(neg))
    else:
        auc = float("nan")
    return {"val_mse": mse, "val_spearman": spearman, "val_success_auc": auc}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    records = load_records(args.data_glob, max_rows=args.max_rows)
    target = "episode_success" if args.loss == "bt" else args.target
    train_records, val_records = split_by_uid(records, args.val_frac, args.seed)
    if args.group_normalize and args.loss == "mse":
        group_normalize(train_records, target)
        group_normalize(val_records, target)
    print(f"train={len(train_records)} val={len(val_records)} target={target} loss={args.loss}")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        StepRMDataset(train_records, tokenizer, target, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    val_loader = DataLoader(
        StepRMDataset(val_records, tokenizer, target, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.squeeze(-1).float()
            if args.loss == "bt":
                loss = bt_pairs_loss(logits, batch["uid"], batch["episode_success"])
            else:
                loss = torch.nn.functional.mse_loss(logits, batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.6f}")
                history.append({"step": step, "loss": float(loss.item())})
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch} done: {metrics}")
        history.append({"epoch": epoch, **metrics})

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history}, f, ensure_ascii=False, indent=2)
    print(f"saved step RM to {args.output_dir}")


if __name__ == "__main__":
    main()
