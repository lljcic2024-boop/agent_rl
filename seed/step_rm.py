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
Step-level reward model (step RM) data pipeline + scoring client.

创新点 1 的基础设施: 以「节点」(一次 action + 一次 feedback = 一个轨迹步) 为单位
训练 reward model, 之后作为 step-wise PPO 的 critic/reward 信号。

Three pieces, deliberately decoupled:

1. `extract_step_rows(rows, ...)` — pure python. Turns the per-step row dicts
   the rollout loop already produces into flat training records with episode
   aggregates (episode_return / return_to_go / episode_success). Unit-testable
   without torch.
2. `StepDatasetDumper` — appends one parquet shard per training step under
   `<dump_dir>/step_rows_<global_step>.parquet`. The trainer calls it right
   after step rewards are known, so a normal RL run doubles as RM data
   collection (no separate rollout pass needed).
3. `StepRewardModelClient` — thin HTTP client for the FastAPI server in
   `scripts/step_rm/serve_step_rm.py`, mirroring the external-teacher client
   pattern (batching, retries, comma-separated replica list).

Training itself lives in `scripts/step_rm/train_step_rm.py`.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Prompt template used consistently at RM training and RM inference time.
STEP_RM_TEXT_TEMPLATE = "{observation}\n\n### Agent step\n{response}"

_TAG_COUNT_KEYS = (
    "tag_plan_count",
    "tag_verify_count",
    "tag_reflect_count",
    "tag_backtrack_count",
)


def build_step_rm_text(observation: str, response: str) -> str:
    """The exact text a step RM scores; keep train/serve in lockstep."""
    return STEP_RM_TEXT_TEMPLATE.format(observation=str(observation), response=str(response))


def _row_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def extract_step_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    gamma: float = 1.0,
    success_threshold: float = 0.5,
    global_step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Turn rollout step rows into flat step-RM training records.

    Args:
        rows: per-step dicts with at least traj_uid, uid, step_num, rewards,
            obs_text, response_text (rows without a decoded response_text are
            skipped — decode before calling, see `rows_from_dataproto`).
        gamma: discount for return_to_go (1.0 = undiscounted).
        success_threshold: episode_return > threshold -> episode_success=1.
        global_step: stamped on every record when given.

    Returns:
        One record per active step, each carrying its episode aggregates:
        episode_return (whole trajectory), return_to_go (this step onward,
        discounted), episode_success, episode_length.
    """
    by_traj: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("active_masks", True)):
            continue
        by_traj.setdefault(str(row.get("traj_uid")), []).append(row)

    records: List[Dict[str, Any]] = []
    for traj_uid, traj_rows in by_traj.items():
        traj_rows = sorted(traj_rows, key=lambda r: int(r.get("step_num", 0)))
        step_rewards = [_row_float(r, "rewards") for r in traj_rows]
        episode_return = float(sum(step_rewards))
        episode_success = 1 if episode_return > success_threshold else 0

        # return-to-go, discounted from each step to the end of the episode
        returns_to_go = [0.0] * len(traj_rows)
        acc = 0.0
        for i in range(len(traj_rows) - 1, -1, -1):
            acc = step_rewards[i] + float(gamma) * acc
            returns_to_go[i] = acc

        for i, row in enumerate(traj_rows):
            response_text = row.get("response_text")
            if response_text is None:
                continue
            record: Dict[str, Any] = {
                "uid": str(row.get("uid")),
                "traj_uid": traj_uid,
                "sample_id": int(row.get("sample_id", -1)),
                "rollout_id": int(row.get("rollout_id", -1)),
                "step_num": int(row.get("step_num", i)),
                "step_id": str(row.get("step_id", f"{traj_uid}_{i}")),
                "obs_text": str(row.get("obs_text", "")),
                "response_text": str(response_text),
                "step_reward": float(step_rewards[i]),
                "episode_return": episode_return,
                "return_to_go": float(returns_to_go[i]),
                "episode_success": int(episode_success),
                "episode_length": int(len(traj_rows)),
                "tag_error_signal": bool(row.get("tag_error_signal", False)),
                "is_teacher_branch": bool(row.get("is_teacher_branch", False)),
            }
            for key in _TAG_COUNT_KEYS:
                if key in row:
                    try:
                        record[key] = int(row[key])
                    except (TypeError, ValueError):
                        record[key] = 0
            if global_step is not None:
                record["global_step"] = int(global_step)
            records.append(record)
    return records


def rows_from_dataproto(batch, tokenizer) -> List[Dict[str, Any]]:
    """
    Flatten a rollout DataProto into row dicts with a decoded response_text.

    Cluster-side helper (imports torch lazily via the batch itself); the pure
    extraction above stays laptop-testable.
    """
    size = len(batch)
    responses = batch.batch["responses"] if "responses" in batch.batch.keys() else None
    attention_mask = batch.batch.get("attention_mask") if hasattr(batch.batch, "get") else None
    rows: List[Dict[str, Any]] = []
    for i in range(size):
        row: Dict[str, Any] = {key: value[i] for key, value in batch.non_tensor_batch.items()}
        if row.get("response_text") is None and responses is not None:
            response_ids = responses[i]
            if attention_mask is not None:
                response_length = response_ids.shape[-1]
                valid = attention_mask[i][-response_length:].bool()
                response_ids = response_ids[valid]
            row["response_text"] = tokenizer.decode(response_ids, skip_special_tokens=True)
        rows.append(row)
    return rows


class StepDatasetDumper:
    """Appends step-RM parquet shards, one per training step."""

    def __init__(self, dump_dir: str, *, gamma: float = 1.0, success_threshold: float = 0.5):
        self.dump_dir = str(dump_dir)
        self.gamma = float(gamma)
        self.success_threshold = float(success_threshold)
        os.makedirs(self.dump_dir, exist_ok=True)

    def dump_rows(self, rows: Sequence[Dict[str, Any]], *, global_step: int) -> int:
        records = extract_step_rows(
            rows,
            gamma=self.gamma,
            success_threshold=self.success_threshold,
            global_step=global_step,
        )
        if not records:
            return 0
        import pandas as pd

        path = os.path.join(self.dump_dir, f"step_rows_{int(global_step):05d}.parquet")
        pd.DataFrame(records).to_parquet(path, index=False)
        return len(records)

    def dump_batch(self, batch, tokenizer, *, global_step: int) -> int:
        return self.dump_rows(
            rows_from_dataproto(batch, tokenizer), global_step=global_step
        )


@dataclass
class StepRMScore:
    scores: List[float]
    elapsed_s: float


class StepRewardModelClient:
    """HTTP client for the step-RM scoring server (serve_step_rm.py).

    Same ergonomics as the external-teacher client: comma-separated replica
    list, per-request batching, retries with endpoint failover.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        batch_size: int = 64,
    ):
        self.base_urls = [u.strip().rstrip("/") for u in str(base_url).split(",") if u.strip()]
        if not self.base_urls:
            raise ValueError(f"step RM base_url is empty: {base_url!r}")
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.batch_size = max(1, int(batch_size))
        import requests

        self._requests = requests
        self._session = requests.Session()
        self._rr = 0

    def _post_scores(self, texts: List[str]) -> List[float]:
        offset = self._rr
        self._rr += 1
        last_error = None
        for attempt in range(self.max_retries):
            base_url = self.base_urls[(offset + attempt) % len(self.base_urls)]
            try:
                response = self._session.post(
                    f"{base_url}/score",
                    json={"texts": texts},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                scores = payload.get("scores")
                if not isinstance(scores, list) or len(scores) != len(texts):
                    raise RuntimeError(
                        f"step RM returned {0 if scores is None else len(scores)} scores for {len(texts)} texts"
                    )
                return [float(s) for s in scores]
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_s * (2**attempt))
        raise RuntimeError(
            f"step RM scoring failed after {self.max_retries} attempts across "
            f"{len(self.base_urls)} endpoint(s): {last_error}"
        )

    def score_texts(self, texts: Sequence[str]) -> StepRMScore:
        start = time.perf_counter()
        scores: List[float] = []
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), self.batch_size):
            scores.extend(self._post_scores(texts[i : i + self.batch_size]))
        return StepRMScore(scores=scores, elapsed_s=time.perf_counter() - start)

    def score_steps(self, observations: Sequence[str], responses: Sequence[str]) -> StepRMScore:
        if len(observations) != len(responses):
            raise ValueError("observations and responses must be the same length")
        return self.score_texts(
            [build_step_rm_text(o, r) for o, r in zip(observations, responses)]
        )
