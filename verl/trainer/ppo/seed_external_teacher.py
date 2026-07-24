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
External teacher scoring for SEED OPD (改造点 2).

Replaces the "policy scores itself under an augmented prompt" step with a real
teacher model served behind a vLLM OpenAI-compatible endpoint. The prompt is
sent as token ids (not text), so token alignment with the student is exact as
long as student and teacher share a tokenizer (e.g. Qwen3-8B vs Qwen3-30B-A3B).
Per-token log-probs come back via vLLM's `prompt_logprobs` extra parameter on
the /v1/completions route.

Notes / assumptions:
- Returned log-probs are raw model log-probs (temperature 1). If the student's
  log-probs were computed with a different temperature, the OPD gap is biased;
  the caller should warn.
- Valid response tokens are assumed to be the trailing valid tokens of the
  concatenated (prompt + response) sequence, which holds for verl's
  left-padded-prompt / right-padded-response layout.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class ExternalTeacherClient:
    """Scores student-sampled tokens with an external teacher server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        concurrency: int = 8,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.concurrency = max(1, int(concurrency))
        import requests  # local import: keep module importable without requests

        self._requests = requests
        self._session = requests.Session()

    # ------------------------------------------------------------------ HTTP

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_prompt_logprobs(self, token_ids: List[int]) -> List[Optional[dict]]:
        """One completions call; returns vLLM prompt_logprobs (list, [0] is None)."""
        payload = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                prompt_logprobs = data["choices"][0].get("prompt_logprobs")
                if prompt_logprobs is None:
                    raise RuntimeError(
                        "Teacher server response has no 'prompt_logprobs'. The endpoint must be a "
                        "vLLM OpenAI-compatible server that supports the prompt_logprobs extra parameter."
                    )
                if len(prompt_logprobs) != len(token_ids):
                    raise RuntimeError(
                        f"prompt_logprobs length {len(prompt_logprobs)} != prompt token count {len(token_ids)}"
                    )
                return prompt_logprobs
            except Exception as error:  # noqa: BLE001 - retry then re-raise
                last_error = error
                if attempt < self.max_retries - 1:
                    sleep_s = self.retry_backoff_s * (2**attempt)
                    logger.warning(
                        "External teacher scoring attempt %s/%s failed (%s); retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries,
                        error,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
        raise RuntimeError(f"External teacher scoring failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _entry_logprob(entry: Optional[dict], token_id: int) -> float:
        """Extract the actual token's logprob from one prompt_logprobs entry.

        vLLM JSON format: {"<token_id>": {"logprob": float, "rank": int, "decoded_token": str}, ...}
        The actual prompt token is always included, even with prompt_logprobs=0.
        """
        if entry is None:
            raise RuntimeError("Got a None prompt_logprobs entry for a response token (position 0 reached?).")
        value = entry.get(str(token_id))
        if value is None:
            value = entry.get(int(token_id))
        if value is None:
            raise RuntimeError(f"prompt_logprobs entry does not contain token id {token_id}: keys={list(entry)[:5]}")
        if isinstance(value, dict):
            return float(value["logprob"])
        return float(value)

    # ------------------------------------------------------------- public API

    def score_response_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (bsz, prompt_len + response_len) concatenated token ids.
            attention_mask: (bsz, prompt_len + response_len) validity mask
                (prompt left-padded, response right-padded).
            response_masks: (bsz, response_len) validity mask of the response part.

        Returns:
            (bsz, response_len) float32 tensor of teacher log-probs, zeros at
            invalid positions. Always returned on CPU; caller moves to device.
        """
        input_ids = input_ids.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        response_masks = response_masks.detach().cpu()
        bsz, response_length = response_masks.shape
        output = torch.zeros((bsz, response_length), dtype=torch.float32)

        jobs = []
        for i in range(bsz):
            valid = attention_mask[i].bool()
            token_ids = input_ids[i][valid].tolist()
            response_positions = torch.nonzero(response_masks[i].bool(), as_tuple=False).flatten().tolist()
            jobs.append((token_ids, response_positions))

        def _run(job):
            token_ids, response_positions = job
            num_response_tokens = len(response_positions)
            if num_response_tokens == 0:
                return None
            if num_response_tokens >= len(token_ids):
                raise RuntimeError(
                    f"Response token count {num_response_tokens} >= total valid tokens {len(token_ids)}; "
                    "the teacher prompt appears to be empty."
                )
            prompt_logprobs = self._request_prompt_logprobs(token_ids)
            tail_entries = prompt_logprobs[-num_response_tokens:]
            tail_token_ids = token_ids[-num_response_tokens:]
            return [self._entry_logprob(entry, token_id) for entry, token_id in zip(tail_entries, tail_token_ids)]

        if len(jobs) == 1:
            results = [_run(jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(jobs))) as pool:
                results = list(pool.map(_run, jobs))

        for i, ((_, response_positions), result) in enumerate(zip(jobs, results)):
            if result is None:
                continue
            output[i, torch.as_tensor(response_positions, dtype=torch.long)] = torch.tensor(
                result, dtype=torch.float32
            )
        return output
