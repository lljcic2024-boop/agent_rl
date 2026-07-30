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

Throughput design (the scoring call sits on the training step's critical path):

- `base_url` accepts a comma-separated list of teacher replicas. Batches are
  spread round-robin; a failed attempt retries on the *next* endpoint, so a
  single dead replica degrades throughput instead of killing the step.
- Rows are packed `batch_size` prompts per HTTP request (vLLM's /completions
  accepts a list of token-id arrays and schedules them with continuous
  batching), so 149 rows are a handful of requests instead of 149 round trips.
- `concurrency` HTTP requests are kept in flight so the server-side scheduler
  always has work queued.
- Every call records `last_stats` (rows/batches/tokens/elapsed/tok-per-s/
  retries/per-endpoint failures) for the trainer to emit as metrics.

Notes / assumptions:
- Returned log-probs are raw model log-probs (temperature 1). If the student's
  log-probs were computed with a different temperature, the OPD gap is biased;
  the caller should warn.
- Valid response tokens are assumed to be the trailing valid tokens of the
  concatenated (prompt + response) sequence, which holds for verl's
  left-padded-prompt / right-padded-response layout.
"""

import itertools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


class ExternalTeacherClient:
    """Scores student-sampled tokens with one or more external teacher replicas."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        concurrency: int = 8,
        batch_size: int = 16,
    ):
        self.base_urls = [u.strip().rstrip("/") for u in str(base_url).split(",") if u.strip()]
        if not self.base_urls:
            raise ValueError(f"external teacher base_url is empty: {base_url!r}")
        self.model = str(model)
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.concurrency = max(1, int(concurrency))
        self.batch_size = max(1, int(batch_size))
        # Stats of the most recent score_response_log_probs() call, flat floats
        # so the trainer can dump them straight into the metrics dict.
        self.last_stats: Dict[str, float] = {}
        self._rr_counter = itertools.count()
        self._rr_lock = threading.Lock()
        import requests  # local import: keep module importable without requests

        self._requests = requests
        # One session per client; requests.Session is thread-safe for our use
        # (independent POSTs, no cookies).
        self._session = requests.Session()

    # ------------------------------------------------------------------ HTTP

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _next_endpoint_offset(self) -> int:
        with self._rr_lock:
            return next(self._rr_counter)

    def _request_prompt_logprobs_batch(
        self,
        prompts: Sequence[List[int]],
        endpoint_failures: Dict[str, int],
    ) -> Tuple[List[List[Optional[dict]]], int]:
        """One /completions call scoring several token-id prompts at once.

        Returns (per-prompt prompt_logprobs lists, retry count). Attempts rotate
        through the endpoint list starting at a round-robin offset, so load is
        spread across replicas and a dead replica is skipped on retry.
        """
        payload = {
            "model": self.model,
            "prompt": [list(p) for p in prompts],
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
        offset = self._next_endpoint_offset()
        last_error = None
        for attempt in range(self.max_retries):
            base_url = self.base_urls[(offset + attempt) % len(self.base_urls)]
            try:
                response = self._session.post(
                    f"{base_url}/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices")
                if not choices or len(choices) != len(prompts):
                    raise RuntimeError(
                        f"expected {len(prompts)} choices, got {0 if not choices else len(choices)}"
                    )
                # vLLM tags each choice with the index of its prompt; don't
                # assume the list comes back ordered.
                by_index: List[Optional[List[Optional[dict]]]] = [None] * len(prompts)
                for position, choice in enumerate(choices):
                    index = int(choice.get("index", position))
                    prompt_logprobs = choice.get("prompt_logprobs")
                    if prompt_logprobs is None:
                        raise RuntimeError(
                            "Teacher server response has no 'prompt_logprobs'. The endpoint must be a "
                            "vLLM OpenAI-compatible server that supports the prompt_logprobs extra parameter."
                        )
                    if len(prompt_logprobs) != len(prompts[index]):
                        raise RuntimeError(
                            f"prompt_logprobs length {len(prompt_logprobs)} != prompt token count "
                            f"{len(prompts[index])} (choice {index})"
                        )
                    by_index[index] = prompt_logprobs
                if any(entry is None for entry in by_index):
                    raise RuntimeError("duplicate/missing choice indices in teacher response")
                return by_index, attempt  # type: ignore[return-value]
            except Exception as error:  # noqa: BLE001 - retry then re-raise
                last_error = error
                endpoint_failures[base_url] = endpoint_failures.get(base_url, 0) + 1
                if attempt < self.max_retries - 1:
                    sleep_s = self.retry_backoff_s * (2**attempt)
                    logger.warning(
                        "External teacher scoring attempt %s/%s on %s failed (%s); retrying on next endpoint in %.1fs.",
                        attempt + 1,
                        self.max_retries,
                        base_url,
                        error,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
        raise RuntimeError(
            f"External teacher scoring failed after {self.max_retries} attempts across "
            f"{len(self.base_urls)} endpoint(s) {self.base_urls}: {last_error}"
        )

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
        start_time = time.perf_counter()
        input_ids = input_ids.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        response_masks = response_masks.detach().cpu()
        bsz, response_length = response_masks.shape
        output = torch.zeros((bsz, response_length), dtype=torch.float32)

        jobs = []  # (row index, token_ids, response_positions)
        rows_skipped = 0
        for i in range(bsz):
            valid = attention_mask[i].bool()
            token_ids = input_ids[i][valid].tolist()
            response_positions = torch.nonzero(response_masks[i].bool(), as_tuple=False).flatten().tolist()
            if not response_positions:
                rows_skipped += 1
                continue
            if len(response_positions) >= len(token_ids):
                raise RuntimeError(
                    f"Response token count {len(response_positions)} >= total valid tokens {len(token_ids)} "
                    f"(row {i}); the teacher prompt appears to be empty."
                )
            jobs.append((i, token_ids, response_positions))

        chunks = [jobs[i : i + self.batch_size] for i in range(0, len(jobs), self.batch_size)]
        endpoint_failures: Dict[str, int] = {}
        retries_total = 0
        retries_lock = threading.Lock()

        def _run_chunk(chunk):
            nonlocal retries_total
            prompts = [token_ids for _, token_ids, _ in chunk]
            per_prompt_logprobs, retries = self._request_prompt_logprobs_batch(prompts, endpoint_failures)
            with retries_lock:
                retries_total += retries
            results = []
            for (row, token_ids, response_positions), prompt_logprobs in zip(chunk, per_prompt_logprobs):
                num_response_tokens = len(response_positions)
                tail_entries = prompt_logprobs[-num_response_tokens:]
                tail_token_ids = token_ids[-num_response_tokens:]
                values = [
                    self._entry_logprob(entry, token_id)
                    for entry, token_id in zip(tail_entries, tail_token_ids)
                ]
                results.append((row, response_positions, values))
            return results

        if len(chunks) <= 1:
            chunk_results = [_run_chunk(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(chunks))) as pool:
                chunk_results = list(pool.map(_run_chunk, chunks))

        for results in chunk_results:
            for row, response_positions, values in results:
                output[row, torch.as_tensor(response_positions, dtype=torch.long)] = torch.tensor(
                    values, dtype=torch.float32
                )

        elapsed = time.perf_counter() - start_time
        tokens_sent = float(sum(len(token_ids) for _, token_ids, _ in jobs))
        tokens_scored = float(sum(len(positions) for _, _, positions in jobs))
        self.last_stats = {
            "rows_scored": float(len(jobs)),
            "rows_skipped": float(rows_skipped),
            "num_batches": float(len(chunks)),
            "num_endpoints": float(len(self.base_urls)),
            "tokens_sent": tokens_sent,
            "tokens_scored": tokens_scored,
            "elapsed_s": float(elapsed),
            "prefill_tokens_per_s": tokens_sent / elapsed if elapsed > 0 else 0.0,
            "retries": float(retries_total),
            "endpoint_failures": float(sum(endpoint_failures.values())),
        }
        if endpoint_failures:
            logger.warning("External teacher endpoint failures this call: %s", endpoint_failures)
        logger.info(
            "External teacher scored %d rows (%d skipped) in %d batch(es) across %d endpoint(s): "
            "%.1fs, %.0f prefill tok/s, %d retries.",
            len(jobs),
            rows_skipped,
            len(chunks),
            len(self.base_urls),
            elapsed,
            self.last_stats["prefill_tokens_per_s"],
            retries_total,
        )
        return output
