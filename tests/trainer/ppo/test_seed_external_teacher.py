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
"""Tests for SEED external teacher scoring against fake vLLM prompt_logprobs servers.

The client packs several rows into one /completions call (vLLM accepts a list
of token-id arrays) and spreads batches across replica endpoints, so the fake
server accepts both the flat single-prompt form and the batched form and tags
each choice with its prompt index.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from verl.trainer.ppo.seed_external_teacher import ExternalTeacherClient

_FAIL_FIRST = {"remaining": 0}
_SEEN_PROMPTS = []  # one entry per prompt scored (flattened across batched calls)
_SEEN_REQUESTS = []  # one entry per HTTP call: list of prompts in that call
_REQUESTS_BY_PORT = {}  # port -> number of HTTP calls served
_DEAD_PORTS = set()  # ports that always answer 500


def _token_logprob(token_id: int) -> float:
    # deterministic per-token value so alignment errors are detectable
    return -float(token_id) / 100.0


def _prompt_logprobs_for(token_ids):
    return [None] + [
        {str(tid): {"logprob": _token_logprob(tid), "rank": 1, "decoded_token": f"t{tid}"}}
        for tid in token_ids[1:]
    ]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == "/v1/completions", self.path
        port = self.server.server_address[1]
        _REQUESTS_BY_PORT[port] = _REQUESTS_BY_PORT.get(port, 0) + 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["prompt"]
        # normalize: flat list of ints (legacy) vs list of token-id lists (batched)
        prompts = [prompt] if prompt and isinstance(prompt[0], int) else prompt
        _SEEN_REQUESTS.append(prompts)
        _SEEN_PROMPTS.extend(prompts)
        if port in _DEAD_PORTS or _FAIL_FIRST["remaining"] > 0:
            if _FAIL_FIRST["remaining"] > 0:
                _FAIL_FIRST["remaining"] -= 1
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        choices = [
            {"index": i, "text": "x", "prompt_logprobs": _prompt_logprobs_for(token_ids)}
            for i, token_ids in enumerate(prompts)
        ]
        data = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture(scope="module")
def servers():
    started = [_start_server() for _ in range(2)]
    yield started
    for server in started:
        server.shutdown()


@pytest.fixture(scope="module")
def client(servers):
    port = servers[0].server_address[1]
    return ExternalTeacherClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="fake-teacher",
        max_retries=3,
        retry_backoff_s=0.01,
        concurrency=4,
        batch_size=16,
    )


def test_alignment_left_padded_prompt(client):
    input_ids = torch.tensor(
        [
            [0, 0, 11, 12, 21, 22, 0],
            [0, 13, 14, 15, 23, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0],
        ]
    )
    response_masks = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
        ]
    )
    out = client.score_response_log_probs(input_ids, attention_mask, response_masks)
    expected = torch.tensor(
        [
            [_token_logprob(21), _token_logprob(22), 0.0],
            [_token_logprob(23), 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(out, expected)
    # server must have received only the valid (non-pad) token ids
    assert [11, 12, 21, 22] in _SEEN_PROMPTS
    assert [13, 14, 15, 23] in _SEEN_PROMPTS


def test_rows_are_batched_into_one_request(client):
    n_before = len(_SEEN_REQUESTS)
    input_ids = torch.tensor([[11, 12, 21], [13, 14, 22], [15, 16, 23]])
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.tensor([[1], [1], [1]])
    out = client.score_response_log_probs(input_ids, attention_mask, response_masks)
    expected = torch.tensor([[_token_logprob(21)], [_token_logprob(22)], [_token_logprob(23)]])
    torch.testing.assert_close(out, expected)
    new_requests = _SEEN_REQUESTS[n_before:]
    assert len(new_requests) == 1  # 3 rows <= batch_size -> exactly one HTTP call
    assert len(new_requests[0]) == 3


def test_batch_size_splits_requests(servers):
    port = servers[0].server_address[1]
    small_batch_client = ExternalTeacherClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="fake-teacher",
        retry_backoff_s=0.01,
        batch_size=2,
        concurrency=2,
    )
    n_before = len(_SEEN_REQUESTS)
    input_ids = torch.tensor([[11, 21], [12, 22], [13, 23], [14, 24], [15, 25]])
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((5, 1), dtype=torch.long)
    out = small_batch_client.score_response_log_probs(input_ids, attention_mask, response_masks)
    expected = torch.tensor([[_token_logprob(20 + i)] for i in range(1, 6)])
    torch.testing.assert_close(out, expected)
    assert len(_SEEN_REQUESTS) - n_before == 3  # ceil(5 / 2)
    assert small_batch_client.last_stats["num_batches"] == 3.0
    assert small_batch_client.last_stats["rows_scored"] == 5.0


def test_empty_response_row_is_skipped(client):
    n_before = len(_SEEN_PROMPTS)
    input_ids = torch.tensor([[11, 12, 21], [13, 14, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    response_masks = torch.tensor([[1], [0]])
    out = client.score_response_log_probs(input_ids, attention_mask, response_masks)
    torch.testing.assert_close(out[1], torch.tensor([0.0]))
    torch.testing.assert_close(out[0], torch.tensor([_token_logprob(21)]))
    assert len(_SEEN_PROMPTS) - n_before == 1  # only the non-empty row is sent
    assert client.last_stats["rows_skipped"] == 1.0


def test_retry_on_server_error(client):
    _FAIL_FIRST["remaining"] = 2  # first two attempts fail, third succeeds
    input_ids = torch.tensor([[11, 21]])
    attention_mask = torch.tensor([[1, 1]])
    response_masks = torch.tensor([[1]])
    out = client.score_response_log_probs(input_ids, attention_mask, response_masks)
    torch.testing.assert_close(out, torch.tensor([[_token_logprob(21)]]))
    assert client.last_stats["retries"] >= 1.0


def test_fails_when_retries_exhausted(client):
    _FAIL_FIRST["remaining"] = 10
    input_ids = torch.tensor([[11, 21]])
    attention_mask = torch.tensor([[1, 1]])
    response_masks = torch.tensor([[1]])
    with pytest.raises(RuntimeError, match="failed after"):
        client.score_response_log_probs(input_ids, attention_mask, response_masks)
    _FAIL_FIRST["remaining"] = 0


def test_round_robin_spreads_batches_across_endpoints(servers):
    ports = [server.server_address[1] for server in servers]
    multi_client = ExternalTeacherClient(
        base_url=",".join(f"http://127.0.0.1:{p}/v1" for p in ports),
        model="fake-teacher",
        retry_backoff_s=0.01,
        batch_size=1,  # one row per request so 4 rows -> 4 requests
        concurrency=1,  # deterministic round-robin order
    )
    counts_before = {p: _REQUESTS_BY_PORT.get(p, 0) for p in ports}
    input_ids = torch.tensor([[11, 21], [12, 22], [13, 23], [14, 24]])
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((4, 1), dtype=torch.long)
    multi_client.score_response_log_probs(input_ids, attention_mask, response_masks)
    served = {p: _REQUESTS_BY_PORT.get(p, 0) - counts_before[p] for p in ports}
    assert served[ports[0]] == 2 and served[ports[1]] == 2


def test_failover_to_healthy_endpoint(servers):
    ports = [server.server_address[1] for server in servers]
    multi_client = ExternalTeacherClient(
        base_url=",".join(f"http://127.0.0.1:{p}/v1" for p in ports),
        model="fake-teacher",
        max_retries=3,
        retry_backoff_s=0.01,
        batch_size=4,
        concurrency=2,
    )
    _DEAD_PORTS.add(ports[0])
    try:
        input_ids = torch.tensor([[11, 21], [12, 22]])
        attention_mask = torch.ones_like(input_ids)
        response_masks = torch.ones((2, 1), dtype=torch.long)
        out = multi_client.score_response_log_probs(input_ids, attention_mask, response_masks)
        expected = torch.tensor([[_token_logprob(21)], [_token_logprob(22)]])
        torch.testing.assert_close(out, expected)  # scored despite one dead replica
    finally:
        _DEAD_PORTS.discard(ports[0])


def test_stats_are_recorded(client):
    input_ids = torch.tensor([[11, 12, 21, 22]])
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.tensor([[1, 1]])
    client.score_response_log_probs(input_ids, attention_mask, response_masks)
    stats = client.last_stats
    assert stats["rows_scored"] == 1.0
    assert stats["tokens_sent"] == 4.0
    assert stats["tokens_scored"] == 2.0
    assert stats["elapsed_s"] > 0.0
    assert stats["prefill_tokens_per_s"] > 0.0
