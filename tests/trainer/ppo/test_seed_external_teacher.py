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
"""Tests for SEED external teacher scoring against a fake vLLM prompt_logprobs server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from verl.trainer.ppo.seed_external_teacher import ExternalTeacherClient

_FAIL_FIRST = {"remaining": 0}
_SEEN_PROMPTS = []


def _token_logprob(token_id: int) -> float:
    # deterministic per-token value so alignment errors are detectable
    return -float(token_id) / 100.0


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == "/v1/completions", self.path
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _SEEN_PROMPTS.append(body["prompt"])
        if _FAIL_FIRST["remaining"] > 0:
            _FAIL_FIRST["remaining"] -= 1
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        token_ids = body["prompt"]
        prompt_logprobs = [None] + [
            {str(tid): {"logprob": _token_logprob(tid), "rank": 1, "decoded_token": f"t{tid}"}}
            for tid in token_ids[1:]
        ]
        payload = {"choices": [{"text": "x", "prompt_logprobs": prompt_logprobs}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def client():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield ExternalTeacherClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="fake-teacher",
        max_retries=3,
        retry_backoff_s=0.01,
        concurrency=4,
    )
    server.shutdown()


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


def test_empty_response_row_is_skipped(client):
    n_before = len(_SEEN_PROMPTS)
    input_ids = torch.tensor([[11, 12, 21], [13, 14, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    response_masks = torch.tensor([[1], [0]])
    out = client.score_response_log_probs(input_ids, attention_mask, response_masks)
    torch.testing.assert_close(out[1], torch.tensor([0.0]))
    torch.testing.assert_close(out[0], torch.tensor([_token_logprob(21)]))
    assert len(_SEEN_PROMPTS) - n_before == 1  # only one HTTP call for the non-empty row


def test_retry_on_server_error(client):
    _FAIL_FIRST["remaining"] = 2  # first two attempts fail, third succeeds
    input_ids = torch.tensor([[11, 21]])
    attention_mask = torch.tensor([[1, 1]])
    response_masks = torch.tensor([[1]])
    out = client.score_response_log_probs(input_ids, attention_mask, response_masks)
    torch.testing.assert_close(out, torch.tensor([[_token_logprob(21)]]))


def test_fails_when_retries_exhausted(client):
    _FAIL_FIRST["remaining"] = 10
    input_ids = torch.tensor([[11, 21]])
    attention_mask = torch.tensor([[1, 1]])
    response_masks = torch.tensor([[1]])
    with pytest.raises(RuntimeError, match="failed after"):
        client.score_response_log_probs(input_ids, attention_mask, response_masks)
    _FAIL_FIRST["remaining"] = 0
