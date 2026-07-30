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
"""Tests for the step-level reward model data pipeline and scoring client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from seed.step_rm import (
    StepDatasetDumper,
    StepRewardModelClient,
    build_step_rm_text,
    extract_step_rows,
)


def _row(traj, step, *, uid="task0", reward=0.0, response="resp", active=True):
    return {
        "traj_uid": traj,
        "uid": uid,
        "sample_id": 0,
        "rollout_id": 0,
        "step_num": step,
        "step_id": f"{traj}_{step}",
        "rewards": reward,
        "obs_text": f"obs {step}",
        "response_text": response,
        "tag_error_signal": False,
        "active_masks": active,
    }


# ------------------------------------------------------------- extraction

def test_extract_step_rows_episode_aggregates():
    rows = [
        _row("t0", 0, reward=0.0),
        _row("t0", 1, reward=0.0),
        _row("t0", 2, reward=1.0),
        _row("t1", 0, reward=0.0),
    ]
    records = extract_step_rows(rows, gamma=1.0)
    by_id = {r["step_id"]: r for r in records}
    assert by_id["t0_0"]["episode_return"] == pytest.approx(1.0)
    assert by_id["t0_0"]["return_to_go"] == pytest.approx(1.0)
    assert by_id["t0_2"]["return_to_go"] == pytest.approx(1.0)
    assert by_id["t0_0"]["episode_success"] == 1
    assert by_id["t0_0"]["episode_length"] == 3
    assert by_id["t1_0"]["episode_success"] == 0


def test_extract_step_rows_discounting():
    rows = [_row("t", 0, reward=0.0), _row("t", 1, reward=1.0)]
    records = extract_step_rows(rows, gamma=0.5)
    by_step = {r["step_num"]: r for r in records}
    assert by_step[0]["return_to_go"] == pytest.approx(0.5)  # 0 + 0.5 * 1
    assert by_step[1]["return_to_go"] == pytest.approx(1.0)


def test_extract_step_rows_skips_inactive_and_unencoded():
    rows = [
        _row("t", 0, active=False),
        {**_row("t", 1), "response_text": None},
        _row("t", 2),
    ]
    records = extract_step_rows(rows)
    assert [r["step_num"] for r in records] == [2]


def test_extract_step_rows_stamps_global_step():
    records = extract_step_rows([_row("t", 0)], global_step=7)
    assert records[0]["global_step"] == 7


# ------------------------------------------------------------------ dumper

def test_dumper_writes_parquet(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    dumper = StepDatasetDumper(str(tmp_path), gamma=1.0)
    n = dumper.dump_rows([_row("t", 0, reward=1.0), _row("t", 1)], global_step=3)
    assert n == 2
    df = pd.read_parquet(tmp_path / "step_rows_00003.parquet")
    assert len(df) == 2
    assert set(["obs_text", "response_text", "return_to_go", "episode_success"]) <= set(df.columns)


def test_dumper_empty_rows_writes_nothing(tmp_path):
    dumper = StepDatasetDumper(str(tmp_path))
    assert dumper.dump_rows([], global_step=1) == 0
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ client

_DEAD_PORTS = set()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == "/score"
        port = self.server.server_address[1]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if port in _DEAD_PORTS:
            self.send_response(500)
            self.end_headers()
            return
        scores = [float(len(t)) for t in body["texts"]]  # deterministic: score = len
        data = json.dumps({"scores": scores}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def rm_servers():
    servers = []
    for _ in range(2):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    yield servers
    for server in servers:
        server.shutdown()


def test_client_scores_and_batches(rm_servers):
    port = rm_servers[0].server_address[1]
    client = StepRewardModelClient(f"http://127.0.0.1:{port}", batch_size=2, retry_backoff_s=0.01)
    result = client.score_texts(["a", "bb", "ccc", "dddd", "eeeee"])
    assert result.scores == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result.elapsed_s > 0


def test_client_score_steps_uses_template(rm_servers):
    port = rm_servers[0].server_address[1]
    client = StepRewardModelClient(f"http://127.0.0.1:{port}", retry_backoff_s=0.01)
    result = client.score_steps(["obs"], ["resp"])
    assert result.scores == [float(len(build_step_rm_text("obs", "resp")))]


def test_client_fails_over_to_healthy_replica(rm_servers):
    ports = [s.server_address[1] for s in rm_servers]
    client = StepRewardModelClient(
        ",".join(f"http://127.0.0.1:{p}" for p in ports),
        retry_backoff_s=0.01,
    )
    _DEAD_PORTS.add(ports[0])
    try:
        result = client.score_texts(["xyz"])
        assert result.scores == [3.0]
    finally:
        _DEAD_PORTS.discard(ports[0])


# --------------------------------------------------------- training helpers

def test_train_script_helpers(tmp_path):
    import importlib.util
    import os
    import sys

    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "step_rm", "train_step_rm.py"
    )
    spec = importlib.util.spec_from_file_location("train_step_rm", os.path.abspath(script))
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_step_rm"] = module
    spec.loader.exec_module(module)

    records = [
        {"uid": "u0", "return_to_go": 1.0},
        {"uid": "u0", "return_to_go": 0.0},
        {"uid": "u1", "return_to_go": 0.5},
        {"uid": "u2", "return_to_go": 0.7},
    ]
    train, val = module.split_by_uid(records, val_frac=0.34, seed=0)
    train_uids = {r["uid"] for r in train}
    val_uids = {r["uid"] for r in val}
    assert train_uids.isdisjoint(val_uids)  # no task leaks across the split
    assert len(train) + len(val) == len(records)

    module.group_normalize(records, "return_to_go")
    u0_values = [r["return_to_go"] for r in records if r["uid"] == "u0"]
    assert sum(u0_values) == pytest.approx(0.0, abs=1e-5)  # zero-mean within group

    import torch

    scores = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = module.bt_pairs_loss(scores, ["g", "g"], torch.tensor([0, 1]))
    assert loss.item() > 0  # success step must outrank the failure step
    loss.backward()  # differentiable
