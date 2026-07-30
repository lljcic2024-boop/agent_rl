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
"""full_step prefix mode: the teacher writes the whole branch step."""

import pytest

from agent_system.multi_turn_rollout.teacher_branch import (
    BranchSpec,
    TeacherPrefixParseError,
    generate_teacher_prefixes,
    parse_teacher_full_step_response,
    run_branch_rollout_batched,
)

GOOD_REPLY = (
    "<think>\n<reflect>The previous query was too broad and returned nothing useful.</reflect>\n"
    "<plan>Search for the specific album title instead.</plan>\n</think>\n"
    "<search>the dark side of the moon release year</search>"
)


# ------------------------------------------------------------------- parsing

def test_parse_full_step_accepts_complete_reply():
    normalized = parse_teacher_full_step_response(GOOD_REPLY)
    assert normalized.startswith("<think>")
    assert "</think>" in normalized
    assert normalized.rstrip().endswith("</search>")


def test_parse_full_step_requires_closed_think():
    with pytest.raises(TeacherPrefixParseError, match="close the thinking block"):
        parse_teacher_full_step_response("<think><plan>x</plan> <search>q</search>")


def test_parse_full_step_requires_exactly_one_action():
    no_action = "<think><plan>look it up</plan></think>"
    with pytest.raises(TeacherPrefixParseError, match="exactly one"):
        parse_teacher_full_step_response(no_action)
    two_actions = GOOD_REPLY + "<answer>1973</answer>"
    with pytest.raises(TeacherPrefixParseError, match="exactly one"):
        parse_teacher_full_step_response(two_actions)


def test_parse_full_step_requires_function_tags():
    reply = "<think>just some untagged text that is long enough</think><search>q</search>"
    with pytest.raises(TeacherPrefixParseError, match="function tag"):
        parse_teacher_full_step_response(reply)


def test_parse_full_step_rejects_commentary_outside_action():
    reply = GOOD_REPLY + "\nHope this helps!"
    with pytest.raises(TeacherPrefixParseError, match="outside the action"):
        parse_teacher_full_step_response(reply)


def test_generate_teacher_prefixes_full_step_mode_uses_full_step_gate():
    spec = BranchSpec(
        parent_traj_uid="t0",
        parent_uid="u0",
        sample_id=0,
        rollout_id=0,
        branch_step_num=1,
        replay_actions=["a0"],
        branch_observation="obs",
    )
    prefixes, failures = generate_teacher_prefixes(
        [spec], lambda messages: GOOD_REPLY, mode="full_step"
    )
    assert failures == {}
    assert prefixes[0].rstrip().endswith("</search>")
    # the same reply fails the think-prefix gate (it closes </think>)
    prefixes2, failures2 = generate_teacher_prefixes(
        [spec], lambda messages: GOOD_REPLY, mode="think_prefix"
    )
    assert prefixes2 == {} and 0 in failures2


def test_generate_teacher_prefixes_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown teacher prefix mode"):
        generate_teacher_prefixes([], lambda m: "", mode="bogus")


# ------------------------------------------------------------------ rollout

def _spec(step=1, traj="t0"):
    return BranchSpec(
        parent_traj_uid=traj,
        parent_uid="u0",
        sample_id=0,
        rollout_id=0,
        branch_step_num=step,
        replay_actions=["replayed action"] * step,
        branch_observation="obs at branch",
        env_kwargs={"question": "q"},
    )


def test_full_step_rollout_skips_policy_at_branch_step():
    spec = _spec(step=1)
    teacher_step = parse_teacher_full_step_response(GOOD_REPLY)
    generate_calls = []
    encode_calls = []

    def env_start(specs):
        return ["obs at branch"]

    def env_step(specs, actions):
        return (["next obs"], [0.5], [False], [{}])

    def generate(obs_texts, prefixes, indices, step_num):
        generate_calls.append(step_num)
        assert all(p is None for p in prefixes)  # never called with the teacher step
        return ["<think><plan>continue</plan>" for _ in obs_texts]

    def encode(obs_texts, response_texts, indices, step_num):
        encode_calls.append((step_num, tuple(response_texts)))

    trajectories = run_branch_rollout_batched(
        [spec],
        {0: teacher_step},
        env_start_batch=env_start,
        env_step_batch=env_step,
        generate_batch=generate,
        max_steps=3,
        prefix_mode="full_step",
        encode_teacher_step=encode,
    )
    assert len(trajectories) == 1
    rows = trajectories[0].rows
    # branch step row is the teacher's text, follow-up row is the student's
    assert rows[0]["response_text"] == teacher_step
    assert rows[0]["branch_prefix_text"] == teacher_step
    assert len(encode_calls) == 1 and encode_calls[0][0] == 1
    assert generate_calls == [2]  # student only generates from the NEXT step


def test_full_step_rollout_requires_encoder_hook():
    with pytest.raises(ValueError, match="encode_teacher_step"):
        run_branch_rollout_batched(
            [_spec()],
            {0: "x"},
            env_start_batch=lambda s: ["obs"],
            env_step_batch=lambda s, a: ([""], [0.0], [True], [{}]),
            generate_batch=lambda o, p, i, s: ["y"],
            max_steps=2,
            prefix_mode="full_step",
        )


def test_think_prefix_mode_unchanged():
    spec = _spec(step=0)
    prefix = "<think>\n<reflect>hm</reflect>\n"

    def generate(obs_texts, prefixes, indices, step_num):
        return [(prefixes[i] or "") + "continued</think><search>q</search>" for i in range(len(obs_texts))]

    trajectories = run_branch_rollout_batched(
        [spec],
        {0: prefix},
        env_start_batch=lambda s: ["obs"],
        env_step_batch=lambda s, a: (["next"], [1.0], [True], [{}]),
        generate_batch=generate,
        max_steps=2,
        prefix_mode="think_prefix",
    )
    assert len(trajectories) == 1
    assert trajectories[0].rows[0]["response_text"].startswith(prefix)
