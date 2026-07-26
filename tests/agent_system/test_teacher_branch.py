"""Tests for the teacher-prefix branch orchestration (改造点 4, offline part).

Covers branch-point selection (error-signal steps, per-traj / global caps,
uid inheritance), the a_T prompt/parse contract, and the text-level branch
continuation loop (env replay hook, prefix injection, episode aggregation)
with fake env / policy hooks. Pure Python: no verl / ray / omegaconf needed.
"""
import importlib.util
import pathlib
import random

import pytest

# Load the module directly from its file to avoid the heavy package __init__
# (agent_system.multi_turn_rollout imports verl).
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "agent_system" / "multi_turn_rollout" / "teacher_branch.py"
)
_spec = importlib.util.spec_from_file_location("teacher_branch", _MODULE_PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def make_row(traj="t0", uid="g0", sample_id=0, rollout_id=0, step=0,
             error=False, active=True, reward=0.0, obs="obs", action="<search>q</search>"):
    return {
        "traj_uid": traj,
        "uid": uid,
        "sample_id": sample_id,
        "rollout_id": rollout_id,
        "step_num": step,
        "active_masks": active,
        "tag_error_signal": error,
        "rewards": reward,
        "obs_text": obs,
        "action_text": action,
    }


GET_ACTION = lambda r: r["action_text"]


# ---------------------------------------------------------------------------
# select_branch_specs
# ---------------------------------------------------------------------------

def test_select_picks_error_signal_step_and_inherits_uid():
    rows = [
        make_row(step=0, action="<search>a</search>", reward=0.1),
        make_row(step=1, error=True, obs="obs-err"),
        make_row(step=2),
    ]
    specs = m.select_branch_specs(rows, get_action_text=GET_ACTION)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.branch_step_num == 1
    assert spec.parent_uid == "g0"
    assert spec.parent_traj_uid == "t0"
    assert spec.branch_traj_uid != "t0"
    assert spec.replay_actions == ["<search>a</search>"]
    assert spec.prefix_reward == pytest.approx(0.1)
    assert spec.branch_observation == "obs-err"


def test_select_skips_trajectories_without_error_signal():
    rows = [make_row(step=0), make_row(step=1)]
    assert m.select_branch_specs(rows, get_action_text=GET_ACTION) == []


def test_select_without_error_requirement_takes_any_step():
    rows = [make_row(step=0), make_row(step=1)]
    specs = m.select_branch_specs(
        rows, get_action_text=GET_ACTION, require_error_signal=False
    )
    assert len(specs) == 1  # capped by max_branches_per_traj=1
    assert specs[0].branch_step_num == 0  # deterministic: earliest candidate


def test_select_ignores_inactive_rows():
    rows = [make_row(step=0, error=True, active=False)]
    assert m.select_branch_specs(rows, get_action_text=GET_ACTION) == []


def test_select_respects_per_traj_cap_with_rng():
    rows = [make_row(step=s, error=True) for s in range(4)]
    specs = m.select_branch_specs(
        rows, get_action_text=GET_ACTION,
        max_branches_per_traj=2, rng=random.Random(7),
    )
    assert len(specs) == 2
    assert specs[0].branch_step_num < specs[1].branch_step_num


def test_select_respects_global_cap_across_trajs():
    rows = []
    for k in range(3):
        rows.append(make_row(traj=f"t{k}", uid=f"g{k}", sample_id=k, step=1, error=True))
        rows.append(make_row(traj=f"t{k}", uid=f"g{k}", sample_id=k, step=0))
    specs = m.select_branch_specs(rows, get_action_text=GET_ACTION, max_total_branches=2)
    assert len(specs) == 2
    # deterministic ordering by (sample_id, rollout_id, step)
    assert [s.parent_traj_uid for s in specs] == ["t0", "t1"]


def test_select_replay_actions_ordered_by_step_num():
    rows = [
        make_row(step=2, error=True),
        make_row(step=0, action="a0"),
        make_row(step=1, action="a1"),
    ]
    specs = m.select_branch_specs(rows, get_action_text=GET_ACTION)
    assert specs[0].replay_actions == ["a0", "a1"]


# ---------------------------------------------------------------------------
# a_T prompt building and parsing
# ---------------------------------------------------------------------------

def test_prefix_messages_contain_observation_and_instruction():
    msgs = m.build_teacher_prefix_messages("OBS-TEXT", tag_instruction="TAG-MENU")
    assert msgs[0]["role"] == "system"
    assert "TAG-MENU" in msgs[0]["content"]
    assert "OBS-TEXT" in msgs[1]["content"]


def test_parse_accepts_tagged_prefix_and_opens_think():
    reply = "<reflect>the search returned nothing useful, the entity name is wrong</reflect>"
    prefix = m.parse_teacher_prefix_response(reply)
    assert prefix.startswith("<think>")
    assert "</think>" not in prefix
    assert "<reflect>" in prefix


def test_parse_strips_teacher_supplied_think_open():
    reply = "<think><reflect>the query was too narrow, need a broader formulation</reflect>"
    prefix = m.parse_teacher_prefix_response(reply)
    assert prefix.count("<think>") == 1


def test_parse_rejects_closed_think():
    with pytest.raises(m.TeacherPrefixParseError):
        m.parse_teacher_prefix_response(
            "<reflect>this went wrong for a clear reason here</reflect></think>"
        )


def test_parse_rejects_action_markup():
    with pytest.raises(m.TeacherPrefixParseError):
        m.parse_teacher_prefix_response(
            "<reflect>bad results, need another query</reflect><search>new query</search>"
        )


def test_parse_rejects_untagged_and_unbalanced():
    with pytest.raises(m.TeacherPrefixParseError):
        m.parse_teacher_prefix_response("just some untagged thinking about the problem")
    with pytest.raises(m.TeacherPrefixParseError):
        m.parse_teacher_prefix_response("<reflect>opened but never closed segment text")


def test_parse_rejects_too_short():
    with pytest.raises(m.TeacherPrefixParseError):
        m.parse_teacher_prefix_response("<reflect>too short</reflect>")


def test_generate_teacher_prefixes_retries_then_drops():
    specs = [
        m.BranchSpec("t0", "g0", 0, 0, 1, [], "obs-a"),
        m.BranchSpec("t1", "g1", 1, 0, 1, [], "obs-b"),
    ]
    calls = {"n": 0}

    def chat_fn(messages):
        calls["n"] += 1
        if "obs-a" in messages[1]["content"]:
            return "<reflect>the previous query missed the actual entity in question</reflect>"
        return "untagged rubbish"  # never parses -> dropped after retries

    prefixes, failures = m.generate_teacher_prefixes(specs, chat_fn, max_retries=2)
    assert set(prefixes) == {0}
    assert set(failures) == {1}
    assert calls["n"] == 3  # 1 success + 2 failed attempts


# ---------------------------------------------------------------------------
# run_branch_rollout (fake env + fake policy)
# ---------------------------------------------------------------------------

PREFIX = "<think>\n<reflect>the query failed, switch to the official name</reflect>\n"


class FakeSearchEnv:
    """Scripted Search-like env: records replay, then serves a fixed episode."""

    def __init__(self, steps_to_answer=1):
        self.replayed = {}
        self.steps_to_answer = steps_to_answer
        self._counts = {}

    def start(self, spec):
        self.replayed[spec.branch_traj_uid] = list(spec.replay_actions)
        self._counts[spec.branch_traj_uid] = 0
        return f"obs@branch:{spec.branch_step_num}"

    def step(self, spec, action):
        self._counts[spec.branch_traj_uid] += 1
        n = self._counts[spec.branch_traj_uid]
        if "<answer>" in action or n >= self.steps_to_answer:
            return "terminal", 1.0, True, {"is_action_valid": True}
        return f"obs@{n}", 0.0, False, {"is_action_valid": True}


def scripted_generate(obs_text, prefix_text):
    if prefix_text is not None:
        return prefix_text + "need the official name</think><search>official name</search>"
    return "<think><verify>result matches all constraints</verify></think><answer>X</answer>"


def _one_spec(**kw):
    defaults = dict(
        parent_traj_uid="t0", parent_uid="g0", sample_id=0, rollout_id=0,
        branch_step_num=1, replay_actions=["<search>old</search>"],
        branch_observation="obs-err", prefix_reward=0.25,
    )
    defaults.update(kw)
    return m.BranchSpec(**defaults)


def test_branch_rollout_replays_env_and_injects_prefix():
    spec = _one_spec()
    env = FakeSearchEnv(steps_to_answer=2)
    trajs = m.run_branch_rollout(
        [spec], {0: PREFIX},
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    assert env.replayed[spec.branch_traj_uid] == ["<search>old</search>"]
    assert len(trajs) == 1
    rows = trajs[0].rows
    # step 1: prefixed continuation; step 2: normal generation reaching <answer>
    assert len(rows) == 2
    assert rows[0]["branch_prefix_text"] == PREFIX
    assert rows[0]["response_text"].startswith(PREFIX)
    assert rows[1]["branch_prefix_text"] is None
    assert trajs[0].done


def test_branch_rows_uid_and_step_numbering():
    spec = _one_spec()
    env = FakeSearchEnv(steps_to_answer=2)
    trajs = m.run_branch_rollout(
        [spec], {0: PREFIX},
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    rows = trajs[0].rows
    for row in rows:
        assert row["uid"] == "g0"  # GRPO group inherited from parent
        assert row["traj_uid"] == spec.branch_traj_uid  # fresh trajectory id
        assert row["is_teacher_branch"] is True
        assert row["branch_parent_traj_uid"] == "t0"
    assert [r["step_num"] for r in rows] == [1, 2]
    assert rows[0]["step_id"] == "0_0_1"


def test_branch_episode_aggregates_include_parent_prefix():
    spec = _one_spec()  # branch at step 1, prefix_reward 0.25
    env = FakeSearchEnv(steps_to_answer=2)
    trajs = m.run_branch_rollout(
        [spec], {0: PREFIX},
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    traj = trajs[0]
    assert traj.episode_reward == pytest.approx(0.25 + 1.0)
    assert traj.episode_length == 1 + 2  # 1 parent step + 2 branch steps
    for row in traj.rows:
        assert row["episode_rewards"] == pytest.approx(1.25)
        assert row["episode_lengths"] == 3


def test_branch_rollout_respects_max_steps_budget():
    spec = _one_spec(branch_step_num=3)  # only 1 step left of a 4-step budget
    env = FakeSearchEnv(steps_to_answer=10)  # env never terminates on its own
    trajs = m.run_branch_rollout(
        [spec], {0: PREFIX},
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    assert len(trajs[0].rows) == 1
    assert not trajs[0].done


def test_branch_rollout_skips_specs_without_prefix():
    specs = [_one_spec(), _one_spec(parent_traj_uid="t1")]
    env = FakeSearchEnv()
    trajs = m.run_branch_rollout(
        specs, {1: PREFIX},  # spec 0 has no a_T (teacher failed) -> skipped
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    assert len(trajs) == 1
    assert trajs[0].spec.parent_traj_uid == "t1"


def test_branch_rollout_rejects_generate_dropping_prefix():
    spec = _one_spec()
    env = FakeSearchEnv()

    def bad_generate(obs_text, prefix_text):
        return "<think>fresh thinking</think><answer>X</answer>"  # lost the prefix

    with pytest.raises(ValueError):
        m.run_branch_rollout(
            [spec], {0: PREFIX},
            env_start=env.start, env_step=env.step,
            generate=bad_generate, max_steps=4,
        )


def test_branch_row_tag_stats_recorded():
    spec = _one_spec()
    env = FakeSearchEnv(steps_to_answer=2)
    trajs = m.run_branch_rollout(
        [spec], {0: PREFIX},
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    rows = trajs[0].rows
    assert rows[0]["tag_reflect_count"] == 1  # from the a_T prefix
    assert rows[1]["tag_verify_count"] == 1
    assert rows[1]["tag_final_answer"] is True


def test_summarize_branch_metrics():
    spec = _one_spec()
    env = FakeSearchEnv(steps_to_answer=2)
    trajs = m.run_branch_rollout(
        [spec], {0: PREFIX},
        env_start=env.start, env_step=env.step,
        generate=scripted_generate, max_steps=4,
    )
    metrics = m.summarize_branch_metrics(trajs)
    assert metrics["seed/teacher_branch/num_branches"] == 1.0
    assert metrics["seed/teacher_branch/done_ratio"] == 1.0
    assert metrics["seed/teacher_branch/mean_episode_reward"] == pytest.approx(1.25)
    empty = m.summarize_branch_metrics([])
    assert empty["seed/teacher_branch/num_branches"] == 0.0
