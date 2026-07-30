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
"""Tests for the pluggable branch-point conditions (branch_selectors.py)."""

import pytest

from agent_system.multi_turn_rollout.branch_selectors import (
    KL_ROW_KEY,
    AllStepsCondition,
    AndCondition,
    ErrorSignalCondition,
    KLGapCondition,
    LowRewardCondition,
    condition_from_config,
    parse_selector_spec,
)
from agent_system.multi_turn_rollout.teacher_branch import select_branch_specs


def _row(traj, step, *, uid="task0", reward=0.0, error=False, kl=None, active=True):
    row = {
        "traj_uid": traj,
        "uid": uid,
        "sample_id": 0,
        "rollout_id": 0,
        "step_num": step,
        "rewards": reward,
        "obs_text": f"obs {traj} {step}",
        "response_text": f"resp {traj} {step}",
        "tag_error_signal": error,
        "active_masks": active,
    }
    if kl is not None:
        row[KL_ROW_KEY] = kl
    return row


# ---------------------------------------------------------------- conditions

def test_error_signal_condition():
    rows = [_row("t0", 0, error=False), _row("t0", 1, error=True)]
    decision = ErrorSignalCondition().evaluate(rows)
    assert decision.eligible == [False, True]


def test_low_reward_condition_uses_episode_return():
    rows = [
        _row("fail", 0, reward=0.0),
        _row("fail", 1, reward=0.0),
        _row("win", 0, reward=0.0),
        _row("win", 1, reward=1.0),
    ]
    decision = LowRewardCondition(threshold=0.0).evaluate(rows)
    assert decision.eligible == [True, True, False, False]
    # priority = deficit below the threshold, same for all steps of the episode
    assert decision.priority[0] == pytest.approx(0.0)


def test_low_reward_threshold_and_priority():
    rows = [_row("a", 0, reward=-2.0), _row("b", 0, reward=0.4)]
    decision = LowRewardCondition(threshold=0.5).evaluate(rows)
    assert decision.eligible == [True, True]
    assert decision.priority[0] > decision.priority[1]  # worse episode -> higher priority


def test_kl_gap_condition_requires_scores():
    rows = [_row("t", 0, kl=1.5), _row("t", 1), _row("t", 2, kl=0.2)]
    decision = KLGapCondition().evaluate(rows)
    assert decision.eligible == [True, False, True]  # unscored row is ineligible
    assert decision.priority[0] == pytest.approx(1.5)


def test_and_condition_composes():
    rows = [
        _row("fail", 0, error=True, kl=2.0),
        _row("fail", 1, error=False, kl=3.0),
        _row("win", 0, error=True, kl=5.0, reward=1.0),
    ]
    condition = AndCondition([ErrorSignalCondition(), LowRewardCondition(), KLGapCondition()])
    decision = condition.evaluate(rows)
    assert decision.eligible == [True, False, False]
    assert condition.requires_kl
    cheap = condition.without_kl()
    assert cheap is not None and not cheap.requires_kl


# ------------------------------------------------------------------- parsing

def test_parse_selector_spec_single_and_combo():
    assert isinstance(parse_selector_spec("error_signal"), ErrorSignalCondition)
    combo = parse_selector_spec("low_reward(threshold=0.5)&kl_gap")
    assert isinstance(combo, AndCondition)
    assert combo.requires_kl
    assert combo.parts[0].threshold == pytest.approx(0.5)


def test_parse_selector_spec_rejects_unknown():
    with pytest.raises(ValueError, match="unknown branch selector"):
        parse_selector_spec("bogus")


def test_condition_from_config_fallbacks():
    assert isinstance(condition_from_config(None, True), ErrorSignalCondition)
    assert isinstance(condition_from_config(None, False), AllStepsCondition)
    assert isinstance(condition_from_config("kl_gap", True), KLGapCondition)


# ------------------------------------------------- select_branch_specs wiring

def _get_action(row):
    return row["response_text"]


def test_select_prefers_highest_priority_step_within_traj():
    rows = [
        _row("t0", 0, kl=0.1),
        _row("t0", 1, kl=3.0),
        _row("t0", 2, kl=1.0),
    ]
    specs = select_branch_specs(
        rows,
        get_action_text=_get_action,
        max_branches_per_traj=1,
        condition=KLGapCondition(),
    )
    assert len(specs) == 1
    assert specs[0].branch_step_num == 1  # the max-KL step
    assert specs[0].selector_priority == pytest.approx(3.0)


def test_select_global_cap_keeps_highest_priority():
    rows = [
        _row("t0", 0, kl=0.5),
        _row("t1", 0, kl=2.5),
        _row("t2", 0, kl=1.5),
    ]
    for i, row in enumerate(rows):
        row["sample_id"] = i
    specs = select_branch_specs(
        rows,
        get_action_text=_get_action,
        max_branches_per_traj=1,
        max_total_branches=2,
        condition=KLGapCondition(),
    )
    assert len(specs) == 2
    assert sorted(s.selector_priority for s in specs) == pytest.approx([1.5, 2.5])


def test_select_legacy_error_signal_still_works():
    rows = [_row("t0", 0, error=False), _row("t0", 1, error=True)]
    specs = select_branch_specs(
        rows,
        get_action_text=_get_action,
        require_error_signal=True,
    )
    assert len(specs) == 1 and specs[0].branch_step_num == 1


def test_select_skips_inactive_rows():
    rows = [_row("t0", 0, error=True, active=False), _row("t0", 1, error=True)]
    specs = select_branch_specs(rows, get_action_text=_get_action, require_error_signal=True)
    assert len(specs) == 1 and specs[0].branch_step_num == 1
