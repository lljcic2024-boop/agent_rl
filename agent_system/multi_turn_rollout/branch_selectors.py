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
Pluggable branch-point conditions for 改造点 4 (teacher-prefix branching).

"什么时候调用老师" is a wrapped condition, chosen by config instead of being
hard-coded. A condition looks at all step rows of the batch and returns, per
row, (a) whether the step is eligible as a branch point and (b) a priority
(higher = branch here first) used to rank candidates within a trajectory and
across the global cap.

Available conditions (spec strings, combined with `&` for AND):

- ``error_signal``      the observation carried an anomaly (Phase 2a selector)
- ``low_reward``        the episode return of the row's trajectory is at or
                        below a threshold (default 0.0, i.e. failed episodes);
                        priority = how far below the threshold
- ``kl_gap``            the per-token normalized KL(student || teacher) of the
                        step, read from the row key ``kl_to_teacher`` which the
                        runner fills by scoring the batch with the external
                        teacher; priority = the KL value. Rows without a score
                        are ineligible.
- ``any``               every active step is eligible (priority 0)

Examples: ``low_reward&kl_gap`` (failed episodes, branch at the step where the
student diverges most from the teacher), ``low_reward(threshold=0.5)``,
``error_signal&low_reward``.

Pure python, no torch/verl imports — unit-testable on a laptop.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

KL_ROW_KEY = "kl_to_teacher"

_SPEC_PART_RE = re.compile(r"^(?P<name>[a-z_]+)(?:\((?P<params>[^)]*)\))?$")


def _row_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _row_bool(row: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    try:
        return bool(value)
    except Exception:
        return default


@dataclass
class SelectorDecision:
    """Per-row verdict of a condition over the whole batch's rows."""

    eligible: List[bool]
    priority: List[float]


class BranchCondition:
    """Base class. Subclasses look at ALL rows so they can build per-trajectory
    aggregates (e.g. episode return) before judging individual steps."""

    name: str = "base"
    requires_kl: bool = False

    def evaluate(self, rows: Sequence[Dict[str, Any]]) -> SelectorDecision:
        raise NotImplementedError

    @property
    def components(self) -> List["BranchCondition"]:
        return [self]

    def describe(self) -> str:
        return self.name


class AllStepsCondition(BranchCondition):
    name = "any"

    def evaluate(self, rows):
        n = len(rows)
        return SelectorDecision(eligible=[True] * n, priority=[0.0] * n)


class ErrorSignalCondition(BranchCondition):
    """Phase 2a: steps whose observation carried an anomaly signal."""

    name = "error_signal"

    def evaluate(self, rows):
        eligible = [_row_bool(row, "tag_error_signal") for row in rows]
        return SelectorDecision(eligible=eligible, priority=[0.0] * len(rows))


class LowRewardCondition(BranchCondition):
    """Steps of trajectories whose episode return <= threshold.

    Search QA rewards are terminal, so "当前 reward 小" at step granularity is
    mostly 0-vs-0; the informative unit is the episode. Priority is the reward
    deficit (threshold - episode_return): the worse the episode, the earlier
    it is branched.
    """

    name = "low_reward"

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def evaluate(self, rows):
        episode_return: Dict[str, float] = {}
        for row in rows:
            traj = str(row.get("traj_uid"))
            episode_return[traj] = episode_return.get(traj, 0.0) + _row_float(row, "rewards")
        eligible: List[bool] = []
        priority: List[float] = []
        for row in rows:
            ret = episode_return.get(str(row.get("traj_uid")), 0.0)
            ok = ret <= self.threshold
            eligible.append(ok)
            priority.append(self.threshold - ret if ok else 0.0)
        return SelectorDecision(eligible=eligible, priority=priority)

    def describe(self) -> str:
        return f"low_reward(threshold={self.threshold:g})"


class KLGapCondition(BranchCondition):
    """Steps where the student diverges most from the teacher.

    Reads ``row[KL_ROW_KEY]``: per-token normalized KL(student || teacher)
    (or the cross-entropy proxy -mean teacher logprob when student rollout
    log-probs are unavailable), injected by the runner before selection.
    Rows without a finite score are ineligible — so if KL scoring failed the
    condition selects nothing rather than selecting blindly.
    """

    name = "kl_gap"
    requires_kl = True

    def __init__(self, min_kl: float = 0.0):
        self.min_kl = float(min_kl)

    def evaluate(self, rows):
        eligible: List[bool] = []
        priority: List[float] = []
        for row in rows:
            value = row.get(KL_ROW_KEY)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = math.nan
            ok = math.isfinite(value) and value >= self.min_kl
            eligible.append(ok)
            priority.append(value if ok else 0.0)
        return SelectorDecision(eligible=eligible, priority=priority)

    def describe(self) -> str:
        return f"kl_gap(min_kl={self.min_kl:g})" if self.min_kl else "kl_gap"


class AndCondition(BranchCondition):
    """AND of eligibilities; priority is the sum of component priorities."""

    name = "and"

    def __init__(self, parts: Sequence[BranchCondition]):
        if not parts:
            raise ValueError("AndCondition needs at least one component")
        self.parts = list(parts)
        self.requires_kl = any(p.requires_kl for p in self.parts)

    @property
    def components(self) -> List[BranchCondition]:
        return list(self.parts)

    def evaluate(self, rows):
        decisions = [p.evaluate(rows) for p in self.parts]
        n = len(rows)
        eligible = [all(d.eligible[i] for d in decisions) for i in range(n)]
        priority = [
            sum(d.priority[i] for d in decisions) if eligible[i] else 0.0 for i in range(n)
        ]
        return SelectorDecision(eligible=eligible, priority=priority)

    def describe(self) -> str:
        return "&".join(p.describe() for p in self.parts)

    def without_kl(self) -> Optional[BranchCondition]:
        """The cheap (non-KL) part, used to pre-filter rows before KL scoring."""
        cheap = [p for p in self.parts if not p.requires_kl]
        if not cheap:
            return None
        if len(cheap) == 1:
            return cheap[0]
        return AndCondition(cheap)


_CONDITION_REGISTRY = {
    "any": AllStepsCondition,
    "error_signal": ErrorSignalCondition,
    "low_reward": LowRewardCondition,
    "kl_gap": KLGapCondition,
}


def _parse_params(raw: Optional[str]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if not raw:
        return params
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"selector parameter {item!r} must be name=value")
        key, value = item.split("=", 1)
        params[key.strip()] = float(value.strip())
    return params


def parse_selector_spec(spec: str) -> BranchCondition:
    """Parse a selector spec string like ``low_reward(threshold=0.5)&kl_gap``."""
    parts: List[BranchCondition] = []
    for raw_part in str(spec).split("&"):
        raw_part = raw_part.strip()
        if not raw_part:
            continue
        match = _SPEC_PART_RE.match(raw_part)
        if match is None:
            raise ValueError(f"cannot parse selector spec part {raw_part!r}")
        name = match.group("name")
        if name not in _CONDITION_REGISTRY:
            raise ValueError(
                f"unknown branch selector {name!r}; known: {sorted(_CONDITION_REGISTRY)}"
            )
        params = _parse_params(match.group("params"))
        parts.append(_CONDITION_REGISTRY[name](**params))
    if not parts:
        raise ValueError(f"selector spec {spec!r} is empty")
    if len(parts) == 1:
        return parts[0]
    return AndCondition(parts)


def condition_from_config(
    selector_spec: Optional[str],
    require_error_signal: bool,
) -> BranchCondition:
    """Resolve the active condition.

    ``selector_spec`` (algorithm.seed.teacher_branch.selector) wins when set;
    otherwise fall back to the legacy ``require_error_signal`` boolean.
    """
    if selector_spec:
        return parse_selector_spec(str(selector_spec))
    return ErrorSignalCondition() if require_error_signal else AllStepsCondition()
