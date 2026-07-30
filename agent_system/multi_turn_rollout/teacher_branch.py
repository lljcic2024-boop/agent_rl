# Copyright 2025
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
改造点 4: teacher-prefix continuation branches, offline-testable orchestration.

This module owns the text-level orchestration of the teacher-prefix rollout:

1. `select_branch_specs`  — pick (trajectory, step) branch points from the main
   rollout rows (Phase 2a: error-signal steps, same signal as `step_selector`).
2. `build_teacher_prefix_messages` / `parse_teacher_prefix_response` — request
   a thinking prefix a_T from the external teacher (OpenAI-compatible chat
   endpoint, same server as the analyzer) and quality-gate the reply.
3. `run_branch_rollout` — replay the environment to the branch point, let the
   student continue from a_T to termination, and emit training-row dicts.

Everything here is text-level and dependency-free (no verl / ray / omegaconf),
so the full flow is unit-testable on a laptop with fake env/generate hooks.
Token-level tensors (prefix generation batch, teacher_token_mask / loss_mask)
are assembled cluster-side from these rows via
`agent_system.multi_turn_rollout.teacher_prefix`.

GRPO grouping contract:
- branch rows inherit the parent trajectory's `uid` (episode-group id), so the
  branch episode's terminal reward is normalized within the same task group as
  its siblings;
- branch rows get a fresh `traj_uid`, so episode reassembly (`build_episode_
  records`, step_rewards discounting) treats the branch as its own trajectory;
- the a_T prefix text is recorded on the first branch row (`branch_prefix_text`)
  and MUST be excluded from the PG loss there: the trainer tokenizes it and
  builds `teacher_token_mask` / `loss_mask` with `assemble_branch_responses`.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Keep in sync with TrajectoryCollector (rollout_loop.py); duplicated here so
# this module stays importable without verl.
TAG_FUNCTION_NAMES = ("plan", "verify", "reflect", "backtrack")
_TAG_ERROR_SIGNAL_RE = re.compile(
    r"<information>\s*</information>|no (?:relevant )?results?|nothing happens|invalid action",
    re.IGNORECASE,
)
_ACTION_TAG_RE = re.compile(r"</?(?:search|answer|action)>", re.IGNORECASE)
_THINK_OPEN = "<think>"


# ---------------------------------------------------------------------------
# 1. Branch point selection
# ---------------------------------------------------------------------------

@dataclass
class BranchSpec:
    """One teacher-prefix branch point inside a main-rollout trajectory."""

    parent_traj_uid: str
    parent_uid: str  # GRPO episode-group id, inherited by the branch
    sample_id: int
    rollout_id: int
    branch_step_num: int  # step index (in the parent trajectory) being replaced
    replay_actions: List[str]  # decoded actions of parent steps < branch_step_num
    branch_observation: str  # recorded observation text at the branch step
    env_kwargs: Any = None  # opaque env reset payload (e.g. the Search task)
    prefix_reward: float = 0.0  # reward accrued on parent steps < branch step
    selector_priority: float = 0.0  # why this point was picked (condition priority)
    branch_traj_uid: str = field(default_factory=lambda: str(uuid.uuid4()))


def _row_bool(row: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    try:
        return bool(value)
    except Exception:
        return default


def select_branch_specs(
    step_rows: Sequence[Dict[str, Any]],
    *,
    get_action_text: Callable[[Dict[str, Any]], str],
    get_env_kwargs: Optional[Callable[[Dict[str, Any]], Any]] = None,
    max_branches_per_traj: int = 1,
    max_total_branches: Optional[int] = None,
    require_error_signal: bool = True,
    condition: Optional[Any] = None,  # BranchCondition; overrides require_error_signal
    rng: Optional[Any] = None,  # random.Random-like; None -> deterministic ranking
) -> List[BranchSpec]:
    """
    Pick branch points from main-rollout step rows.

    Args:
        step_rows: per-step row dicts as produced by the rollout loop /
            gather_rollout_data. Required keys: traj_uid, uid, sample_id,
            rollout_id, step_num, active_masks; optional: tag_error_signal,
            rewards, kl_to_teacher.
        get_action_text: decodes a row into the env action text (used to
            replay the trajectory up to the branch point).
        get_env_kwargs: extracts the env reset payload for the row's task.
        max_branches_per_traj: at most this many branch points per trajectory.
        max_total_branches: global cap across the batch (None = no cap).
        require_error_signal: legacy Phase 2a selector, used only when
            `condition` is None. True -> only error-signal steps; False -> all.
        condition: a `branch_selectors.BranchCondition`; when given it decides
            eligibility AND ranking (priority high -> picked first, both within
            a trajectory and under the global cap).
        rng: random.Random-like sampler for candidate choice; None ranks by
            (priority desc, step asc) deterministically.

    Returns:
        BranchSpecs ordered by (sample_id, rollout_id, branch_step_num), each
        carrying its `selector_priority`.
    """
    if condition is None:
        from agent_system.multi_turn_rollout.branch_selectors import condition_from_config

        condition = condition_from_config(None, require_error_signal)

    active_rows = [row for row in step_rows if _row_bool(row, "active_masks", True)]
    decision = condition.evaluate(active_rows)

    by_traj: Dict[str, List[Tuple[Dict[str, Any], bool, float]]] = {}
    for row, ok, prio in zip(active_rows, decision.eligible, decision.priority):
        by_traj.setdefault(str(row["traj_uid"]), []).append((row, ok, prio))

    specs: List[BranchSpec] = []
    for traj_uid, entries in by_traj.items():
        entries = sorted(entries, key=lambda e: int(e[0]["step_num"]))
        candidates = [idx for idx, (_, ok, _) in enumerate(entries) if ok]
        if not candidates:
            continue
        if len(candidates) > max_branches_per_traj:
            if rng is not None:
                candidates = sorted(rng.sample(candidates, max_branches_per_traj))
            else:
                # priority desc, then earliest step; stable for equal priorities
                candidates = sorted(candidates, key=lambda idx: (-entries[idx][2], idx))
                candidates = sorted(candidates[:max_branches_per_traj])

        for idx in candidates:
            row, _, prio = entries[idx]
            prior_rows = [e[0] for e in entries[:idx]]
            replay_actions = [get_action_text(r) for r in prior_rows]
            prefix_reward = 0.0
            for r in prior_rows:
                try:
                    prefix_reward += float(r.get("rewards", 0.0))
                except (TypeError, ValueError):
                    pass
            specs.append(
                BranchSpec(
                    parent_traj_uid=traj_uid,
                    parent_uid=str(row["uid"]),
                    sample_id=int(row["sample_id"]),
                    rollout_id=int(row["rollout_id"]),
                    branch_step_num=int(row["step_num"]),
                    replay_actions=replay_actions,
                    branch_observation=str(row.get("obs_text", "")),
                    env_kwargs=None if get_env_kwargs is None else get_env_kwargs(row),
                    prefix_reward=prefix_reward,
                    selector_priority=float(prio),
                )
            )

    specs.sort(key=lambda s: (s.sample_id, s.rollout_id, s.branch_step_num))
    if max_total_branches is not None and len(specs) > max_total_branches:
        if rng is not None:
            specs = sorted(
                rng.sample(specs, max_total_branches),
                key=lambda s: (s.sample_id, s.rollout_id, s.branch_step_num),
            )
        else:
            # keep the highest-priority branch points under the global cap
            kept = sorted(
                specs,
                key=lambda s: (-s.selector_priority, s.sample_id, s.rollout_id, s.branch_step_num),
            )[:max_total_branches]
            specs = sorted(kept, key=lambda s: (s.sample_id, s.rollout_id, s.branch_step_num))
    return specs


# ---------------------------------------------------------------------------
# 2. a_T generation (external teacher, OpenAI-compatible chat endpoint)
# ---------------------------------------------------------------------------

_DEFAULT_TAG_INSTRUCTION_FALLBACK = (
    "Organize the thinking with the four function tags <plan>, <verify>, "
    "<reflect>, <backtrack>; every segment must belong to exactly one tag."
)


def _tag_think_instruction() -> str:
    try:
        from agent_system.environments.prompts.search import SEARCH_TAG_THINK_INSTRUCTION

        return SEARCH_TAG_THINK_INSTRUCTION
    except Exception:
        return _DEFAULT_TAG_INSTRUCTION_FALLBACK


TEACHER_PREFIX_SYSTEM_PROMPT = """You are an expert teacher demonstrating high-quality agent reasoning.
You will be shown the exact observation an agent sees at one step of a multi-turn task. The agent's previous attempt at this step went in an unproductive direction.
Write ONLY the beginning of an ideal reasoning process for this step (a thinking prefix). A student model will continue from your prefix, finish the reasoning, and choose the action.

{tag_instruction}

Hard requirements for your reply:
- Output ONLY the thinking prefix: one or two complete tagged segments (e.g. a <reflect> followed by a <backtrack>).
- Do NOT close the reasoning: no </think> tag anywhere.
- Do NOT choose or mention a concrete action markup: no <search>, <answer> or <action> tags.
- Do not add any commentary outside the tagged segments."""


def build_teacher_prefix_messages(
    branch_observation: str,
    *,
    tag_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build chat messages asking the teacher to write the a_T thinking prefix."""
    instruction = tag_instruction if tag_instruction is not None else _tag_think_instruction()
    system = TEACHER_PREFIX_SYSTEM_PROMPT.format(tag_instruction=instruction)
    user = (
        "The agent's current observation (task, history, and instructions) is:\n\n"
        f"{branch_observation}\n\n"
        "Write the thinking prefix now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


TEACHER_FULL_STEP_SYSTEM_PROMPT = """You are an expert teacher demonstrating high-quality agent behaviour.
You will be shown the exact observation an agent sees at one step of a multi-turn task. The agent's previous attempt at this step went in an unproductive direction.
Write the COMPLETE ideal response for this single step: the full reasoning followed by exactly one action. A student model will take over from the NEXT step onward.

{tag_instruction}

Hard requirements for your reply:
- Start with <think>, put the tagged reasoning segments inside, and close with </think>.
- After </think>, output exactly ONE action: either <search>query</search> or <answer>final answer</answer>.
- Do not add any commentary outside the thinking block and the single action."""


def build_teacher_full_step_messages(
    branch_observation: str,
    *,
    tag_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build chat messages asking the teacher to write one complete step."""
    instruction = tag_instruction if tag_instruction is not None else _tag_think_instruction()
    system = TEACHER_FULL_STEP_SYSTEM_PROMPT.format(tag_instruction=instruction)
    user = (
        "The agent's current observation (task, history, and instructions) is:\n\n"
        f"{branch_observation}\n\n"
        "Write the complete response for this step now (reasoning + one action)."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class TeacherPrefixParseError(ValueError):
    """The teacher reply violates the a_T prefix contract."""


def parse_teacher_prefix_response(
    text: str,
    *,
    min_chars: int = 20,
    max_chars: int = 2000,
) -> str:
    """
    Quality-gate the teacher's reply and normalize it into the a_T prefix text.

    The returned prefix is `<think>` + tagged segments, intentionally left
    unclosed so the student's continuation stays inside the thinking block.

    Raises:
        TeacherPrefixParseError: closed thinking, action markup, missing
            function tags, or out-of-bounds length.
    """
    content = str(text or "").strip()
    # Tolerate teachers that wrap the reply in <think> ... (</think> is rejected below).
    if content.lower().startswith(_THINK_OPEN):
        content = content[len(_THINK_OPEN):].strip()

    if "</think>" in content.lower():
        raise TeacherPrefixParseError("teacher prefix must not close the thinking block (</think> found)")
    if _ACTION_TAG_RE.search(content):
        raise TeacherPrefixParseError("teacher prefix must not contain action markup (<search>/<answer>/<action>)")

    open_tags = re.findall(r"<(plan|verify|reflect|backtrack)>", content, re.IGNORECASE)
    if not open_tags:
        raise TeacherPrefixParseError("teacher prefix contains no function tag segment")
    for tag in set(t.lower() for t in open_tags):
        if content.lower().count(f"<{tag}>") != content.lower().count(f"</{tag}>"):
            raise TeacherPrefixParseError(f"unbalanced <{tag}> tags in teacher prefix")
    stripped = re.sub(r"</?(?:plan|verify|reflect|backtrack)>", "", content, flags=re.IGNORECASE)
    if not (min_chars <= len(stripped.strip()) <= max_chars):
        raise TeacherPrefixParseError(
            f"teacher prefix length {len(stripped.strip())} outside [{min_chars}, {max_chars}]"
        )

    return f"{_THINK_OPEN}\n{content}\n"


_FULL_STEP_ACTION_RE = re.compile(
    r"<(search|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)


def parse_teacher_full_step_response(
    text: str,
    *,
    min_chars: int = 20,
    max_chars: int = 6000,
) -> str:
    """
    Quality-gate a full-step teacher reply: `<think>` tagged reasoning
    `</think>` followed by exactly one `<search>`/`<answer>` action.

    Returns the normalized complete response text (think block + action).

    Raises:
        TeacherPrefixParseError: missing/unclosed think block, zero or
            multiple actions, missing function tags, out-of-bounds length.
    """
    content = str(text or "").strip()
    if not content.lower().startswith(_THINK_OPEN):
        content = f"{_THINK_OPEN}\n{content}"
    if content.lower().count("</think>") != 1:
        raise TeacherPrefixParseError("full-step reply must close the thinking block exactly once")
    think_part, after = re.split(r"</think>", content, maxsplit=1, flags=re.IGNORECASE)
    think_body = think_part[len(_THINK_OPEN):].strip() if think_part.lower().startswith(_THINK_OPEN) else think_part.strip()

    open_tags = re.findall(r"<(plan|verify|reflect|backtrack)>", think_body, re.IGNORECASE)
    if not open_tags:
        raise TeacherPrefixParseError("full-step reply contains no function tag segment")
    for tag in set(t.lower() for t in open_tags):
        if think_body.lower().count(f"<{tag}>") != think_body.lower().count(f"</{tag}>"):
            raise TeacherPrefixParseError(f"unbalanced <{tag}> tags in full-step reply")

    actions = _FULL_STEP_ACTION_RE.findall(after)
    if len(actions) != 1:
        raise TeacherPrefixParseError(
            f"full-step reply must contain exactly one <search>/<answer> action, found {len(actions)}"
        )
    action_tag, action_body = actions[0]
    action_body = action_body.strip()
    if not action_body:
        raise TeacherPrefixParseError("full-step action is empty")
    leftover = _FULL_STEP_ACTION_RE.sub("", after).strip()
    if leftover:
        raise TeacherPrefixParseError(f"unexpected text outside the action: {leftover[:80]!r}")

    stripped = re.sub(r"</?(?:plan|verify|reflect|backtrack)>", "", think_body, flags=re.IGNORECASE)
    if not (min_chars <= len(stripped.strip()) <= max_chars):
        raise TeacherPrefixParseError(
            f"full-step reasoning length {len(stripped.strip())} outside [{min_chars}, {max_chars}]"
        )

    tag = action_tag.lower()
    return f"{_THINK_OPEN}\n{think_body}\n</think>\n<{tag}>{action_body}</{tag}>"


def _generate_one_prefix(
    spec: BranchSpec,
    chat_fn: Callable[[List[Dict[str, str]]], str],
    *,
    tag_instruction: Optional[str],
    max_retries: int,
    mode: str = "think_prefix",
) -> Tuple[Optional[str], str]:
    """Request a_T for one spec. Returns (prefix or None, error message)."""
    if mode == "full_step":
        messages = build_teacher_full_step_messages(
            spec.branch_observation, tag_instruction=tag_instruction
        )
        parse = parse_teacher_full_step_response
    else:
        messages = build_teacher_prefix_messages(
            spec.branch_observation, tag_instruction=tag_instruction
        )
        parse = parse_teacher_prefix_response
    last_error = "no attempt made"
    for _ in range(max(1, int(max_retries))):
        try:
            reply = chat_fn(messages)
            return parse(reply), ""
        except TeacherPrefixParseError as exc:
            last_error = str(exc)
        except Exception as exc:  # transport errors: retry, then drop
            last_error = f"teacher call failed: {exc}"
    return None, last_error


def generate_teacher_prefixes(
    specs: Sequence[BranchSpec],
    chat_fn: Callable[[List[Dict[str, str]]], str],
    *,
    tag_instruction: Optional[str] = None,
    max_retries: int = 2,
    max_workers: int = 1,
    mode: str = "think_prefix",
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Request a_T for every spec through `chat_fn` (an OpenAI-compatible chat
    call returning the assistant message content, e.g. the analyzer client).

    `mode` selects the contract: "think_prefix" (default) asks for an unclosed
    thinking prefix the student continues within the step; "full_step" asks for
    the complete response of the branch step (closed thinking + one action) —
    the student then takes over from the NEXT step.

    `max_workers > 1` issues the teacher calls from a thread pool; the teacher
    is a remote HTTP server, so this is the difference between one training step
    waiting seconds and waiting minutes. `chat_fn` must then be thread-safe
    (the OpenAI client is).

    Returns:
        (prefixes, failures): spec index -> prefix text, spec index -> error.
        Failed specs should simply be dropped (no branch for that point).
    """
    if mode not in ("think_prefix", "full_step"):
        raise ValueError(f"unknown teacher prefix mode {mode!r}")
    prefixes: Dict[int, str] = {}
    failures: Dict[int, str] = {}

    def record(index: int, result: Tuple[Optional[str], str]) -> None:
        prefix, error = result
        if prefix is None:
            failures[index] = error
        else:
            prefixes[index] = prefix

    workers = max(1, int(max_workers))
    if workers == 1 or len(specs) <= 1:
        for i, spec in enumerate(specs):
            record(
                i,
                _generate_one_prefix(
                    spec,
                    chat_fn,
                    tag_instruction=tag_instruction,
                    max_retries=max_retries,
                    mode=mode,
                ),
            )
        return prefixes, failures

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        futures = {
            pool.submit(
                _generate_one_prefix,
                spec,
                chat_fn,
                tag_instruction=tag_instruction,
                max_retries=max_retries,
                mode=mode,
            ): i
            for i, spec in enumerate(specs)
        }
        for future, index in futures.items():
            try:
                record(index, future.result())
            except Exception as exc:  # pragma: no cover - defensive
                failures[index] = f"teacher call failed: {exc}"
    return prefixes, failures


# ---------------------------------------------------------------------------
# 3. Branch continuation rollout (env replay + student continuation)
# ---------------------------------------------------------------------------

def _count_tag_segments(text: str, tag: str) -> int:
    return len(re.findall(rf"<{re.escape(tag)}>", str(text or ""), re.IGNORECASE))


def _make_branch_row(
    spec: BranchSpec,
    *,
    step_num: int,
    obs_text: str,
    response_text: str,
    reward: float,
    done: bool,
    is_action_valid: bool,
    prev_action_invalid: bool,
    prefix_text: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "uid": spec.parent_uid,
        "traj_uid": spec.branch_traj_uid,
        "sample_id": spec.sample_id,
        "rollout_id": spec.rollout_id,
        "step_num": step_num,
        "step_id": f"{spec.sample_id}_{spec.rollout_id}_{step_num}",
        "obs_text": obs_text,
        "response_text": response_text,
        "rewards": float(reward),
        "active_masks": True,
        "is_action_valid": bool(is_action_valid),
        "is_teacher_branch": True,
        "branch_parent_traj_uid": spec.parent_traj_uid,
        "branch_step_num": spec.branch_step_num,
        "branch_prefix_text": prefix_text,  # non-None only on the branch's first row
        "tag_final_answer": ("<answer>" in str(response_text).lower()) or bool(done),
        "tag_error_signal": bool(prev_action_invalid)
        or _TAG_ERROR_SIGNAL_RE.search(str(obs_text)) is not None,
    }
    for tag_name in TAG_FUNCTION_NAMES:
        row[f"tag_{tag_name}_count"] = _count_tag_segments(response_text, tag_name)
    return row


@dataclass
class BranchTrajectory:
    """One completed teacher-prefix branch: rows + episode aggregates."""

    spec: BranchSpec
    rows: List[Dict[str, Any]]
    episode_reward: float  # prefix (parent) reward + branch continuation reward
    episode_length: int  # parent steps before branch + branch steps
    done: bool


def run_branch_rollout(
    specs: Sequence[BranchSpec],
    prefix_texts: Dict[int, str],
    *,
    env_start: Callable[[BranchSpec], str],
    env_step: Callable[[BranchSpec, str], Tuple[str, float, bool, Dict[str, Any]]],
    generate: Callable[[str, Optional[str]], str],
    max_steps: int,
) -> List[BranchTrajectory]:
    """
    Roll every branch from its branch point to termination.

    The env and the policy are injected, which keeps this loop testable
    offline and lets the cluster glue back it with the real pieces:
    - `env_start(spec)`: reset the env for the spec's task and REPLAY
      `spec.replay_actions`; return the observation text at the branch step.
      (Search: retrieval is stateless, replay = re-run the recorded queries;
      ALFWorld: snapshot restore or action replay.)
    - `env_step(spec, action_text)` -> (next_obs, reward, done, info).
    - `generate(obs_text, prefix_text)` -> FULL response text for the step;
      `prefix_text` is non-None only for the branch step, where generation
      must continue from a_T (`build_prefix_generation_batch` cluster-side)
      and the returned text must include the prefix.

    Steps are numbered continuing the parent trajectory (branch_step_num,
    branch_step_num + 1, ...) so step-position-aware consumers stay coherent.
    """
    trajectories: List[BranchTrajectory] = []
    for i, spec in enumerate(specs):
        prefix_text = prefix_texts.get(i)
        if prefix_text is None:
            continue

        obs_text = env_start(spec)
        rows: List[Dict[str, Any]] = []
        branch_reward = 0.0
        prev_action_invalid = False
        done = False
        step_num = spec.branch_step_num
        steps_taken = 0
        remaining_steps = max(0, int(max_steps) - spec.branch_step_num)

        while steps_taken < remaining_steps:
            step_prefix = prefix_text if steps_taken == 0 else None
            response_text = generate(obs_text, step_prefix)
            if step_prefix is not None and not str(response_text).startswith(step_prefix):
                raise ValueError(
                    "generate() must return the full response including the a_T prefix "
                    "for the branch step"
                )
            next_obs, reward, done, info = env_step(spec, response_text)
            info = info or {}
            is_action_valid = bool(info.get("is_action_valid", True))
            rows.append(
                _make_branch_row(
                    spec,
                    step_num=step_num,
                    obs_text=obs_text,
                    response_text=response_text,
                    reward=float(reward),
                    done=bool(done),
                    is_action_valid=is_action_valid,
                    prev_action_invalid=prev_action_invalid,
                    prefix_text=step_prefix,
                )
            )
            branch_reward += float(reward)
            prev_action_invalid = not is_action_valid
            obs_text = next_obs
            step_num += 1
            steps_taken += 1
            if done:
                break

        for row in rows:
            row["episode_rewards"] = spec.prefix_reward + branch_reward
            row["episode_lengths"] = spec.branch_step_num + steps_taken

        trajectories.append(
            BranchTrajectory(
                spec=spec,
                rows=rows,
                episode_reward=spec.prefix_reward + branch_reward,
                episode_length=spec.branch_step_num + steps_taken,
                done=bool(done),
            )
        )
    return trajectories


def bucket_specs_by_replay_length(
    specs: Sequence[BranchSpec],
    prefix_texts: Dict[int, str],
) -> List[List[int]]:
    """
    Group spec indices by replay length so each bucket can be driven in lockstep.

    All specs in one bucket need the same number of replayed parent actions,
    which is what lets a single batched environment replay them together and
    then continue every branch with one `generate_batch` call per step.
    Specs without a teacher prefix are dropped (no branch for that point).

    Returns:
        Buckets ordered by replay length; indices inside a bucket keep the
        original `specs` order.
    """
    buckets: Dict[int, List[int]] = {}
    for i, spec in enumerate(specs):
        if prefix_texts.get(i) is None:
            continue
        buckets.setdefault(len(spec.replay_actions), []).append(i)
    return [buckets[key] for key in sorted(buckets)]


def run_branch_rollout_batched(
    specs: Sequence[BranchSpec],
    prefix_texts: Dict[int, str],
    *,
    env_start_batch: Callable[[List[BranchSpec]], List[str]],
    env_step_batch: Callable[
        [List[BranchSpec], List[str]],
        Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]],
    ],
    generate_batch: Callable[[List[str], List[Optional[str]], List[int], int], List[str]],
    max_steps: int,
    prefix_mode: str = "think_prefix",
    encode_teacher_step: Optional[
        Callable[[List[str], List[str], List[int], int], None]
    ] = None,
) -> List[BranchTrajectory]:
    """
    Batched counterpart of `run_branch_rollout`, one generate call per step.

    Same row/episode contract as `run_branch_rollout` (rows come from the shared
    `_make_branch_row`), but branches are processed in replay-length buckets and
    every step of a bucket issues a single batched policy call, which is what
    the cluster rollout worker needs to stay efficient.

    Injected hooks (bucket-wide, all lists aligned with the bucket order):
    - `env_start_batch(specs)`: reset the batched env for these specs and replay
      `spec.replay_actions`; return the observation text at each branch step.
    - `env_step_batch(specs, action_texts)` -> (next_obs, rewards, dones, infos);
      finished branches are still stepped (like the main rollout loop does) and
      simply masked out here.
    - `generate_batch(obs_texts, prefix_texts, spec_indices, step_num)` -> full
      response text per row. `prefix_texts[i]` is non-None only on the branch
      step, where the returned text must start with that prefix. `spec_indices`
      are indices into `specs`, so the cluster glue can stash the token-level
      tensors it produced for each row.

    `prefix_mode`:
    - "think_prefix" (default): a_T is an unclosed thinking prefix; the student
      continues within the branch step via `generate_batch`.
    - "full_step": a_T is the COMPLETE response of the branch step (teacher
      picked the action and finished the step). No policy call is made at the
      branch step; `encode_teacher_step(obs_texts, response_texts, spec_indices,
      step_num)` is invoked instead so the cluster glue can tokenize the
      teacher-written row (all its response tokens are teacher tokens). The
      student takes over from the next step onward.
    """
    if prefix_mode not in ("think_prefix", "full_step"):
        raise ValueError(f"unknown prefix_mode {prefix_mode!r}")
    if prefix_mode == "full_step" and encode_teacher_step is None:
        raise ValueError("prefix_mode='full_step' requires the encode_teacher_step hook")
    trajectories: List[BranchTrajectory] = []
    for bucket in bucket_specs_by_replay_length(specs, prefix_texts):
        bucket_specs = [specs[i] for i in bucket]
        obs_texts = list(env_start_batch(bucket_specs))
        if len(obs_texts) != len(bucket_specs):
            raise ValueError(
                f"env_start_batch returned {len(obs_texts)} observations for {len(bucket_specs)} specs"
            )

        branch_step_num = bucket_specs[0].branch_step_num
        remaining_steps = max(0, int(max_steps) - len(bucket_specs[0].replay_actions))
        n = len(bucket_specs)
        rows: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
        branch_rewards = [0.0] * n
        steps_taken = [0] * n
        prev_action_invalid = [False] * n
        active = [True] * n
        done_flags = [False] * n

        for offset in range(remaining_steps):
            step_num = branch_step_num + offset
            step_prefixes: List[Optional[str]] = [
                prefix_texts[bucket[i]] if offset == 0 else None for i in range(n)
            ]
            if prefix_mode == "full_step" and offset == 0:
                # The teacher already wrote the whole step: the prefix IS the
                # response. Tokenize/bookkeep it, but do not call the policy.
                response_texts = [str(step_prefixes[i]) for i in range(n)]
                encode_teacher_step(obs_texts, response_texts, list(bucket), step_num)
            else:
                response_texts = list(generate_batch(obs_texts, step_prefixes, bucket, step_num))
                if len(response_texts) != n:
                    raise ValueError(
                        f"generate_batch returned {len(response_texts)} responses for {n} rows"
                    )
                for i in range(n):
                    expected_prefix = step_prefixes[i]
                    if expected_prefix is not None and not str(response_texts[i]).startswith(expected_prefix):
                        raise ValueError(
                            "generate_batch() must return the full response including the a_T prefix "
                            "for the branch step"
                        )

            next_obs, rewards, dones, infos = env_step_batch(bucket_specs, response_texts)
            infos = list(infos or [{} for _ in range(n)])

            for i in range(n):
                if not active[i]:
                    continue
                info = infos[i] or {}
                is_action_valid = bool(info.get("is_action_valid", True))
                rows[i].append(
                    _make_branch_row(
                        bucket_specs[i],
                        step_num=step_num,
                        obs_text=obs_texts[i],
                        response_text=response_texts[i],
                        reward=float(rewards[i]),
                        done=bool(dones[i]),
                        is_action_valid=is_action_valid,
                        prev_action_invalid=prev_action_invalid[i],
                        prefix_text=step_prefixes[i],
                    )
                )
                branch_rewards[i] += float(rewards[i])
                steps_taken[i] += 1
                prev_action_invalid[i] = not is_action_valid
                if bool(dones[i]):
                    done_flags[i] = True
                    active[i] = False

            obs_texts = list(next_obs)
            if not any(active):
                break

        for i, spec in enumerate(bucket_specs):
            episode_reward = spec.prefix_reward + branch_rewards[i]
            episode_length = spec.branch_step_num + steps_taken[i]
            for row in rows[i]:
                row["episode_rewards"] = episode_reward
                row["episode_lengths"] = episode_length
            trajectories.append(
                BranchTrajectory(
                    spec=spec,
                    rows=rows[i],
                    episode_reward=episode_reward,
                    episode_length=episode_length,
                    done=done_flags[i],
                )
            )
    return trajectories


def summarize_branch_metrics(trajectories: Sequence[BranchTrajectory]) -> Dict[str, float]:
    """Aggregate metrics for logging under the `seed/teacher_branch/*` prefix."""
    total = len(trajectories)
    if total == 0:
        return {
            "seed/teacher_branch/num_branches": 0.0,
            "seed/teacher_branch/done_ratio": 0.0,
            "seed/teacher_branch/mean_episode_reward": 0.0,
            "seed/teacher_branch/mean_branch_steps": 0.0,
        }
    return {
        "seed/teacher_branch/num_branches": float(total),
        "seed/teacher_branch/done_ratio": sum(1.0 for t in trajectories if t.done) / total,
        "seed/teacher_branch/mean_episode_reward": sum(t.episode_reward for t in trajectories) / total,
        "seed/teacher_branch/mean_branch_steps": sum(len(t.rows) for t in trajectories) / total,
    }
