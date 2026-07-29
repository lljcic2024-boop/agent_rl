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
改造点 4 cluster glue: run teacher-prefix branches against the real env/policy.

`teacher_branch.py` owns the text-level orchestration with injected hooks so it
stays laptop-testable; `teacher_prefix.py` owns the token bookkeeping. This
module is the only place that touches verl runtime objects (DataProto, the
rollout worker group, the environment manager) and wires the three together:

    main rollout rows
      -> select_branch_specs                (error-signal steps)
      -> generate_teacher_prefixes          (external 30B teacher, a_T)
      -> run_branch_rollout_batched         (env replay + student continuation)
      -> DataProto rows with teacher_token_mask / loss_mask

Branch rows are concatenated into the training batch: they inherit the parent
`uid` (so GRPO normalizes them inside the same task group) with a fresh
`traj_uid`, and their a_T tokens are excluded from the PG `loss_mask` while
being marked in `teacher_token_mask` for the FKL term (`compute_opd_fkl_loss`).
"""

import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional, Tuple

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn

from agent_system.multi_turn_rollout.teacher_branch import (
    BranchSpec,
    generate_teacher_prefixes,
    run_branch_rollout_batched,
    select_branch_specs,
    summarize_branch_metrics,
)
from agent_system.multi_turn_rollout.teacher_prefix import (
    assemble_branch_responses,
    build_prefix_generation_batch,
)

module_logger = logging.getLogger(__name__)

# Every way `run()` can come back with no branch rows. The funnel below is
# always emitted in full (one-hot over these), because the failure mode this
# guards against is silence: a run that produces zero branches while
# `num_candidates` looks healthy still trains, just with an empty FKL term.
_EXIT_REASONS = (
    "disabled",
    "schedule",
    "no_rows",
    "no_env_kwargs",
    "no_prefix",
    "no_tensors",
    "ok",
)

# Row fields that a branch row must define itself; everything else is copied
# from the parent trajectory's row so the concatenated batch keeps one key set.
_BRANCH_OWNED_KEYS = (
    "uid",
    "traj_uid",
    "sample_id",
    "rollout_id",
    "step_num",
    "step_id",
    "obs_text",
    "obs_text_base",
    "anchor_obs",
    "rewards",
    "active_masks",
    "is_action_valid",
    "episode_rewards",
    "episode_lengths",
    "tag_final_answer",
    "tag_error_signal",
    "tag_plan_count",
    "tag_verify_count",
    "tag_reflect_count",
    "tag_backtrack_count",
)

def _select(config, path, default=None):
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _batch_to_rows(batch: DataProto) -> List[Dict[str, Any]]:
    """
    Flatten a rollout DataProto into per-row dicts for `select_branch_specs`.

    Only the non-tensor columns the selector reads are materialized; `responses`
    is carried along so the action text can be decoded for the replay.
    """
    size = len(batch)
    rows: List[Dict[str, Any]] = []
    responses = batch.batch["responses"] if "responses" in batch.batch.keys() else None
    for i in range(size):
        row: Dict[str, Any] = {
            key: value[i] for key, value in batch.non_tensor_batch.items()
        }
        if responses is not None:
            row["responses"] = responses[i]
        rows.append(row)
    return rows


def attach_default_masks(batch: DataProto) -> DataProto:
    """
    Give main-rollout rows the columns branch rows carry.

    `DataProto.concat` requires identical key sets, and `dp_actor` needs
    `loss_mask` once `meta_info["multi_turn"]` is on. For ordinary rows the PG
    mask is just the attention mask (so the loss is unchanged from the
    non-multi_turn path) and no token is teacher-generated.

    `is_teacher_branch` has to be here too: `concat` takes its key set from the
    first DataProto, so a column that only the branch rows carry trips
    `list_of_dict_to_dict_of_list`'s `assert key in output`.
    """
    if "loss_mask" not in batch.batch.keys():
        batch.batch["loss_mask"] = batch.batch["attention_mask"].clone()
    if "teacher_token_mask" not in batch.batch.keys():
        response_length = batch.batch["responses"].size(1)
        batch.batch["teacher_token_mask"] = torch.zeros(
            (len(batch), response_length),
            dtype=batch.batch["attention_mask"].dtype,
            device=batch.batch["attention_mask"].device,
        )
    if "is_teacher_branch" not in batch.non_tensor_batch:
        batch.non_tensor_batch["is_teacher_branch"] = np.array(
            [False] * len(batch), dtype=object
        )
    return batch


def branch_config(config) -> Dict[str, Any]:
    """Read the `algorithm.seed.teacher_branch` block with defaults."""
    return {
        "enable": bool(_select(config, "algorithm.seed.teacher_branch.enable", False)),
        "max_branches_per_traj": int(_select(config, "algorithm.seed.teacher_branch.max_branches_per_traj", 1)),
        "max_total_branches": _select(config, "algorithm.seed.teacher_branch.max_total_branches", None),
        "require_error_signal": bool(_select(config, "algorithm.seed.teacher_branch.require_error_signal", True)),
        "prefix_concurrency": int(_select(config, "algorithm.seed.teacher_branch.prefix_concurrency", 8)),
        "prefix_max_retries": int(_select(config, "algorithm.seed.teacher_branch.prefix_max_retries", 2)),
        "start_after_steps": _select(config, "algorithm.seed.teacher_branch.start_after_steps", None),
        "stop_after_steps": _select(config, "algorithm.seed.teacher_branch.stop_after_steps", None),
        # When on, an unscheduled empty result raises instead of silently
        # degrading to plain PPO. Use it for smoke runs; leave it off for
        # production, where one bad step should not kill a 150-step job.
        "strict": bool(_select(config, "algorithm.seed.teacher_branch.strict", False)),
    }


class BranchProducedNothing(RuntimeError):
    """Raised in strict mode when a scheduled-on branch step yields no rows."""


def _init_funnel() -> Dict[str, float]:
    """
    Zero-initialize the whole funnel so every step logs every field.

    Metrics that only appear on the paths that reach them are unreadable in
    aggregate: a missing key looks the same as a key that was never reached,
    and wandb plots a gap rather than a zero. Declaring them up front makes
    "candidates 21 -> specs 0" visible at a glance.
    """
    funnel = {
        "seed/teacher_branch/num_candidates": 0.0,
        "seed/teacher_branch/num_specs_with_env_kwargs": 0.0,
        "seed/teacher_branch/num_prefixes": 0.0,
        "seed/teacher_branch/num_trajectories": 0.0,
        "seed/teacher_branch/num_rows": 0.0,
        "seed/teacher_branch/prefix_failure_ratio": 0.0,
    }
    for reason in _EXIT_REASONS:
        funnel[f"seed/teacher_branch/exit_{reason}"] = 0.0
    return funnel


def _assemble_float_response(
    prefix_lengths: List[int],
    continuation_values: torch.Tensor,
    continuation_mask: torch.Tensor,
    response_length: int,
    fill_value: float,
) -> torch.Tensor:
    """
    Right-pad `a_T + continuation` for a float column (e.g. rollout_log_probs).

    Mirrors `assemble_branch_responses`'s layout so float columns stay aligned
    with `responses`. a_T positions get `fill_value` (the teacher generated
    them, so the student rollout has no log-prob for them).
    """
    bsz = continuation_values.size(0)
    out = torch.full(
        (bsz, response_length),
        fill_value,
        dtype=continuation_values.dtype,
        device=continuation_values.device,
    )
    for i in range(bsz):
        valid = continuation_mask[i].bool()
        values = continuation_values[i][valid]
        start = min(int(prefix_lengths[i]), response_length)
        room = response_length - start
        if room > 0 and values.numel() > 0:
            values = values[:room]
            out[i, start : start + values.numel()] = values
    return out


class SearchBranchEnv:
    """
    A batched Search environment used only for branch replay.

    The main rollout's env manager is mid-episode when branches run, so branches
    need their own instance. Capacity is reused across buckets: the underlying
    `SearchMultiProcessEnv` pads short reset/step calls, so one manager sized to
    the largest bucket serves all of them.
    """

    def __init__(self, config, capacity: int):
        from agent_system.environments.env_manager import SearchEnvironmentManager
        from agent_system.environments.env_package.search import (
            build_search_envs,
            search_projection,
        )
        from functools import partial

        self.capacity = int(capacity)
        envs = build_search_envs(
            seed=int(_select(config, "env.seed", 0)) + 7777,
            env_num=self.capacity,
            group_n=1,
            is_train=True,
            env_config=config.env,
        )
        self.manager = SearchEnvironmentManager(envs, partial(search_projection), config)

    def reset(self, env_kwargs: List[Any]) -> Tuple[List[str], List[Any]]:
        obs, _ = self.manager.reset(kwargs=list(env_kwargs))
        return list(obs["text"]), self._anchors(obs)

    def step(self, action_texts: List[str]):
        obs, rewards, dones, infos = self.manager.step(list(action_texts))
        return (
            list(obs["text"]),
            list(rewards),
            list(dones),
            list(infos),
            self._anchors(obs),
        )

    @staticmethod
    def _anchors(obs: Dict[str, Any]) -> List[Any]:
        """
        Per-row anchor observation (history-free state) for GiGPO/SEED step grouping.

        Falls back to the rendered text when the env does not expose an anchor,
        which still groups identical states — just less aggressively.
        """
        anchor = obs.get("anchor")
        if anchor is None:
            return list(obs["text"])
        return list(anchor)

    def close(self):
        try:
            self.manager.envs.close()
        except Exception:
            pass


class TeacherBranchRunner:
    """
    Cluster-side driver for 改造点 4.

    Owns the tokenizer/worker-group/env plumbing behind the injected hooks of
    `run_branch_rollout_batched`, and turns the resulting rows into a DataProto
    that can be concatenated with the main rollout batch.
    """

    def __init__(self, config, tokenizer, collector, actor_rollout_wg):
        self.config = config
        self.tokenizer = tokenizer
        self.collector = collector  # TrajectoryCollector, for prompt construction
        self.actor_rollout_wg = actor_rollout_wg
        self.cfg = branch_config(config)
        self.response_length = int(_select(config, "data.max_response_length", 512))
        self.max_prefix_tokens = int(
            _select(
                config,
                "algorithm.seed.teacher_branch.max_prefix_tokens",
                max(1, self.response_length // 2),
            )
        )
        self.env_max_steps = int(_select(config, "env.max_steps", 4))
        self._branch_env: Optional[SearchBranchEnv] = None
        # (spec index, step_num) -> assembled tensors for that branch row
        self._row_tensors: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        # (spec index, step_num) -> anchor observation for that row
        self._row_anchors: Dict[Tuple[int, int], Any] = {}
        # anchors of the observation the next generate_batch call will see
        self._pending_anchors: List[Any] = []
        # env reset payloads for the current training step, indexed by sample_id
        self._env_kwargs: Optional[Any] = None
        # meta_info forwarded to generate_sequences; the rollout worker fills in
        # eos/pad ids itself, this only carries sampling switches.
        self._meta_info: Dict[str, Any] = {}

    # -- teacher prefix ----------------------------------------------------
    def _chat_fn(self):
        """OpenAI-compatible chat callable pointing at the external teacher."""
        import os

        from utils import chat_completion_with_retry, create_openai_client

        base_url = str(
            _select(self.config, "algorithm.seed.external_teacher.base_url", None)
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        )
        model = str(
            _select(self.config, "algorithm.seed.external_teacher.model", None)
            or os.environ.get("OPENAI_MODEL")
            or ""
        )
        timeout = float(_select(self.config, "algorithm.seed.external_teacher.timeout", 600.0))
        retries = int(_select(self.config, "algorithm.seed.external_teacher.max_retries", 3))
        client = create_openai_client(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )

        def chat_fn(messages):
            return chat_completion_with_retry(
                client=client,
                model=model,
                messages=messages,
                retries=max(1, retries),
            )

        return chat_fn

    def _prefix_token_ids(self, prefix_text: str) -> List[int]:
        ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        return list(ids[: self.max_prefix_tokens])

    def _canonicalize_prefixes(self, prefix_texts: Dict[int, str]) -> Dict[int, str]:
        """
        Re-decode each a_T through the tokenizer.

        The prefix is fed to the policy as tokens but compared as text (the
        branch step's response must start with it), so the text has to be
        exactly what those tokens decode back to — including the truncation at
        `max_prefix_tokens`.
        """
        canonical: Dict[int, str] = {}
        for index, text in prefix_texts.items():
            token_ids = self._prefix_token_ids(text)
            if not token_ids:
                continue
            canonical[index] = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return canonical

    # -- student continuation ---------------------------------------------
    def _generate_batch(
        self,
        obs_texts: List[str],
        step_prefixes: List[Optional[str]],
        spec_indices: List[int],
        step_num: int,
    ) -> List[str]:
        """
        One batched policy call: prompts from `obs_texts`, generation continuing
        after a_T where present. Stores the token-level tensors for each row and
        returns the full response texts (a_T included).
        """
        prompt_batch = self.collector.build_text_prompt_batch(obs_contents=list(obs_texts))
        prompt_input_ids = prompt_batch.batch["input_ids"]
        prompt_attention_mask = prompt_batch.batch["attention_mask"]
        pad_token_id = self.tokenizer.pad_token_id

        prefix_token_ids = [
            self._prefix_token_ids(text) if text else [] for text in step_prefixes
        ]
        prefix_batch = build_prefix_generation_batch(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            prefix_token_ids=prefix_token_ids,
            pad_token_id=pad_token_id,
        )

        # raw_prompt_ids is intentionally NOT provided: the rollout worker
        # re-derives it from input_ids, so a_T becomes part of the prompt and
        # generation continues right after it.
        gen_input = DataProto.from_single_dict(
            data={
                "input_ids": prefix_batch.input_ids,
                "attention_mask": prefix_batch.attention_mask,
                "position_ids": prefix_batch.position_ids,
            },
            meta_info=dict(self._meta_info),
        )
        padded, pad_size = pad_dataproto_to_divisor(gen_input, self.actor_rollout_wg.world_size)
        output_padded = self.actor_rollout_wg.generate_sequences(padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        continuation_ids = output.batch["responses"]
        cont_len = continuation_ids.size(1)
        continuation_mask = output.batch["attention_mask"][:, -cont_len:]

        assembled = assemble_branch_responses(
            prefix_token_ids=prefix_token_ids,
            continuation_ids=continuation_ids,
            continuation_attention_mask=continuation_mask,
            pad_token_id=pad_token_id,
            response_length=self.response_length,
        )

        prefix_lengths = [len(ids) for ids in prefix_token_ids]
        rollout_log_probs = None
        if "rollout_log_probs" in output.batch.keys():
            rollout_log_probs = _assemble_float_response(
                prefix_lengths=prefix_lengths,
                continuation_values=output.batch["rollout_log_probs"],
                continuation_mask=continuation_mask,
                response_length=self.response_length,
                fill_value=-1.0,
            )

        full_attention_mask = torch.cat(
            [prompt_attention_mask, assembled.response_attention_mask.to(prompt_attention_mask.dtype)],
            dim=-1,
        )
        full_input_ids = torch.cat([prompt_input_ids, assembled.responses], dim=-1)
        from verl.utils.model import compute_position_id_with_mask

        full_position_ids = compute_position_id_with_mask(full_attention_mask)

        response_texts: List[str] = []
        for i, spec_index in enumerate(spec_indices):
            row_tensors = {
                "prompts": prompt_input_ids[i],
                "responses": assembled.responses[i],
                "input_ids": full_input_ids[i],
                "attention_mask": full_attention_mask[i],
                "position_ids": full_position_ids[i],
                "loss_mask": torch.cat(
                    [
                        torch.zeros_like(prompt_attention_mask[i]),
                        assembled.loss_mask[i].to(prompt_attention_mask.dtype),
                    ],
                    dim=-1,
                ),
                "teacher_token_mask": assembled.teacher_token_mask[i],
            }
            if rollout_log_probs is not None:
                row_tensors["rollout_log_probs"] = rollout_log_probs[i]
            self._row_tensors[(spec_index, step_num)] = row_tensors
            if i < len(self._pending_anchors):
                self._row_anchors[(spec_index, step_num)] = self._pending_anchors[i]

            continuation_text = self.tokenizer.decode(
                continuation_ids[i][continuation_mask[i].bool()], skip_special_tokens=True
            )
            prefix_text = step_prefixes[i] or ""
            response_texts.append(prefix_text + continuation_text)

        return response_texts

    # -- env replay --------------------------------------------------------
    def _ensure_env(self, capacity: int) -> SearchBranchEnv:
        if self._branch_env is None or self._branch_env.capacity < capacity:
            if self._branch_env is not None:
                self._branch_env.close()
            self._branch_env = SearchBranchEnv(self.config, capacity)
        return self._branch_env

    def _env_start_batch(self, bucket_specs: List[BranchSpec]) -> List[str]:
        """
        Reset the branch env for this bucket and replay the parent's actions.

        Search retrieval is stateless, so replaying the recorded queries
        reproduces both the env state and the observation history that
        `build_text_obs` renders into the branch-step prompt.
        """
        env = self._ensure_env(len(bucket_specs))
        obs_texts, anchors = env.reset([spec.env_kwargs for spec in bucket_specs])
        replay_length = len(bucket_specs[0].replay_actions)
        for step_idx in range(replay_length):
            actions = [spec.replay_actions[step_idx] for spec in bucket_specs]
            obs_texts, _, _, _, anchors = env.step(actions)
        self._pending_anchors = list(anchors)
        return obs_texts

    def _env_step_batch(self, bucket_specs: List[BranchSpec], action_texts: List[str]):
        env = self._ensure_env(len(bucket_specs))
        obs_texts, rewards, dones, infos, anchors = env.step(action_texts)
        # Anchors of the *next* observation; the row for this step already
        # recorded the pre-step anchor in `_generate_batch`.
        self._pending_anchors = list(anchors)
        return obs_texts, rewards, dones, infos

    # -- driver ------------------------------------------------------------
    def run(
        self,
        batch: DataProto,
        global_step: int,
        env_kwargs: Optional[Any] = None,
    ) -> Tuple[Optional[DataProto], Dict[str, float]]:
        """
        Select branch points from `batch`, roll them, and return the branch rows.

        Args:
            batch: the gathered main-rollout batch (one row per step).
            global_step: current training step, for the start/stop schedule.
            env_kwargs: the env reset payloads of the prompt batch, indexed by
                `sample_id`. The rollout loop pops them before stepping the env,
                so the trainer has to pass them in for the replay.

        Returns:
            (branch DataProto or None, metrics dict). None means no branch was
            produced this step (no candidate, teacher refused, disabled by
            schedule); the caller then trains on the main batch unchanged.

            The metrics always carry the full funnel and exactly one
            `exit_<reason>` set to 1.0, so an empty result is never silent.
        """
        metrics: Dict[str, float] = _init_funnel()

        def _bail(reason: str, detail: str):
            """Mark the exit reason, log it, and (in strict mode) refuse to go on."""
            metrics[f"seed/teacher_branch/exit_{reason}"] = 1.0
            if reason in ("disabled", "schedule"):
                return None, metrics
            module_logger.warning(
                "改造点 4 produced no branch rows at step %s (reason=%s): %s",
                global_step,
                reason,
                detail,
            )
            if self.cfg["strict"]:
                raise BranchProducedNothing(
                    f"teacher branch produced nothing at step {global_step} "
                    f"(reason={reason}): {detail}. Funnel: "
                    + ", ".join(
                        f"{key.rsplit('/', 1)[-1]}={value:g}"
                        for key, value in metrics.items()
                        if key.startswith("seed/teacher_branch/num_")
                    )
                )
            return None, metrics

        if not self.cfg["enable"]:
            return _bail("disabled", "algorithm.seed.teacher_branch.enable=False")
        start_after = self.cfg["start_after_steps"]
        stop_after = self.cfg["stop_after_steps"]
        if start_after is not None and global_step < int(start_after):
            metrics["seed/teacher_branch/skipped_by_schedule"] = 1.0
            return _bail("schedule", f"step {global_step} < start_after_steps {start_after}")
        if stop_after is not None and global_step > int(stop_after):
            metrics["seed/teacher_branch/skipped_by_schedule"] = 1.0
            return _bail("schedule", f"step {global_step} > stop_after_steps {stop_after}")
        metrics["seed/teacher_branch/skipped_by_schedule"] = 0.0

        self._env_kwargs = env_kwargs
        rows = _batch_to_rows(batch)
        if not rows:
            return _bail("no_rows", "the main-rollout batch is empty")

        specs = select_branch_specs(
            rows,
            get_action_text=lambda row: self._decode_response(row),
            get_env_kwargs=self._lookup_env_kwargs,
            max_branches_per_traj=self.cfg["max_branches_per_traj"],
            max_total_branches=(
                None if self.cfg["max_total_branches"] is None else int(self.cfg["max_total_branches"])
            ),
            require_error_signal=self.cfg["require_error_signal"],
        )
        metrics["seed/teacher_branch/num_candidates"] = float(len(specs))
        specs = [spec for spec in specs if spec.env_kwargs is not None]
        metrics["seed/teacher_branch/num_specs_with_env_kwargs"] = float(len(specs))
        if not specs:
            metrics.update(summarize_branch_metrics([]))
            # This is the pair that looks healthy and is not: candidates were
            # found, then every one was dropped because its env reset payload
            # could not be resolved (trainer passed env_kwargs=None, or the
            # sample_id did not index into it).
            return _bail(
                "no_env_kwargs",
                f"{int(metrics['seed/teacher_branch/num_candidates'])} candidate(s) selected but none "
                f"resolved an env payload (env_kwargs is "
                f"{'None' if env_kwargs is None else f'len={len(env_kwargs)}'})",
            )

        prefix_texts, failures = generate_teacher_prefixes(
            specs,
            self._chat_fn(),
            max_retries=self.cfg["prefix_max_retries"],
            max_workers=self.cfg["prefix_concurrency"],
        )
        metrics["seed/teacher_branch/prefix_failure_ratio"] = (
            float(len(failures)) / max(1, len(specs))
        )
        prefix_texts = self._canonicalize_prefixes(prefix_texts)
        metrics["seed/teacher_branch/num_prefixes"] = float(len(prefix_texts))
        if not prefix_texts:
            metrics.update(summarize_branch_metrics([]))
            return _bail(
                "no_prefix",
                f"all {len(specs)} teacher prefix request(s) were rejected or empty "
                f"({len(failures)} hard failure(s)); check the teacher server and the quality gate",
            )

        self._row_tensors.clear()
        self._row_anchors.clear()
        self._pending_anchors = []
        trajectories = run_branch_rollout_batched(
            specs,
            prefix_texts,
            env_start_batch=self._env_start_batch,
            env_step_batch=self._env_step_batch,
            generate_batch=self._generate_batch,
            max_steps=self.env_max_steps,
        )
        metrics.update(summarize_branch_metrics(trajectories))
        metrics["seed/teacher_branch/num_trajectories"] = float(len(trajectories))

        spec_index = {id(spec): i for i, spec in enumerate(specs)}
        branch_proto = self._rows_to_dataproto(trajectories, spec_index, batch)
        if branch_proto is None:
            return _bail(
                "no_tensors",
                f"{len(trajectories)} branch trajectory/ies rolled but none survived DataProto "
                "assembly (missing generated tensors for every row)",
            )
        metrics["seed/teacher_branch/num_rows"] = float(len(branch_proto))
        metrics["seed/teacher_branch/exit_ok"] = 1.0
        return branch_proto, metrics

    def _decode_response(self, row: Dict[str, Any]) -> str:
        text = row.get("response_text")
        if text is not None:
            return str(text)
        responses = row.get("responses")
        if responses is None:
            return ""
        return self.tokenizer.decode(responses, skip_special_tokens=True)

    def _lookup_env_kwargs(self, row: Dict[str, Any]) -> Any:
        """
        Resolve the env reset payload for a row.

        The rollout loop pops `env_kwargs` before stepping the env, so rows do
        not carry it; the trainer hands the pre-pop array to `run()` and rows are
        matched back by `sample_id` (index into the prompt batch).
        """
        if row.get("env_kwargs") is not None:
            return row["env_kwargs"]
        if self._env_kwargs is None:
            return None
        try:
            sample_id = int(row["sample_id"])
        except (KeyError, TypeError, ValueError):
            return None
        if sample_id < 0 or sample_id >= len(self._env_kwargs):
            return None
        return self._env_kwargs[sample_id]

    # -- batch assembly ----------------------------------------------------
    def _rows_to_dataproto(
        self,
        trajectories: List[Any],
        spec_index: Dict[int, int],
        batch: DataProto,
    ) -> Optional[DataProto]:
        """
        Turn branch rows into a DataProto that can be concatenated with `batch`.

        Every column of the main batch must be present with the same key set
        (`DataProto.concat` requirement), so each branch row starts as a copy of
        its parent trajectory's row and then overrides the fields it owns
        (identity, observation, reward, tags) plus all tensor columns, which come
        from the tensors `_generate_batch` stashed for that (spec, step).
        """
        parent_row_index = self._parent_row_index(batch)
        tensor_keys = set(batch.batch.keys()) | {"loss_mask", "teacher_token_mask"}
        non_tensor_keys = set(batch.non_tensor_batch.keys()) | {"is_teacher_branch"}

        samples: List[Dict[str, Any]] = []
        for traj in trajectories:
            index = spec_index.get(id(traj.spec))
            if index is None:
                continue
            parent = parent_row_index.get(traj.spec.parent_traj_uid)
            for row in traj.rows:
                tensors = self._row_tensors.get((index, int(row["step_num"])))
                if tensors is None:
                    # generation for this row failed; dropping it keeps the batch
                    # consistent (the trajectory just contributes fewer steps).
                    continue
                sample: Dict[str, Any] = {}
                if parent is not None:
                    for key in non_tensor_keys:
                        if key in _BRANCH_OWNED_KEYS or key not in batch.non_tensor_batch:
                            continue
                        sample[key] = batch.non_tensor_batch[key][parent]
                for key in non_tensor_keys:
                    if key in row:
                        sample[key] = row[key]
                sample["is_teacher_branch"] = True
                anchor = self._row_anchors.get((index, int(row["step_num"])))
                if "anchor_obs" in non_tensor_keys:
                    sample["anchor_obs"] = anchor if anchor is not None else row.get("obs_text", "")
                if "obs_text_base" in non_tensor_keys and "obs_text_base" not in row:
                    # branch observations are never skill-augmented, so the base
                    # text is the observation itself.
                    sample["obs_text_base"] = row.get("obs_text", "")
                if parent is not None:
                    # owned columns the branch path does not produce (e.g.
                    # obs_text_base, tag_*_count) still have to exist, otherwise
                    # DataProto.concat rejects the mismatched key set.
                    for key in non_tensor_keys:
                        if key in sample or key not in batch.non_tensor_batch:
                            continue
                        sample[key] = batch.non_tensor_batch[key][parent]
                missing_non_tensor = non_tensor_keys - set(sample.keys())
                if missing_non_tensor:
                    raise RuntimeError(
                        f"branch row is missing non-tensor columns: {sorted(missing_non_tensor)}"
                    )
                for key in tensor_keys:
                    if key in tensors:
                        sample[key] = tensors[key]
                    elif parent is not None and key in batch.batch.keys():
                        # column the branch path does not rebuild (e.g. a cached
                        # log-prob); inherit the parent's so shapes line up.
                        sample[key] = batch.batch[key][parent]
                missing = tensor_keys - set(sample.keys())
                if missing:
                    raise RuntimeError(f"branch row is missing tensor columns: {sorted(missing)}")
                samples.append(sample)

        if not samples:
            return None
        return DataProto.from_single_dict(data=collate_fn(samples))

    @staticmethod
    def _parent_row_index(batch: DataProto) -> Dict[str, int]:
        """First row index of each parent trajectory, used as the copy template."""
        traj_uids = batch.non_tensor_batch.get("traj_uid")
        if traj_uids is None:
            return {}
        index: Dict[str, int] = {}
        for i, traj_uid in enumerate(traj_uids):
            index.setdefault(str(traj_uid), i)
        return index
