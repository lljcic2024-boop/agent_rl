"""Tests for the 改造点 4 cluster glue (`branch_runner`).

Covers the pieces that touch verl runtime objects: default-mask attachment for
main-rollout rows, the a_T token bookkeeping inside `_generate_batch` (prompt
assembly, teacher/PG mask split, float-column alignment), and the row ->
DataProto conversion that must stay `DataProto.concat`-compatible with the main
batch. The tokenizer, the rollout worker group and the env are stubbed, so this
runs on CPU without ray/vllm/GPUs.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn

from agent_system.multi_turn_rollout import branch_runner as br
from agent_system.multi_turn_rollout.teacher_branch import BranchSpec, BranchTrajectory

PROMPT_LEN = 6
RESP_LEN = 8
PAD = 0


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Char-level tokenizer: token id = ord(char), so decode(encode(x)) == x."""

    pad_token_id = PAD

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids if int(i) != PAD)


class FakeCollector:
    """Stands in for TrajectoryCollector.build_text_prompt_batch (left-padded)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def build_text_prompt_batch(self, obs_contents):
        self.calls.append(list(obs_contents))
        rows = []
        for text in obs_contents:
            ids = self.tokenizer.encode(text)[:PROMPT_LEN]
            pad = PROMPT_LEN - len(ids)
            rows.append(
                {
                    "input_ids": torch.tensor([PAD] * pad + ids, dtype=torch.long),
                    "attention_mask": torch.tensor([0] * pad + [1] * len(ids), dtype=torch.long),
                }
            )
        return DataProto.from_single_dict(data=collate_fn(rows))


class FakeWorkerGroup:
    """Echoes a scripted continuation; records the prompts it was handed."""

    world_size = 1

    def __init__(self, continuation="!next", with_log_probs=True):
        self.continuation = continuation
        self.with_log_probs = with_log_probs
        self.seen_input_ids = []

    def generate_sequences(self, proto):
        prompt_ids = proto.batch["input_ids"]
        prompt_mask = proto.batch["attention_mask"]
        self.seen_input_ids.append(prompt_ids.clone())
        bsz = prompt_ids.size(0)
        cont = [ord(c) for c in self.continuation]
        responses = torch.full((bsz, RESP_LEN), PAD, dtype=torch.long)
        resp_mask = torch.zeros((bsz, RESP_LEN), dtype=torch.long)
        for i in range(bsz):
            responses[i, : len(cont)] = torch.tensor(cont, dtype=torch.long)
            resp_mask[i, : len(cont)] = 1
        out = {
            "prompts": prompt_ids,
            "responses": responses,
            "input_ids": torch.cat([prompt_ids, responses], dim=-1),
            "attention_mask": torch.cat([prompt_mask, resp_mask], dim=-1),
            "position_ids": torch.zeros((bsz, PROMPT_LEN + RESP_LEN), dtype=torch.long),
        }
        if self.with_log_probs:
            log_probs = torch.full((bsz, RESP_LEN), -1.0)
            log_probs[:, : len(cont)] = -0.5
            out["rollout_log_probs"] = log_probs
        return DataProto.from_single_dict(data=out)


def make_config(**overrides):
    cfg = OmegaConf.create(
        {
            "data": {"max_response_length": RESP_LEN},
            "env": {"max_steps": 4, "seed": 0},
            "algorithm": {
                "seed": {
                    "teacher_branch": {
                        "enable": True,
                        "max_branches_per_traj": 1,
                        "max_total_branches": None,
                        "require_error_signal": True,
                        "max_prefix_tokens": 4,
                        "prefix_concurrency": 2,
                        "prefix_max_retries": 1,
                        "start_after_steps": None,
                        "stop_after_steps": None,
                    },
                    "external_teacher": {"base_url": None, "model": "t", "timeout": 1.0, "max_retries": 1},
                }
            },
        }
    )
    for path, value in overrides.items():
        OmegaConf.update(cfg, path, value, merge=True)
    return cfg


def make_runner(**overrides):
    tokenizer = FakeTokenizer()
    return br.TeacherBranchRunner(
        config=make_config(**overrides),
        tokenizer=tokenizer,
        collector=FakeCollector(tokenizer),
        actor_rollout_wg=FakeWorkerGroup(),
    )


def make_main_batch(n=2, traj_uids=("t0", "t1")):
    """A minimal main-rollout batch with the columns the glue reads/copies."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompts": torch.full((PROMPT_LEN,), 65 + i, dtype=torch.long),
                "responses": torch.full((RESP_LEN,), 97 + i, dtype=torch.long),
                "input_ids": torch.full((PROMPT_LEN + RESP_LEN,), 65 + i, dtype=torch.long),
                "attention_mask": torch.ones(PROMPT_LEN + RESP_LEN, dtype=torch.long),
                "position_ids": torch.arange(PROMPT_LEN + RESP_LEN, dtype=torch.long),
                "uid": f"g{i}",
                "traj_uid": traj_uids[i],
                "sample_id": i,
                "rollout_id": 0,
                "step_num": 0,
                "step_id": f"{i}_0_0",
                "obs_text": f"obs-{i}",
                "obs_text_base": f"obs-{i}",
                "anchor_obs": f"anchor-{i}",
                "rewards": 0.0,
                "active_masks": True,
                "is_action_valid": True,
                "episode_rewards": 0.0,
                "episode_lengths": 1,
                "tag_error_signal": True,
                "tag_final_answer": False,
                "data_source": "nq",
                "index": i,
                "tool_callings": 1.0,
                "success_rate": 0.5,
            }
        )
    return DataProto.from_single_dict(data=collate_fn(rows))


# ---------------------------------------------------------------------------
# attach_default_masks
# ---------------------------------------------------------------------------

def test_attach_default_masks_is_a_noop_for_pg_loss():
    batch = make_main_batch()
    br.attach_default_masks(batch)
    assert torch.equal(batch.batch["loss_mask"], batch.batch["attention_mask"])
    assert batch.batch["teacher_token_mask"].shape == (2, RESP_LEN)
    assert batch.batch["teacher_token_mask"].sum() == 0


def test_attach_default_masks_does_not_overwrite_existing():
    batch = make_main_batch()
    marker = torch.zeros_like(batch.batch["attention_mask"])
    batch.batch["loss_mask"] = marker
    br.attach_default_masks(batch)
    assert torch.equal(batch.batch["loss_mask"], marker)


def test_attach_default_masks_marks_main_rows_as_non_branch():
    # `DataProto.concat` reads its key set off the FIRST proto, so a column only
    # the branch rows carry raises `assert key in output`. The real trainer
    # concatenates without a select(), so the main batch must declare it.
    batch = make_main_batch()
    br.attach_default_masks(batch)
    assert batch.non_tensor_batch["is_teacher_branch"].tolist() == [False, False]


def test_main_and_branch_batches_concat_without_reselecting_columns():
    runner = make_runner()
    batch = make_main_batch()
    spec = _spec()
    branch = runner._rows_to_dataproto([_branch_traj(runner, spec)], {id(spec): 0}, batch)
    batch = br.attach_default_masks(batch)
    merged = DataProto.concat([batch, branch])
    assert len(merged) == 3
    assert merged.non_tensor_batch["is_teacher_branch"].tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# branch_config
# ---------------------------------------------------------------------------

def test_branch_config_defaults_to_disabled():
    cfg = OmegaConf.create({"algorithm": {"seed": {}}})
    assert br.branch_config(cfg)["enable"] is False
    assert br.branch_config(cfg)["prefix_concurrency"] == 8


# ---------------------------------------------------------------------------
# _assemble_float_response
# ---------------------------------------------------------------------------

def test_float_response_places_continuation_after_prefix():
    values = torch.tensor([[0.1, 0.2, 0.9]])
    mask = torch.tensor([[1, 1, 0]])
    out = br._assemble_float_response([2], values, mask, response_length=6, fill_value=-1.0)
    assert out.shape == (1, 6)
    assert out[0].tolist() == pytest.approx([-1.0, -1.0, 0.1, 0.2, -1.0, -1.0], abs=1e-6)


def test_float_response_truncates_at_response_length():
    values = torch.tensor([[0.1, 0.2, 0.3]])
    mask = torch.ones((1, 3), dtype=torch.long)
    out = br._assemble_float_response([2], values, mask, response_length=3, fill_value=0.0)
    assert out[0].tolist() == pytest.approx([0.0, 0.0, 0.1], abs=1e-6)


# ---------------------------------------------------------------------------
# _canonicalize_prefixes
# ---------------------------------------------------------------------------

def test_canonicalize_truncates_prefix_to_token_budget():
    runner = make_runner()  # max_prefix_tokens = 4
    canonical = runner._canonicalize_prefixes({0: "abcdefgh", 1: ""})
    assert canonical == {0: "abcd"}  # empty prefix drops out entirely


# ---------------------------------------------------------------------------
# _generate_batch
# ---------------------------------------------------------------------------

def test_generate_batch_masks_split_teacher_and_student_tokens():
    runner = make_runner()
    texts = runner._generate_batch(
        obs_texts=["obs"], step_prefixes=["a_T"], spec_indices=[0], step_num=1
    )
    assert texts == ["a_T" + "!next"]  # a_T is echoed back with the continuation

    tensors = runner._row_tensors[(0, 1)]
    teacher = tensors["teacher_token_mask"]
    loss = tensors["loss_mask"][-RESP_LEN:]
    assert teacher.tolist()[:3] == [1, 1, 1]          # 3 chars of a_T
    assert teacher.sum().item() == 3
    assert loss.tolist()[:3] == [0, 0, 0]             # a_T excluded from PG
    assert loss.tolist()[3:8] == [1, 1, 1, 1, 1]      # 5 chars of "!next"
    # the two masks partition the valid response and never overlap
    assert (teacher * loss).sum().item() == 0
    response_mask = tensors["attention_mask"][-RESP_LEN:]
    assert torch.equal(response_mask, (teacher + loss).clamp(max=1))


def test_generate_batch_feeds_prefix_as_part_of_the_prompt():
    runner = make_runner()
    runner._generate_batch(["obs"], ["a_T"], [0], 1)
    seen = runner.actor_rollout_wg.seen_input_ids[0][0].tolist()
    # prompt row is [pad..., obs, a_T]: generation continues right after a_T
    assert "".join(chr(i) for i in seen if i != PAD) == "obsa_T"


def test_generate_batch_without_prefix_is_a_plain_step():
    runner = make_runner()
    runner._generate_batch(["obs"], [None], [0], 2)
    tensors = runner._row_tensors[(0, 2)]
    assert tensors["teacher_token_mask"].sum().item() == 0
    assert tensors["loss_mask"][-RESP_LEN:].sum().item() == len("!next")


def test_generate_batch_aligns_rollout_log_probs_with_responses():
    runner = make_runner()
    runner._generate_batch(["obs"], ["a_T"], [0], 1)
    log_probs = runner._row_tensors[(0, 1)]["rollout_log_probs"]
    assert log_probs.shape == (RESP_LEN,)
    assert log_probs[:3].tolist() == [-1.0, -1.0, -1.0]  # a_T: no student log-prob
    assert log_probs[3:8].tolist() == [-0.5] * 5


def test_generate_batch_sequence_columns_are_consistent():
    runner = make_runner()
    runner._generate_batch(["obs"], ["a_T"], [0], 1)
    t = runner._row_tensors[(0, 1)]
    assert t["input_ids"].numel() == PROMPT_LEN + RESP_LEN
    assert torch.equal(t["input_ids"][-RESP_LEN:], t["responses"])
    assert torch.equal(t["input_ids"][:PROMPT_LEN], t["prompts"])
    assert t["position_ids"].numel() == PROMPT_LEN + RESP_LEN
    # loss_mask spans the whole sequence; the prompt half is never trained on
    assert t["loss_mask"].numel() == PROMPT_LEN + RESP_LEN
    assert t["loss_mask"][:PROMPT_LEN].sum().item() == 0


def test_generate_batch_records_anchor_for_the_row():
    runner = make_runner()
    runner._pending_anchors = ["anchor-x"]
    runner._generate_batch(["obs"], ["a_T"], [3], 1)
    assert runner._row_anchors[(3, 1)] == "anchor-x"


# ---------------------------------------------------------------------------
# _lookup_env_kwargs
# ---------------------------------------------------------------------------

def test_lookup_env_kwargs_by_sample_id():
    runner = make_runner()
    runner._env_kwargs = np.array([{"q": "a"}, {"q": "b"}], dtype=object)
    assert runner._lookup_env_kwargs({"sample_id": 1}) == {"q": "b"}
    assert runner._lookup_env_kwargs({"sample_id": 9}) is None
    assert runner._lookup_env_kwargs({}) is None


def test_lookup_env_kwargs_prefers_the_row_column():
    runner = make_runner()
    runner._env_kwargs = np.array([{"q": "a"}], dtype=object)
    assert runner._lookup_env_kwargs({"sample_id": 0, "env_kwargs": {"q": "row"}}) == {"q": "row"}


# ---------------------------------------------------------------------------
# _rows_to_dataproto
# ---------------------------------------------------------------------------

def _branch_traj(runner, spec, step_nums=(1,)):
    """Roll fake tensors for `spec` and build a matching BranchTrajectory."""
    rows = []
    for offset, step_num in enumerate(step_nums):
        prefix = "a_T" if offset == 0 else None
        runner._pending_anchors = [f"branch-anchor-{step_num}"]
        runner._generate_batch(["obs"], [prefix], [0], step_num)
        rows.append(
            {
                "uid": spec.parent_uid,
                "traj_uid": spec.branch_traj_uid,
                "sample_id": spec.sample_id,
                "rollout_id": spec.rollout_id,
                "step_num": step_num,
                "step_id": f"{spec.sample_id}_{spec.rollout_id}_{step_num}",
                "obs_text": "branch-obs",
                "response_text": "a_T!next",
                "rewards": 1.0,
                "active_masks": True,
                "is_action_valid": True,
                "episode_rewards": 1.25,
                "episode_lengths": 2,
                "tag_final_answer": False,
                "tag_error_signal": False,
            }
        )
    return BranchTrajectory(spec=spec, rows=rows, episode_reward=1.25, episode_length=2, done=True)


def _spec():
    return BranchSpec(
        parent_traj_uid="t0", parent_uid="g0", sample_id=0, rollout_id=0,
        branch_step_num=1, replay_actions=["<search>old</search>"],
        branch_observation="obs-err", env_kwargs={"q": "a"}, prefix_reward=0.25,
    )


def test_rows_to_dataproto_concats_with_the_main_batch():
    runner = make_runner()
    batch = make_main_batch()
    br.attach_default_masks(batch)
    spec = _spec()
    traj = _branch_traj(runner, spec)

    branch = runner._rows_to_dataproto([traj], {id(spec): 0}, batch)
    assert branch is not None
    assert len(branch) == 1
    assert set(branch.batch.keys()) == set(batch.batch.keys())
    assert set(branch.non_tensor_batch.keys()) >= set(batch.non_tensor_batch.keys())

    merged = DataProto.concat([batch, branch.select(
        batch_keys=list(batch.batch.keys()),
        non_tensor_batch_keys=list(batch.non_tensor_batch.keys()),
    )])
    assert len(merged) == 3


def test_branch_rows_inherit_parent_uid_with_a_fresh_traj_uid():
    runner = make_runner()
    batch = make_main_batch()
    spec = _spec()
    branch = runner._rows_to_dataproto([_branch_traj(runner, spec)], {id(spec): 0}, batch)
    assert branch.non_tensor_batch["uid"][0] == "g0"          # same GRPO group
    assert branch.non_tensor_batch["traj_uid"][0] != "t0"     # own trajectory
    assert branch.non_tensor_batch["is_teacher_branch"][0] is True


def test_branch_rows_copy_unowned_columns_from_the_parent():
    runner = make_runner()
    batch = make_main_batch()
    spec = _spec()
    branch = runner._rows_to_dataproto([_branch_traj(runner, spec)], {id(spec): 0}, batch)
    # parent row 0 has traj_uid t0, data_source nq, success_rate 0.5
    assert branch.non_tensor_batch["data_source"][0] == "nq"
    assert branch.non_tensor_batch["success_rate"][0] == 0.5
    assert branch.non_tensor_batch["tool_callings"][0] == 1.0
    # but the identity/observation columns are the branch's own
    assert branch.non_tensor_batch["obs_text"][0] == "branch-obs"
    assert branch.non_tensor_batch["anchor_obs"][0] == "branch-anchor-1"
    assert branch.non_tensor_batch["episode_rewards"][0] == 1.25


def test_branch_rows_carry_the_fkl_masks():
    runner = make_runner()
    batch = make_main_batch()
    spec = _spec()
    branch = runner._rows_to_dataproto([_branch_traj(runner, spec)], {id(spec): 0}, batch)
    teacher = branch.batch["teacher_token_mask"][0]
    loss = branch.batch["loss_mask"][0][-RESP_LEN:]
    assert teacher.sum().item() == 3
    assert (teacher * loss).sum().item() == 0
    assert loss.sum().item() == len("!next")


def test_rows_to_dataproto_keeps_multi_step_branches_in_order():
    runner = make_runner()
    batch = make_main_batch()
    spec = _spec()
    traj = _branch_traj(runner, spec, step_nums=(1, 2))
    branch = runner._rows_to_dataproto([traj], {id(spec): 0}, batch)
    assert len(branch) == 2
    assert list(branch.non_tensor_batch["step_num"]) == [1, 2]
    # only the branch step has teacher tokens
    assert branch.batch["teacher_token_mask"][0].sum().item() == 3
    assert branch.batch["teacher_token_mask"][1].sum().item() == 0


def test_rows_to_dataproto_drops_rows_without_generated_tensors():
    runner = make_runner()
    batch = make_main_batch()
    spec = _spec()
    traj = _branch_traj(runner, spec)
    runner._row_tensors.clear()  # generation results lost
    assert runner._rows_to_dataproto([traj], {id(spec): 0}, batch) is None


def test_rows_to_dataproto_returns_none_for_no_trajectories():
    runner = make_runner()
    assert runner._rows_to_dataproto([], {}, make_main_batch()) is None


# ---------------------------------------------------------------------------
# run(): schedule gating
# ---------------------------------------------------------------------------

def test_run_returns_none_when_disabled():
    runner = make_runner(**{"algorithm.seed.teacher_branch.enable": False})
    proto, metrics = runner.run(make_main_batch(), global_step=1)
    assert proto is None
    assert metrics["seed/teacher_branch/exit_disabled"] == 1.0


def test_run_respects_start_and_stop_schedule():
    runner = make_runner(**{"algorithm.seed.teacher_branch.start_after_steps": 5})
    proto, metrics = runner.run(make_main_batch(), global_step=1)
    assert proto is None
    assert metrics["seed/teacher_branch/skipped_by_schedule"] == 1.0
    assert metrics["seed/teacher_branch/exit_schedule"] == 1.0

    runner = make_runner(**{"algorithm.seed.teacher_branch.stop_after_steps": 3})
    proto, metrics = runner.run(make_main_batch(), global_step=9)
    assert proto is None
    assert metrics["seed/teacher_branch/skipped_by_schedule"] == 1.0
    assert metrics["seed/teacher_branch/exit_schedule"] == 1.0


def test_run_reports_no_candidates_without_env_kwargs():
    """Every candidate is dropped when the env payload cannot be resolved."""
    runner = make_runner()
    proto, metrics = runner.run(make_main_batch(), global_step=1, env_kwargs=None)
    assert proto is None
    assert metrics["seed/teacher_branch/num_branches"] == 0.0


# ---------------------------------------------------------------------------
# run(): exit accounting
#
# The bug this guards against: a step that emitted num_candidates=21 and then
# returned None in 1ms. Candidates are not branches, and every one of the seven
# exit paths used to be indistinguishable in the metrics.
# ---------------------------------------------------------------------------

def _exit_reason(metrics):
    """The single reason flagged, so a test never passes on a wrong-path exit."""
    flagged = [
        key.rsplit("exit_", 1)[-1]
        for key, value in metrics.items()
        if "/exit_" in key and value == 1.0
    ]
    assert len(flagged) == 1, f"expected exactly one exit reason, got {flagged}"
    return flagged[0]


def test_every_exit_emits_the_whole_funnel():
    # A missing key and a zero key look the same in a dashboard, so all of the
    # funnel fields have to be present even on the earliest exit.
    runner = make_runner(**{"algorithm.seed.teacher_branch.enable": False})
    _, metrics = runner.run(make_main_batch(), global_step=1)
    for name in (
        "num_candidates",
        "num_specs_with_env_kwargs",
        "num_prefixes",
        "num_trajectories",
        "num_rows",
    ):
        assert f"seed/teacher_branch/{name}" in metrics
    for reason in br._EXIT_REASONS:
        assert f"seed/teacher_branch/exit_{reason}" in metrics


def test_funnel_pins_the_env_kwargs_drop():
    """num_candidates > 0 while num_specs_with_env_kwargs == 0 is now visible."""
    runner = make_runner()
    proto, metrics = runner.run(make_main_batch(), global_step=1, env_kwargs=None)
    assert proto is None
    assert _exit_reason(metrics) == "no_env_kwargs"
    assert metrics["seed/teacher_branch/num_candidates"] > 0.0
    assert metrics["seed/teacher_branch/num_specs_with_env_kwargs"] == 0.0


def test_funnel_pins_the_prefix_rejection(monkeypatch):
    """env payloads resolve, but the teacher returns nothing usable."""
    runner = make_runner()
    # `run()` builds the teacher chat client before it knows whether it will be
    # used, so it has to be stubbed even though no request is made here.
    monkeypatch.setattr(runner, "_chat_fn", lambda: (lambda messages: ""))
    seen = {}

    def _no_prefixes(specs, chat_fn, max_retries, max_workers, **kwargs):
        seen["specs"] = len(specs)
        return {}, {i: "rejected by the quality gate" for i in range(len(specs))}

    monkeypatch.setattr(br, "generate_teacher_prefixes", _no_prefixes)
    proto, metrics = runner.run(
        make_main_batch(),
        global_step=1,
        env_kwargs=np.array([{"q": "a"}, {"q": "b"}], dtype=object),
    )

    assert proto is None
    assert seen["specs"] > 0
    assert _exit_reason(metrics) == "no_prefix"
    assert metrics["seed/teacher_branch/num_specs_with_env_kwargs"] > 0.0
    assert metrics["seed/teacher_branch/num_prefixes"] == 0.0
    assert metrics["seed/teacher_branch/prefix_failure_ratio"] == 1.0


def test_strict_mode_raises_instead_of_degrading_silently():
    # Without this, a 150-step production run can finish with an empty FKL term
    # and no failure anywhere in the logs.
    runner = make_runner(**{"algorithm.seed.teacher_branch.strict": True})
    with pytest.raises(br.BranchProducedNothing) as excinfo:
        runner.run(make_main_batch(), global_step=1, env_kwargs=None)
    assert "no_env_kwargs" in str(excinfo.value)
    assert "num_candidates=" in str(excinfo.value)


def test_strict_mode_still_allows_scheduled_off_steps():
    """Being switched off by the schedule is not a failure, even in strict mode."""
    runner = make_runner(
        **{
            "algorithm.seed.teacher_branch.strict": True,
            "algorithm.seed.teacher_branch.start_after_steps": 5,
        }
    )
    proto, metrics = runner.run(make_main_batch(), global_step=1)
    assert proto is None
    assert _exit_reason(metrics) == "schedule"
