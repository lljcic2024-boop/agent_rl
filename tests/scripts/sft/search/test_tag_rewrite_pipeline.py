"""Tests for the tag-SFT rewrite pipeline QC gates and transforms (改造点 1, Search 侧).

Covers the pure, teacher-independent logic of
``scripts/sft/search/tag_rewrite_pipeline.py``: the three QC gates
(format / action fidelity / segment length), response parsing and
reassembly, the original->tag-menu prompt transform, trajectory helpers,
and the end-to-end ``qc_and_export`` bucketing / failure-oversampling /
metrics accounting.
"""
import re

from scripts.sft.search import tag_rewrite_pipeline as m

# The instruction sentence the non-tag Search templates carry; ``to_tag_prompt``
# rewrites this (followed by a newline) into the tag-menu instruction.
ORIG_INSTRUCTION = (
    "You should first conduct reasoning process. "
    "This process MUST be enclosed within <think> </think> tags.\n"
)
PROMPT_WITH_ORIG = "Question: who won?\n" + ORIG_INSTRUCTION + "Answer format: ...\n"

DEFAULT_ERROR_RE = re.compile(
    r"(no\s+(?:relevant\s+)?(?:results?|information)|not\s+found|error|invalid|"
    r"<information>\s*</information>)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Gate 1: parse_tag_segments (format)
# ---------------------------------------------------------------------------

def test_parse_single_segment():
    segs = m.parse_tag_segments("<plan>decompose the question into sub goals</plan>")
    assert segs == [("plan", "decompose the question into sub goals")]


def test_parse_multiple_segments_with_whitespace_between():
    text = "<plan>make a plan</plan>\n  <verify>check the result</verify>"
    segs = m.parse_tag_segments(text)
    assert segs == [("plan", "make a plan"), ("verify", "check the result")]


def test_parse_is_case_insensitive_and_lowercases_tag():
    segs = m.parse_tag_segments("<PLAN>Mixed Case</PLAN>")
    assert segs == [("plan", "Mixed Case")]


def test_parse_rejects_untagged_text_between_segments():
    assert m.parse_tag_segments("<plan>a</plan> junk <verify>b</verify>") is None


def test_parse_rejects_leading_untagged_text():
    assert m.parse_tag_segments("lead in <plan>a</plan>") is None


def test_parse_rejects_trailing_untagged_text():
    assert m.parse_tag_segments("<plan>a</plan> trailing words") is None


def test_parse_rejects_unpaired_tag():
    assert m.parse_tag_segments("<plan>a</plan><verify>b") is None


def test_parse_rejects_unknown_tag():
    assert m.parse_tag_segments("<foo>a</foo>") is None


def test_parse_rejects_empty_and_whitespace():
    assert m.parse_tag_segments("") is None
    assert m.parse_tag_segments("   \n  ") is None


# ---------------------------------------------------------------------------
# Gate 2: check_action_fidelity
# ---------------------------------------------------------------------------

def test_fidelity_true_for_clean_thinking():
    assert m.check_action_fidelity(
        "<plan>decompose the question</plan>", "<search>who won</search>"
    ) is True


def test_fidelity_rejects_action_tag_in_thinking():
    assert m.check_action_fidelity(
        "<plan>let me <search>peek</search></plan>", "<search>who won</search>"
    ) is False


def test_fidelity_rejects_think_or_information_tag_in_thinking():
    assert m.check_action_fidelity(
        "<plan><think>nested</think></plan>", "<search>q</search>"
    ) is False
    assert m.check_action_fidelity(
        "<plan>saw <information>x</information></plan>", "<search>q</search>"
    ) is False


def test_fidelity_rejects_non_normalized_action():
    # action carries stray whitespace; reassembly normalizes and no longer matches
    assert m.check_action_fidelity(
        "<plan>decompose the question</plan>", "<search> who won </search>"
    ) is False


# ---------------------------------------------------------------------------
# Gate 3: check_segment_lengths + make_token_counter
# ---------------------------------------------------------------------------

def test_make_token_counter_falls_back_to_whitespace():
    counter = m.make_token_counter(None)
    assert counter("a b c d") == 4


def test_segment_lengths_within_bounds():
    counter = m.make_token_counter(None)
    segs = [("plan", "a b c"), ("verify", "d e")]
    assert m.check_segment_lengths(segs, counter, 2, 5) is True


def test_segment_lengths_too_short():
    counter = m.make_token_counter(None)
    assert m.check_segment_lengths([("plan", "a")], counter, 2, 5) is False


def test_segment_lengths_too_long():
    counter = m.make_token_counter(None)
    assert m.check_segment_lengths([("plan", "a b c d e f")], counter, 2, 5) is False


# ---------------------------------------------------------------------------
# Response parsing / reassembly
# ---------------------------------------------------------------------------

def test_extract_action_block_search_is_stripped():
    assert m.extract_action_block("<search> who won </search>") == "<search>who won</search>"


def test_extract_action_block_answer():
    assert m.extract_action_block("noise <answer> 42 </answer>") == "<answer>42</answer>"


def test_extract_action_block_prefers_search_when_both_present():
    assert (
        m.extract_action_block("<search>s</search><answer>a</answer>")
        == "<search>s</search>"
    )


def test_extract_action_block_none():
    assert m.extract_action_block("just thinking, no action") is None


def test_split_response_with_think_and_action():
    think, action = m.split_response("<think> reasoning here </think>\n<search>q</search>")
    assert think == "reasoning here"
    assert action == "<search>q</search>"


def test_split_response_without_think():
    think, action = m.split_response("<search>q</search>")
    assert think is None
    assert action == "<search>q</search>"


def test_reassemble_response_shape():
    out = m.reassemble_response("<plan>a</plan>", "<search>q</search>")
    assert out == "<think>\n<plan>a</plan>\n</think>\n<search>q</search>"


# ---------------------------------------------------------------------------
# Prompt transform: original -> tag menu
# ---------------------------------------------------------------------------

def test_to_tag_prompt_replaces_instruction():
    out = m.to_tag_prompt(PROMPT_WITH_ORIG)
    assert out is not None
    assert "This process MUST be enclosed within <think> </think> tags." not in out
    assert m.SEARCH_TAG_THINK_INSTRUCTION in out


def test_to_tag_prompt_passthrough_when_already_tagged():
    already = "prefix\n" + m.SEARCH_TAG_THINK_INSTRUCTION + "\nsuffix"
    assert m.to_tag_prompt(already) == already


def test_to_tag_prompt_none_when_no_instruction():
    assert m.to_tag_prompt("some unrelated prompt text") is None


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def test_build_history_text_first_step():
    assert m.build_history_text([], 0) == "(this is the first step)"


def test_build_history_text_lists_actions():
    steps = [
        {"step_idx": 0, "model_response": "<search>q0</search>", "observation": "obs0"},
    ]
    out = m.build_history_text(steps, 1)
    assert "Step 0" in out
    assert "<search>q0</search>" in out
    assert "obs0" in out


def test_build_history_text_marks_invalid_action():
    steps = [{"step_idx": 0, "model_response": "no action here", "observation": "o"}]
    assert "(invalid action)" in m.build_history_text(steps, 1)


def test_build_history_text_truncates():
    steps = [
        {"step_idx": 0, "model_response": "<search>q</search>", "observation": "x" * 7000},
    ]
    out = m.build_history_text(steps, 1)
    assert out.startswith("...(truncated)...")


def test_is_error_observation():
    assert m.is_error_observation("no relevant results found", DEFAULT_ERROR_RE) is True
    assert m.is_error_observation("<information></information>", DEFAULT_ERROR_RE) is True
    assert m.is_error_observation("here is a solid answer", DEFAULT_ERROR_RE) is False


def test_step_key():
    traj = {"task_id": "t1", "rollout_id": 3}
    assert m.step_key(traj, {"step_idx": 2}) == "t1:3:2"


# ---------------------------------------------------------------------------
# qc_and_export: bucketing / metrics / oversampling / file outputs
# ---------------------------------------------------------------------------

def _candidate(**overrides):
    base = {
        "step_key": "t:0:0",
        "task_id": "t",
        "rollout_id": 0,
        "step_index": 0,
        "data_source": "nq",
        "rewrite_error": None,
        "tagged_think": "<plan>decompose the question into two sub goals</plan>",
        "action_block": "<search>a query here</search>",
        "observation_prompt": PROMPT_WITH_ORIG,
        "observation": "ok",
        "source_success": True,
    }
    base.update(overrides)
    return base


def _run_qc(tmp_path, candidates, **kwargs):
    counter = m.make_token_counter(None)
    params = dict(
        candidates=candidates,
        output_dir=tmp_path,
        count_tokens=counter,
        min_segment_tokens=3,
        max_segment_tokens=100,
        failure_oversample=1.0,
        sft_val_ratio=0.1,
        min_tag_ratio=0.05,
        error_regex=DEFAULT_ERROR_RE,
        seed=2026,
    )
    params.update(kwargs)
    return m.qc_and_export(**params)


def test_qc_buckets_each_failure_category(tmp_path):
    candidates = [
        _candidate(  # pass + final answer + verify
            step_key="t:0:0",
            tagged_think="<verify>check the candidate answer against every constraint</verify>",
            action_block="<answer>42</answer>",
            observation="done",
            source_success=True,
        ),
        _candidate(  # pass + error step + reflect
            step_key="t:0:1",
            tagged_think="<reflect>the search returned no results so this route failed</reflect>",
            action_block="<search>alternate query terms</search>",
            observation="no relevant results found",
            source_success=False,
        ),
        _candidate(  # fail: format
            step_key="t:0:2",
            tagged_think="just plain untagged reasoning text",
        ),
        _candidate(  # fail: api / no action
            step_key="t:0:3",
            rewrite_error="timeout",
            tagged_think="",
        ),
        _candidate(  # fail: action fidelity (non-normalized action)
            step_key="t:0:4",
            tagged_think="<plan>decompose the question into sub goals</plan>",
            action_block="<search> spaced query </search>",
        ),
        _candidate(  # fail: segment length (single token < min 3)
            step_key="t:0:5",
            tagged_think="<plan>tiny</plan>",
        ),
        _candidate(  # fail: prompt transform (no instruction to replace)
            step_key="t:0:6",
            tagged_think="<plan>decompose the question into sub goals</plan>",
            observation_prompt="prompt without the reasoning instruction",
        ),
    ]
    metrics = _run_qc(tmp_path, candidates)
    qc = metrics["qc"]
    assert qc["total"] == 7
    assert qc["pass"] == 2
    assert qc["fail_format"] == 1
    assert qc["fail_api_or_no_action"] == 1
    assert qc["fail_action_fidelity"] == 1
    assert qc["fail_segment_length"] == 1
    assert qc["fail_prompt_transform"] == 1
    assert metrics["qc_pass_rate"] == 2 / 7
    assert metrics["unique_passed_records"] == 2
    assert metrics["exported_records"] == 2


def test_qc_tag_distribution_and_conditional_rates(tmp_path):
    candidates = [
        _candidate(
            step_key="t:0:0",
            tagged_think="<verify>check the candidate answer against every constraint</verify>",
            action_block="<answer>42</answer>",
            observation="done",
        ),
        _candidate(
            step_key="t:0:1",
            tagged_think="<reflect>the search returned no results so this route failed</reflect>",
            action_block="<search>alternate query terms</search>",
            observation="no relevant results found",
            source_success=False,
        ),
    ]
    metrics = _run_qc(tmp_path, candidates)
    dist = metrics["tag_distribution"]
    assert dist["verify"] == 0.5
    assert dist["reflect"] == 0.5
    assert dist["plan"] == 0.0
    assert dist["backtrack"] == 0.0
    assert set(metrics["low_coverage_tags"]) == {"plan", "backtrack"}
    assert metrics["reflect_given_error_rate"] == 1.0
    assert metrics["verify_given_final_rate"] == 1.0


def test_qc_failure_oversampling_and_files(tmp_path):
    candidates = [
        _candidate(  # passing, from a successful trajectory (not oversampled)
            step_key="t:0:0",
            tagged_think="<plan>decompose the question into two sub goals</plan>",
            action_block="<search>first query</search>",
            source_success=True,
        ),
        _candidate(  # passing, from a failed trajectory (oversampled)
            step_key="t:0:1",
            tagged_think="<reflect>the search returned no results so this route failed</reflect>",
            action_block="<search>second query</search>",
            observation="no relevant results found",
            source_success=False,
        ),
    ]
    metrics = _run_qc(tmp_path, candidates, failure_oversample=3.0)
    # 2 unique passing + 2 extra copies of the single failed-trajectory record
    assert metrics["unique_passed_records"] == 2
    assert metrics["exported_records"] == 4

    all_jsonl = tmp_path / "tag_sft_all.jsonl"
    assert all_jsonl.exists()
    assert sum(1 for _ in all_jsonl.open()) == 4
    assert (tmp_path / "tag_metrics.json").exists()
    assert (tmp_path / "tag_sft_train.parquet").exists()
    assert (tmp_path / "tag_sft_val.parquet").exists()
