#!/usr/bin/env python3
"""Build tag-SFT data for Search QA: rewrite rollout thinking into function-typed tags.

Consumes `baseline_rollouts.jsonl` produced by scripts/sft/search/pipeline.py
(stage 1, baseline rollouts). For every step of every trajectory, a teacher
model rewrites the free-form `<think>` content into function-typed segments
(<plan>/<verify>/<reflect>/<backtrack>) while the action and observation are
preserved verbatim. Three QC gates are applied per the experiment plan:

  1. format: the rewritten thinking must consist solely of properly paired
     tag segments, with no untagged text;
  2. action fidelity: the reassembled response must parse to exactly the
     original action, and no action tags may appear inside the thinking;
  3. segment length: every tag segment must be within [min, max] tokens.

Exports per-step SFT records (prompt uses the tag-menu template so that
SFT/RL/eval share the same prompt) as JSONL + train/val parquet compatible
with scripts/sft/search/train_sft.sh (data.prompt_key=prompt,
data.response_key=response).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from agent_system.environments.prompts.search import (  # noqa: E402
    SEARCH_TAG_THINK_INSTRUCTION,
)
from scripts.sft._common.pipeline import (  # noqa: E402
    ChatEndpoint,
    OpenAITextClient,
    append_jsonl,
    log_stage,
    read_jsonl,
    setup_logging,
    update_progress,
    write_json,
)

TAG_NAMES = ("plan", "verify", "reflect", "backtrack")
TAG_SEGMENT_RE = re.compile(
    r"<(plan|verify|reflect|backtrack)>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# The instruction sentence in the original (non-tag) SEARCH templates,
# replaced by the tag-menu instruction when exporting SFT prompts.
# Matched as a regex to be robust to trailing-whitespace differences.
ORIGINAL_THINK_INSTRUCTION_RE = re.compile(
    r"You should first conduct reasoning process\. "
    r"This process MUST be enclosed within <think> </think> tags\.[ \t]*\n"
)

REWRITE_SYSTEM_PROMPT = """You are an expert at reorganizing an agent's reasoning into function-typed thinking segments. You will be given one step of a search agent's trajectory: the task, the interaction history, the current-step prompt the agent saw, and the agent's original response (free-form thinking plus one action).

Rewrite ONLY the thinking. Your output must follow these rules exactly:
1. Output one or more thinking segments, each enclosed in exactly one of these tags:
   <plan> Planning: decompose the question into sub-goals and decide what to do next. Used when the task starts, or the previous step went well and the agent needs to move forward.
   <verify> Verification: check whether the latest search results match expectations, and check the candidate answer against ALL constraints in the question one by one. Used every time search results arrive, and always before the final answer.
   <reflect> Reflection: diagnose what went wrong (empty or contradictory search results, errors) and analyze the cause. Used when any anomalous signal appears.
   <backtrack> Backtracking: conclude the current approach is a dead end, explicitly abandon it, and state the alternative route. Used when reflection shows the current direction is useless.
2. Preserve the meaning and factual content of the original thinking; do not invent new facts, do not change the conclusion that led to the action.
3. Every part of your output must be inside exactly one tag. No text outside tags. Do not nest tags.
4. Do NOT output <think>, <search>, <answer>, the action, or anything else — only the tagged thinking segments.
5. Keep each segment concise (roughly 20 to 300 tokens)."""

REWRITE_USER_TEMPLATE = """Task question: {task_description}

Interaction history before this step:
{history}

Current-step observation/prompt shown to the agent:
{observation}

The agent's original response at this step:
{response}

The action the agent took (must remain unchanged, do NOT output it):
{action}

Now rewrite the agent's thinking for this step into function-typed segments. Remember: output ONLY the tagged segments."""


# ---------------------------------------------------------------------------
# Response parsing / reassembly
# ---------------------------------------------------------------------------

def split_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (think_text, action_block) from an original model response."""
    think_match = THINK_RE.search(response)
    think_text = think_match.group(1).strip() if think_match else None
    action_block = extract_action_block(response)
    return think_text, action_block


def extract_action_block(text: str) -> Optional[str]:
    """First complete <search>/<answer> block, normalized like the env projection."""
    m = SEARCH_RE.search(text)
    if m:
        return f"<search>{m.group(1).strip()}</search>"
    m = ANSWER_RE.search(text)
    if m:
        return f"<answer>{m.group(1).strip()}</answer>"
    return None


def reassemble_response(tagged_think: str, action_block: str) -> str:
    return f"<think>\n{tagged_think.strip()}\n</think>\n{action_block}"


# ---------------------------------------------------------------------------
# QC gates
# ---------------------------------------------------------------------------

def parse_tag_segments(tagged_think: str) -> Optional[List[Tuple[str, str]]]:
    """Gate 1 (format): text must be fully covered by paired tag segments.

    Returns the list of (tag, content) segments, or None if the format is invalid.
    """
    text = tagged_think.strip()
    if not text:
        return None
    segments: List[Tuple[str, str]] = []
    cursor = 0
    for match in TAG_SEGMENT_RE.finditer(text):
        between = text[cursor:match.start()]
        if between.strip():
            return None  # untagged content between segments
        segments.append((match.group(1).lower(), match.group(2).strip()))
        cursor = match.end()
    if text[cursor:].strip():
        return None  # trailing untagged content
    if not segments:
        return None
    # any leftover raw tag tokens indicate unpaired/nested tags
    stripped = TAG_SEGMENT_RE.sub("", text)
    if re.search(r"</?(plan|verify|reflect|backtrack)>", stripped, re.IGNORECASE):
        return None
    return segments


def check_action_fidelity(tagged_think: str, original_action: str) -> bool:
    """Gate 2: no action/think tags inside the thinking; reassembled action identical."""
    if re.search(r"</?(search|answer|think|information)>", tagged_think, re.IGNORECASE):
        return False
    reassembled = reassemble_response(tagged_think, original_action)
    return extract_action_block(reassembled) == original_action


def make_token_counter(tokenizer_path: Optional[str]):
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    return lambda text: len(text.split())


def check_segment_lengths(
    segments: Sequence[Tuple[str, str]],
    count_tokens,
    min_tokens: int,
    max_tokens: int,
) -> bool:
    """Gate 3: every segment within [min_tokens, max_tokens]."""
    return all(min_tokens <= count_tokens(content) <= max_tokens for _, content in segments)


# ---------------------------------------------------------------------------
# Prompt transformation (original template -> tag-menu template)
# ---------------------------------------------------------------------------

def to_tag_prompt(observation_prompt: str) -> Optional[str]:
    """Rewrite the recorded per-step prompt to declare the tag capability menu."""
    transformed, n_replaced = ORIGINAL_THINK_INSTRUCTION_RE.subn(
        SEARCH_TAG_THINK_INSTRUCTION + "\n",
        observation_prompt,
        count=1,
    )
    if n_replaced:
        return transformed
    if SEARCH_TAG_THINK_INSTRUCTION in observation_prompt:
        return observation_prompt  # rollouts already used the tag template
    return None


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def build_history_text(steps: Sequence[Dict[str, Any]], upto: int, max_chars: int = 6000) -> str:
    parts: List[str] = []
    for step in steps[:upto]:
        action = extract_action_block(str(step.get("model_response", ""))) or "(invalid action)"
        observation = str(step.get("observation", "")).strip()
        parts.append(f"Step {step.get('step_idx', '?')}: {action}\nResult: {observation}")
    history = "\n".join(parts) if parts else "(this is the first step)"
    if len(history) > max_chars:
        history = "...(truncated)...\n" + history[-max_chars:]
    return history


def is_error_observation(observation: str, error_regex: re.Pattern) -> bool:
    return bool(error_regex.search(observation or ""))


def step_key(traj: Dict[str, Any], step: Dict[str, Any]) -> str:
    return f"{traj['task_id']}:{traj['rollout_id']}:{step.get('step_idx')}"


# ---------------------------------------------------------------------------
# Rewrite stage
# ---------------------------------------------------------------------------

def build_rewrite_messages(
    traj: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    step_pos: int,
) -> Optional[List[Dict[str, str]]]:
    step = steps[step_pos]
    response = str(step.get("model_response", ""))
    _, action_block = split_response(response)
    if action_block is None:
        return None  # invalid-action step; cannot supervise an action-preserving rewrite
    user = REWRITE_USER_TEMPLATE.format(
        task_description=traj.get("task_description", ""),
        history=build_history_text(steps, step_pos),
        observation=str(step.get("observation", "")).strip()[:4000],
        response=response.strip()[:6000],
        action=action_block,
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def rewrite_one_step(
    *,
    traj: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    step_pos: int,
    client: OpenAITextClient,
) -> Dict[str, Any]:
    step = steps[step_pos]
    record: Dict[str, Any] = {
        "step_key": step_key(traj, step),
        "task_id": traj["task_id"],
        "rollout_id": int(traj["rollout_id"]),
        "step_index": int(step.get("step_idx", step_pos)),
        "num_steps": len(steps),
        "is_last_step": step_pos == len(steps) - 1,
        "source_success": bool(traj.get("success", False)),
        "data_source": traj.get("data_source", "unknown"),
        "task_description": traj.get("task_description", ""),
        "observation": str(step.get("observation", "")),
        "observation_prompt": str(step.get("observation_prompt") or step.get("observation", "")),
        "original_response": str(step.get("model_response", "")),
    }
    messages = build_rewrite_messages(traj, steps, step_pos)
    if messages is None:
        record.update({"rewrite_error": "no_parsable_action", "tagged_think": ""})
        return record
    _, action_block = split_response(record["original_response"])
    record["action_block"] = action_block
    raw_output, api_error = client.complete(messages)
    record["tagged_think"] = raw_output.strip()
    record["rewrite_error"] = api_error
    return record


def run_rewrites(
    *,
    rollouts: Sequence[Dict[str, Any]],
    endpoint: ChatEndpoint,
    output_dir: Path,
    workers: int,
    resume: bool,
    max_steps_per_traj: Optional[int],
) -> List[Dict[str, Any]]:
    candidates_path = output_dir / "tag_rewrite_candidates.jsonl"
    existing = read_jsonl(candidates_path) if (resume and candidates_path.exists()) else []
    done_keys = {record["step_key"] for record in existing}
    results = list(existing)

    specs: List[Tuple[Dict[str, Any], List[Dict[str, Any]], int]] = []
    for traj in rollouts:
        steps = list(traj.get("steps", []))
        if max_steps_per_traj:
            steps = steps[:max_steps_per_traj]
        for pos in range(len(steps)):
            key = step_key(traj, steps[pos])
            if key in done_keys:
                continue
            specs.append((traj, steps, pos))

    log_stage(
        output_dir,
        "tag_rewrite",
        "running",
        existing=len(existing),
        pending=len(specs),
    )
    if not specs:
        return results

    client = OpenAITextClient(endpoint)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(rewrite_one_step, traj=traj, steps=steps, step_pos=pos, client=client)
            for traj, steps, pos in specs
        ]
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(candidates_path, record)
            results.append(record)
            completed += 1
            if completed % 50 == 0 or completed == len(specs):
                update_progress(
                    output_dir,
                    stage="tag_rewrite",
                    status="running",
                    completed_rewrites=len(results),
                    pending_rewrites=len(specs) - completed,
                )
    log_stage(output_dir, "tag_rewrite", "complete", total=len(results))
    return results


# ---------------------------------------------------------------------------
# QC + export stage
# ---------------------------------------------------------------------------

def qc_and_export(
    *,
    candidates: Sequence[Dict[str, Any]],
    output_dir: Path,
    count_tokens,
    min_segment_tokens: int,
    max_segment_tokens: int,
    failure_oversample: float,
    sft_val_ratio: float,
    min_tag_ratio: float,
    error_regex: re.Pattern,
    seed: int,
) -> Dict[str, Any]:
    qc_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    passed: List[Dict[str, Any]] = []
    reflect_given_error = Counter()
    verify_given_final = Counter()

    for record in candidates:
        qc_counter["total"] += 1
        if record.get("rewrite_error"):
            qc_counter["fail_api_or_no_action"] += 1
            continue
        tagged_think = str(record.get("tagged_think", ""))
        action_block = record.get("action_block")
        segments = parse_tag_segments(tagged_think)
        if segments is None:
            qc_counter["fail_format"] += 1
            continue
        if not action_block or not check_action_fidelity(tagged_think, action_block):
            qc_counter["fail_action_fidelity"] += 1
            continue
        if not check_segment_lengths(segments, count_tokens, min_segment_tokens, max_segment_tokens):
            qc_counter["fail_segment_length"] += 1
            continue

        prompt = to_tag_prompt(record["observation_prompt"])
        if prompt is None:
            qc_counter["fail_prompt_transform"] += 1
            continue

        qc_counter["pass"] += 1
        seg_tags = [tag for tag, _ in segments]
        tag_counter.update(seg_tags)

        is_error_step = is_error_observation(record.get("observation", ""), error_regex)
        if is_error_step:
            reflect_given_error["total"] += 1
            if "reflect" in seg_tags:
                reflect_given_error["hit"] += 1
        is_final_answer = bool(action_block and action_block.startswith("<answer>"))
        if is_final_answer:
            verify_given_final["total"] += 1
            if "verify" in seg_tags:
                verify_given_final["hit"] += 1

        passed.append(
            {
                "prompt": prompt,
                "response": reassemble_response(tagged_think, action_block),
                "step_key": record["step_key"],
                "task_id": record["task_id"],
                "rollout_id": int(record["rollout_id"]),
                "step_index": int(record["step_index"]),
                "data_source": record.get("data_source", "unknown"),
                "source_success": bool(record.get("source_success", False)),
                "is_error_step": is_error_step,
                "is_final_answer": is_final_answer,
                "tags": ",".join(seg_tags),
            }
        )

    # Failure-trajectory oversampling (failed trajectories carry the rare
    # reflect/backtrack supervision).
    export_records = list(passed)
    if failure_oversample > 1.0:
        rng = random.Random(seed)
        failed_records = [r for r in passed if not r["source_success"]]
        extra_copies = failure_oversample - 1.0
        for record in failed_records:
            copies = int(extra_copies)
            if rng.random() < (extra_copies - copies):
                copies += 1
            export_records.extend(dict(record) for _ in range(copies))

    all_jsonl = output_dir / "tag_sft_all.jsonl"
    if all_jsonl.exists():
        all_jsonl.unlink()
    for record in export_records:
        append_jsonl(all_jsonl, record)

    if export_records:
        shuffled = list(export_records)
        random.Random(seed).shuffle(shuffled)
        val_size = int(round(len(shuffled) * sft_val_ratio))
        if len(shuffled) > 1:
            val_size = max(1, min(val_size, len(shuffled) - 1))
        val_records, train_records = shuffled[:val_size], shuffled[val_size:]
        try:
            import pandas as pd

            pd.DataFrame(train_records).to_parquet(output_dir / "tag_sft_train.parquet")
            pd.DataFrame(val_records).to_parquet(output_dir / "tag_sft_val.parquet")
        except Exception as exc:  # pragma: no cover
            logging.warning("Could not write parquet SFT exports: %s", exc)
    else:
        logging.warning("No records passed QC; nothing exported.")

    total_tags = sum(tag_counter.values()) or 1
    tag_distribution = {tag: tag_counter.get(tag, 0) / total_tags for tag in TAG_NAMES}
    low_tags = [tag for tag, ratio in tag_distribution.items() if ratio < min_tag_ratio]
    if low_tags:
        logging.warning(
            "Tags below the %.0f%% coverage threshold: %s. Consider synthesizing extra "
            "demonstrations for them (see experiment plan 2.2).",
            min_tag_ratio * 100,
            low_tags,
        )

    metrics = {
        "qc": dict(qc_counter),
        "qc_pass_rate": qc_counter["pass"] / max(qc_counter["total"], 1),
        "tag_distribution": tag_distribution,
        "low_coverage_tags": low_tags,
        "reflect_given_error_rate": (
            reflect_given_error["hit"] / reflect_given_error["total"]
            if reflect_given_error["total"]
            else None
        ),
        "verify_given_final_rate": (
            verify_given_final["hit"] / verify_given_final["total"]
            if verify_given_final["total"]
            else None
        ),
        "exported_records": len(export_records),
        "unique_passed_records": len(passed),
    }
    write_json(output_dir / "tag_metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, help="Path to baseline_rollouts.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-api-key", default="EMPTY")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-temperature", type=float, default=0.3)
    parser.add_argument("--teacher-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--teacher-timeout", type=float, default=180.0)
    parser.add_argument("--teacher-retries", type=int, default=3)
    parser.add_argument("--teacher-retry-delay", type=float, default=1.0)
    parser.add_argument("--rewrite-workers", type=int, default=32)
    parser.add_argument("--max-steps-per-traj", type=int, default=None)
    parser.add_argument("--tokenizer-path", default=None, help="HF tokenizer for segment length QC; falls back to whitespace tokens")
    parser.add_argument("--min-segment-tokens", type=int, default=20)
    parser.add_argument("--max-segment-tokens", type=int, default=300)
    parser.add_argument("--failure-oversample", type=float, default=2.0, help="Duplication factor for records from failed trajectories (1.0 disables)")
    parser.add_argument("--min-tag-ratio", type=float, default=0.05)
    parser.add_argument("--sft-val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--error-obs-regex",
        default=r"(no\s+(?:relevant\s+)?(?:results?|information)|not\s+found|error|invalid|<information>\s*</information>)",
        help="Case-insensitive regex marking a tool-error/empty observation",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "INFO")

    rollouts = read_jsonl(Path(args.rollouts))
    logging.info("Loaded %d rollout trajectories from %s", len(rollouts), args.rollouts)

    endpoint = ChatEndpoint(
        base_url=args.teacher_base_url,
        api_key=args.teacher_api_key,
        model=args.teacher_model,
        temperature=args.teacher_temperature,
        max_completion_tokens=args.teacher_max_completion_tokens,
        timeout=args.teacher_timeout,
        retries=args.teacher_retries,
        retry_delay=args.teacher_retry_delay,
        extra_body=None,
    )

    candidates = run_rewrites(
        rollouts=rollouts,
        endpoint=endpoint,
        output_dir=output_dir,
        workers=args.rewrite_workers,
        resume=args.resume,
        max_steps_per_traj=args.max_steps_per_traj,
    )

    count_tokens = make_token_counter(args.tokenizer_path)
    error_regex = re.compile(args.error_obs_regex, re.IGNORECASE)
    metrics = qc_and_export(
        candidates=candidates,
        output_dir=output_dir,
        count_tokens=count_tokens,
        min_segment_tokens=args.min_segment_tokens,
        max_segment_tokens=args.max_segment_tokens,
        failure_oversample=args.failure_oversample,
        sft_val_ratio=args.sft_val_ratio,
        min_tag_ratio=args.min_tag_ratio,
        error_regex=error_regex,
        seed=args.seed,
    )
    logging.info("tag-SFT metrics: %s", json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
