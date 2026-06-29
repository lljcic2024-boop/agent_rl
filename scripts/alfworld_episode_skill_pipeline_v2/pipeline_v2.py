#!/usr/bin/env python3
"""Generate episode-skill SFT data from ALFWorld baseline rollouts.

Version 2 intentionally skips downstream skill validation. It reads existing
baseline rollouts, regenerates candidate episode skills with the current OPID
episode-only analyzer prompt, and exports parseable candidates directly as SFT
records.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.alfworld_episode_skill_pipeline.pipeline import (  # noqa: E402
    ChatEndpoint,
    append_jsonl,
    build_candidate_skill_record,
    json_safe,
    load_env_file,
    log_stage,
    read_jsonl,
    resolve_endpoint,
    setup_logging,
    update_progress,
    write_json,
)
from concurrent.futures import ThreadPoolExecutor, as_completed


OUTPUT_FILES = [
    "baseline_rollouts.jsonl",
    "candidate_skills.jsonl",
    "sft_episode_skill_all.jsonl",
    "sft_episode_skill_train.parquet",
    "sft_episode_skill_val.parquet",
    "metrics.json",
    "progress.json",
    "run_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs/alfworld_episode_skill_pipeline_v2")
    parser.add_argument(
        "--baseline-rollouts",
        default="outputs/alfworld_episode_skill_pipeline_qwen25_3b/baseline_rollouts.jsonl",
        help="Existing baseline_rollouts.jsonl to reuse.",
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--skill-gen-workers", type=int, default=128)
    parser.add_argument("--sft-val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--regenerate-candidates",
        action="store_true",
        help="Delete existing candidate/SFT outputs before generating candidates.",
    )
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--skill-base-url", default=None)
    parser.add_argument("--skill-api-key", default=None)
    parser.add_argument("--skill-model", default=None)
    parser.add_argument("--skill-temperature", type=float, default=0.0)
    parser.add_argument("--skill-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--skill-timeout", type=float, default=120.0)
    parser.add_argument("--skill-retries", type=int, default=5)
    parser.add_argument("--skill-retry-delay", type=float, default=1.0)
    parser.add_argument("--skill-extra-body-json", default=None)
    return parser.parse_args()


def prepare_output_dir(
    output_dir: Path,
    *,
    overwrite: bool,
    resume: bool,
    regenerate_candidates: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if overwrite:
        for path in existing:
            path.unlink()
        return
    if regenerate_candidates:
        for name in (
            "candidate_skills.jsonl",
            "sft_episode_skill_all.jsonl",
            "sft_episode_skill_train.parquet",
            "sft_episode_skill_val.parquet",
            "metrics.json",
        ):
            path = output_dir / name
            if path.exists():
                path.unlink()
        return
    if existing and not resume:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. "
            "Use --resume, --overwrite, or --regenerate-candidates."
        )


def copy_baseline_rollouts(source: Path, output_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"baseline rollouts not found: {source}")
    target = output_dir / "baseline_rollouts.jsonl"
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


def generate_candidate_skills_v2(
    *,
    baseline_rollouts: Sequence[Dict[str, Any]],
    output_dir: Path,
    skill_endpoint: ChatEndpoint,
    max_candidates: int | None,
    max_workers: int,
    resume: bool,
) -> List[Dict[str, Any]]:
    path = output_dir / "candidate_skills.jsonl"
    existing = read_jsonl(path) if resume else []
    existing_ids = {record["skill_id"] for record in existing}
    records = list(existing)
    rollouts = list(baseline_rollouts)
    if max_candidates is not None:
        rollouts = rollouts[: max(0, max_candidates)]

    pending = [
        trajectory
        for trajectory in rollouts
        if f"{trajectory['task_id']}:{trajectory['rollout_id']}" not in existing_ids
    ]
    expected_total = len(rollouts)
    log_stage(
        output_dir,
        "skill_generation",
        "running",
        existing_skills=len(existing),
        pending_skills=len(pending),
        expected_skills=expected_total,
        skill_gen_workers=int(max_workers),
    )
    if not pending:
        log_stage(
            output_dir,
            "skill_generation",
            "complete",
            completed_skills=len(records),
            expected_skills=expected_total,
            parse_ok_skills=sum(1 for record in records if record.get("parse_ok")),
        )
        return records

    worker_count = max(1, min(int(max_workers), len(pending)))
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_trajectory = {
            executor.submit(
                build_candidate_skill_record,
                trajectory=trajectory,
                skill_endpoint=skill_endpoint,
            ): trajectory
            for trajectory in pending
        }
        for future in as_completed(future_to_trajectory):
            trajectory = future_to_trajectory[future]
            skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
            completed += 1
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "skill_id": skill_id,
                    "task_id": trajectory["task_id"],
                    "task_type": trajectory["task_type"],
                    "game_file": trajectory["game_file"],
                    "source_rollout_id": trajectory["rollout_id"],
                    "source_success": bool(trajectory.get("success", False)),
                    "source_num_steps": int(trajectory.get("num_steps", 0)),
                    "task_description": trajectory.get("task_description", ""),
                    "analysis_prompt": None,
                    "llm_raw_output": "",
                    "episode_summary": "",
                    "episode_skill": "",
                    "parse_ok": False,
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(path, record)
            records.append(record)
            logging.info(
                "Generated candidate skill %d/%d: %s parse_ok=%s.",
                completed,
                len(pending),
                skill_id,
                record.get("parse_ok"),
            )
            update_progress(
                output_dir,
                stage="skill_generation",
                status="running",
                completed_in_current_run=completed,
                pending_in_current_run=len(pending),
                completed_skills=len(records),
                expected_skills=expected_total,
                last_skill_id=skill_id,
                last_parse_ok=record.get("parse_ok"),
            )

    log_stage(
        output_dir,
        "skill_generation",
        "complete",
        completed_skills=len(records),
        expected_skills=expected_total,
        parse_ok_skills=sum(1 for record in records if record.get("parse_ok")),
    )
    return records


def build_sft_exports_from_candidates(
    *,
    candidate_skills: Sequence[Dict[str, Any]],
    output_dir: Path,
    sft_val_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    sft_records: List[Dict[str, Any]] = []
    for candidate in candidate_skills:
        if not candidate.get("parse_ok"):
            continue
        prompt = candidate.get("analysis_prompt", {})
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        if not messages:
            continue
        response_payload = {
            "episode_summary": candidate.get("episode_summary", ""),
            "episode_skill": candidate.get("episode_skill", ""),
        }
        sft_records.append(
            {
                "prompt": str(messages[-1].get("content", "")),
                "response": json.dumps(response_payload, ensure_ascii=False),
                "skill_id": candidate["skill_id"],
                "task_id": candidate["task_id"],
                "task_type": candidate["task_type"],
                "source_success": bool(candidate.get("source_success", False)),
                "source_num_steps": int(candidate.get("source_num_steps", 0)),
                "parse_ok": bool(candidate.get("parse_ok")),
            }
        )

    all_jsonl = output_dir / "sft_episode_skill_all.jsonl"
    if all_jsonl.exists():
        all_jsonl.unlink()
    for record in sft_records:
        append_jsonl(all_jsonl, record)

    if not sft_records:
        logging.warning("No parseable candidate skills; SFT parquet export skipped.")
        return sft_records

    shuffled = list(sft_records)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * sft_val_ratio))
    if len(shuffled) > 1:
        val_size = max(1, min(val_size, len(shuffled) - 1))
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    try:
        import pandas as pd

        pd.DataFrame(train_records).to_parquet(output_dir / "sft_episode_skill_train.parquet")
        pd.DataFrame(val_records).to_parquet(output_dir / "sft_episode_skill_val.parquet")
    except Exception as exc:
        logging.warning("Could not write parquet SFT exports: %s", exc)
    return sft_records


def write_metrics_v2(
    *,
    baseline_rollouts: Sequence[Dict[str, Any]],
    candidate_skills: Sequence[Dict[str, Any]],
    sft_records: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    payload = {
        "version": "v2_no_skill_validation",
        "baseline_rollouts": len(baseline_rollouts),
        "candidate_skills": len(candidate_skills),
        "parse_ok_skills": sum(1 for record in candidate_skills if record.get("parse_ok")),
        "parse_error_skills": sum(1 for record in candidate_skills if not record.get("parse_ok")),
        "sft_records": len(sft_records),
        "candidate_by_task_type": dict(Counter(record.get("task_type", "unknown") for record in candidate_skills)),
        "sft_by_task_type": dict(Counter(record.get("task_type", "unknown") for record in sft_records)),
        "source_success_counts": dict(Counter(str(bool(record.get("source_success", False))) for record in candidate_skills)),
    }
    write_json(output_dir / "metrics.json", payload)


def write_run_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    skill_endpoint: ChatEndpoint,
) -> None:
    redacted_args = vars(args).copy()
    if redacted_args.get("skill_api_key"):
        redacted_args["skill_api_key"] = "<redacted>"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "v2_no_skill_validation",
        "project_root": str(PROJECT_ROOT),
        "argv": sys.argv,
        "args": redacted_args,
        "skill_endpoint": {
            "base_url": skill_endpoint.base_url,
            "api_key": "<redacted>" if skill_endpoint.api_key else None,
            "model": skill_endpoint.model,
            "temperature": skill_endpoint.temperature,
            "max_completion_tokens": skill_endpoint.max_completion_tokens,
        },
    }
    write_json(output_dir / "run_config.json", payload)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    prepare_output_dir(
        output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        regenerate_candidates=args.regenerate_candidates,
    )
    setup_logging(output_dir, args.log_level)

    skill_endpoint = resolve_endpoint(
        prefix="skill",
        args=args,
        default_base_url_env="SKILL_OPENAI_BASE_URL",
        default_model_env="SKILL_OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.skill_temperature,
        max_completion_tokens=args.skill_max_completion_tokens,
        timeout=args.skill_timeout,
        retries=args.skill_retries,
        retry_delay=args.skill_retry_delay,
        extra_body_json=args.skill_extra_body_json,
    )
    write_run_config(args=args, output_dir=output_dir, skill_endpoint=skill_endpoint)

    baseline_path = copy_baseline_rollouts(Path(args.baseline_rollouts), output_dir)
    baseline_rollouts = read_jsonl(baseline_path)
    if args.max_candidates is not None:
        baseline_rollouts_for_generation = baseline_rollouts[: max(0, args.max_candidates)]
    else:
        baseline_rollouts_for_generation = baseline_rollouts
    log_stage(
        output_dir,
        "baseline_rollouts",
        "complete",
        baseline_rollouts=len(baseline_rollouts),
        used_for_generation=len(baseline_rollouts_for_generation),
        source=str(Path(args.baseline_rollouts)),
    )

    candidate_skills = generate_candidate_skills_v2(
        baseline_rollouts=baseline_rollouts_for_generation,
        output_dir=output_dir,
        skill_endpoint=skill_endpoint,
        max_candidates=None,
        max_workers=args.skill_gen_workers,
        resume=args.resume and not args.regenerate_candidates,
    )

    log_stage(output_dir, "sft_export", "running", candidate_skills=len(candidate_skills))
    sft_records = build_sft_exports_from_candidates(
        candidate_skills=candidate_skills,
        output_dir=output_dir,
        sft_val_ratio=args.sft_val_ratio,
        seed=args.seed,
    )
    log_stage(output_dir, "sft_export", "complete", sft_records=len(sft_records))

    write_metrics_v2(
        baseline_rollouts=baseline_rollouts_for_generation,
        candidate_skills=candidate_skills,
        sft_records=sft_records,
        output_dir=output_dir,
    )
    log_stage(
        output_dir,
        "complete",
        "complete",
        baseline_rollouts=len(baseline_rollouts_for_generation),
        candidate_skills=len(candidate_skills),
        parse_ok_skills=sum(1 for record in candidate_skills if record.get("parse_ok")),
        sft_records=len(sft_records),
    )
    logging.info("Pipeline v2 complete. Outputs are in %s.", output_dir)


if __name__ == "__main__":
    main()
