from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class SkillGenRewardConfig:
    downstream_gain_coef: float = 1.0
    valid_json_bonus: float = 0.2
    non_empty_skill_bonus: float = 0.2
    too_long_penalty: float = 0.2
    max_output_chars: int = 1200
    reward_clip: Optional[float] = 2.0


def has_non_empty_skill(episode_skill: object, step_skills: object) -> bool:
    if str(episode_skill or "").strip():
        return True
    if isinstance(step_skills, Mapping):
        return any(str(skill or "").strip() for skill in step_skills.values())
    if isinstance(step_skills, Iterable) and not isinstance(step_skills, (str, bytes)):
        return any(str(skill or "").strip() for skill in step_skills)
    return bool(str(step_skills or "").strip())


def build_skill_text(episode_skill: object, step_skills: object) -> str:
    parts = [str(episode_skill or "").strip()]
    if isinstance(step_skills, Mapping):
        parts.extend(str(skill or "").strip() for _, skill in sorted(step_skills.items(), key=lambda item: str(item[0])))
    elif isinstance(step_skills, Iterable) and not isinstance(step_skills, (str, bytes)):
        parts.extend(str(skill or "").strip() for skill in step_skills)
    else:
        parts.append(str(step_skills or "").strip())
    return "\n".join(part for part in parts if part)


def compute_skill_gen_reward(
    *,
    downstream_logprob_gain: float,
    episode_success: Optional[float],
    valid_json: bool,
    episode_skill: object,
    step_skills: object,
    raw_output: object,
    config: SkillGenRewardConfig,
) -> Dict[str, float]:
    success = episode_success is None or float(episode_success) >= 1.0
    gain = float(downstream_logprob_gain)
    if not success:
        gain = 0.0

    non_empty_skill = has_non_empty_skill(episode_skill, step_skills)
    raw_output_chars = len(str(raw_output or ""))
    too_long = config.max_output_chars > 0 and raw_output_chars > int(config.max_output_chars)

    downstream_component = config.downstream_gain_coef * gain
    valid_json_component = config.valid_json_bonus if valid_json else 0.0
    non_empty_component = config.non_empty_skill_bonus if non_empty_skill else 0.0
    too_long_component = config.too_long_penalty if too_long else 0.0
    reward = (
        downstream_component
        + valid_json_component
        + non_empty_component
        - too_long_component
    )
    if config.reward_clip is not None and float(config.reward_clip) > 0:
        clip = float(config.reward_clip)
        reward = max(min(reward, clip), -clip)

    return {
        "reward": float(reward),
        "downstream_logprob_gain_on_success_steps": float(downstream_component),
        "valid_json_bonus": float(valid_json_component),
        "non_empty_skill_bonus": float(non_empty_component),
        "too_long_penalty": float(too_long_component),
        "valid_json": 1.0 if valid_json else 0.0,
        "non_empty_skill": 1.0 if non_empty_skill else 0.0,
        "too_long": 1.0 if too_long else 0.0,
        "raw_output_chars": float(raw_output_chars),
    }
