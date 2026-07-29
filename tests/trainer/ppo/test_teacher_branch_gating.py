"""Gating tests for the 改造点 4 hook inside `RayPPOTrainer`.

`_is_teacher_branch_enabled` and `_use_loss_mask` decide whether branch rollout
runs and whether the whole downstream pipeline (`apply_kl_penalty`,
`compute_advantage`, `update_actor`) reads `loss_mask` instead of
`attention_mask`. Getting them wrong either silently trains on a_T tokens as if
the student had written them, or silently drops the FKL term. The methods only
touch `self.config`, so they are exercised unbound against a stub.
"""
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _Stub:
    """Carries just the config attribute the two methods read."""

    def __init__(self, config):
        self.config = config

    _config_bool = staticmethod(RayPPOTrainer._config_bool)
    _is_teacher_branch_enabled = RayPPOTrainer._is_teacher_branch_enabled
    _use_loss_mask = RayPPOTrainer._use_loss_mask


def _stub(branch_enable=False, fkl_coef=0.0, multi_turn=False):
    return _Stub(
        OmegaConf.create(
            {
                "algorithm": {"seed": {"teacher_branch": {"enable": branch_enable}}},
                "actor_rollout_ref": {
                    "actor": {"opd_fkl_loss_coef": fkl_coef},
                    "rollout": {"multi_turn": {"enable": multi_turn}},
                },
            }
        )
    )


def test_branch_needs_both_the_switch_and_the_fkl_coefficient():
    assert _stub(branch_enable=True, fkl_coef=0.05)._is_teacher_branch_enabled() is True
    assert _stub(branch_enable=False, fkl_coef=0.05)._is_teacher_branch_enabled() is False


def test_branch_self_disables_when_fkl_is_off(capsys):
    """a_T tokens are excluded from the PG mask, so fkl=0 means no loss at all."""
    stub = _stub(branch_enable=True, fkl_coef=0.0)
    assert stub._is_teacher_branch_enabled() is False
    assert "opd_fkl_loss_coef=0" in capsys.readouterr().out


def test_branch_config_missing_entirely_is_disabled():
    stub = _Stub(
        OmegaConf.create(
            {
                "algorithm": {"seed": {}},
                "actor_rollout_ref": {
                    "actor": {"opd_fkl_loss_coef": 0.05},
                    "rollout": {"multi_turn": {"enable": False}},
                },
            }
        )
    )
    assert stub._is_teacher_branch_enabled() is False


def test_loss_mask_is_used_when_branches_run():
    assert _stub(branch_enable=True, fkl_coef=0.05)._use_loss_mask() is True


def test_loss_mask_still_follows_multi_turn_rollout():
    assert _stub(multi_turn=True)._use_loss_mask() is True
    assert _stub()._use_loss_mask() is False
