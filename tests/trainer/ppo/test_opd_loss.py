import torch

from verl.trainer.ppo.core_algos import compute_opd_fkl_loss, compute_opd_loss


def test_opd_loss_uses_gated_teacher_gap_on_selected_steps():
    log_prob = torch.tensor(
        [[-2.0, -1.0], [-0.5, -0.25]],
        requires_grad=True,
    )
    teacher_log_prob = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    response_mask = torch.ones_like(log_prob)
    step_mask = torch.tensor([1.0, 0.0])

    loss, active_ratio, gate_mean, gate_active_ratio, gap_mean = compute_opd_loss(
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        opd_step_mask=step_mask,
        gate_beta=1.0,
        loss_agg_mode="token-mean",
    )

    active_gap = teacher_log_prob[0] - log_prob.detach()[0]
    expected_gate = torch.sigmoid(active_gap)
    expected_loss = (expected_gate * (teacher_log_prob[0] - log_prob[0])).mean()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(active_ratio, torch.tensor(0.5))
    torch.testing.assert_close(gate_mean, expected_gate.mean())
    torch.testing.assert_close(gate_active_ratio, torch.tensor(0.5))
    torch.testing.assert_close(gap_mean, torch.tensor(0.0))

    loss.backward()
    expected_grad = torch.zeros_like(log_prob)
    expected_grad[0] = -expected_gate / expected_gate.numel()
    torch.testing.assert_close(log_prob.grad, expected_grad)


def test_opd_loss_rkl_mode_uses_signed_advantage():
    log_prob = torch.tensor(
        [[-2.0, -1.0], [-0.5, -0.25]],
        requires_grad=True,
    )
    teacher_log_prob = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    response_mask = torch.ones_like(log_prob)
    step_mask = torch.tensor([1.0, 0.0])

    loss, active_ratio, weight_mean, weight_active_ratio, gap_mean = compute_opd_loss(
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        opd_step_mask=step_mask,
        loss_mode="rkl",
        loss_agg_mode="token-mean",
    )

    # advantage = teacher_lp - student_lp (signed, no gate)
    advantage = (teacher_log_prob[0] - log_prob.detach()[0])  # [1.0, -1.0]
    expected_loss = (advantage * (teacher_log_prob[0] - log_prob[0])).mean()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(active_ratio, torch.tensor(0.5))
    torch.testing.assert_close(weight_mean, advantage.mean())
    # one positive-advantage token out of two selected tokens
    torch.testing.assert_close(weight_active_ratio, torch.tensor(0.5))
    torch.testing.assert_close(gap_mean, torch.tensor(0.0))

    loss.backward()
    # gradient = -advantage / n on selected tokens: positive-advantage tokens are
    # reinforced, negative-advantage tokens are suppressed
    expected_grad = torch.zeros_like(log_prob)
    expected_grad[0] = -advantage / advantage.numel()
    torch.testing.assert_close(log_prob.grad, expected_grad)


def test_opd_loss_rkl_mode_clips_advantage():
    log_prob = torch.tensor([[-5.0, -1.0]], requires_grad=True)
    teacher_log_prob = torch.tensor([[-1.0, -1.5]])
    response_mask = torch.ones_like(log_prob)

    loss, _, weight_mean, _, _ = compute_opd_loss(
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        loss_mode="rkl",
        rkl_adv_clip=2.0,
        loss_agg_mode="token-mean",
    )

    raw_advantage = (teacher_log_prob - log_prob.detach())[0]  # [4.0, -0.5]
    clipped = raw_advantage.clamp(min=-2.0, max=2.0)  # [2.0, -0.5]
    expected_loss = (clipped * (teacher_log_prob[0] - log_prob[0])).mean()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(weight_mean, clipped.mean())


def test_opd_loss_rejects_unknown_mode():
    log_prob = torch.tensor([[-1.0]])
    teacher_log_prob = torch.tensor([[-1.0]])
    response_mask = torch.ones_like(log_prob)

    try:
        compute_opd_loss(
            log_prob=log_prob,
            teacher_log_prob=teacher_log_prob,
            response_mask=response_mask,
            loss_mode="forward",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown loss_mode")


def test_opd_loss_returns_zero_when_no_steps_are_selected():
    log_prob = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    teacher_log_prob = torch.tensor([[-1.0, -2.0]])
    response_mask = torch.ones_like(log_prob)
    step_mask = torch.tensor([0.0])

    outputs = compute_opd_loss(
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        opd_step_mask=step_mask,
    )

    for value in outputs:
        torch.testing.assert_close(value, torch.tensor(0.0))


def test_opd_fkl_loss_is_cross_entropy_on_teacher_tokens():
    log_prob = torch.tensor(
        [[-2.0, -1.0, -0.5], [-0.5, -0.25, -3.0]],
        requires_grad=True,
    )
    validity = torch.ones_like(log_prob)
    # token-level mask: only two tokens are teacher-generated
    teacher_token_mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    loss, token_ratio, student_lp_mean = compute_opd_fkl_loss(
        log_prob=log_prob,
        teacher_token_mask=teacher_token_mask,
        response_validity_mask=validity,
        loss_agg_mode="token-mean",
    )

    # CE on the two selected tokens: mean(-(-2.0), -(-3.0)) = 2.5
    torch.testing.assert_close(loss, torch.tensor(2.5))
    torch.testing.assert_close(token_ratio, torch.tensor(2.0 / 6.0))
    torch.testing.assert_close(student_lp_mean, torch.tensor(-2.5))

    loss.backward()
    expected_grad = -teacher_token_mask / teacher_token_mask.sum()
    torch.testing.assert_close(log_prob.grad, expected_grad)


def test_opd_fkl_loss_sample_level_mask():
    log_prob = torch.tensor([[-1.0, -2.0], [-4.0, -6.0]], requires_grad=True)
    validity = torch.ones_like(log_prob)
    sample_mask = torch.tensor([0.0, 1.0])

    loss, token_ratio, _ = compute_opd_fkl_loss(
        log_prob=log_prob,
        teacher_token_mask=sample_mask,
        response_validity_mask=validity,
    )

    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(token_ratio, torch.tensor(0.5))


def test_opd_fkl_loss_zero_when_no_teacher_tokens():
    log_prob = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    validity = torch.ones_like(log_prob)
    teacher_token_mask = torch.zeros_like(log_prob)

    outputs = compute_opd_fkl_loss(
        log_prob=log_prob,
        teacher_token_mask=teacher_token_mask,
        response_validity_mask=validity,
    )
    for value in outputs:
        torch.testing.assert_close(value, torch.tensor(0.0))
