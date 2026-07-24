"""Tests for the teacher-prefix continuation token bookkeeping (改造点 4 offline core)."""
import torch

from agent_system.multi_turn_rollout import teacher_prefix as mod

PAD = 0
def test_build_prefix_generation_batch_keeps_left_padding():
    prompt_ids = torch.tensor([
        [PAD, PAD, 11, 12],
        [PAD, 13, 14, 15],
    ])
    prompt_mask = torch.tensor([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ])
    prefixes = [[91, 92], [93]]
    out = mod.build_prefix_generation_batch(prompt_ids, prompt_mask, prefixes, PAD)
    # width = 4 + max_prefix(2) = 6, rows right-aligned
    assert out.input_ids.shape == (2, 6)
    assert out.input_ids[0].tolist() == [PAD, PAD, 11, 12, 91, 92]
    assert out.attention_mask[0].tolist() == [0, 0, 1, 1, 1, 1]
    assert out.input_ids[1].tolist() == [PAD, PAD, 13, 14, 15, 93]
    assert out.attention_mask[1].tolist() == [0, 0, 1, 1, 1, 1]
    assert out.prefix_lengths.tolist() == [2, 1]
    # position ids: cumsum-1 clamped, contiguous over valid区
    assert out.position_ids[0].tolist() == [0, 0, 0, 1, 2, 3]


def test_build_prefix_generation_batch_empty_prefix():
    prompt_ids = torch.tensor([[11, 12]])
    prompt_mask = torch.tensor([[1, 1]])
    out = mod.build_prefix_generation_batch(prompt_ids, prompt_mask, [[]], PAD)
    assert out.input_ids.shape == (1, 2)
    assert out.input_ids[0].tolist() == [11, 12]
    assert out.prefix_lengths.tolist() == [0]


def test_assemble_branch_responses_masks():
    prefixes = [[91, 92], []]
    cont_ids = torch.tensor([
        [21, 22, PAD],
        [23, 24, 25],
    ])
    cont_mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
    ])
    out = mod.assemble_branch_responses(prefixes, cont_ids, cont_mask, PAD)
    # natural length = max(2+2, 0+3) = 4
    assert out.responses.shape == (2, 4)
    assert out.responses[0].tolist() == [91, 92, 21, 22]
    assert out.teacher_token_mask[0].tolist() == [1, 1, 0, 0]
    assert out.loss_mask[0].tolist() == [0, 0, 1, 1]
    assert out.response_attention_mask[0].tolist() == [1, 1, 1, 1]
    # row without prefix: pure student tokens, all PG, no FKL
    assert out.responses[1].tolist() == [23, 24, 25, PAD]
    assert out.teacher_token_mask[1].tolist() == [0, 0, 0, 0]
    assert out.loss_mask[1].tolist() == [1, 1, 1, 0]
    # contract: masks are disjoint and union = validity
    assert ((out.teacher_token_mask & out.loss_mask) == 0).all()
    assert ((out.teacher_token_mask | out.loss_mask) == out.response_attention_mask).all()


def test_assemble_branch_responses_truncation():
    prefixes = [[91, 92, 93]]
    cont_ids = torch.tensor([[21, 22]])
    cont_mask = torch.tensor([[1, 1]])
    out = mod.assemble_branch_responses(prefixes, cont_ids, cont_mask, PAD, response_length=4)
    assert out.responses[0].tolist() == [91, 92, 93, 21]
    assert out.teacher_token_mask[0].tolist() == [1, 1, 1, 0]
    assert out.loss_mask[0].tolist() == [0, 0, 0, 1]


