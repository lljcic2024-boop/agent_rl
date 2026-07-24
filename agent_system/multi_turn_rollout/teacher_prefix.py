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
改造点 4: teacher-prefix continuation rollout, offline-testable core.

At a selected step, the teacher writes a thinking prefix a_T. The student then
continues generation from that prefix, and the episode is rolled forward to
termination (the env-replay part lives in the trainer / rollout loop and needs
cluster runtime; this module owns the token bookkeeping).

Contract with the FKL loss (compute_opd_fkl_loss / dp_actor):
- a_T tokens are marked in `teacher_token_mask` (forward-KL / CE supervision);
- a_T tokens are EXCLUDED from the PG `loss_mask` (they were not sampled from
  the student, so importance ratios are meaningless there);
- both masks are (bsz, response_length) over the assembled branch response
  `a_T + student continuation`.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class PrefixGenerationBatch:
    """Prompt tensors with a_T appended as a generation prefix (left-padded)."""

    input_ids: torch.Tensor  # (bsz, prompt_len + max_prefix_len)
    attention_mask: torch.Tensor  # same shape
    position_ids: torch.Tensor  # same shape
    prefix_lengths: torch.Tensor  # (bsz,) number of a_T tokens per sample


def build_prefix_generation_batch(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prefix_token_ids: List[List[int]],
    pad_token_id: int,
) -> PrefixGenerationBatch:
    """
    Append teacher-written prefix tokens (a_T) to left-padded prompts so the
    student's generation continues right after a_T.

    Keeps left padding: each row's layout is [pad ... pad, prompt, a_T], so
    generation starts immediately after the a_T tokens for every sample.

    Args:
        prompt_input_ids: (bsz, prompt_len) left-padded prompt token ids.
        prompt_attention_mask: (bsz, prompt_len) prompt validity mask.
        prefix_token_ids: per-sample a_T token id lists (may be empty).
        pad_token_id: pad token id used for re-padding.

    Returns:
        PrefixGenerationBatch with tensors of width prompt_len + max_prefix_len.
    """
    if prompt_input_ids.shape != prompt_attention_mask.shape:
        raise ValueError(
            f"prompt_input_ids shape {tuple(prompt_input_ids.shape)} does not match "
            f"prompt_attention_mask shape {tuple(prompt_attention_mask.shape)}"
        )
    bsz, prompt_len = prompt_input_ids.shape
    if len(prefix_token_ids) != bsz:
        raise ValueError(f"prefix_token_ids has {len(prefix_token_ids)} entries for batch size {bsz}")

    max_prefix_len = max((len(ids) for ids in prefix_token_ids), default=0)
    total_len = prompt_len + max_prefix_len
    device = prompt_input_ids.device

    input_ids = torch.full((bsz, total_len), pad_token_id, dtype=prompt_input_ids.dtype, device=device)
    attention_mask = torch.zeros((bsz, total_len), dtype=prompt_attention_mask.dtype, device=device)
    prefix_lengths = torch.zeros(bsz, dtype=torch.long, device=device)

    for i in range(bsz):
        valid = prompt_attention_mask[i].bool()
        prompt_tokens = prompt_input_ids[i][valid]
        prefix = torch.as_tensor(prefix_token_ids[i], dtype=prompt_input_ids.dtype, device=device)
        prefix_lengths[i] = prefix.numel()
        row = torch.cat([prompt_tokens, prefix])
        # keep left padding: place the row at the right edge
        input_ids[i, total_len - row.numel():] = row
        attention_mask[i, total_len - row.numel():] = 1

    # standard left-padded position ids: cumsum over the attention mask
    position_ids = torch.clamp(attention_mask.long().cumsum(dim=-1) - 1, min=0)
    return PrefixGenerationBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prefix_lengths=prefix_lengths,
    )


@dataclass
class BranchResponses:
    """Assembled branch responses `a_T + continuation` with FKL/PG masks."""

    responses: torch.Tensor  # (bsz, response_length)
    response_attention_mask: torch.Tensor  # (bsz, response_length) validity
    loss_mask: torch.Tensor  # (bsz, response_length) PG mask, a_T excluded
    teacher_token_mask: torch.Tensor  # (bsz, response_length) a_T marked


def assemble_branch_responses(
    prefix_token_ids: List[List[int]],
    continuation_ids: torch.Tensor,
    continuation_attention_mask: torch.Tensor,
    pad_token_id: int,
    response_length: Optional[int] = None,
) -> BranchResponses:
    """
    Merge a_T and the student continuation into one right-padded response block
    and produce the masks required by the OPD-FKL training contract.

    Args:
        prefix_token_ids: per-sample a_T token id lists.
        continuation_ids: (bsz, cont_len) right-padded student continuation ids
            (generation output after the a_T prefix).
        continuation_attention_mask: (bsz, cont_len) continuation validity mask.
        pad_token_id: pad token id.
        response_length: target width; defaults to max(a_T + continuation) length.
            Sequences longer than this are truncated on the right.

    Returns:
        BranchResponses. For every row:
        - teacher_token_mask marks the first len(a_T) valid positions,
        - loss_mask marks the continuation positions only,
        - response_attention_mask = teacher_token_mask | loss_mask.
    """
    if continuation_ids.shape != continuation_attention_mask.shape:
        raise ValueError(
            f"continuation_ids shape {tuple(continuation_ids.shape)} does not match "
            f"continuation_attention_mask shape {tuple(continuation_attention_mask.shape)}"
        )
    bsz = continuation_ids.size(0)
    if len(prefix_token_ids) != bsz:
        raise ValueError(f"prefix_token_ids has {len(prefix_token_ids)} entries for batch size {bsz}")
    device = continuation_ids.device

    rows = []
    prefix_lens = []
    for i in range(bsz):
        prefix = torch.as_tensor(prefix_token_ids[i], dtype=continuation_ids.dtype, device=device)
        continuation = continuation_ids[i][continuation_attention_mask[i].bool()]
        prefix_lens.append(prefix.numel())
        rows.append(torch.cat([prefix, continuation]))

    natural_length = max((row.numel() for row in rows), default=0)
    if response_length is None:
        response_length = natural_length
    response_length = max(int(response_length), 1)

    responses = torch.full((bsz, response_length), pad_token_id, dtype=continuation_ids.dtype, device=device)
    response_attention_mask = torch.zeros((bsz, response_length), dtype=torch.long, device=device)
    loss_mask = torch.zeros((bsz, response_length), dtype=torch.long, device=device)
    teacher_token_mask = torch.zeros((bsz, response_length), dtype=torch.long, device=device)

    for i in range(bsz):
        row = rows[i][:response_length]
        n = row.numel()
        n_prefix = min(prefix_lens[i], n)
        responses[i, :n] = row
        response_attention_mask[i, :n] = 1
        teacher_token_mask[i, :n_prefix] = 1
        loss_mask[i, n_prefix:n] = 1

    return BranchResponses(
        responses=responses,
        response_attention_mask=response_attention_mask,
        loss_mask=loss_mask,
        teacher_token_mask=teacher_token_mask,
    )
