# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Offline, unpacked DiffusionGemma SFT with checkpointed default RNG streams."""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import Iterable

import torch
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model

from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState


@dataclass(frozen=True)
class DiffusionGemmaBatch:
    """Owned clean encoder and one response-relative canvas per row."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    noisy_tokens: torch.Tensor
    position_ids: torch.Tensor
    canvas_positions: torch.Tensor
    encoder_valid_mask: torch.Tensor
    prefix_lengths: torch.Tensor
    self_conditioning_mask: torch.Tensor


def prepare_batch(
    batch: dict,
    *,
    canvas_length: int,
    eos_token_id: int,
    vocab_size: int,
    noise_epsilon: float = 1e-3,
    self_conditioning_probability: float = 0.5,
) -> DiffusionGemmaBatch:
    """Reconstruct standard GPTSFT shifted rows and sample one complete canvas.

    ``token_count`` counts the unshifted clean tokens. The final token is
    recovered from the final valid shifted label. EOS fill completes the final
    response block and is valid clean encoder context and an AR target. Noise
    uses the default device RNG, whose state Bridge checkpoints per DP rank.
    """
    if canvas_length < 1 or vocab_size < 1 or not 0 <= eos_token_id < vocab_size:
        raise ValueError("positive canvas/vocabulary sizes and an in-vocabulary EOS are required")
    if not 0 < noise_epsilon <= 1 or not 0 <= self_conditioning_probability <= 1:
        raise ValueError("invalid corruption epsilon or self-conditioning probability")
    if any(batch.get(key) is not None for key in ("cu_seqlens", "packed_seq_params")):
        raise ValueError("DiffusionGemma does not support sequence packing")
    required = ("tokens", "labels", "loss_mask", "token_count", "context_lengths")
    if any(batch.get(key) is None for key in required):
        raise ValueError("DiffusionGemma requires the standard unpacked GPTSFT collator fields")
    tokens, labels, mask = (batch[key] for key in required[:3])
    if tokens.ndim != 2 or tokens.shape != labels.shape or tokens.shape != mask.shape or not tokens.shape[0]:
        raise ValueError("tokens, labels and loss_mask must have matching nonempty [B,S] shapes")
    if len(batch["token_count"]) != len(tokens) or len(batch["context_lengths"]) != len(tokens):
        raise ValueError("one token_count and context_length are required per row")
    rows, targets, starts = [], [], []
    for row, (count, context) in enumerate(zip(batch["token_count"], batch["context_lengths"], strict=True)):
        count, context = int(count), int(context)
        if count - 1 > tokens.shape[1]:
            raise ValueError("truncated GPTSFT row: final clean token is unavailable")
        if not 0 < context < count or count < 2:
            raise ValueError("each sample needs a nonempty prompt and response")
        expected = torch.zeros_like(mask[row])
        expected[context - 1 : count - 1] = 1
        if not torch.equal(mask[row], expected):
            raise ValueError("DiffusionGemma requires one contiguous, fully supervised final response")
        if not torch.equal(labels[row, : count - 2], tokens[row, 1 : count - 1]):
            raise ValueError("GPTSFT labels must be shifted clean tokens")
        clean = torch.cat((tokens[row, : count - 1], labels[row, count - 2 : count - 1]))
        if clean[-1].item() != eos_token_id:
            raise ValueError("the fully supervised final response must end with tokenizer EOS")
        padding = (-(count - context)) % canvas_length
        clean = torch.nn.functional.pad(clean, (0, padding), value=eos_token_id)
        blocks = (count - context + padding) // canvas_length
        block = int(torch.randint(blocks, (), device=tokens.device))
        start = context + block * canvas_length
        rows.append(clean)
        targets.append(clean[start : start + canvas_length])
        starts.append(start)
    lengths = torch.tensor([row.numel() for row in rows], device=tokens.device)
    clean = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=eos_token_id)
    target_ids = torch.stack(targets)
    positions = torch.arange(clean.shape[1], device=tokens.device)[None].expand(len(rows), -1)
    prefix_lengths = torch.tensor(starts, device=tokens.device)
    canvas_positions = prefix_lengths[:, None] + torch.arange(canvas_length, device=tokens.device)[None]
    timestep = noise_epsilon + (1 - noise_epsilon) * torch.rand(len(rows), 1, device=tokens.device)
    corrupt = torch.rand(target_ids.shape, device=tokens.device) < timestep
    replacements = torch.randint(vocab_size, target_ids.shape, device=tokens.device)
    return DiffusionGemmaBatch(
        input_ids=clean,
        targets=target_ids,
        noisy_tokens=torch.where(corrupt, replacements, target_ids),
        position_ids=positions,
        canvas_positions=canvas_positions,
        encoder_valid_mask=positions < lengths[:, None],
        prefix_lengths=prefix_lengths,
        self_conditioning_mask=torch.rand(len(rows), device=tokens.device) < self_conditioning_probability,
    )


def diffusion_gemma_loss(
    valid_ar: torch.Tensor,
    encoder_loss_weight: float,
    output_tensor: tuple[torch.Tensor, torch.Tensor],
    *,
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize diffusion and encoder objectives independently across DP.

    Return the legacy two-item schedule contract. Megatron's schedule divides
    by the number of microbatches and DDP averages gradients across DP ranks;
    scaling local sums by DP size/global token counts therefore yields the
    average of per-microbatch objectives. This is deliberately not one global
    token mean over the entire accumulation window when row lengths differ.
    Reporting leaves local sums/counts for Bridge's existing metric reduction.
    """
    diffusion_losses, encoder_losses = output_tensor
    sums = torch.stack((diffusion_losses.float().sum(), (encoder_losses.float() * valid_ar).sum()))
    counts = torch.stack((sums.new_tensor(diffusion_losses.numel()), valid_ar.sum().to(sums.dtype)))
    global_counts = counts.detach().clone()
    dp_size = 1
    if torch.distributed.is_initialized():
        group = parallel_state.get_data_parallel_group()
        dp_size = torch.distributed.get_world_size(group)
        torch.distributed.all_reduce(global_counts, group=group)
    if bool((global_counts <= 0).any()):
        raise ValueError("both diffusion and clean encoder objectives need valid tokens")
    loss = dp_size * (sums[0] / global_counts[0] + encoder_loss_weight * sums[1] / global_counts[1])
    # Reuse the standard rerun checks on the final scalar microbatch objective.
    masked_next_token_loss(
        torch.ones_like(loss),
        loss,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )
    metrics = {
        "lm loss": torch.stack((loss.detach(), counts.new_ones(()))),
        "diffusion loss": torch.stack((sums[0].detach(), counts[0])),
        "encoder loss": torch.stack((sums[1].detach(), counts[1])),
    }
    return loss, metrics


class DiffusionGemmaStep:
    """Native DiffusionGemma full-parameter BF16 SFT forward step.

    All canvas tokens contribute equally, including retained tokens and EOS
    fill. The separate clean encoder AR loss includes prompt-adjacent tokens.
    """

    def __init__(
        self,
        *,
        canvas_length: int = 256,
        noise_epsilon: float = 1e-3,
        self_conditioning_probability: float = 0.5,
        encoder_loss_weight: float = 1.0,
    ) -> None:
        if not isinstance(canvas_length, int) or canvas_length < 1:
            raise ValueError("canvas_length must be a positive integer")
        if not 0 < noise_epsilon <= 1 or not 0 <= self_conditioning_probability <= 1:
            raise ValueError("invalid corruption epsilon or self-conditioning probability")
        if not isfinite(encoder_loss_weight) or encoder_loss_weight < 0:
            raise ValueError("encoder_loss_weight must be finite and nonnegative")
        self.canvas_length = canvas_length
        self.noise_epsilon = noise_epsilon
        self.self_conditioning_probability = self_conditioning_probability
        self.encoder_loss_weight = encoder_loss_weight

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: torch.nn.Module, return_schedule_plan: bool = False
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], partial]:
        """Consume a standard shifted GPTSFT batch and return per-token losses."""
        if return_schedule_plan or state.cfg.model.overlap_moe_expert_parallel_comm:
            raise ValueError("DiffusionGemma does not support overlap schedule plans")
        if state.cfg.model.calculate_per_token_loss:
            raise ValueError("DiffusionGemma requires calculate_per_token_loss=False")
        if state.cfg.peft is not None:
            raise ValueError("DiffusionGemma does not support PEFT")
        if state.cfg.model.recompute_granularity is not None:
            raise ValueError("DiffusionGemma does not support activation recompute")
        if any(
            getattr(state.cfg.dataset, key, False)
            for key in ("enable_offline_packing", "enable_in_batch_packing", "enable_global_batch_packing")
        ):
            raise ValueError("DiffusionGemma does not support sequence packing")
        core_model = unwrap_model(model)
        eos = state.tokenizer.eod
        raw_batch = next(data_iterator)
        device = next(core_model.parameters()).device
        raw_batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        batch = prepare_batch(
            raw_batch,
            canvas_length=self.canvas_length,
            eos_token_id=eos,
            vocab_size=core_model.config.vocab_size,
            noise_epsilon=self.noise_epsilon,
            self_conditioning_probability=self.self_conditioning_probability,
        )
        encoder_logits, decoder_logits = model(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            noisy_tokens=batch.noisy_tokens,
            canvas_positions=batch.canvas_positions,
            encoder_valid_mask=batch.encoder_valid_mask,
            prefix_lengths=batch.prefix_lengths,
            self_conditioning_mask=batch.self_conditioning_mask,
        )
        diffusion_losses = core_model.compute_language_model_loss(
            batch.targets, decoder_logits.transpose(0, 1).contiguous()
        )
        encoder_losses = core_model.compute_language_model_loss(
            batch.input_ids[:, 1:].contiguous(), encoder_logits[:, :-1].transpose(0, 1).contiguous()
        )
        valid_ar = batch.encoder_valid_mask[:, :-1] & batch.encoder_valid_mask[:, 1:]
        return (diffusion_losses, encoder_losses), partial(
            diffusion_gemma_loss,
            valid_ar,
            self.encoder_loss_weight,
            check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
            check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        )
