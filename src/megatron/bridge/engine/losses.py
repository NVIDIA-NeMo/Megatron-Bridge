# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Predefined loss math over the ``(PreparedMicrobatch, ModelOutput)`` contract.

Every loss returns the additive FP32 contribution of one microbatch; the engine divides once at
``optim_step``. Losses select active positions (``loss_mask & ~padding_mask``) before any arithmetic so
padding never produces NaN/Inf that a later multiply-by-zero could not remove.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from megatron.bridge.engine.datum import ModelOutput, PreparedMicrobatch


def cross_entropy(batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
    """Negative log-likelihood summed over active positions."""
    active = batch.active_mask
    return -output.logprobs[active].float().sum()


def importance_sampling(batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
    """``-(exp(logp - old) * advantage)`` summed over active positions (phase C1)."""
    # TODO(C1): mirror megatron/rl calculate_grpo_loss math (ratio, optional TIS) with per-token advantages.
    raise NotImplementedError("importance_sampling is implemented in phase C1.")


def ppo_clip(batch: PreparedMicrobatch, output: ModelOutput, *, clip_low: float, clip_high: float) -> torch.Tensor:
    """Clipped PPO objective summed over active positions (phase C1)."""
    # TODO(C1): asymmetric clipping, optional TIS correction from rollout_logprobs, k3 KL to reference_logprobs.
    raise NotImplementedError("ppo_clip is implemented in phase C1.")


def chunked_vocab_apply(
    logits: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], *, chunk_tokens: int
) -> torch.Tensor:
    """Apply a full-vocabulary op per token chunk, preserving gradient flow, to bound peak memory (phase C1)."""
    # TODO(C1): copy megatron/rl vocab_parallel_selective_log_softmax (TP-aware, chunked) with attribution.
    raise NotImplementedError("chunked_vocab_apply is implemented in phase C1.")
