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

"""Block-diffusion generation for LLaDA1.5 loaded into a Megatron ``GPTModel``.

Mirrors the official sampling loop in the ML-GSAI/LLaDA repo
(``generate.py``): the prompt is concatenated with a sequence of ``<MASK>``
tokens, and the model is repeatedly invoked on the full sequence (with
fully bidirectional attention — see :class:`LLaDA15TEDotProductAttention`)
to predict the masked positions. Each iteration unmasks the most confident
predictions inside the current block; once a block is fully unmasked the
loop advances to the next block.

Note: unlike LLaDA2, no block-diagonal attention mask is constructed. The
"block" structure is purely a sampling-time choice (which positions to
unmask per step). The model itself sees the full sequence with zero
attention bias.
"""

from typing import Optional

import torch

from megatron.bridge.diffusion.common.dllm import get_num_transfer_tokens, get_transfer_index
from megatron.bridge.diffusion.models.llada15.llada15_attention import LLaDA15TEDotProductAttention


def _unwrap(model):
    """Unwrap Float16Module / DDP / VLM wrappers to reach the raw GPTModel."""
    if hasattr(model, "module"):
        return _unwrap(model.module)
    if hasattr(model, "language_model"):
        return _unwrap(model.language_model)
    return model


def _iter_llada15_attentions(model):
    """Yield each layer's LLaDA15TEDotProductAttention core-attention module."""
    for layer in _unwrap(model).decoder.layers:
        ca = layer.self_attention.core_attention
        if isinstance(ca, LLaDA15TEDotProductAttention):
            yield ca


def _set_padding_mask(model, mask: Optional[torch.Tensor]) -> None:
    """Broadcast a boolean key-padding mask ``[B, S]`` to every attention layer."""
    for attn in _iter_llada15_attentions(model):
        attn.set_padding_mask(mask)


def _clear_attention_state(model) -> None:
    """Drop any stored mask state so it does not leak into the next batch."""
    for attn in _iter_llada15_attentions(model):
        attn.reset_inference_state()


@torch.no_grad()
def generate_block_diffusion(
    model,
    input_ids: torch.Tensor,
    *,
    gen_length: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    neg_entropy: bool = False,
    threshold: Optional[float] = None,
    mask_token_id: int = 126336,
    eos_token_id: Optional[int] = 126081,
    eos_early_stop: bool = False,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Sample tokens from a LLaDA1.5 Megatron ``GPTModel`` via block diffusion.

    The model attends **fully bidirectionally** over the entire sequence at every
    step (LLaDA1.5's reference uses a zero attention bias), so each iteration
    re-forwards the whole ``x``. The block structure governs only *which*
    positions are eligible to be unmasked per step, not the attention pattern.
    Per-step token selection is delegated to the shared diffusion sampler in
    :mod:`megatron.bridge.diffusion.common.dllm`, the same primitives used by
    NemotronLabsDiffusion and verified by the generation-parity test.

    The defaults (``remasking="low_confidence"``, ``neg_entropy=False``,
    ``threshold=None``, ``temperature=0``) reproduce greedy ML-GSAI sampling
    exactly. Set ``temperature > 0`` for Gumbel sampling, ``neg_entropy=True`` to
    rank by distribution entropy, or ``threshold`` for confidence-gated transfer.

    Args:
        model: Megatron ``GPTModel`` built with :class:`LLaDA15ModelProvider`.
        input_ids: Prompt tokens ``[B, prompt_len]``.
        gen_length: Number of new tokens to generate.
        block_length: Tokens unmasked per outer block iteration.
        steps: Total denoising steps (split evenly across blocks).
        temperature: Gumbel sampling temperature (``0`` = greedy).
        remasking: Confidence source, ``"low_confidence"`` or ``"random"``.
        neg_entropy: Rank by negative entropy instead of chosen-token probability.
        threshold: Optional confidence threshold for gated transfer.
        mask_token_id: LLaDA1.5 mask token id (default 126336).
        eos_token_id: EOS id (default 126081) for early stopping.
        eos_early_stop: Stop generation once *every* sample in the batch has
            emitted at least one EOS in its generated region. Evaluated at block
            boundaries (not per step), so the current block is always fully
            unmasked before stopping and no mask id survives before a sample's
            EOS. Disable for fixed-length outputs.
        pad_token_id: If given and the batched prompt contains padding, a boolean
            key-padding mask is installed so padded positions are never attended
            to. Required for correct mixed-length batched generation (LLaDA1.5
            attends fully bidirectionally and has no implicit padding mask).

    Returns:
        Token ids ``[B, prompt_len + gen_length]`` — always full width. When
        ``eos_early_stop`` triggers, generation halts but the tensor is *not*
        truncated; positions past each sample's first EOS keep their generated
        ids, so callers should trim at the first ``eos_token_id`` per row.
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    total_length = prompt_len + gen_length

    x = torch.full((B, total_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = (gen_length + block_length - 1) // block_length
    steps_per_block = max(1, steps // max(num_blocks, 1))

    position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(B, -1)

    # Build a key-padding mask over the full sequence: only prompt positions can
    # be padding (the generated region is filled with mask/real tokens). Install
    # it on every attention layer so padded keys are blocked.
    pad_key_mask = None
    if pad_token_id is not None:
        prompt_pad = input_ids == pad_token_id
        if bool(prompt_pad.any()):
            pad_key_mask = torch.zeros(B, total_length, dtype=torch.bool, device=device)
            pad_key_mask[:, :prompt_len] = prompt_pad
    _set_padding_mask(model, pad_key_mask)

    try:
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = min(block_start + block_length, total_length)
            block_slice = slice(block_start, block_end)

            # Schedule computed once per block from its initial (fully-masked) state.
            block_mask0 = x[:, block_slice] == mask_token_id
            num_transfer = get_num_transfer_tokens(block_mask0, steps_per_block)

            for step_idx in range(steps_per_block):
                mask_now = x[:, block_slice] == mask_token_id
                if mask_now.sum() == 0:
                    break

                output = model(input_ids=x, position_ids=position_ids, attention_mask=None)
                logits = output if isinstance(output, torch.Tensor) else output[0]
                block_logits = logits[:, block_slice, :]

                x0, transfer_index = get_transfer_index(
                    block_logits,
                    temperature,
                    remasking,
                    mask_now,
                    x[:, block_slice],
                    num_transfer_tokens=num_transfer[:, step_idx],
                    threshold=threshold,
                    neg_entropy=neg_entropy,
                )
                cur = x[:, block_slice].clone()
                cur[transfer_index] = x0[transfer_index]
                x[:, block_slice] = cur

            # EOS early-stop, evaluated at the block boundary (not per step):
            # stop only once *every* sample has emitted at least one EOS. The
            # per-sample reduction ``.any(dim=1).all()`` avoids halting the whole
            # batch when a single sample finishes first (which would return other
            # samples still full of mask ids), and checking after the block is
            # fully unmasked guarantees no ``mask_token_id`` remains before any
            # sample's first EOS.
            if eos_early_stop and eos_token_id is not None:
                if (x[:, prompt_len:] == eos_token_id).any(dim=1).all():
                    return x

        return x
    finally:
        # Always clear stored mask state so it can't leak into the next batch.
        _clear_attention_state(model)
