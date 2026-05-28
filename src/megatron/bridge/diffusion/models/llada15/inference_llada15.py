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
import torch.nn.functional as F


def _sample_with_temperature(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens and return ``(token_ids, token_probs)`` both shaped ``[B, L]``."""
    B, L, V = logits.shape
    flat = logits.reshape(-1, V)

    if temperature == 0.0 and (top_k is None or top_k <= 0) and (top_p is None or top_p >= 1.0):
        probs = F.softmax(flat, dim=-1)
        tokens = flat.argmax(dim=-1, keepdim=True)
        token_probs = probs.gather(-1, tokens)
        return tokens.squeeze(-1).view(B, L), token_probs.squeeze(-1).view(B, L)

    if temperature > 0 and temperature != 1.0:
        flat = flat / temperature
    if top_k is not None and top_k > 0:
        vals, _ = torch.topk(flat, top_k)
        flat = flat.masked_fill(flat < vals[..., -1:], float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(flat, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(flat, dtype=torch.bool).scatter(-1, sorted_idx, remove)
        flat = flat.masked_fill(mask, float("-inf"))

    probs = F.softmax(flat, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1)
    token_probs = probs.gather(-1, tokens)
    return tokens.squeeze(-1).view(B, L), token_probs.squeeze(-1).view(B, L)


def _get_transfer_schedule(block_length: int, steps: int) -> torch.Tensor:
    """Per-step counts of tokens to unmask, summing to ``block_length``."""
    if steps == 0:
        return torch.tensor([], dtype=torch.int64)
    base = block_length // steps
    rem = block_length % steps
    sched = torch.full((steps,), base, dtype=torch.int64)
    sched[:rem] += 1
    return sched


@torch.no_grad()
def generate_block_diffusion(
    model,
    input_ids: torch.Tensor,
    *,
    gen_length: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    mask_token_id: int = 126336,
    eos_token_id: Optional[int] = 126081,
    eos_early_stop: bool = False,
) -> torch.Tensor:
    """Sample tokens from a LLaDA1.5 Megatron ``GPTModel`` via block diffusion.

    Args:
        model: Megatron ``GPTModel`` built with :class:`LLaDA15ModelProvider`.
        input_ids: Prompt tokens ``[B, prompt_len]``.
        gen_length: Number of new tokens to generate.
        block_length: Tokens unmasked per outer block iteration.
        steps: Denoising steps per block.
        temperature, top_k, top_p: Standard sampling controls (greedy when
            ``temperature == 0``).
        mask_token_id: LLaDA1.5 mask token id (default 126336).
        eos_token_id: EOS id (default 126081) used for early stopping when
            ``eos_early_stop`` is set.
        eos_early_stop: Stop as soon as EOS is fully confirmed in the
            generated region.

    Returns:
        Token ids ``[B, prompt_len + gen_length]`` (or shorter if EOS-stopped).
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    total_length = prompt_len + gen_length

    x = torch.full((B, total_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = (gen_length + block_length - 1) // block_length
    steps_per_block = max(1, steps // max(num_blocks, 1))
    transfer_schedule = _get_transfer_schedule(block_length, steps_per_block)

    position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(B, -1)

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(block_start + block_length, total_length)
        cur_block_len = block_end - block_start

        for step_idx in range(steps_per_block):
            block_slice = x[:, block_start:block_end]
            if (block_slice != mask_token_id).all():
                break

            output = model(input_ids=x, position_ids=position_ids, attention_mask=None)
            logits = output if isinstance(output, torch.Tensor) else output[0]
            block_logits = logits[:, block_start:block_end, :]

            x0, x0_p = _sample_with_temperature(block_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            n_to_transfer = min(int(transfer_schedule[step_idx].item()), cur_block_len)
            active = block_slice == mask_token_id
            confidence = torch.where(active, x0_p, torch.full_like(x0_p, float("-inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for b in range(B):
                n = min(n_to_transfer, int(active[b].sum().item()))
                if n > 0:
                    _, idx = torch.topk(confidence[b], k=n)
                    transfer_index[b, idx] = True
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]

            if eos_early_stop and eos_token_id is not None:
                if (x[:, prompt_len:] == eos_token_id).any():
                    return x

    return x
