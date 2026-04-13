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

"""CTC inference for dLLM: single-pass generation with blank collapse."""

import time
import torch

from megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion import (
    _set_inference_mode,
    _set_inference_params,
    _clear_kv_cache,
    _model_forward,
)


def ctc_collapse(tokens, blank_id):
    """Collapse CTC output: remove blanks and deduplicate consecutive tokens.

    Args:
        tokens: [batch, seq_len] token ids (argmax of CTC output).
        blank_id: the blank token id.

    Returns:
        collapsed: list of 1-D tensors (variable length per batch element).
    """
    collapsed = []
    for b in range(tokens.shape[0]):
        seq = tokens[b]
        # Remove blanks
        non_blank = seq[seq != blank_id]
        if non_blank.numel() == 0:
            collapsed.append(non_blank)
            continue
        # Deduplicate consecutive tokens
        mask = torch.ones(non_blank.shape[0], dtype=torch.bool, device=non_blank.device)
        mask[1:] = non_blank[1:] != non_blank[:-1]
        collapsed.append(non_blank[mask])
    return collapsed


@torch.no_grad()
def generate_ctc(
    model,
    prompt: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 128,
    mask_id: int = 100,
    eos_token_id: int = None,
):
    """CTC-based generation: single forward pass per block, then collapse.

    Args:
        model: Megatron GPTModel on CUDA.
        prompt: [batch, prompt_len] token ids.
        gen_length: total number of tokens to generate (approximate, CTC
            collapse may produce fewer).
        block_length: CTC output block size (e.g. 128). After collapse,
            each block yields ~block_length/2 tokens.
        mask_id: blank token id (used as CTC blank and input fill).
        eos_token_id: stop generation when this token appears.

    Returns:
        x_accum: [batch, prompt_len + generated_len] full sequence (padded).
        nfe: number of forward evaluations.
        timing: dict with prefill_ms, decode_ms, kv_update_ms.
    """
    batch_size = prompt.shape[0]
    assert batch_size == 1, "CTC generation currently supports batch_size=1"

    # Target tokens per block after collapse (block_length/2 is the CTC target size)
    target_per_block = block_length // 2
    num_blocks = (gen_length + target_per_block - 1) // target_per_block

    nfe = 0
    _t_prefill_ms = 0.0
    _t_decode_ms = 0.0
    _t_kv_update_ms = 0.0

    # --- Prefill: build KV cache for the prompt ---
    _set_inference_mode(model, True)
    _set_inference_params(model, causal=True, cache_enabled=True)
    _clear_kv_cache(model)

    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    _model_forward(model, prompt)
    torch.cuda.synchronize()
    _t_prefill_ms = (time.perf_counter() - _t0) * 1000.0

    x_accum = prompt.clone()
    total_generated = 0

    for _ in range(num_blocks):
        if total_generated >= gen_length:
            break

        # Create blank block
        blank_block = torch.full(
            (batch_size, block_length),
            mask_id,
            dtype=prompt.dtype,
            device=prompt.device,
        )

        # Single forward pass (bidirectional within block, using KV cache for context)
        nfe += 1
        _set_inference_params(model, causal=False, cache_enabled=False)
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        logits = _model_forward(model, blank_block)  # [1, block_length, V]
        torch.cuda.synchronize()
        _t_decode_ms += (time.perf_counter() - _t0) * 1000.0

        # Argmax decode
        predicted = torch.argmax(logits, dim=-1)  # [1, block_length]

        # CTC collapse
        collapsed_list = ctc_collapse(predicted, blank_id=mask_id)
        collapsed = collapsed_list[0]  # [variable_len]

        if collapsed.numel() == 0:
            # Model produced only blanks — stop
            break

        # Append collapsed tokens to accumulated sequence
        collapsed_2d = collapsed.unsqueeze(0)  # [1, collapsed_len]
        x_accum = torch.cat([x_accum, collapsed_2d], dim=1)
        total_generated += collapsed.numel()

        # Check for EOS
        if eos_token_id is not None and (collapsed == eos_token_id).any():
            eos_pos = (collapsed == eos_token_id).nonzero(as_tuple=True)[0][0]
            # Trim to EOS
            trim_len = collapsed.numel() - eos_pos.item() - 1
            if trim_len > 0:
                x_accum = x_accum[:, :-trim_len]
            break

        # Update KV cache with the collapsed tokens (causal)
        _set_inference_params(model, causal=True, cache_enabled=True)
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        _model_forward(model, collapsed_2d)
        torch.cuda.synchronize()
        _t_kv_update_ms += (time.perf_counter() - _t0) * 1000.0

    _set_inference_mode(model, False)
    _timing = {
        "prefill_ms": _t_prefill_ms,
        "denoise_ms": _t_decode_ms,
        "kv_update_ms": _t_kv_update_ms,
    }
    return x_accum, nfe, _timing
