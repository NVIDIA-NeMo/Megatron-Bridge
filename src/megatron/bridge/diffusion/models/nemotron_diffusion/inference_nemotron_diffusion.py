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
"""
Generation utilities for Megatron GPTModel with NemotronDiffusionAttention inference support.

Supports:
  - AR (autoregressive) generation with KV cache
  - dLLM (block diffusion) generation with prefix cache + iterative denoising

The KV cache lives inside each layer's NemotronDiffusionAttention via its
_inference_mode / _kv_cache_* attributes. No Megatron InferenceContext is used.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers (ported from the original eval.py)
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits, temperature):
    """Apply Gumbel noise to logits for stochastic sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Compute the number of tokens to unmask at each diffusion step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
    neg_entropy=False,
):
    """Select token indices to transfer from masked to unmasked at each diffusion step."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    if neg_entropy:
        p = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        confidence_scores = torch.sum(p * log_probs, dim=-1)
    else:
        confidence_scores = x0_p

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


# ---------------------------------------------------------------------------
# Core attention-layer helpers
# ---------------------------------------------------------------------------


def _unwrap(model):
    """Unwrap Float16Module, DDP, or VLM wrappers to get the raw GPTModel."""
    if hasattr(model, "module"):
        return _unwrap(model.module)
    if hasattr(model, "language_model"):
        return _unwrap(model.language_model)
    return model


def _get_core_attentions(model):
    """Return list of NemotronDiffusionAttention modules from the Megatron GPT model."""
    m = _unwrap(model)
    attns = []
    for layer in m.decoder.layers:
        attns.append(layer.self_attention.core_attention)
    return attns


def _set_inference_mode(model, enabled: bool):
    _tp_send_cmd(_CMD_SET_INF_MODE_ON if enabled else _CMD_SET_INF_MODE_OFF)
    for attn in _get_core_attentions(model):
        attn.set_inference_mode(enabled)


def _set_inference_params(model, causal: bool, cache_enabled: bool):
    _tp_send_cmd(_CMD_SET_PARAMS, extra=[int(causal), int(cache_enabled)])
    for attn in _get_core_attentions(model):
        attn.set_inference_params(causal, cache_enabled)


def _clear_kv_cache(model):
    _tp_send_cmd(_CMD_CLEAR_CACHE)
    for attn in _get_core_attentions(model):
        attn.clear_kv_cache()


_TP_GROUP = None
_TP_SRC_GLOBAL_RANK = 0

# Command codes for TP follower loop
_CMD_FORWARD = 1
_CMD_SET_INF_MODE_ON = 2
_CMD_SET_INF_MODE_OFF = 3
_CMD_SET_PARAMS = 4  # followed by 2 ints: causal, cache_enabled
_CMD_CLEAR_CACHE = 5
_CMD_STOP = 0


def set_tp_group(group, src_global_rank=0):
    """Set the TP process group for _model_forward token broadcasts."""
    global _TP_GROUP, _TP_SRC_GLOBAL_RANK
    _TP_GROUP = group
    _TP_SRC_GLOBAL_RANK = src_global_rank


def _tp_send_cmd(cmd, extra=None):
    """Broadcast a command to TP followers (no-op if TP=1)."""
    if _TP_GROUP is None:
        return
    t = torch.tensor([cmd], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(t, src=_TP_SRC_GLOBAL_RANK, group=_TP_GROUP)
    if extra is not None:
        e = torch.tensor(extra, dtype=torch.long, device="cuda")
        torch.distributed.broadcast(e, src=_TP_SRC_GLOBAL_RANK, group=_TP_GROUP)


def _broadcast_tensor(tensor, src, group):
    """Broadcast shape then data so all peers have an identically-shaped tensor."""
    rank = torch.distributed.get_rank()
    if rank == src:
        shape_t = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
    else:
        shape_t = torch.zeros(2, dtype=torch.long, device="cuda")
    torch.distributed.broadcast(shape_t, src=src, group=group)
    if rank != src:
        tensor = torch.zeros(shape_t.tolist(), dtype=torch.long, device="cuda")
    torch.distributed.broadcast(tensor, src=src, group=group)
    return tensor


def _model_forward(model, input_ids):
    """Call GPTModel.forward with minimal args (no inference_context).

    When a TP group is set, broadcasts input_ids from the TP-rank-0 process
    so all TP peers call forward with identical inputs.

    Args:
        model: Megatron GPTModel (already on CUDA).
        input_ids: [batch, seq_len] token ids.

    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    if _TP_GROUP is not None:
        _tp_send_cmd(_CMD_FORWARD)
        input_ids = _broadcast_tensor(input_ids, _TP_SRC_GLOBAL_RANK, _TP_GROUP)

    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
    )
    logits = output if isinstance(output, torch.Tensor) else output[0]
    return logits


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_ar(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    eos_token_id: int = None,
):
    """Standard left-to-right autoregressive generation with KV cache.

    Args:
        model: Megatron GPTModel on CUDA.
        prompt: [batch, prompt_len] token ids.
        max_new_tokens: number of tokens to generate.
        temperature: sampling temperature (0 = greedy).
        eos_token_id: stop generation when this token is produced.

    Returns:
        generated: [batch, prompt_len + new_tokens] full sequence.
    """
    _set_inference_mode(model, True)
    _set_inference_params(model, causal=True, cache_enabled=True)
    _clear_kv_cache(model)

    # Prefill: process the prompt and cache KV
    logits = _model_forward(model, prompt)  # [b, prompt_len, V]
    generated = prompt.clone()

    # Decode tokens one at a time
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]  # [b, V]
        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [b, 1]
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [b, 1]

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # Forward just the new token (KV cache has everything before it)
        logits = _model_forward(model, next_token)  # [b, 1, V]

    _set_inference_mode(model, False)
    return generated


# ---------------------------------------------------------------------------
# dLLM (Block Diffusion) generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_dllm(
    model,
    prompt: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 32,
    steps: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 100,
    threshold: float = None,
    shift_logits: bool = True,
    neg_entropy: bool = True,
):
    """Block-diffusion generation with prefix KV cache.

    Replicates generate_with_prefix_cache_block_diff_sbd from the original eval.py
    but uses NemotronDiffusionAttention's inference mode instead of HF past_key_values.

    Args:
        model: Megatron GPTModel on CUDA.
        prompt: [batch, prompt_len] token ids.
        gen_length: total number of tokens to generate.
        block_length: size of each denoising block.
        steps: total denoising steps across all blocks.
        temperature: sampling temperature for Gumbel noise.
        remasking: remasking strategy ("low_confidence" or "random").
        mask_id: mask token id.
        threshold: optional denoising confidence threshold.
        shift_logits: if True, use dream-style shifted logits (position i-1
            predicts token i, i.e. next-token prediction). If False, each
            masked position's logits predict its own token directly. For dLLM
            this should typically be False; for AR-style models use True.
        neg_entropy: if True, use negative entropy for confidence scoring.

    Returns:
        x_accum: [batch, prompt_len + gen_length] full sequence with generated tokens.
        nfe: number of forward evaluations used.
    """
    dream_style = shift_logits
    x_accum = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    _t_prefill_ms = 0.0
    _t_denoise_ms = 0.0
    _t_kv_update_ms = 0.0

    # --- Prefill: build KV cache for the prompt (causal attention) ---
    _set_inference_mode(model, True)
    _set_inference_params(model, causal=True, cache_enabled=True)
    _clear_kv_cache(model)

    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    logits = _model_forward(model, prompt)  # [b, prompt_len, V]
    torch.cuda.synchronize()
    _t_prefill_ms += (time.perf_counter() - _t0) * 1000.0

    next_logits_context = None
    if dream_style:
        next_logits_context = logits[:, -1:, :]  # [b, 1, V]

    # --- Generate each block ---
    for num_block in range(num_blocks):
        # Create mask block
        mask_block = torch.full(
            (prompt.shape[0], block_length),
            mask_id,
            dtype=prompt.dtype,
            device=prompt.device,
        )
        x_accum = torch.cat([x_accum, mask_block], dim=1)
        current_block_start = prompt.size(1) + num_block * block_length
        block_slice = slice(current_block_start, current_block_start + block_length)

        mask_block_idx0 = x_accum[:, block_slice] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(mask_block_idx0, steps_per_block)

        # --- Denoise the block iteratively ---
        for i in range(steps_per_block):
            mask_block_idx = x_accum[:, block_slice] == mask_id
            if mask_block_idx.sum() == 0:
                break

            nfe += 1
            _set_inference_params(model, causal=False, cache_enabled=False)
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
            logits_block = _model_forward(model, x_accum[:, block_slice])  # [b, block_length, V]
            torch.cuda.synchronize()
            _t_denoise_ms += (time.perf_counter() - _t0) * 1000.0

            if dream_style:
                if block_length == 1:
                    logits_use = next_logits_context
                else:
                    logits_use = torch.cat([next_logits_context, logits_block[:, :-1, :]], dim=1)
                mask_use = mask_block_idx
                x_use = x_accum[:, block_slice]

                x0, transfer_idx = get_transfer_index(
                    logits_use,
                    temperature,
                    remasking,
                    mask_use,
                    x_use,
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    threshold=threshold,
                    neg_entropy=neg_entropy,
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur
            else:
                x0, transfer_idx = get_transfer_index(
                    logits_block,
                    temperature,
                    remasking,
                    mask_block_idx,
                    x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    threshold=threshold,
                    neg_entropy=neg_entropy,
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

        # --- After block is denoised, update KV cache with the clean block ---
        _set_inference_params(model, causal=True, cache_enabled=True)
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        output_logits = _model_forward(model, x_accum[:, block_slice])  # [b, block_length, V]
        torch.cuda.synchronize()
        _t_kv_update_ms += (time.perf_counter() - _t0) * 1000.0

        if dream_style and num_block < num_blocks - 1:
            next_logits_context = output_logits[:, -1:, :]

    _set_inference_mode(model, False)
    _timing = {
        "prefill_ms": _t_prefill_ms,
        "denoise_ms": _t_denoise_ms,
        "kv_update_ms": _t_kv_update_ms,
    }
    return x_accum, nfe, _timing


# ---------------------------------------------------------------------------
# TP follower loop (for tp_local > 0 ranks)
# ---------------------------------------------------------------------------


def tp_send_stop():
    """Tell TP followers to exit their loop."""
    _tp_send_cmd(_CMD_STOP)


def tp_follower_loop(model):
    """Blocking loop for TP-non-rank-0 processes.

    Waits for commands from the TP-rank-0 process and mirrors all model
    operations (set_inference_mode, set_inference_params, clear_kv_cache,
    model forward) so Megatron TP communication stays in sync.
    """
    while True:
        cmd = torch.zeros(1, dtype=torch.long, device="cuda")
        torch.distributed.broadcast(cmd, src=_TP_SRC_GLOBAL_RANK, group=_TP_GROUP)
        cmd = cmd.item()

        if cmd == _CMD_STOP:
            break
        elif cmd == _CMD_FORWARD:
            input_ids = _broadcast_tensor(
                torch.zeros(1, 1, dtype=torch.long, device="cuda"),
                _TP_SRC_GLOBAL_RANK,
                _TP_GROUP,
            )
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        elif cmd == _CMD_SET_INF_MODE_ON:
            for attn in _get_core_attentions(model):
                attn.set_inference_mode(True)
        elif cmd == _CMD_SET_INF_MODE_OFF:
            for attn in _get_core_attentions(model):
                attn.set_inference_mode(False)
        elif cmd == _CMD_SET_PARAMS:
            extra = torch.zeros(2, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(extra, src=_TP_SRC_GLOBAL_RANK, group=_TP_GROUP)
            causal = bool(extra[0].item())
            cache_enabled = bool(extra[1].item())
            for attn in _get_core_attentions(model):
                attn.set_inference_params(causal, cache_enabled)
        elif cmd == _CMD_CLEAR_CACHE:
            for attn in _get_core_attentions(model):
                attn.clear_kv_cache()
