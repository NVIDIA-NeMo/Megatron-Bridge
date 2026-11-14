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

"""Numerical operators of the Inkling2 language decoder.

The operators follow the reference implementation's arithmetic: FP32
accumulation, and rounding back to the activation dtype only where the
reference does. Sequence tensors are sequence-first, ``[S, B, ...]``. A
``positions`` tensor of shape ``[S, B]`` holds each token's offset inside its
document, so packed documents never exchange information through the short
convolutions.

With ``compiled=True`` an operator runs through ``torch.compile`` as one full
graph, a few fused kernels instead of a chain of FP32 passes. Dynamo builds a
graph for every new call signature (shapes, strides, source count, grad mode),
so compiled operators suit training at fixed sequence shapes.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Protocol

import torch


ATTN_RES_EPS = 1e-6
L2_CLIP_EPS = 1e-6

# The AttnRes mixture alone needs a graph per source count, query count, source stride, and grad mode.
_RECOMPILE_LIMIT = 512
_ACCUMULATED_RECOMPILE_LIMIT = 8192
_COMPILED: dict[tuple[Callable, bool | None], Callable] = {}


def _compiled(fn: Callable, *, dynamic: bool | None = False) -> Callable:
    """Return ``fn`` compiled as one full graph, built once per process."""
    key = (fn, dynamic)
    compiled = _COMPILED.get(key)
    if compiled is None:
        import torch._dynamo

        # Dynamo keeps config overrides in context variables, which do not reach autograd's thread, where full
        # activation recompute replays the forward pass; raise the limits' defaults, which every thread reads.
        config = torch._dynamo.config
        for name, limit in (
            ("recompile_limit", _RECOMPILE_LIMIT),
            ("accumulated_recompile_limit", _ACCUMULATED_RECOMPILE_LIMIT),
        ):
            entry = getattr(config, "_config", {}).get(name)
            if entry is not None and isinstance(getattr(entry, "default", None), int):
                entry.default = max(entry.default, limit)
        kwargs: dict[str, object] = {"dynamic": dynamic, "fullgraph": True}
        if "recompile_limit" in inspect.signature(torch.compile).parameters:
            kwargs["recompile_limit"] = _RECOMPILE_LIMIT
        compiled = torch.compile(fn, **kwargs)
        _COMPILED[key] = compiled
    return compiled


class ShortConvHalo(Protocol):
    """Supplies the tokens before a context-parallel rank's chunks to a causal short convolution."""

    def extend(self, x: torch.Tensor) -> torch.Tensor:
        """Return the local rows ``[S_local, ...]`` followed by the halo rows of every rank."""
        ...

    def shift(self, extended: torch.Tensor, x: torch.Tensor, lag: int) -> torch.Tensor:
        """Return ``x[t - lag]`` for every local token, zero before the start of its document."""
        ...


def attn_res_mix(
    sources: Sequence[torch.Tensor],
    queries: torch.Tensor,
    *,
    normalize_values: bool,
    partial_addend: torch.Tensor | None = None,
    compiled: bool = False,
) -> torch.Tensor:
    """Mix residual sources with Block Attention Residuals.

    Each pseudo-query attends over the sources independently at every token:
    keys are the RMS-normalized sources, the query is scaled by
    ``hidden_size**-0.5``, and the values are the sources themselves or, with
    ``normalize_values``, their RMS-normalized form.

    Args:
        sources: Completed block representations followed by the current
            partial block, each ``[..., H]`` with identical shapes.
        queries: Pseudo-queries, ``[H]`` or ``[Q, H]``.
        normalize_values: Whether to mix RMS-normalized values.
        partial_addend: Optional sublayer output added to the last source in
            FP32 before mixing.
        compiled: Whether to run the mixture through ``torch.compile``.

    Returns:
        FP32 mixtures of shape ``[Q, ..., H]``.
    """
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    if queries.ndim != 2:
        raise ValueError(f"AttnRes queries must be [H] or [Q, H], got {tuple(queries.shape)}")
    mix = _compiled(_attn_res_mix) if compiled else _attn_res_mix
    return mix(list(sources), queries, normalize_values=normalize_values, partial_addend=partial_addend)


def _attn_res_mix(
    sources: list[torch.Tensor],
    queries: torch.Tensor,
    *,
    normalize_values: bool,
    partial_addend: torch.Tensor | None,
) -> torch.Tensor:
    values = torch.stack(sources, dim=0).float()
    if partial_addend is not None:
        values = torch.cat((values[:-1], (values[-1] + partial_addend.float()).unsqueeze(0)), dim=0)
    hidden_size = values.shape[-1]
    if queries.shape[-1] != hidden_size:
        raise ValueError(f"AttnRes query width {queries.shape[-1]} != source width {hidden_size}")
    inv_rms = torch.rsqrt(values.square().mean(dim=-1) + ATTN_RES_EPS)
    effective_queries = queries.float() * hidden_size**-0.5
    logits = torch.einsum("n...d,qd->q...n", values, effective_queries) * inv_rms.movedim(0, -1).unsqueeze(0)
    probabilities = torch.softmax(logits, dim=-1)
    if normalize_values:
        values = values * inv_rms.unsqueeze(-1)
    return torch.einsum("q...n,n...d->q...d", probabilities, values)


def residual_add(partial: torch.Tensor, *addends: torch.Tensor, compiled: bool = False) -> torch.Tensor:
    """Add sublayer outputs to the partial block in FP32 and round once, optionally compiled."""
    add = _compiled(_residual_add) if compiled else _residual_add
    return add(partial, *addends)


def _residual_add(partial: torch.Tensor, *addends: torch.Tensor) -> torch.Tensor:
    result = partial.float()
    for addend in addends:
        result = result + addend.float()
    return result.to(partial.dtype)


def _lagged(x: torch.Tensor, lag: int, positions: torch.Tensor) -> torch.Tensor:
    """Return ``x[t - lag]``, zero where that token lies before the document start."""
    if lag == 0:
        return x
    shifted = torch.zeros_like(x)
    shifted[lag:] = x[:-lag]
    keep = (positions >= lag).to(x.dtype)
    return shifted * keep.reshape(*keep.shape, *([1] * (x.ndim - keep.ndim)))


def _shifted(
    current: torch.Tensor,
    lag: int,
    positions: torch.Tensor,
    halo: ShortConvHalo | None,
    extended: torch.Tensor | None,
) -> torch.Tensor:
    if halo is None:
        return _lagged(current, lag, positions)
    return halo.shift(extended, current, lag)


def static_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    halo: ShortConvHalo | None = None,
    compiled: bool = False,
) -> torch.Tensor:
    """Apply Inkling2's static residual short convolution.

    ``y[t, c] = (x[t, c] + sum_lag w[c, W - 1 - lag] * x[t - lag, c]) / sqrt(W + 1)``.

    Args:
        x: Input of shape ``[S, B, C]``.
        weight: Per-channel taps of shape ``[C, W]`` (the HF ``[C, 1, W]`` layout squeezed).
        positions: Offsets inside each document, ``[S, B]``.
        halo: Under context parallelism, supplies the tokens before the local chunks.
        compiled: Whether to run the convolution through ``torch.compile``; with a
            halo, which exchanges rows between ranks, it runs eagerly.

    Returns:
        Output with the dtype of ``x``.
    """
    if compiled and halo is None:
        return _compiled(_static_short_conv)(x, weight, positions, halo=None)
    return _static_short_conv(x, weight, positions, halo=halo)


def _static_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    halo: ShortConvHalo | None,
) -> torch.Tensor:
    channels, width = weight.shape
    if x.shape[-1] != channels:
        raise ValueError(f"Short convolution width {channels} != input width {x.shape[-1]}")
    # The reference rounds static taps to the activation dtype before FP32 accumulation.
    taps = weight.to(x.dtype).float()
    current = x.float()
    extended = None if halo is None else halo.extend(current)
    accumulator = torch.zeros_like(current)
    for tap in range(width):
        accumulator = accumulator + _shifted(current, width - 1 - tap, positions, halo, extended) * taps[:, tap]
    return ((accumulator + current) * (width + 1) ** -0.5).to(x.dtype)


def dynamic_short_conv(
    x: torch.Tensor,
    static_weight: torch.Tensor,
    dynamic_weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    halo: ShortConvHalo | None = None,
    compiled: bool = False,
) -> torch.Tensor:
    """Apply the full-attention K/V short convolution with per-token dynamic taps.

    ``y[t, h, c] = (x[t, h, c] + sum_lag (s[lag, h, c] + d[t, h, lag]) * x[t - lag, h, c]) / sqrt(2W + 1)``.

    Args:
        x: Input of shape ``[S, B, H * D]``.
        static_weight: Static taps of shape ``[W, H * D]``.
        dynamic_weight: Dynamic taps of shape ``[S, B, H, W]``, shared within each head.
        positions: Offsets inside each document, ``[S, B]``.
        halo: Under context parallelism, supplies the tokens before the local chunks.
        compiled: Whether to run the convolution through ``torch.compile``; with a
            halo, which exchanges rows between ranks, it runs eagerly.

    Returns:
        Output with the dtype of ``x``.
    """
    if compiled and halo is None:
        return _compiled(_dynamic_short_conv)(x, static_weight, dynamic_weight, positions, halo=None)
    return _dynamic_short_conv(x, static_weight, dynamic_weight, positions, halo=halo)


def _dynamic_short_conv(
    x: torch.Tensor,
    static_weight: torch.Tensor,
    dynamic_weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    halo: ShortConvHalo | None,
) -> torch.Tensor:
    width, channels = static_weight.shape
    heads = dynamic_weight.shape[-2]
    if dynamic_weight.shape[-1] != width or channels % heads or x.shape[-1] != channels:
        raise ValueError(
            f"Inconsistent dynamic convolution shapes: x {tuple(x.shape)}, static {tuple(static_weight.shape)}, "
            f"dynamic {tuple(dynamic_weight.shape)}"
        )
    head_dim = channels // heads
    current = x.float()
    extended = None if halo is None else halo.extend(current)
    static = static_weight.float().view(width, heads, head_dim)
    dynamic = dynamic_weight.float()
    accumulator = None
    for lag in range(width):
        value = _shifted(current, lag, positions, halo, extended).unflatten(-1, (heads, head_dim))
        term = value * (static[lag] + dynamic[..., lag : lag + 1])
        accumulator = term if accumulator is None else accumulator + term
    return ((accumulator.flatten(-2) + current) * (2 * width + 1) ** -0.5).to(x.dtype)


def l2_clip(
    x: torch.Tensor,
    threshold: float,
    *,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    compiled: bool = False,
) -> torch.Tensor:
    """Scale each vector down to ``threshold`` when its L2 norm exceeds it.

    Args:
        x: Input of shape ``[..., heads, D]``.
        threshold: Maximum norm.
        scale: Optional learned per-head, per-channel scale ``[heads, D]``
            applied before clipping.
        output_scale: Optional FP32 per-token factor broadcast over the last
            two dimensions, applied after rounding the clipped vector.
        compiled: Whether to run the clip through ``torch.compile``.

    Returns:
        Output with the dtype of ``x``.
    """
    clip = _compiled(_l2_clip) if compiled else _l2_clip
    return clip(x, threshold, scale=scale, output_scale=output_scale)


def _l2_clip(
    x: torch.Tensor,
    threshold: float,
    *,
    scale: torch.Tensor | None,
    output_scale: torch.Tensor | None,
) -> torch.Tensor:
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    output = (value * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(x.dtype)
    if output_scale is not None:
        output = (output.float() * output_scale.float()[..., None, None]).to(x.dtype)
    return output


def length_scaling_tau(positions: torch.Tensor, n_floor: float, alpha: float) -> torch.Tensor:
    """Return the full-attention length scaling ``1 + alpha * log(max((p + 1) / n_floor, 1))``."""
    return 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / n_floor, min=1.0))


def situ_and_mul(
    inputs: torch.Tensor, gate_softcap: float, up_softcap: float, *, compiled: bool = False
) -> torch.Tensor:
    """Softcapped SwiGLU over concatenated ``[gate, up]`` halves.

    The sigmoid uses the raw gate, while the multiplier and the up projection are
    tanh soft-capped: ``cap_g(g) * sigmoid(g) * cap_u(u)``. A zero cap disables it.
    With ``compiled``, the token count, which changes with every batch's routing,
    becomes a dynamic dimension after its first change.
    """
    activation = _compiled(_situ_and_mul, dynamic=None) if compiled else _situ_and_mul
    return activation(inputs, gate_softcap, up_softcap)


def _situ_and_mul(inputs: torch.Tensor, gate_softcap: float, up_softcap: float) -> torch.Tensor:
    gate, up = torch.chunk(inputs.float(), 2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    if gate_softcap > 0:
        gate = gate_softcap * torch.tanh(gate / gate_softcap)
    if up_softcap > 0:
        up = up_softcap * torch.tanh(up / up_softcap)
    return (gate * sigmoid * up).to(inputs.dtype)


def document_positions(
    seq_len: int,
    batch: int,
    device: torch.device,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each token's offset inside its document as ``[S, B]`` int64.

    Without ``cu_seqlens`` every batch row is one document. With packed
    sequences (``B == 1``), ``cu_seqlens`` delimits the documents; tokens past
    its last boundary form their own document.
    """
    if cu_seqlens is None:
        return torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch)
    if batch != 1:
        raise ValueError("Packed sequences require batch size 1")
    boundaries = cu_seqlens.to(device=device, dtype=torch.int64)
    tokens = torch.arange(seq_len, device=device)
    starts = boundaries[torch.searchsorted(boundaries, tokens, right=True) - 1]
    return (tokens - starts).unsqueeze(1)


def document_ids(seq_len: int, device: torch.device, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    """Return a document index per token of a packed sequence (all zeros without ``cu_seqlens``)."""
    if cu_seqlens is None:
        return torch.zeros(seq_len, dtype=torch.int64, device=device)
    boundaries = cu_seqlens.to(device=device, dtype=torch.int64)
    return torch.searchsorted(boundaries, torch.arange(seq_len, device=device), right=True) - 1
