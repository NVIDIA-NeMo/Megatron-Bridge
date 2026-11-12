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

"""Dispatch of the ERNIE 5.5 fused Triton kernels (``ernie_fused_ops``).

The reference implementation runs three Triton kernels wherever their
inputs fit: SiTU-GLU, the multimax head, and the local RoPE of context-parallel THD rows.
:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_ops` holds Bridge's copy of the kernels. This module
decides when they run, behind ``Ernie55TransformerConfig.ernie_fused_ops``. The flag defaults to ``True``, the
reference's behavior; ``model.ernie_fused_ops=false`` on the command line restores the eager paths.

The eager paths stay the fallback. A kernel runs only for a CUDA tensor in bf16, fp16 or fp32, with Triton
importable, and in the layout the kernel takes. Everything else runs the eager code, including every CPU tensor.

Where each kernel plugs in:

* **SiTU-GLU** (``hidden_act="situ"``). Megatron-Core computes it with ``situ_glu`` from
  ``megatron.core.activations``, which three modules bind as a module global: ``megatron.core.transformer.mlp``
  (``MLP.forward``: the dense ``-`` layers, the shared expert, whose ``SharedExpertMLP.forward`` is
  ``MLP.forward``, and SequentialMLP experts), ``megatron.core.transformer.moe.experts`` (TEGroupedMLP, the grouped
  routed experts) and ``megatron.core.transformer.moe.shared_experts`` (the shared expert's
  ``moe_shared_expert_overlap`` path). :func:`install_ernie55_fused_ops` replaces each binding with a wrapper.
  Megatron-Core passes the wrapper the clamp scales, not the config, so the wrapper goes by the scales: every
  SiTU-GLU model built with ``ernie_fused_ops=True`` records its ``(activation_func_tanh_clamp_scale,
  activation_func_tanh_clamp_scale_linear)`` pair, and a model built with the flag off removes its pair again. The
  wrapper runs the Triton kernel for a call with a recorded pair, a contiguous CUDA input and no linear offset, and
  calls Megatron-Core's ``situ_glu`` otherwise, so a model with other scales keeps Megatron-Core's function. Models
  with the same scales cannot be told apart: among them, the one built last decides (a training process builds one
  model). Only the unfused activation path calls ``situ_glu`` (``bias_activation_fusion=False``, as in every ERNIE
  recipe); Megatron-Core's fused bias-SiTU kernels keep running unchanged. The activation slot
  (``use_te_activation_func``) is not an option: Megatron-Core rejects it together with
  ``activation_func_tanh_clamp_scale``.
* **Multimax.** The output layer calls
  :func:`~megatron.bridge.models.ernie.modeling_ernie55.output_layer.ernie55_multimax` with ``fused`` set from the
  config; on CUDA logits it runs the Triton forward and backward instead of the chunked fp32 autograd function.
* **THD-CP local RoPE.** At context_parallel_size > 1 without ``apply_rope_fusion``, Megatron-Core's DSv4 attention
  (the H layers' q, kv and inverse output RoPE), its CSA indexer and its CP-balanced indexer call
  ``cp_utils.apply_thd_cp_local_rope_unfused``, looked up as a module attribute. :func:`install_ernie55_fused_ops`
  wraps it. The wrapper receives the layer's config and runs the Triton kernel under the reference's conditions:
  a config with ``ernie_fused_ops=True`` (per-layer config copies keep the field), ``rotary_interleaved=False``,
  a CUDA input and an angle table exactly ``pos_dim`` wide. The input must also be in a layout the kernel takes
  (:func:`thd_cp_rope_layout_fits`). Every other call keeps Megatron-Core's function.
  ERNIE 5.5 D layers apply their RoPE in Megatron-Core's AbsorbedMLA, in the zigzag partition, and do not use it.

The reference's column-sliced remote top-k rewrites the Python top-k loop of its own
D-layer core. Bridge's D layers have no such loop: Megatron-Core's cuDNN DSA plans select their top-k, and at
context_parallel_size > 1 the ``PACKED_THD_GENERAL`` plan already scores only each document's causal columns.

Proof of use: every fused launch counts in :data:`FUSED_OP_CALLS` under ``"<op>@<site>"``, and the first launch at
each site logs one INFO line on global rank 0.
"""

from __future__ import annotations

import collections
import functools
import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from megatron.core.utils import log_single_rank
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55 import fused_ops


logger = logging.getLogger(__name__)

__all__ = [
    "CP_UTILS_MODULE",
    "FUSED_OP_CALLS",
    "SITU_GLU_SITES",
    "ernie55_fused_ops_enabled",
    "fused_kernels_support",
    "fused_multimax",
    "fused_situ_glu",
    "fused_thd_cp_local_rope",
    "install_ernie55_fused_ops",
    "thd_cp_rope_layout_fits",
    "triton_available",
]

FUSED_OP_CALLS: collections.Counter[str] = collections.Counter()
"""Fused forward launches per ``"<op>@<site>"``: ``situ_glu@<Megatron-Core module>`` (see :data:`SITU_GLU_SITES`),
``multimax@output_layer``, and ``thd_cp_local_rope@forward`` / ``thd_cp_local_rope@inverse``."""

SITU_GLU_SITES = (
    "megatron.core.transformer.mlp",
    "megatron.core.transformer.moe.experts",
    "megatron.core.transformer.moe.shared_experts",
)
"""Megatron-Core modules whose ``situ_glu`` binding computes an ERNIE 5.5 MLP activation."""

CP_UTILS_MODULE = "megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils"
"""Megatron-Core module whose ``apply_thd_cp_local_rope_unfused`` applies the RoPE of context-parallel THD rows."""

_KERNEL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@dataclass
class _DispatchState:
    """Process-wide dispatch state (see the module docstring)."""

    situ_glu_scales: set[tuple[float, float | None]] = field(default_factory=set)
    """``(gate_scale, linear_scale)`` of every SiTU-GLU model built with ``ernie_fused_ops=True`` and not rebuilt
    with the flag off since: the ``situ_glu`` wrappers run the kernel only for calls with one of these pairs."""
    logged: set[str] = field(default_factory=set)


_STATE = _DispatchState()


def ernie55_fused_ops_enabled(config: object) -> bool:
    """Whether ``config`` asks for the fused kernels (``ernie_fused_ops=True``; configs without the field do not).

    Args:
        config: A transformer config, or a per-layer copy of one.

    Returns:
        The flag, ``False`` for configs without it.
    """
    return bool(getattr(config, "ernie_fused_ops", False))


@functools.cache
def triton_available() -> bool:
    """Whether Triton with its CUDA ``libdevice`` math imports (checked once per process).

    Returns:
        ``True`` if the kernels can be defined.
    """
    try:
        import triton.language  # noqa: F401
        from triton.language.extra.cuda import libdevice  # noqa: F401
    except ImportError:
        return False
    return True


def fused_kernels_support(tensor: Tensor) -> bool:
    """Whether the Triton kernels can take ``tensor``: CUDA, bf16, fp16 or fp32, and Triton importable.

    Args:
        tensor: Kernel input.

    Returns:
        ``True`` if the device, the dtype and the Triton installation allow a kernel launch; the caller checks
        the layout.
    """
    return tensor.is_cuda and tensor.dtype in _KERNEL_DTYPES and triton_available()


def thd_cp_rope_layout_fits(x: Tensor, rotary_pos_emb: Tensor, nope_dim: int, pos_dim: int) -> bool:
    """Whether the THD-CP local RoPE kernel takes this layout, as :func:`fused_ops.fused_thd_cp_local_rope` assumes.

    The kernel takes rows ``[t, heads, dim]``, ``[t, 1, heads, dim]`` or ``[t, dim]``. Their width must be
    ``dim = nope_dim + pos_dim``, with an even ``pos_dim`` (two rotary halves). The angle table must hold exactly one
    row of ``pos_dim`` angles per position, such as ``[positions, 1, 1, pos_dim]`` from Megatron-Core's
    ``RotaryEmbedding``; the table width is also one of the reference's own conditions. Megatron-Core's function
    takes more, for example a batch dimension other than 1, and fails in its own way on a width other than
    ``nope_dim + pos_dim``. On such inputs the kernel would raise or read the wrong angles.

    Args:
        x: Rows to rotate.
        rotary_pos_emb: The RoPE angle table.
        nope_dim: Leading channels that pass through.
        pos_dim: Rotary channels.

    Returns:
        ``True`` if the kernel can take ``x`` and ``rotary_pos_emb``; the caller checks the device and the dtype.
    """
    return (
        (x.ndim in (2, 3) or (x.ndim == 4 and x.shape[1] == 1))
        and nope_dim >= 0
        and pos_dim > 0
        and pos_dim % 2 == 0
        and x.shape[-1] == nope_dim + pos_dim
        and rotary_pos_emb.ndim >= 2
        and rotary_pos_emb.shape[-1] == pos_dim
        and rotary_pos_emb.numel() == rotary_pos_emb.shape[0] * pos_dim
    )


def _log_once(key: str, message: str, *args: object, level: int = logging.INFO) -> None:
    """Log ``message`` once per process and ``key``, on global rank 0."""
    if key in _STATE.logged:
        return
    _STATE.logged.add(key)
    log_single_rank(logger, level, message, *args)


def _record(op: str, site: str, tensor: Tensor) -> None:
    """Count a fused launch and log the first one at each site."""
    key = f"{op}@{site}"
    FUSED_OP_CALLS[key] += 1
    _log_once(
        key,
        "ERNIE 5.5 fused ops: first Triton %s launch at %s (input %s %s).",
        op,
        site,
        tuple(tensor.shape),
        str(tensor.dtype).removeprefix("torch."),
    )


def fused_situ_glu(x: Tensor, gate_scale: float, linear_scale: float | None = None, *, site: str = "direct") -> Tensor:
    """Count, log and run the Triton SiTU-GLU (:func:`fused_ops.fused_situ_glu`).

    Args:
        x: Contiguous CUDA input ``[..., 2 * n]``: the gate half, then the up half.
        gate_scale: Megatron-Core's ``activation_func_tanh_clamp_scale`` (the reference's ``beta``).
        linear_scale: ``activation_func_tanh_clamp_scale_linear`` (``linear_beta``); ``None`` for a linear up half.
        site: Call-site label for :data:`FUSED_OP_CALLS`.

    Returns:
        ``[..., n]`` in the dtype of ``x``.
    """
    _record("situ_glu", site, x)
    return fused_ops.fused_situ_glu(x, gate_scale, linear_scale)


def fused_multimax(logits: Tensor, ranges: Tensor, scales: Tensor, *, site: str = "output_layer") -> Tensor:
    """Count, log and run the Triton multimax (:func:`fused_ops.fused_multimax`) on CUDA logits.

    Args:
        logits: CUDA logits ``[..., vocab]`` (made contiguous here).
        ranges: ``[4]`` ranges ``r``.
        scales: ``[4]`` scales ``t``.
        site: Call-site label for :data:`FUSED_OP_CALLS`.

    Returns:
        Corrected logits in a new tensor with the shape and dtype of ``logits``.
    """
    _record("multimax", site, logits)
    return fused_ops.fused_multimax(logits.contiguous(), ranges, scales)


def fused_thd_cp_local_rope(
    x: Tensor,
    rotary_pos_emb: Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: Tensor,
    global_start: int,
    inverse: bool = False,
) -> Tensor:
    """Count, log and run the Triton THD-CP local RoPE (:func:`fused_ops.fused_thd_cp_local_rope`).

    Arguments are those of ``cp_utils.apply_thd_cp_local_rope_unfused`` without the config.

    Returns:
        A new tensor with the shape and dtype of ``x``.
    """
    _record("thd_cp_local_rope", "inverse" if inverse else "forward", x)
    return fused_ops.fused_thd_cp_local_rope(
        x, rotary_pos_emb, nope_dim, pos_dim, cu_seqlens_padded, global_start, inverse
    )


def _situ_glu_dispatcher(eager: Callable[..., Tensor], site: str) -> Callable[..., Tensor]:
    """Wrap Megatron-Core's ``situ_glu`` as bound in ``site``: the Triton kernel when it applies, else ``eager``."""

    def situ_glu(
        x: Tensor, gate_scale: float, linear_scale: float | None = None, linear_offset: float = 0.0
    ) -> Tensor:
        if (
            (gate_scale, linear_scale) in _STATE.situ_glu_scales
            and linear_offset == 0.0
            and x.ndim >= 1
            and x.shape[-1] % 2 == 0
            and x.is_contiguous()
            and fused_kernels_support(x)
        ):
            return fused_situ_glu(x, gate_scale, linear_scale, site=site)
        return eager(x, gate_scale, linear_scale, linear_offset)

    situ_glu.__doc__ = f"ERNIE 5.5 dispatcher around Megatron-Core's ``situ_glu`` in ``{site}`` (see fused_dispatch)."
    situ_glu.ernie55_eager = eager
    situ_glu.ernie55_site = site
    return situ_glu


def _thd_cp_local_rope_dispatcher(unfused: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Wrap ``cp_utils.apply_thd_cp_local_rope_unfused``: the Triton kernel for ERNIE configs, else ``unfused``."""

    def apply_thd_cp_local_rope_unfused(
        x: Tensor,
        rotary_pos_emb: Tensor,
        nope_dim: int,
        pos_dim: int,
        cu_seqlens_padded: Tensor,
        global_start: int,
        config: object,
        inverse: bool = False,
    ) -> Tensor:
        if (
            ernie55_fused_ops_enabled(config)
            and not getattr(config, "rotary_interleaved", False)
            and thd_cp_rope_layout_fits(x, rotary_pos_emb, nope_dim, pos_dim)
            and fused_kernels_support(x)
        ):
            return fused_thd_cp_local_rope(
                x, rotary_pos_emb, nope_dim, pos_dim, cu_seqlens_padded, global_start, inverse
            )
        return unfused(x, rotary_pos_emb, nope_dim, pos_dim, cu_seqlens_padded, global_start, config, inverse)

    apply_thd_cp_local_rope_unfused.__doc__ = (
        "ERNIE 5.5 dispatcher around Megatron-Core's apply_thd_cp_local_rope_unfused (see fused_dispatch)."
    )
    apply_thd_cp_local_rope_unfused.ernie55_unfused = unfused
    return apply_thd_cp_local_rope_unfused


def _install_situ_glu_dispatchers() -> list[str]:
    """Wrap ``situ_glu`` in every importable site of :data:`SITU_GLU_SITES` (idempotent); return the wrapped sites."""
    wrapped = []
    for site in SITU_GLU_SITES:
        try:
            module = importlib.import_module(site)
        except ImportError:
            continue
        current = getattr(module, "situ_glu", None)
        if current is None:
            continue
        if getattr(current, "ernie55_site", None) is None:
            module.situ_glu = _situ_glu_dispatcher(current, site)
        wrapped.append(site)
    return wrapped


def _install_thd_cp_local_rope_dispatcher() -> bool:
    """Wrap ``cp_utils.apply_thd_cp_local_rope_unfused`` (idempotent); ``False`` if Megatron-Core has none."""
    try:
        cp_utils = importlib.import_module(CP_UTILS_MODULE)
    except ImportError:
        return False
    current = getattr(cp_utils, "apply_thd_cp_local_rope_unfused", None)
    if current is None:
        return False
    if getattr(current, "ernie55_unfused", None) is None:
        cp_utils.apply_thd_cp_local_rope_unfused = _thd_cp_local_rope_dispatcher(current)
    return True


def _situ_glu_scales(config: object) -> tuple[float, float | None] | None:
    """``config``'s SiTU-GLU clamp scales as Megatron-Core passes them to ``situ_glu``; ``None`` without SiTU-GLU."""
    gate_scale = getattr(config, "activation_func_tanh_clamp_scale", None)
    if gate_scale is None:
        return None
    return gate_scale, getattr(config, "activation_func_tanh_clamp_scale_linear", None)


def install_ernie55_fused_ops(config: object) -> None:
    """Apply ``config.ernie_fused_ops`` to the process: install the dispatchers, record or drop the SiTU-GLU scales.

    :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` calls this after it is built.
    With ``ernie_fused_ops=True`` it wraps ``cp_utils.apply_thd_cp_local_rope_unfused``. For a SiTU-GLU model
    (``activation_func_tanh_clamp_scale`` set) it also wraps the ``situ_glu`` binding of every
    :data:`SITU_GLU_SITES` module and records the model's ``(activation_func_tanh_clamp_scale,
    activation_func_tanh_clamp_scale_linear)`` pair, so that the wrappers run the kernel for calls with these
    scales. Each wrapper is installed once per process. With ``False`` nothing is installed, and the model's SiTU-GLU
    scales are dropped if an earlier model recorded them, so calls with them run Megatron-Core's ``situ_glu`` again.
    The multimax head reads the flag itself (see
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.output_layer.install_ernie55_multimax`).

    Args:
        config: The model's transformer config.
    """
    enabled = ernie55_fused_ops_enabled(config)
    scales = _situ_glu_scales(config)
    if not enabled:
        if scales in _STATE.situ_glu_scales:
            _STATE.situ_glu_scales.discard(scales)
            _log_once(
                f"situ_glu_off:{scales}",
                "ERNIE 5.5 fused ops: a model with ernie_fused_ops=False and the SiTU-GLU scales %s of an earlier "
                "model was built; SiTU-GLU with these scales runs eagerly in this process from now on.",
                scales,
                level=logging.WARNING,
            )
        _log_once("off", "ERNIE 5.5 fused ops: off (ernie_fused_ops=False); the eager paths run.")
        return
    rope = _install_thd_cp_local_rope_dispatcher()
    situ_sites = []
    if scales is not None:
        situ_sites = _install_situ_glu_dispatchers()
        _STATE.situ_glu_scales.add(scales)
    _log_once(
        f"on:{scales}",
        "ERNIE 5.5 fused ops: on (ernie_fused_ops=True). Triton kernels for multimax%s%s, on CUDA tensors.",
        "; THD-CP local RoPE" if rope else "",
        "" if scales is None else f"; SiTU-GLU with scales {scales} in {', '.join(situ_sites)}",
    )
    if torch.cuda.is_available() and not triton_available():
        _log_once(
            "no_triton",
            "ERNIE 5.5 fused ops: ernie_fused_ops=True, but Triton does not import; the eager paths run.",
            level=logging.WARNING,
        )
