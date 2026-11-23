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

"""ERNIE 5.5 RMSNorms: the downcast of fp32 streams, and the reference implementation's arithmetic.

With hyper-connections, ERNIE 5.5 carries the decoder's residual streams in fp32: the embedding is
cast to fp32 at the stack entry, and every mix of the streams runs in fp32. The layers themselves
compute in bf16. The reference model's RMSNorm therefore casts its input to the weight dtype first,
then computes fp32 statistics and returns the weight dtype. A norm built here does the same: it is
the configured norm module (Transformer Engine's ``TENorm`` in the stack spec), unchanged and under
its own parameter names, plus a forward pre-hook that casts the input to the norm weight's dtype.

That cast is the only rounding between an fp32 stream and a layer, so a downcasting norm sits in
every place that can receive a stream: each layer's ``input_layernorm`` and ``pre_mlp_layernorm``,
the decoder's final norm, and the MTP ``hnorm``. Inputs that already have the weight dtype pass
through unchanged, so the norms behave like the plain ones in stacks without hyper-connections.
Autograd casts the gradient back to the input dtype.

Reference arithmetic (``ernie_reference_rmsnorm``). Transformer Engine's RMSNorm and the reference
implementation's ``ErnieRMSNorm`` compute the same function, ``bf16(x * rsqrt(mean(x^2) + eps) * w)``
with fp32 statistics, but they sum the squares in different orders: TE's kernel adds each thread's
squares in sequence and then reduces across a warp, PyTorch's reduction kernel uses its own tree.
The mean square then differs by an fp32 ulp on about a third of the rows, and when ``x * rsqrt`` lands
on a bf16 rounding tie, the output differs by a bf16 ulp. TE also switches kernels when the row count
is a multiple of 4, so a token's norm output depends on how many tokens its micro-batch holds.
:class:`Ernie55RMSNorm` is the reference's eager expression, statement for statement, and reproduces
its outputs and gradients bitwise. It is slower than TE's kernel (it keeps fp32 copies for the
backward), so it runs only when the config sets ``ernie_reference_rmsnorm``; the norm builders here
(:func:`ernie55_rmsnorm`, :func:`ernie55_rmsnorm_spec` and :func:`downcasting_norm`) pick it or the
configured norm per module. The reference builds the norms of every layer, the attention latents, the
H compressor and the decoder's final norm as ``ErnieRMSNorm``, but its MTP block (``enorm``, ``hnorm``
and the MTP final norm) comes from Megatron-Core's Transformer Engine MTP spec, so those stay ``TENorm``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from megatron.core.transformer.spec_utils import ModuleSpec


__all__ = [
    "Ernie55RMSNorm",
    "Ernie55RMSNormSelector",
    "downcast_norm_input",
    "downcasting_norm",
    "ernie55_rmsnorm",
    "ernie55_rmsnorm_spec",
    "is_downcasting_norm",
    "uses_reference_rmsnorm",
]

# Set on a norm module once its forward pre-hook is registered.
_DOWNCAST_ATTR = "_ernie55_downcasts_input"


def uses_reference_rmsnorm(config: object) -> bool:
    """Whether ``config`` selects the reference implementation's RMSNorm arithmetic (``ernie_reference_rmsnorm``)."""
    return bool(getattr(config, "ernie_reference_rmsnorm", False))


def _norm_device(config: object) -> torch.device | str:
    """Device for the norm weight, chosen like Megatron-Core's TE modules (CPU when CUDA is missing)."""
    if getattr(config, "use_cpu_initialization", False):
        return "cpu"
    if getattr(config, "init_model_with_meta_device", False):
        return "meta"
    if not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda", torch.cuda.current_device())


class Ernie55RMSNorm(torch.nn.Module):
    """RMSNorm with the reference implementation's eager arithmetic (``ErnieRMSNorm``).

    ``forward`` computes ``x = x.to(w.dtype)``, then ``(xf * rsqrt(mean(xf^2) + eps) * w.float())`` with
    ``xf = x.float()``, rounded once to the weight dtype: the reference's expression in the reference's
    order, so its outputs and gradients are bitwise the reference's. The parameter is ``weight``
    (``params_dtype``, initialized to ones), the name and layout of Transformer Engine's RMSNorm, so
    checkpoints and weight mappings are unchanged.

    Args:
        config: Transformer config; ``normalization`` must be ``"RMSNorm"``, without
            ``layernorm_zero_centered_gamma``.
        hidden_size: Normalized width.
        eps: Epsilon added to the mean square.
        has_residual: Must be False (no fused residual).

    Raises:
        ValueError: If the config asks for another normalization, a zero-centered gamma or a fused
            residual.
    """

    def __init__(self, config: Any, hidden_size: int, eps: float = 1e-5, has_residual: bool = False) -> None:
        super().__init__()
        if getattr(config, "normalization", "RMSNorm") != "RMSNorm":
            raise ValueError(
                f"ERNIE 5.5 reference RMSNorm needs normalization='RMSNorm', got {config.normalization!r}."
            )
        if getattr(config, "layernorm_zero_centered_gamma", False):
            raise ValueError("ERNIE 5.5 reference RMSNorm does not support layernorm_zero_centered_gamma.")
        if has_residual:
            raise ValueError("ERNIE 5.5 reference RMSNorm does not fuse a residual (has_residual=True).")
        self.eps = eps
        self.returns_residual = False
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=config.params_dtype, device=_norm_device(config))
        )
        # Transformer Engine's norms tag their weight like this; Megatron-Core's gradient reduction reads it.
        self.weight.sequence_parallel = bool(getattr(config, "sequence_parallel", False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``; returns the weight dtype."""
        x = x.to(self.weight.dtype)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(
            self.weight.dtype
        )

    def extra_repr(self) -> str:
        """Width and epsilon."""
        return f"{self.weight.numel()}, eps={self.eps}"


def _config_argument(args: tuple, kwargs: dict) -> object:
    if "config" in kwargs:
        return kwargs["config"]
    if args:
        return args[0]
    raise TypeError("ERNIE 5.5 norm builders need the transformer config (config=...).")


def _build_ernie55_rmsnorm(*args, norm: Callable[..., torch.nn.Module], **kwargs) -> torch.nn.Module:
    if uses_reference_rmsnorm(_config_argument(args, kwargs)):
        return Ernie55RMSNorm(*args, **kwargs)
    return norm(*args, **kwargs)


class Ernie55RMSNormSelector:
    """Norm "class" whose construction returns :class:`Ernie55RMSNorm` or ``norm``, like ``TENorm`` itself.

    ``Ernie55RMSNormSelector(config=..., hidden_size=..., eps=..., norm=TENorm)`` builds
    :class:`Ernie55RMSNorm` when ``config`` sets ``ernie_reference_rmsnorm`` and ``norm(...)`` otherwise. It is
    a class so that Megatron-Core's ``build_module`` accepts it inside a ``ModuleSpec``
    (:func:`ernie55_rmsnorm_spec`); slots that Megatron-Core calls directly take :func:`ernie55_rmsnorm`.
    """

    def __new__(cls, *args, norm: Callable[..., torch.nn.Module], **kwargs) -> torch.nn.Module:
        return _build_ernie55_rmsnorm(*args, norm=norm, **kwargs)


def ernie55_rmsnorm(norm: Callable[..., torch.nn.Module]) -> functools.partial:
    """Return a norm builder that builds :class:`Ernie55RMSNorm` under ``ernie_reference_rmsnorm``, else ``norm``.

    The result goes into layer-spec norm slots that Megatron-Core calls directly with ``config``,
    ``hidden_size`` and ``eps``; the config of that call decides. ``norm`` stays visible as the
    ``norm`` keyword, so spec tools can replace it. Slots built with ``build_module`` take
    :func:`ernie55_rmsnorm_spec`.

    Args:
        norm: The norm builder used without the switch, for example ``TENorm``.

    Returns:
        ``functools.partial`` of :class:`Ernie55RMSNormSelector` with ``norm`` bound.
    """
    return functools.partial(Ernie55RMSNormSelector, norm=norm)


def ernie55_rmsnorm_spec(norm: Callable[..., torch.nn.Module]) -> ModuleSpec:
    """:func:`ernie55_rmsnorm` as a ``ModuleSpec``, for slots Megatron-Core builds with ``build_module``.

    Args:
        norm: The norm builder used without the switch, for example ``TENorm``.

    Returns:
        ``ModuleSpec(module=Ernie55RMSNormSelector, params={"norm": norm})``.
    """
    return ModuleSpec(module=Ernie55RMSNormSelector, params={"norm": norm})


def _cast_input_to_weight_dtype(module: torch.nn.Module, args: tuple) -> tuple:
    if not args:
        raise TypeError(f"ERNIE 5.5 downcasting norm {type(module).__name__}: the input must be passed positionally.")
    weight_dtype = module.weight.dtype
    hidden_states = args[0]
    if hidden_states.dtype == weight_dtype:
        return args
    return (hidden_states.to(weight_dtype), *args[1:])


def downcast_norm_input(module: torch.nn.Module) -> torch.nn.Module:
    """Make ``module`` cast its input to its weight dtype before normalizing (in place, idempotent).

    Args:
        module: A norm module with a ``weight`` parameter.

    Returns:
        ``module``.

    Raises:
        TypeError: If ``module`` has no ``weight`` parameter.
    """
    if not isinstance(getattr(module, "weight", None), torch.Tensor):
        raise TypeError(f"ERNIE 5.5 downcasting norm: {type(module).__name__} has no weight to take the dtype from.")
    if not getattr(module, _DOWNCAST_ATTR, False):
        module.register_forward_pre_hook(_cast_input_to_weight_dtype)
        setattr(module, _DOWNCAST_ATTR, True)
    return module


def is_downcasting_norm(module: torch.nn.Module) -> bool:
    """Whether :func:`downcast_norm_input` was applied to ``module``."""
    return bool(getattr(module, _DOWNCAST_ATTR, False))


def _build_downcasting_norm(*args, norm: Callable[..., torch.nn.Module], **kwargs) -> torch.nn.Module:
    return downcast_norm_input(_build_ernie55_rmsnorm(*args, norm=norm, **kwargs))


def _build_fixed_downcasting_norm(*args, norm: Callable[..., torch.nn.Module], **kwargs) -> torch.nn.Module:
    return downcast_norm_input(norm(*args, **kwargs))


def downcasting_norm(norm: Callable[..., torch.nn.Module], *, reference_switch: bool = True) -> functools.partial:
    """Return a norm builder that builds ``norm`` and applies :func:`downcast_norm_input`.

    The result goes into layer-spec norm slots (``input_layernorm``, ``pre_mlp_layernorm``, the MTP
    ``hnorm``), which Megatron-Core calls with ``config``, ``hidden_size`` and ``eps``. Under
    ``ernie_reference_rmsnorm`` it builds :class:`Ernie55RMSNorm` instead of ``norm`` (see
    :func:`ernie55_rmsnorm`), unless ``reference_switch`` is False: the reference implementation builds
    its MTP block norms with Transformer Engine, so the MTP ``hnorm`` always builds ``norm``. The wrapped
    builder stays visible as the ``norm`` keyword, so spec tools can replace it.

    Args:
        norm: The norm builder, for example ``TENorm``.
        reference_switch: Whether ``ernie_reference_rmsnorm`` selects :class:`Ernie55RMSNorm`.

    Returns:
        ``functools.partial`` of the internal builder with ``norm`` bound.
    """
    builder = _build_downcasting_norm if reference_switch else _build_fixed_downcasting_norm
    return functools.partial(builder, norm=norm)
