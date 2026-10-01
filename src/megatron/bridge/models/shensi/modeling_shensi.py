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

"""Shensi custom components.

Shensi keeps the Megatron-Core transformer stack and adds the pieces its architecture needs:

* **mHC hyper-connections** — the residual stream is a pool of ``hc_mult`` streams; each sublayer
  reads the pool and writes back into a fixed plus a routed subset of streams.
* **Attention residuals** — layers are grouped into blocks; a block-writing layer publishes its
  stream context and the later layers of the block read it. The state travels across pipeline
  stages on its own communication group.
* **Mixture-of-experts layers** — routed experts run through a low-rank bottleneck, the leading
  layers are dense MLPs gated by a token-embedding lookup, and the ERC term needs a global view of
  the expert weights.
* **Selective fp32 keeping** — a small set of parameters stays in fp32 while the model trains in
  bf16, matching the ``_keep_in_fp32_modules_strict`` list of the HF checkpoint.
* **Decoder layer, layer specs and the model** — the wiring that assembles the above into a
  Megatron-Core ``GPTModel`` with a multi-token-prediction head.
"""

import copy
import functools
import hashlib
import math
import os
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.backends import get_backend_from_config
from megatron.core.models.gpt.gpt_model import GPTModel as DeepSeekModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention_module_specs import (
    get_dsv4_hybrid_module_spec_for_backend,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    LayerNormBuilder,
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.training import print_rank_0
from torch import Tensor

from megatron.bridge.models.shensi.configuration_shensi import (
    SHENSI_PENDING_FIELDS,
    ShensiTransformerConfig,
)


# Selective fp32 keeping
# ----------------------
# Selective fp32 parameter keeping.
#
# Some Shensi parameters are numerically sensitive enough to be held in fp32 while the rest of the
# model trains in bf16: the mHC stream mixing (its Gram-Schmidt branch divides by norms), the
# attention-residual readers, and the strict list of norms / sinks / position biases that the HF
# checkpoint also keeps in fp32 (``_keep_in_fp32_modules_strict``).
#
# The module marks the selected parameters and, when the surrounding module is moved by a caller
# (``.apply``, ``.to``, meta-device init), restores them; a forward wrapper casts the marked weights
# to the compute dtype for the matmul and casts the gradient back. Groups are selectable through
# ``SHENSI_FP32_KEEP_GROUPS`` (or ``SHENSI_HC_FP32_KEEP=0`` to disable the mechanism).

FP32_KEEP_ATTR = "_shensi_keep_fp32"
FP32_KEEP_GROUP_ATTR = "SHENSI_FP32_KEEP_GROUP"
GROUP_MHC = "mhc"
GROUP_ATTN_RES = "attn_res"
GROUP_STRICT = "strict"
DEFAULT_GROUPS: Tuple[str, ...] = (GROUP_MHC, GROUP_STRICT)
STRICT_SUFFIXES: Tuple[str, ...] = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_mlp_layernorm",
    "q_a_norm",
    "kv_norm",
    "norm",
    "sinks",
    "position_bias",
)
STRICT_EXTRA_SUBSTRINGS: Tuple[str, ...] = (
    ".core_attention.attn_sink",
    ".core_attention.compressor.ape",
    ".core_attention.indexer.compressor.ape",
)
CAST_ORIG_FORWARD_ATTR = "_shensi_cast_orig_forward"
CAST_DTYPE_ATTR = "_shensi_cast_model_dtype"
APPLY_GUARD_ATTR = "_shensi_orig_apply"
ENV_SWITCH = "SHENSI_HC_FP32_KEEP"
ENV_GROUPS = "SHENSI_FP32_KEEP_GROUPS"
ENV_NO_CAST = "SHENSI_FP32_KEEP_NO_CAST"


class ShensiFp32KeepMixin:
    """Marks a module whose parameters stay in fp32 during bf16 training."""

    def _apply(self, fn, recurse=True):  # noqa: ANN001
        saved: List[Tuple[torch.nn.Module, str, torch.Tensor]] = []
        for mod in self.modules():
            for name, p in mod._parameters.items():
                if p is not None and getattr(p, FP32_KEEP_ATTR, False) and p.dtype == torch.float32:
                    saved.append((mod, name, p.detach().clone()))
        out = super()._apply(fn, recurse=recurse)
        for mod, name, fp32 in saved:
            cur = mod._parameters.get(name)
            if cur is None:
                continue
            if cur.dtype != torch.float32 or cur.device != fp32.device:
                cur.data = fp32.to(cur.device)
        return out


def fp32_keep_enabled(config=None, env: Optional[str] = None) -> bool:
    """Whether fp32 keeping is active for this config / environment."""
    return bool(fp32_keep_groups(config, env))


def fp32_keep_groups(config=None, env: Optional[str] = None) -> Tuple[str, ...]:
    """Parameter groups that stay in fp32 (the environment overrides the config field)."""
    import os

    raw_groups = os.environ.get(ENV_GROUPS, env)
    if raw_groups is not None and str(raw_groups).strip() != "":
        return tuple(g.strip() for g in str(raw_groups).replace(",", " ").split() if g.strip())
    raw = os.environ.get(ENV_SWITCH, None)
    if raw is not None and str(raw).strip() != "":
        return DEFAULT_GROUPS if str(raw).strip().lower() in ("1", "true", "yes", "on") else ()
    if config is not None and hasattr(config, "hc_fp32_keep"):
        return DEFAULT_GROUPS if bool(getattr(config, "hc_fp32_keep")) else ()
    return DEFAULT_GROUPS


def _keep_targets(module: torch.nn.Module, groups) -> List[torch.nn.Module]:
    want = tuple(groups)
    return [m for m in module.modules() if getattr(m, FP32_KEEP_GROUP_ATTR, None) in want]


def apply_fp32_keep(model: torch.nn.Module, *, groups=None, verbose: bool = False) -> List[str]:
    """Cast the selected parameters of a model to fp32 and return their names."""
    groups = DEFAULT_GROUPS if groups is None else tuple(groups)
    if not groups:
        return []
    names: List[str] = []
    targets = _keep_targets(model, groups)
    for mod in targets:
        for pname, p in list(mod.named_parameters(recurse=True)):
            if p is None:
                continue
            if p.dtype != torch.float32:
                p.data = p.data.float()
            setattr(p, FP32_KEEP_ATTR, True)
            names.append(f"{type(mod).__name__}:{pname}")
    if GROUP_STRICT in groups:
        names.extend(apply_strict_fp32_keep(model))
    if verbose and names:
        from megatron.training import print_rank_0

        print_rank_0(
            f"[shensi][fp32keep] kept {len(names)} parameters in fp32 across {len(targets)} family "
            f"modules (groups={','.join(groups)})"
        )
    return sorted(names)


def install_apply_guard(mod: torch.nn.Module) -> bool:
    """Install a guard that re-applies fp32 keeping after ``.apply`` moves the module."""
    if getattr(mod, APPLY_GUARD_ATTR, None) is not None:
        return False
    if not any(p is not None and getattr(p, FP32_KEEP_ATTR, False) for p in mod._parameters.values()):
        return False
    orig_apply = mod._apply

    def guarded_apply(fn, recurse=True):
        saved = [
            (n, p, p.detach().clone())
            for n, p in mod._parameters.items()
            if p is not None and getattr(p, FP32_KEEP_ATTR, False) and p.dtype == torch.float32
        ]
        out = orig_apply(fn, recurse=recurse)
        for n, p, fp32 in saved:
            cur = mod._parameters.get(n)
            if cur is None or cur is fp32:
                continue
            if cur.dtype != torch.float32 or cur.device != fp32.device:
                cur.data = fp32.to(cur.device)
        return out

    setattr(mod, APPLY_GUARD_ATTR, orig_apply)
    mod._apply = guarded_apply
    return True


def compute_dtype_for(inputs, kwargs, fallback: torch.dtype) -> torch.dtype:
    """Dtype of the current compute (autocast) context, falling back to the tensor dtypes."""
    try:
        if torch.is_autocast_enabled():
            return torch.get_autocast_dtype(torch.get_autocast_device_type())
    except Exception:  # noqa: BLE001
        pass
    for t in list(inputs) + list(kwargs.values()):
        if torch.is_tensor(t) and t.is_floating_point() and t.dtype != torch.float32:
            return t.dtype
    for t in list(inputs) + list(kwargs.values()):
        if torch.is_tensor(t) and t.is_floating_point():
            return t.dtype
    return fallback


class _ShensiFp32CastToComputeDtype(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dtype: torch.dtype):
        ctx.input_dtype = tensor.dtype
        return tensor.to(dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.to(ctx.input_dtype), None


def compute_dtype_weight(p: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast a weight to the compute dtype for the forward pass."""
    if p.dtype == dtype:
        return p
    if not p.requires_grad:
        return p.detach().to(dtype)
    return _ShensiFp32CastToComputeDtype.apply(p, dtype)


def install_forward_cast(mod: torch.nn.Module) -> bool:
    """Make a module cast its fp32-kept weights to the compute dtype on forward."""
    if getattr(mod, CAST_ORIG_FORWARD_ATTR, None) is not None:
        return False
    import os

    if str(os.environ.get(ENV_NO_CAST, "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    # The kept names and the model dtype are fixed once the module is marked, so both are resolved
    # here rather than on every forward pass.
    kept = [
        name
        for name, p in mod._parameters.items()
        if p is not None and getattr(p, FP32_KEEP_ATTR, False) and p.dtype == torch.float32
    ]
    if not kept:
        return False
    orig = mod.forward
    model_dtype = next((p.dtype for p in mod._parameters.values() if p is not None and p.dtype != torch.float32), None)

    def casted_forward(*args, **kwargs):
        fallback = getattr(mod, CAST_DTYPE_ATTR, None) or model_dtype
        dtype = compute_dtype_for(args, kwargs, fallback)
        if dtype is None or dtype == torch.float32:
            return orig(*args, **kwargs)
        saved = [(name, mod._parameters[name]) for name in kept]
        for name, p in saved:
            mod._parameters[name] = compute_dtype_weight(p, dtype)
        try:
            return orig(*args, **kwargs)
        finally:
            for name, p in saved:
                mod._parameters[name] = p

    casted_forward.__name__ = "shensi_fp32_keep_cast_forward"
    setattr(mod, CAST_ORIG_FORWARD_ATTR, orig)
    mod.forward = casted_forward
    return True


def set_cast_model_dtype(model: torch.nn.Module) -> None:
    """Record the compute dtype used by the forward casts of a model."""
    dtypes = {p.dtype for p in model.parameters() if p.dtype != torch.float32}
    dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
    for mod in model.modules():
        if getattr(mod, CAST_ORIG_FORWARD_ATTR, None) is not None:
            setattr(mod, CAST_DTYPE_ATTR, dtype)


def _is_family_keep_module(mod: torch.nn.Module) -> bool:
    return getattr(mod, FP32_KEEP_GROUP_ATTR, None) in (GROUP_MHC, GROUP_ATTN_RES)


def _ancestor_has_family_group(model: torch.nn.Module, mod_name: str) -> bool:
    mods = dict(model.named_modules())
    parts = mod_name.split(".") if mod_name else []
    for i in range(len(parts), -1, -1):
        m = mods.get(".".join(parts[:i]))
        if m is not None and _is_family_keep_module(m):
            return True
    return False


def strict_keep_targets(model: torch.nn.Module, suffixes=STRICT_SUFFIXES):
    """Modules whose non-family parameters should stay in fp32."""
    pats = tuple(suffixes) + tuple(STRICT_EXTRA_SUBSTRINGS)
    mods = dict(model.named_modules())
    out = []
    for key, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if not any(s in key for s in pats):
            continue
        head = key.rsplit(".", 1)[0] if "." in key else ""
        if _ancestor_has_family_group(model, head):
            continue
        mod = mods.get(head, model)
        out.append((key, head, mod))
    return out


def apply_strict_fp32_keep(model: torch.nn.Module, *, suffixes=None, verbose=False) -> List[str]:
    """Apply fp32 keeping to the strict parameter suffixes of a model."""
    suffixes = STRICT_SUFFIXES if suffixes is None else tuple(suffixes)
    names: List[str] = []
    touched = {}
    for key, mod_name, mod in strict_keep_targets(model, suffixes):
        p = dict(mod.named_parameters(recurse=False)).get(key.rsplit(".", 1)[-1])
        if p is None:
            p = model.get_parameter(key)
        if p.dtype != torch.float32:
            p.data = p.data.float()
        setattr(p, FP32_KEEP_ATTR, True)
        names.append(key)
        touched[id(mod)] = (mod_name, mod)
    for mod_name, mod in touched.values():
        install_forward_cast(mod)
        install_apply_guard(mod)
    set_cast_model_dtype(model)
    if verbose and names:
        print(f"[shensi][fp32keep] strict: {len(names)} parameters in {len(touched)} modules")
    return sorted(names)


def strict_keep_param_names(model: torch.nn.Module) -> List[str]:
    """Names of the non-family parameters kept in fp32."""
    return sorted(k for k, _m, _mod in strict_keep_targets(model))


def keep_param_names(model: torch.nn.Module) -> List[str]:
    """All parameter names kept in fp32 (family and strict groups)."""
    return sorted(
        f"{mod_name}.{pname}" if mod_name else pname
        for mod_name, mod in model.named_modules()
        for pname, p in mod._parameters.items()
        if p is not None and getattr(p, FP32_KEEP_ATTR, False)
    )


def dtype_counts(model: torch.nn.Module) -> dict:
    """How many parameters of each dtype a model has."""
    out: dict = {}
    for _, p in model.named_parameters():
        key = str(p.dtype).replace("torch.", "")
        n, numel = out.get(key, (0, 0))
        out[key] = (n + 1, numel + p.numel())
    return out


# mHC hyper-connections
# -----------------------
# mHC hyper-connections.
#
# The residual stream of Shensi is not a single tensor but ``hc_mult`` parallel streams. Each
# sublayer (attention or MLP) reads the streams through a hyper-connection, collapses them for the
# sublayer, and writes the result back into a subset of streams:
#
# * ``pre_fn`` collapses all streams into the sublayer input;
# * ``write_back`` refreshes ``hc_active_streams`` streams per token: the first ``hc_fixed_streams``
#   are always refreshed and the remaining ``routed_streams`` are picked by a top-k router over the
#   stream pool. On MLP layers the write is additionally augmented with depth convolutions and their
#   Gram-Schmidt orthogonalizations, which give the mixer a local (in depth) view of the history;
# * `ShensiHyperHead` collapses the streams back to ``hidden_size`` at the end of the stack, which is
#   also the contraction the MTP head consumes.
#
# The stream mixing parameters stay in fp32 (see the fp32-keeping section above) because the
# Gram-Schmidt branch normalizes by stream norms.


def _packed_bounds(packed_seq_params, total: int):
    """Document boundaries of a packed (THD) batch, or None when the batch is dense."""
    if packed_seq_params is None:
        return None
    cu = packed_seq_params.cu_seqlens_kv_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_kv
    bounds = [int(v) for v in cu.tolist()]
    if bounds[-1] < total:
        bounds.append(total)
    return bounds


# The mHC router normalizes the flattened stream pool before it scores the streams.
ROUTE_NORM_EPS = 1.0e-5


class ShensiUnweightedRMSNorm(nn.Module):
    """RMSNorm without a learnable scale, used inside mHC stream mixing."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last dimension in fp32 and return in the input dtype."""
        return F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).to(x.dtype)


class ShensiHyperConnection(ShensiFp32KeepMixin, MegatronModule):
    """Reads the HC streams before a sublayer and writes the sublayer output back into them."""

    SHENSI_FP32_KEEP_GROUP = GROUP_MHC

    def __init__(self, config, layer_number: int = 1, is_mlp: bool = False) -> None:
        super().__init__(config)
        self.hc_mult = int(config.num_residual_streams)
        self.active_streams = int(config.hc_active_streams)
        self.fixed_streams = int(config.hc_fixed_streams)
        self.routed_streams = self.active_streams - self.fixed_streams
        hidden = int(config.hidden_size)
        if not 0 < self.fixed_streams <= self.active_streams <= self.hc_mult:
            raise ValueError(
                "requires 0 < hc_fixed_streams <= hc_active_streams <= hc_mult, got "
                f"fixed={self.fixed_streams} active={self.active_streams} mult={self.hc_mult}"
            )
        self.input_norm = ShensiUnweightedRMSNorm(eps=config.layernorm_epsilon)
        self.pre_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden))
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult))
        self.pre_scale = nn.Parameter(torch.empty(1))
        self.route_norm = nn.LayerNorm(self.hc_mult * hidden, eps=ROUTE_NORM_EPS)
        self.route_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden))
        self.route_base = nn.Parameter(torch.empty(self.hc_mult))
        self.route_scale = nn.Parameter(torch.empty(1))
        self.is_mlp = is_mlp
        self.hc_conv_kernels: Tuple[int, ...] = tuple(int(k) for k in config.hc_conv_kernels)
        self.kr = (len(self.hc_conv_kernels) + 1) if is_mlp else 1
        if is_mlp:
            if not self.hc_conv_kernels:
                raise ValueError("hc_conv_kernels must not be empty: the MLP-side mHC needs at least one kernel")
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(hidden, hidden, ks, padding=ks - 1, groups=hidden, bias=False)
                    for ks in self.hc_conv_kernels
                ]
            )
        self.post_fn = nn.Parameter(torch.empty(self.active_streams * self.kr, self.active_streams * hidden))
        self.post_base = nn.Parameter(torch.empty(self.active_streams * self.kr))
        self.post_scale = nn.Parameter(torch.empty(1))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre = torch.sigmoid(F.linear(flat, self.pre_fn.float()) * self.pre_scale.float() + self.pre_base.float())
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return collapsed

    def _select_active_streams(self, hidden_streams: torch.Tensor):
        """Pick the streams this sublayer may write: fixed ones first, then the router's top-k."""
        s_len, b, hc, _ = hidden_streams.shape
        flat = self.route_norm(hidden_streams.flatten(start_dim=2).to(self.route_norm.weight.dtype))
        route_scores = torch.sigmoid(
            F.linear(flat, self.route_fn.float()) * self.route_scale.float() + self.route_base.float()
        )
        fixed_mask = torch.arange(hc, device=route_scores.device) < self.fixed_streams
        route_scores = route_scores.masked_fill(fixed_mask.view(1, 1, -1), float("-inf"))
        fixed_idx = torch.arange(self.fixed_streams, device=hidden_streams.device).view(1, 1, -1).expand(s_len, b, -1)
        routed_idx = route_scores.topk(self.routed_streams, dim=-1).indices
        active_idx = torch.cat([fixed_idx, routed_idx], dim=-1)
        p = torch.cat(
            [
                torch.ones_like(fixed_idx, dtype=route_scores.dtype),
                route_scores.gather(-1, routed_idx),
            ],
            dim=-1,
        )
        return active_idx, p

    def _augmented_features(
        self,
        sublayer_output: torch.Tensor,
        s_len: int,
        b: int,
        hidden: int,
        packed_seq_params=None,
    ):
        """Depth views of the sublayer output: the signal itself plus orthogonalized convolutions."""
        x = sublayer_output.permute(1, 2, 0).contiguous().to(self.temporal_convs[0].weight.dtype)
        bounds = _packed_bounds(packed_seq_params, x.size(-1))
        conv_outs = []
        for conv in self.temporal_convs:
            if bounds is None:
                conv_outs.append(conv(x)[..., :s_len])
            else:
                conv_outs.append(
                    torch.cat(
                        [conv(x[..., s:e])[..., : e - s] for s, e in zip(bounds[:-1], bounds[1:]) if e > s],
                        dim=-1,
                    )
                )
        ortho = []
        prevs = [x]
        for g in conv_outs:
            v = g
            for prev in prevs:
                denom = (prev * prev).sum(dim=1, keepdim=True).clamp_min(self.input_norm.eps)
                v = v - ((prev * v).sum(dim=1, keepdim=True) / denom) * prev
            ortho.append(v)
            prevs.append(v)
        out_aug = torch.cat([x] + ortho, dim=1).transpose(1, 2).reshape(b, s_len, self.kr, hidden).float()
        return out_aug.permute(1, 0, 2, 3)

    def write_back(
        self,
        hidden_streams: torch.Tensor,
        sublayer_output: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        s_len, b, _, hidden = hidden_streams.shape
        active_idx, p = self._select_active_streams(hidden_streams)
        if self.is_mlp:
            out_aug = self._augmented_features(sublayer_output, s_len, b, hidden, packed_seq_params)
        else:
            out_aug = sublayer_output.float().unsqueeze(-2)
        active_streams = hidden_streams.gather(2, active_idx.unsqueeze(-1).expand(-1, -1, -1, hidden))
        post = 2 * torch.sigmoid(
            F.linear(
                self.input_norm(active_streams.flatten(start_dim=2).float()),
                self.post_fn.float(),
            ).view(s_len, b, self.active_streams, self.kr)
            * self.post_scale.float()
            + self.post_base.float().view(self.active_streams, self.kr)
        )
        delta = torch.einsum("bskr,bsrh->bskh", post, out_aug) * p.unsqueeze(-1)
        return hidden_streams.scatter(
            2,
            active_idx.unsqueeze(-1).expand(-1, -1, -1, hidden),
            delta.to(hidden_streams.dtype),
        )

    def init_shensi_weights(self, std: float = 0.02, hidden_size=None) -> None:
        with torch.no_grad():
            nn.init.normal_(self.pre_fn, mean=0.0, std=std)
            nn.init.zeros_(self.pre_base)
            nn.init.constant_(self.pre_scale, 0.01)
            nn.init.normal_(self.route_fn, mean=0.0, std=std)
            nn.init.zeros_(self.route_base)
            nn.init.ones_(self.route_scale)
            nn.init.normal_(self.post_fn, mean=0.0, std=std)
            nn.init.zeros_(self.post_base)
            nn.init.constant_(self.post_scale, 0.01)
            if self.is_mlp:
                for conv in self.temporal_convs:
                    nn.init.kaiming_uniform_(conv.weight, a=5**0.5)


class ShensiHyperHead(ShensiFp32KeepMixin, MegatronModule):
    """Collapses the HC streams back into a single residual stream."""

    SHENSI_FP32_KEEP_GROUP = GROUP_MHC

    def __init__(self, config, layer_number: int = 1) -> None:
        super().__init__(config)
        self.hc_mult = int(config.num_residual_streams)
        hidden = int(config.hidden_size)
        self.input_norm = ShensiUnweightedRMSNorm(eps=config.layernorm_epsilon)
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size=None) -> None:
        with torch.no_grad():
            nn.init.normal_(self.hc_fn, mean=0.0, std=std)
            nn.init.zeros_(self.hc_base)
            nn.init.ones_(self.hc_scale)


# Attention residuals
# ---------------------


class ShensiAttentionResidual(ShensiFp32KeepMixin, MegatronModule):
    """Attention residual: mixes a block's stream context into the layer input."""

    SHENSI_FP32_KEEP_GROUP = GROUP_ATTN_RES

    def __init__(self, config, layer_number: int = 1) -> None:
        super().__init__(config)
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.eps = float(config.layernorm_epsilon)
        self.read_heads = int(getattr(config, "attn_res_read_heads", 8) or 8)
        rank = int(config.routed_expert_hidden_size)
        self.g_a_proj = nn.Linear(hidden, rank, bias=True)
        self.g_b_proj = nn.Linear(rank, 3 * hidden, bias=True)
        self.q_a_proj = nn.Linear(hidden, rank, bias=False)
        self.q_b_proj = nn.Linear(rank, hidden, bias=False)
        self.k_a_proj = nn.Linear(hidden, rank, bias=False)
        self.k_b_proj = nn.Linear(rank, hidden, bias=False)
        self.g_scale = nn.Parameter(torch.zeros(4))
        self.t = nn.Parameter(torch.linspace(0.0, 1.0, hidden) * math.log(2.0 * int(config.num_layers)))

    def forward(
        self,
        prefix: torch.Tensor,
        delta: Optional[torch.Tensor],
        blocks: torch.Tensor,
        output_norm_weight: Optional[torch.Tensor],
        num_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self._norm(prefix + (delta if delta is not None else 0.0))
        g0, g1 = self.g_a_proj, self.g_b_proj
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()
        gates = F.linear(F.linear(state, g0.weight.float(), g0.bias.float()), g1.weight.float(), g1.bias.float())
        r_decay, r_erase, r_write = gates.reshape(*state.shape[:-1], 3, -1).unbind(-2)
        decay = torch.exp(-F.softplus(r_decay) * decay_scale * self.t.exp())
        erase = F.softplus(r_erase) * erase_scale
        write = 1.0 + torch.tanh(r_write) * write_scale
        k0, k1 = self.k_a_proj, self.k_b_proj
        khat = F.normalize(
            F.linear(F.linear(delta if delta is not None else state, k0.weight.float()), k1.weight.float()),
            dim=-1,
        )
        m = decay * prefix + write * (delta if delta is not None else 0.0)
        lam = erase.mean(dim=-1, keepdim=True).clamp(min=-0.5)
        updated = m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)
        if num_blocks > 0:
            values = torch.cat([blocks[..., :num_blocks, :].float(), updated.unsqueeze(-2)], dim=-2)
            q0, q1 = self.q_a_proj, self.q_b_proj
            query = F.linear(F.linear(state, q0.weight.float()), q1.weight.float())
            dim = values.shape[-1]
            head_dim = dim // self.read_heads
            flat_v = values.reshape(-1, self.read_heads, head_dim)
            flat_q = query.reshape(-1, self.read_heads, head_dim)
            with torch.no_grad():
                V = flat_v.detach()
                n = V.shape[0]
                cov = torch.einsum("nhd,nhe->hde", V, V) / n
                scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
                ridge = max(head_dim, n) * torch.finfo(cov.dtype).eps * scale
                cov.diagonal(dim1=-2, dim2=-1).add_(ridge.unsqueeze(-1))
                evals, evecs = torch.linalg.eigh(cov)
                floor = (evals[..., -1:] * head_dim * torch.finfo(cov.dtype).eps).clamp_min(
                    torch.finfo(cov.dtype).tiny
                )
                whiten = evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor))) @ evecs.transpose(-1, -2)
                v = torch.einsum("nhd,hde->nhe", flat_v, whiten)
                q = torch.einsum("nhd,hde->nhe", flat_q, whiten)
            v = v.view(*values.shape[:-1], self.read_heads, -1)
            q = q.view(*query.shape[:-1], self.read_heads, -1)
            logits = (v * q.unsqueeze(-3)).sum(dim=-1) * torch.rsqrt(v.square().mean(dim=-1) + self.eps)
            s = torch.logsumexp(logits, dim=-2, keepdim=True)
            scores = torch.exp(logits - F.softplus(s))
            routed = (scores.unsqueeze(-1) * values.view_as(v)).sum(dim=-3).reshape(
                *values.shape[:-2], values.shape[-1]
            ) * read_scale
        else:
            routed = torch.zeros_like(updated)
        output = updated + routed
        if output_norm_weight is not None:
            output = F.rms_norm(output, (output.shape[-1],), weight=output_norm_weight.float(), eps=self.eps)
        return output.to(out_dtype), updated.to(out_dtype)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self.eps)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        d = int(hidden_size or self.hidden_size)
        with torch.no_grad():
            nn.init.zeros_(self.g_a_proj.weight)
            nn.init.zeros_(self.g_b_proj.weight)
            bias = self.g_b_proj.bias
            bias[:d] = 2.0
            bias[d : 2 * d] = -2.0
            bias[2 * d :] = -2.0
            nn.init.zeros_(self.q_a_proj.weight)
            nn.init.zeros_(self.q_b_proj.weight)
            nn.init.normal_(self.k_a_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.k_b_proj.weight, mean=0.0, std=std)


class ShensiAttnResState:
    """Per-block AttnRes bookkeeping shared between layers of the same block."""

    def __init__(
        self,
        num_blocks: int,
        *,
        hc_mult: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.hc_mult = None if hc_mult is None else int(hc_mult)
        self.hidden_size = None if hidden_size is None else int(hidden_size)
        self.prefix_sum: Optional[torch.Tensor] = None
        self.residual: Optional[torch.Tensor] = None
        self.num_written = 0
        self.pp_imported = False

    def begin(self, stream: torch.Tensor) -> None:
        if stream.dim() != 4:
            raise ValueError(f"ShensiAttnResState.begin expects a 4D n-stream input, got {stream.shape}")
        self.prefix_sum = stream
        s, b, hc, hidden = stream.shape
        self.residual = stream.new_zeros(s, b, hc, self.num_blocks, hidden)
        self.hc_mult, self.hidden_size = int(hc), int(hidden)
        self.num_written = 0
        self.pp_imported = False

    def write_block(self, slot: int, prefix_sum: torch.Tensor) -> None:
        if self.residual is None:
            raise RuntimeError("attention-residual state is unset: the first layer of a block did not call begin()")
        self.residual[..., slot, :] = prefix_sum.to(self.residual.dtype)
        self.num_written = max(self.num_written, int(slot) + 1)

    def check_ready(self, layer_number: int) -> None:
        if self.prefix_sum is None or self.residual is None:
            raise RuntimeError(
                f"attention-residual state is unset: layer {layer_number} is not the first layer "
                "of the stack, no layer called begin(), and nothing was imported from the "
                "previous pipeline stage."
            )

    def state_for_pp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Header and state tensors for cross-stage transfer.

        The header carries the shape, the number of written slots and the dtype code; the state
        travels in its own dtype, so a bf16 run transfers bf16 values.
        """
        if self.prefix_sum is None or self.residual is None:
            raise RuntimeError("attention-residual state is unset: cannot export a pipeline payload")
        ps, rs = self.prefix_sum, self.residual
        s, b, hc, hidden = ps.shape
        nb = int(rs.shape[-2])
        header = torch.tensor(
            [MAGIC, VERSION, s, b, hc, hidden, nb, self.num_written, _DTYPE_TO_CODE[ps.dtype]],
            dtype=torch.int64,
            device=ps.device,
        )
        state = torch.cat([ps.reshape(-1), rs.reshape(-1)], dim=0)
        _LIVE_PAYLOADS.add(state)
        return header, state

    def load_from_pp(
        self,
        header: torch.Tensor,
        state: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        expected_hidden_flat_numel: Optional[int] = None,
    ) -> Dict[str, object]:
        if header.dtype != torch.int64:
            raise ValueError(f"the state header is an int64 tensor, got {header.dtype}")
        if header.numel() < HEADER_LEN:
            raise ValueError(f"header too short ({header.numel()} < {HEADER_LEN})")
        fields = header.tolist()
        magic, version = int(fields[0]), int(fields[1])
        if magic != MAGIC:
            raise ValueError(f"bad payload magic ({magic} != {MAGIC}): not an attention-residual payload")
        if version != VERSION:
            raise ValueError(f"unsupported payload version ({version} != {VERSION})")
        s, b, hc, hidden, nb = (int(v) for v in fields[2:7])
        num_written = int(fields[7])
        code = int(fields[8])
        if code not in _CODE_TO_DTYPE:
            raise ValueError(f"unknown state dtype code {code} in the payload header")
        if state.dtype != _CODE_TO_DTYPE[code]:
            raise ValueError(f"payload dtype {state.dtype} does not match the header dtype {_CODE_TO_DTYPE[code]}")
        if not torch.is_tensor(state) or state.dim() != 1:
            raise ValueError(f"payload state must be a 1-D tensor, got {type(state).__name__}")
        expect = s * b * hc * hidden * (1 + nb)
        if int(state.numel()) != expect:
            raise ValueError(
                f"payload size mismatch: header declares {expect} state elements (s={s}, b={b}, "
                f"hc={hc}, H={hidden}, nb={nb}), got {int(state.numel())}; check "
                "num_residual_streams / hidden_size / the attention-residual block count on both sides"
            )
        if expected_hidden_flat_numel is not None and s * b * hc * hidden != int(expected_hidden_flat_numel):
            raise ValueError(
                f"payload holds {s * b * hc * hidden} n-stream elements but this layer received "
                f"{int(expected_hidden_flat_numel)} hidden elements"
            )
        if nb != self.num_blocks:
            raise ValueError(
                f"payload declares {nb} block slots but the local state has "
                f"{self.num_blocks}: with pipeline parallelism every stage must build its stack "
                "from the global block count (bind_attn_res_state(num_blocks=num_attn_res_blocks(...))), "
                "not from its own layer count"
            )
        if num_written < 0 or num_written > nb:
            raise ValueError(f"invalid written-slot count in payload: {num_written} (slots {nb})")
        n1 = s * b * hc * hidden
        prefix_sum = state[:n1].reshape(s, b, hc, hidden).clone()
        residual = state[n1:].reshape(s, b, hc, nb, hidden).clone()
        if dtype is not None:
            prefix_sum = prefix_sum.to(dtype)
            residual = residual.to(dtype)
        self.prefix_sum = prefix_sum
        self.residual = residual
        self.hc_mult, self.hidden_size = int(hc), int(hidden)
        self.num_written = num_written
        self.pp_imported = True
        return {
            "shape": (s, b, hc, hidden),
            "num_blocks": nb,
            "num_written": num_written,
            "numel": int(state.numel()),
            "dtype": str(self.prefix_sum.dtype),
        }


def bind_attn_res_state(block, config, num_blocks: int, *, handoff=None, plan=None) -> ShensiAttnResState:
    """Create and attach the AttnRes state for the layers of one transformer block."""
    check_attn_res_pp_support(config, plan=plan)
    state = ShensiAttnResState(
        num_blocks=num_blocks,
        hc_mult=int(getattr(config, "num_residual_streams", 1) or 1),
        hidden_size=int(getattr(config, "hidden_size", 0) or 0) or None,
    )
    layers = getattr(block, "layers", None)
    if layers is None:
        raise AttributeError(f"{type(block).__name__} has no layers, cannot bind attention-residual state")
    for layer in layers:
        layer.attn_res_state = state
        layer.attn_res_handoff = handoff
    block.shensi_attn_res_state = state
    block.shensi_attn_res_handoff = handoff
    return state


def bind_mtp_attn_res_state(mtp_block, config) -> dict[int, ShensiAttnResState]:
    """Attach one AttnRes state per MTP layer and return them by layer id."""
    layers = getattr(mtp_block, "layers", None) or []
    states: dict[int, ShensiAttnResState] = {}
    for idx, mtp_layer in enumerate(layers):
        inner = getattr(mtp_layer, "mtp_model_layer", None)
        if inner is None:
            raise AttributeError(f"{type(mtp_layer).__name__} has no mtp_model_layer to bind state to")
        state = ShensiAttnResState(
            num_blocks=1,
            hc_mult=int(getattr(config, "num_residual_streams", 1) or 1),
            hidden_size=int(getattr(config, "hidden_size", 0) or 0) or None,
        )
        inner.attn_res_state = state
        inner.attn_res_handoff = None
        states[idx] = state
    return states


def contract_output_streams(
    hidden_states: torch.Tensor,
    *,
    state: ShensiAttnResState,
    output_attn_res,
    hc_head,
    output_norm,
    hc_mult: int,
    hidden_size: int,
    num_blocks: int,
) -> torch.Tensor:
    """Contract the mHC streams of a block into the single hidden state the output head consumes."""
    if state is None or state.prefix_sum is None:
        raise RuntimeError("attention-residual state is empty: the block has not run forward yet")
    s, b = hidden_states.shape[0], hidden_states.shape[1]
    if hidden_states.dim() == 4:
        streams = hidden_states
    elif hidden_states.shape[-1] == hc_mult * hidden_size:
        streams = hidden_states.view(s, b, hc_mult, hidden_size)
    elif hidden_states.shape[-1] == hidden_size:
        streams = hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1)
    else:
        raise ValueError(
            f"cannot contract attention-residual streams from shape {tuple(hidden_states.shape)}: "
            f"expected a 4D n-stream tensor, a flat n-stream [s, b, {hc_mult * hidden_size}] or a "
            f"single [s, b, {hidden_size}] hidden state"
        )
    collapsed, _ = output_attn_res(
        state.prefix_sum,
        streams,
        state.residual,
        output_norm_weight=None,
        num_blocks=num_blocks,
    )
    return output_norm(hc_head(collapsed))


def bind_mtp_stream_contract(
    mtp_block,
    *,
    states: Dict[int, ShensiAttnResState],
    output_attn_res,
    hc_head,
    output_norm,
    hc_mult: int,
    hidden_size: int,
) -> int:
    """Give every MTP layer the stream contract that precedes its own final norm."""
    bound = 0
    for idx, mtp_layer in enumerate(getattr(mtp_block, "layers", None) or []):
        attach = getattr(mtp_layer, "attach_stream_contract", None)
        if attach is None:
            continue
        attach(
            functools.partial(
                contract_output_streams,
                state=states.get(idx),
                output_attn_res=output_attn_res,
                hc_head=hc_head,
                output_norm=output_norm,
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                num_blocks=1,
            )
        )
        bound += 1
    return bound


class ShensiMultiTokenPredictionLayer(MultiTokenPredictionLayer):
    """MTP layer that contracts the mHC streams ahead of its own final norm.

    The family contracts streams in the model post-process, so every MTP depth performs the same
    step before the shared head; mcore performs that contraction itself only for its
    ``enable_mhc_connections`` implementation, which the family does not use.
    """

    def attach_stream_contract(self, contract) -> None:
        """Attach the callable that turns this layer's multi-stream output into one hidden state."""
        self.stream_contract = contract

    def _postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        contract = getattr(self, "stream_contract", None)
        if contract is None:
            raise RuntimeError(
                "the MTP layer has no stream contract: build the model through ShensiModel so it "
                "can bind the attention-residual reader and the hyper head, or keep mtp_num_layers "
                "at 0"
            )
        return super()._postprocess(contract(hidden_states))


def attn_res_block_layer_types(num_layers: int, n_hash_layers: int, block_size: int) -> List[str]:
    """Layer types of every AttnRes block, in depth order."""
    return [
        (
            "block_write_layer"
            if i == 0 or (i >= n_hash_layers and (i - n_hash_layers) % block_size == 0)
            else "block_read_layer"
        )
        for i in range(num_layers)
    ]


def num_attn_res_blocks(num_layers: int, n_hash_layers: int, block_size: int) -> int:
    """Number of AttnRes blocks for a stack of ``num_layers`` layers."""
    return attn_res_block_layer_types(num_layers, n_hash_layers, block_size).count("block_write_layer")


def prev_valid_blocks(layer_idx: int, num_layers: int, n_hash_layers: int, block_size: int) -> int:
    """Blocks a layer may read from (all earlier blocks, skipping the hash prefix)."""
    types = attn_res_block_layer_types(num_layers, n_hash_layers, block_size)
    return sum(1 for t in types[:layer_idx] if t == "block_write_layer")


MAGIC = 21320
VERSION = 2
HEADER_LEN = 9
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float64: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


class AttnResPPLayoutError(ValueError):
    """Raised when the AttnRes block layout cannot be split across pipeline stages."""

    pass


@dataclass(frozen=True)
class AttnResBlock:
    """One AttnRes block: its depth range and the layers that belong to it."""

    index: int
    write_layer: int
    read_layers: Tuple[int, ...]

    @property
    def layers(self) -> Tuple[int, ...]:
        return (self.write_layer,) + self.read_layers


def blocks_of(num_layers: int, n_hash_layers: int, block_size: int) -> Tuple[AttnResBlock, ...]:
    """Slice a stack into AttnRes blocks (hash layers first, then fixed-size blocks)."""
    types = attn_res_block_layer_types(num_layers, n_hash_layers, block_size)
    writes = [i for i, t in enumerate(types) if t == "block_write_layer"]
    out: List[AttnResBlock] = []
    for bi, w in enumerate(writes):
        end = writes[bi + 1] if bi + 1 < len(writes) else num_layers
        out.append(AttnResBlock(index=bi, write_layer=w, read_layers=tuple(range(w + 1, end))))
    return tuple(out)


def _pipeline_layout_uses_virtual_stages(vpp_size: int) -> bool:
    if vpp_size <= 1:
        return True
    try:
        from megatron.core import parallel_state as ps

        return ps.get_virtual_pipeline_model_parallel_world_size() is not None
    except Exception:
        return False


def _stage_layers_from_layout(layout, pp_size: int, vpp_size: int):
    if _pipeline_layout_uses_virtual_stages(vpp_size):
        return {
            (pp, vp): tuple(layer_ids_from_layout(layout, vp_stage=vp, pp_rank=pp))
            for vp in range(vpp_size)
            for pp in range(pp_size)
        }
    from megatron.core.transformer.enums import LayerType

    out = {}
    offset = 0
    for vp in range(vpp_size):
        for pp in range(pp_size):
            n = layout.layout[pp][vp].count(LayerType.decoder)
            out[(pp, vp)] = tuple(range(offset, offset + n))
            offset += n
    return out


@dataclass
class LayoutReport:
    """Result of validating a pipeline layout against the AttnRes block boundaries."""

    ok: bool
    pp_size: int = 1
    vpp_size: int = 1
    num_layers: int = 0
    stage_layers: Dict[Tuple[int, int], Tuple[int, ...]] = field(default_factory=dict)
    boundaries: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    crossing_blocks: List[Tuple[AttnResBlock, Tuple[Tuple[int, int], ...]]] = field(default_factory=list)
    handoff_required: bool = False
    handoff_enabled: Optional[bool] = None
    problems: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - log formatting only
        lines = [
            f"[attn_res_pp] layout check: {'PASS' if self.ok else 'FAIL'} "
            f"(pp={self.pp_size}, vpp={self.vpp_size}, num_layers={self.num_layers})",
            "  stage -> global layer ids: "
            + ", ".join(
                f"pp{pp}.vp{vp}=[{ids[0]}..{ids[-1]}]" if ids else f"pp{pp}.vp{vp}=[]"
                for (pp, vp), ids in sorted(self.stage_layers.items())
            ),
            f"  boundaries (adjacent stage pairs) = {self.boundaries}; state handoff needed = "
            f"{self.handoff_required} (handoff implemented = {self.handoff_enabled})",
            "  blocks crossing a stage boundary = "
            + (
                ", ".join(
                    f"block{b.index}(write={b.write_layer}, readers={list(b.read_layers)}) spans "
                    + "/".join(f"pp{p}.vp{v}" for p, v in stages)
                    for b, stages in self.crossing_blocks
                )
                or "none (every block fits inside one stage)"
            ),
        ]
        for n in self.notes:
            lines.append(f"  [note] {n}")
        for i, p in enumerate(self.problems, 1):
            lines.append(f"  [problem {i}] {p}")
        for i, s in enumerate(self.suggestions, 1):
            lines.append(f"  [suggest {i}] {s}")
        return "\n".join(lines)


@dataclass
class AttnResStagePlan:
    """Assignment of AttnRes blocks to pipeline / virtual-pipeline stages."""

    num_layers: int
    n_hash_layers: int
    block_size: int
    pp_size: int
    vpp_size: int
    stage_layers: Dict[Tuple[int, int], Tuple[int, ...]]
    source: str = "layout"
    local_stage: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.stage_layers = {k: tuple(sorted(v)) for k, v in dict(self.stage_layers).items()}
        self._layer_to_stage = {layer: key for key, ids in self.stage_layers.items() for layer in ids}
        self.blocks = blocks_of(self.num_layers, self.n_hash_layers, self.block_size)

    @classmethod
    def from_stage_layers(
        cls,
        stage_layers: Dict[Tuple[int, int], Sequence[int]],
        *,
        num_layers: int,
        n_hash_layers: int,
        block_size: int,
        local_stage: Tuple[int, int] = (0, 0),
        source: str = "manual",
    ) -> "AttnResStagePlan":
        pp_size = 1 + max((k[0] for k in stage_layers), default=0)
        vpp_size = 1 + max((k[1] for k in stage_layers), default=0)
        return cls(
            num_layers=num_layers,
            n_hash_layers=n_hash_layers,
            block_size=block_size,
            pp_size=pp_size,
            vpp_size=vpp_size,
            stage_layers={k: tuple(sorted(v)) for k, v in stage_layers.items()},
            source=source,
            local_stage=local_stage,
        )

    @classmethod
    def from_config(
        cls, config, *, pp_rank: Optional[int] = None, vp_stage: Optional[int] = None
    ) -> "AttnResStagePlan":
        pp_size = int(getattr(config, "pipeline_model_parallel_size", 1) or 1)
        vpp_size = int(getattr(config, "virtual_pipeline_model_parallel_size", None) or 1)
        num_layers = getattr(config, "num_layers", None)
        if num_layers is None:
            raise AttnResPPLayoutError(
                "config has no num_layers, cannot check the attention-residual layout; pass a full "
                "Megatron-Core TransformerConfig or plan=AttnResStagePlan.from_stage_layers(...)"
            )
        num_layers = int(num_layers)
        n_hash = int(getattr(config, "moe_n_hash_layers", 0) or 0)
        block_size = int(getattr(config, "attn_res_block_size", 4) or 4)
        layout = getattr(config, "pipeline_model_parallel_layout", None)
        stage_layers: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        if pp_size <= 1 and vpp_size <= 1:
            stage_layers[(0, 0)] = tuple(range(num_layers))
            source = "single-stage"
        elif layout is not None:
            stage_layers = _stage_layers_from_layout(layout, pp_size, vpp_size)
            source = "layout"
        else:
            source = "offsets"
            for vp in range(vpp_size):
                for pp in range(pp_size):
                    stage_layers[(pp, vp)] = tuple(local_layer_ids(config, vp_stage=vp, pp_rank=pp))
        if pp_rank is None:
            try:
                from megatron.core import parallel_state as ps

                if ps.is_initialized():
                    pp_rank = int(ps.get_pipeline_model_parallel_rank())
                    vp_stage = int(ps.get_virtual_pipeline_model_parallel_rank() or 0)
                else:
                    pp_rank, vp_stage = 0, 0
            except Exception:  # noqa: BLE001
                pp_rank, vp_stage = 0, 0
        return cls(
            num_layers=num_layers,
            n_hash_layers=n_hash,
            block_size=block_size,
            pp_size=pp_size,
            vpp_size=vpp_size,
            stage_layers=stage_layers,
            source=source,
            local_stage=(int(pp_rank or 0), int(vp_stage or 0)),
        )

    @property
    def local_layers(self) -> Tuple[int, ...]:
        return self.stage_layers.get(self.local_stage, ())

    def stage_of(self, layer: int) -> Optional[Tuple[int, int]]:
        return self._layer_to_stage.get(int(layer))

    def crossing_blocks(self) -> List[Tuple[AttnResBlock, Tuple[Tuple[int, int], ...]]]:
        out = []
        for block in self.blocks:
            stages = []
            for layer in block.layers:
                st = self.stage_of(layer)
                if st is not None and st not in stages:
                    stages.append(st)
            if len(stages) > 1:
                out.append((block, tuple(stages)))
        return out

    def block_of(self, layer: int) -> int:
        layer = int(layer)
        for block in self.blocks:
            end = block.read_layers[-1] + 1 if block.read_layers else block.write_layer + 1
            if block.write_layer <= layer < end:
                return block.index
        return -1

    def block_crosses(self, layer: int) -> bool:
        bi = self.block_of(layer)
        return any(b.index == bi for b, _ in self.crossing_blocks())

    def payload_numel(
        self,
        *,
        seq_len: int,
        micro_batch: int,
        hc_mult: int,
        hidden_size: int,
        num_blocks: Optional[int] = None,
    ) -> int:
        nb = len(self.blocks) if num_blocks is None else int(num_blocks)
        return HEADER_LEN + int(seq_len) * int(micro_batch) * int(hc_mult) * int(hidden_size) * (1 + nb)

    def boundaries(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        out: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for layer in range(1, self.num_layers):
            prev, cur = self.stage_of(layer - 1), self.stage_of(layer)
            if prev is not None and cur is not None and prev != cur and (prev, cur) not in out:
                out.append((prev, cur))
        return out

    def needs_import(self, layer: int) -> bool:
        layer = int(layer)
        if layer <= 0 or layer >= self.num_layers:
            return False
        prev, cur = self.stage_of(layer - 1), self.stage_of(layer)
        return prev is not None and cur is not None and prev != cur

    def needs_export(self, layer: int) -> bool:
        layer = int(layer)
        if layer < 0 or layer >= self.num_layers - 1:
            return False
        nxt, cur = self.stage_of(layer + 1), self.stage_of(layer)
        return nxt is not None and cur is not None and nxt != cur

    def local_import_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.local_layers if self.needs_import(i))

    def local_export_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.local_layers if self.needs_export(i))

    def report(self, *, handoff_enabled: Optional[bool] = None) -> LayoutReport:
        problems: List[str] = []
        suggestions: List[str] = []
        notes: List[str] = []
        crossing = self.crossing_blocks()
        boundaries = self.boundaries()
        all_ids = [i for ids in self.stage_layers.values() for i in ids]
        if sorted(all_ids) != list(range(self.num_layers)):
            problems.append(
                f"the layout must cover decoder layer ids 0..{self.num_layers - 1} exactly, "
                f"got {len(all_ids)} entries ({len(set(all_ids))} unique, max {max(all_ids, default=-1)})"
            )
            suggestions.append("check that pipeline_model_parallel_layout lists num_layers decoder entries")
        embedding_stages = [k for k, ids in self.stage_layers.items() if ids and min(ids) == 0]
        hash_layers = set(range(self.n_hash_layers))
        if embedding_stages:
            emb_stage = embedding_stages[0]
            emb_ids = set(self.stage_layers[emb_stage])
            missing = sorted(hash_layers - emb_ids)
            if missing:
                problems.append(
                    f"hash layers {missing} are not on the embedding stage pp{emb_stage[0]}.vp{emb_stage[1]}"
                    " (hash layers must share a stage with the embedding)"
                )
                suggestions.append(
                    f"move the {len(missing)} hash layers into the first stage (see recommended_layout_str)"
                )
        if crossing:
            if handoff_enabled is False:
                problems.append(
                    "these blocks cross a pipeline boundary while attn_res_pp_state_transfer is off: "
                    " "
                    + "; ".join(
                        f"block{b.index}(write={b.write_layer}, readers={list(b.read_layers)}) -> "
                        + "/".join(f"pp{p}.vp{v}" for p, v in st)
                        for b, st in crossing
                    )
                )
                suggestions.append(
                    "either (a) enable attn_res_pp_state_transfer (default) so block state is handed "
                    "over, or (b) use a block-aligned layout:"
                    + recommended_layout_str(
                        self.num_layers,
                        self.n_hash_layers,
                        self.block_size,
                        self.pp_size,
                        self.vpp_size,
                    )
                )
            else:
                notes.append(
                    f"{len(crossing)} blocks cross a stage boundary and are handled by the handoff:"
                    + "; ".join(f"block{b.index}" for b, _ in crossing)
                )
            for b, st in crossing:
                vps = {v for _, v in st}
                if len(vps) > 1:
                    problems.append(
                        f"block{b.index} crosses virtual stages (vps {sorted(vps)}): with "
                        "interleaved scheduling the handoff only covers adjacent pipeline stages"
                    )
                    suggestions.append(
                        "align block boundaries with virtual stage boundaries (see "
                        "recommended_layout_str) or verify with vpp=1 first"
                    )
                elif self.vpp_size > 1:
                    notes.append(f"block{b.index} crosses a pp stage but stays inside one vp stage: pp handoff")
        handoff_required = self.pp_size > 1 or self.vpp_size > 1
        if handoff_required and handoff_enabled is False:
            problems.append(
                f"pipeline parallelism (pp={self.pp_size}, vpp={self.vpp_size}) requires block-state handoff: "
                "the final contraction on ShensiModel reads the complete residual stack, including "
                "the slots written on earlier stages, but the state is stage-private memory"
            )
            suggestions.append("enable config.attn_res_pp_state_transfer (default True)")
        ok = not problems
        return LayoutReport(
            ok=ok,
            pp_size=self.pp_size,
            vpp_size=self.vpp_size,
            num_layers=self.num_layers,
            stage_layers={k: v for k, v in sorted(self.stage_layers.items())},
            boundaries=boundaries,
            crossing_blocks=crossing,
            handoff_required=handoff_required,
            handoff_enabled=handoff_enabled,
            problems=problems,
            suggestions=suggestions,
            notes=notes,
        )


def _block_spans(num_layers: int, n_hash_layers: int, block_size: int) -> List[Tuple[int, int]]:
    blocks = blocks_of(num_layers, n_hash_layers, block_size)
    spans = []
    for bi, b in enumerate(blocks):
        end = blocks[bi + 1].write_layer if bi + 1 < len(blocks) else num_layers
        spans.append((b.write_layer, end))
    return spans


def recommended_layout_str(
    num_layers: int,
    n_hash_layers: int,
    block_size: int,
    pp_size: int,
    vpp_size: int = 1,
) -> str:
    """Human readable summary of a recommended pipeline layout."""
    n_chunks = int(pp_size) * int(vpp_size)
    if n_chunks < 1:
        raise ValueError(f"invalid pp_size/vpp_size: {pp_size}/{vpp_size}")
    spans = _block_spans(int(num_layers), int(n_hash_layers), int(block_size))
    if n_chunks > len(spans):
        raise AttnResPPLayoutError(
            f"PP*VPP={n_chunks} exceeds the {len(spans)} attention-residual blocks of this layer "
            f"plan (num_layers={num_layers}, n_hash_layers={n_hash_layers}, "
            f"block_size={block_size}), and every pipeline chunk holds at least one block. "
            "Reduce pp/vpp, grow num_layers or shrink block_size."
        )
    target = int(num_layers) / n_chunks
    counts: List[int] = []
    bi = 0
    for stage_i in range(n_chunks):
        remaining_stages = n_chunks - stage_i
        if remaining_stages == 1:
            take = len(spans) - bi
        else:
            take = 0
            acc = 0
            while bi + take < len(spans) - (remaining_stages - 1):
                size = spans[bi + take][1] - spans[bi + take][0]
                if take > 0 and acc + size > target:
                    break
                acc += size
                take += 1
            take = max(take, 1)
        layers = 0
        for k in range(take):
            w, e = spans[bi + k]
            layers += e - w
        counts.append(layers)
        bi += take
    if sum(counts) != num_layers:
        raise AttnResPPLayoutError(
            f"internal error: stage layer counts sum to {sum(counts)} != num_layers {num_layers}"
        )
    chunks = ["t" * c for c in counts]
    chunks[0] = "E" + chunks[0]
    chunks[-1] = chunks[-1] + "L"
    return "|".join(chunks)


RECOMPUTE_METHOD_BLOCK = "block"
RECOMPUTE_METHOD_UNIFORM = "uniform"


@dataclass(frozen=True)
class AttnResRecomputePlan:
    """Which AttnRes states must be recomputed when activation recompute is on."""

    granularity: Optional[str]
    method: Optional[str]
    recompute_num_layers: Optional[int]
    num_layers_per_stage: int
    pp_size: int
    vpp_size: int
    chunks: Tuple[Tuple[int, int], ...]
    safe: bool
    reason: str

    def describe(self) -> str:
        chunks = ", ".join(f"[{a},{b})" for a, b in self.chunks)
        return (
            f"granularity={self.granularity!r} method={self.method!r} "
            f"recompute_num_layers={self.recompute_num_layers} "
            f"layers per stage={self.num_layers_per_stage} pp={self.pp_size} vpp={self.vpp_size} "
            f"chunks=({chunks}) safe={self.safe} ({self.reason})"
        )


def _norm_granularity(v) -> Optional[str]:
    if v is None:
        return None
    return str(v)


def build_attn_res_recompute_plan(config) -> AttnResRecomputePlan:
    """Derive the recompute plan for the current parallelism configuration."""
    gran = _norm_granularity(getattr(config, "recompute_granularity", None))
    method = _norm_granularity(getattr(config, "recompute_method", None)) or RECOMPUTE_METHOD_UNIFORM
    k = getattr(config, "recompute_num_layers", None)
    k = None if k is None else int(k)
    pp = int(getattr(config, "pipeline_model_parallel_size", 1) or 1)
    vpp = int(getattr(config, "virtual_pipeline_model_parallel_size", None) or 1)
    n_stage = int(getattr(config, "num_layers", 0) or 0)
    if pp > 1:
        n_stage = int(getattr(config, "num_layers", 0) or 0)
    chunks: list = []
    if gran == "full" and n_stage > 0 and k:
        if method == RECOMPUTE_METHOD_UNIFORM:
            i = 0
            while i < n_stage:
                chunks.append((i, min(i + k, n_stage)))
                i += k
        elif method == RECOMPUTE_METHOD_BLOCK:
            chunks = [(i, i + 1) for i in range(min(k, n_stage))]
        else:
            chunks = []
    safe, reason = True, "full recompute is off (no layer forward is replayed)"
    if gran == "full":
        if pp > 1 or vpp > 1:
            safe, reason = (
                False,
                "pp/vpp>1: replaying the first layer would import again (recv is not idempotent) and the cross-stage state is not among the saved tensors",
            )
        elif not k:
            safe, reason = (
                False,
                "full recompute without recompute_num_layers (chunking is undefined)",
            )
        elif method == RECOMPUTE_METHOD_UNIFORM and k >= n_stage:
            safe, reason = (
                True,
                f"a single uniform chunk covers the stage ({k}>={n_stage}): replay rebuilds state from the first layer",
            )
        elif method == RECOMPUTE_METHOD_BLOCK and k == 1:
            safe, reason = (
                True,
                "block checkpoints only the first layer (k=1): replay rebuilds state from it",
            )
        elif method == RECOMPUTE_METHOD_UNIFORM:
            safe, reason = (
                False,
                f"the uniform chunk does not start at the first layer (k={k}<{n_stage}): replay reads the end-of-block state",
            )
        elif method == RECOMPUTE_METHOD_BLOCK:
            safe, reason = (
                False,
                f"block k={k}>1: layers 1..{k - 1} are replayed before the first layer, so state is not rebuilt",
            )
        else:
            safe, reason = (
                False,
                f"unknown recompute_method={method!r} (uniform/block are the supported ones)",
            )
    return AttnResRecomputePlan(
        granularity=gran,
        method=method if gran == "full" else None,
        recompute_num_layers=k,
        num_layers_per_stage=n_stage,
        pp_size=pp,
        vpp_size=vpp,
        chunks=tuple(chunks),
        safe=safe,
        reason=reason,
    )


def check_attn_res_recompute_support(config) -> AttnResRecomputePlan:
    """Validate that recompute can be used with this AttnRes layout."""
    plan = build_attn_res_recompute_plan(config)
    if plan.safe:
        return plan
    raise NotImplementedError(
        "[shensi][attn_res] full activation recompute does not rebuild the attention-residual "
        f"block state: {plan.describe()}\n"
        "A replayed forward refreshes the state only when the replay starts at the layer that "
        "closed the block, and the first layer of a stage must not import its block state twice. "
        "Recompute shapes that keep both properties:\n"
        "  (1) pp=1 with one chunk over the whole stage: uniform recompute with "
        "recompute_num_layers >= num_layers, or block recompute with recompute_num_layers=1;\n"
        "  (2) full recompute off and selective recompute on;\n"
        "  (3) pp>1: keep full recompute off, because the cross-stage state is not part of the "
        "saved tensors a replay restores."
    )


def check_attn_res_pp_support(
    config,
    *,
    plan: Optional[AttnResStagePlan] = None,
    handoff_enabled: Optional[bool] = None,
    raise_on_error: bool = True,
) -> LayoutReport:
    """Validate pipeline splits against the AttnRes block boundaries."""
    if raise_on_error:
        check_attn_res_recompute_support(config)
    if handoff_enabled is None:
        handoff_enabled = bool(getattr(config, "attn_res_pp_state_transfer", True))
    if plan is None:
        try:
            plan = AttnResStagePlan.from_config(config)
        except AttnResPPLayoutError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AttnResPPLayoutError(
                "cannot derive the attention-residual pipeline layout: "
                f"{type(e).__name__}: {e}. Pipeline parallelism needs a full Megatron-Core "
                "TransformerConfig (num_layers / pipeline_model_parallel_size / "
                "pipeline_model_parallel_layout) or plan=AttnResStagePlan.from_stage_layers(...)"
            ) from e
    report = plan.report(handoff_enabled=handoff_enabled)
    if report.ok or not raise_on_error:
        return report
    header = f"attention-residual layout check failed (pp={report.pp_size}, vpp={report.vpp_size}):\n" + str(report)
    transfer_disabled = any("attn_res_pp_state_transfer" in p for p in report.problems)
    if transfer_disabled:
        raise NotImplementedError(
            header + "\n\nfix: set config.attn_res_pp_state_transfer to True so block state is handed "
            "over between stages, or use one of the block-aligned layouts suggested above."
        )
    raise AttnResPPLayoutError(header)


_GROUP_CACHE: Dict[tuple, object] = {}
_flush_registry: List["ShensiAttnResPPHandoff"] = []
_PENDING_SENDS: List[tuple] = []
_PENDING_BYTES: List[int] = [0]
_RELEASED_BYTES: List[int] = [0]
_LIVE_PAYLOADS: "weakref.WeakSet" = weakref.WeakSet()


def _payload_nbytes(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else int(t.numel()) * int(t.element_size())


def _background_send(group, tensor: torch.Tensor) -> bool:
    """Whether an asynchronous send of `tensor` on `group` progresses on its own.

    Gloo moves an asynchronous send only while the process group that owns it runs, so the
    payload leaves the exporting stage at its next process-group call, usually the backward
    pass that the receiving stage's forward pass is itself waiting on. NCCL keeps device
    tensors travelling on the group's own stream, so there the sender continues freely.
    """
    if not tensor.is_cuda:
        return False
    try:
        return "nccl" in torch.distributed.get_backend(group)
    except Exception:  # noqa: BLE001 - an unresolvable group falls back to a blocking send
        return False


def pending_transfer_bytes() -> int:
    """Bytes queued for cross-stage AttnRes state transfer."""
    return int(_PENDING_BYTES[0])


def released_transfer_bytes() -> int:
    """Bytes of cross-stage AttnRes state already released."""
    return int(_RELEASED_BYTES[0])


def live_payload_stats() -> Dict[str, int]:
    """Number and size of live AttnRes state payloads, by device."""
    live = [t for t in _LIVE_PAYLOADS]
    return {
        "n_live": len(live),
        "bytes_live": sum(_payload_nbytes(t) for t in live),
        "n_registered": len(_LIVE_PAYLOADS),
    }


def _track_payloads() -> bool:
    return str(os.environ.get("SHENSI_ATTN_RES_PP_TRACK_PAYLOAD", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def payload_device(hint: Optional[torch.Tensor] = None) -> torch.device:
    """Device the AttnRes state payloads live on."""
    if hint is not None:
        return hint.device
    try:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - cpu path when the CUDA context is unavailable
        pass
    return torch.device("cpu")


def _release_send(entry) -> None:
    work, buf, nbytes = entry
    work.wait()
    _PENDING_BYTES[0] -= int(nbytes)
    _RELEASED_BYTES[0] += int(nbytes)
    del buf


def _flush_all() -> None:
    while _PENDING_SENDS:
        _release_send(_PENDING_SENDS.pop(0))


class _ShensiAttnResPPBridge(torch.autograd.Function):
    """Send the block state on `tag` (header) and `tag + 1` (state); receive its gradient on `tag + 2`."""

    @staticmethod
    def forward(
        ctx,
        stage_output: torch.Tensor,
        header: torch.Tensor,
        state: torch.Tensor,
        dst,
        group,
        tag,
        use_isend: bool,
    ):
        ctx.dst, ctx.group, ctx.tag = dst, group, tag
        ctx.state_numel = int(state.numel())
        ctx.state_dtype = state.dtype
        ctx.state_device = state.device
        ctx.work = None
        ctx.nbytes = 0
        bufs = (header.detach().contiguous(), state.detach().contiguous())
        ctx.bufs = bufs
        if dst is not None:
            nbytes = sum(_payload_nbytes(buf) for buf in bufs)
            if use_isend and _background_send(group, bufs[1]):
                ctx.work = (
                    torch.distributed.isend(bufs[0], dst=dst, group=group, tag=tag),
                    torch.distributed.isend(bufs[1], dst=dst, group=group, tag=tag + 1),
                )
                ctx.nbytes = nbytes
                _PENDING_BYTES[0] += nbytes
            else:
                torch.distributed.send(bufs[0], dst=dst, group=group, tag=tag)
                torch.distributed.send(bufs[1], dst=dst, group=group, tag=tag + 1)
                _RELEASED_BYTES[0] += nbytes
        return stage_output

    @staticmethod
    def backward(ctx, grad_stage_output: torch.Tensor):
        if ctx.dst is None:
            return grad_stage_output, None, None, None, None, None, None
        if ctx.work is not None:
            for work in ctx.work:
                work.wait()
            _PENDING_BYTES[0] -= int(ctx.nbytes)
            _RELEASED_BYTES[0] += int(ctx.nbytes)
            ctx.work, ctx.bufs, ctx.nbytes = None, (), 0
        grad = torch.empty(ctx.state_numel, dtype=ctx.state_dtype, device=ctx.state_device)
        torch.distributed.recv(grad, src=ctx.dst, group=ctx.group, tag=ctx.tag + 2)
        return grad_stage_output, None, grad, None, None, None, None


def _recv_payload(
    numel: int,
    src: Optional[int],
    group,
    tag: int,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Receive the header and the state of one block, and wire the gradient channel back."""
    dev = payload_device() if device is None else device
    header = torch.empty(HEADER_LEN, dtype=torch.int64, device=dev)
    torch.distributed.recv(header, src=src, group=group, tag=tag)
    code = int(header.tolist()[8])
    if code not in _CODE_TO_DTYPE:
        raise ValueError(f"unknown state dtype code {code} in the payload header")
    state = torch.empty(int(numel), dtype=_CODE_TO_DTYPE[code], device=dev)
    torch.distributed.recv(state, src=src, group=group, tag=tag + 1)
    state.requires_grad_(True)

    def _send_back(grad: torch.Tensor):
        g = grad.detach().contiguous()
        nbytes = _payload_nbytes(g)
        if _background_send(group, g):
            work = torch.distributed.isend(g, dst=src, group=group, tag=tag + 2)
            _PENDING_SENDS.append((work, g, nbytes))
            _PENDING_BYTES[0] += nbytes
        else:
            torch.distributed.send(g, dst=src, group=group, tag=tag + 2)
            _RELEASED_BYTES[0] += nbytes
        return None

    state.register_hook(_send_back)
    return header, state


class ShensiAttnResPPHandoff:
    """Cross-stage handoff of AttnRes state for PP > 1."""

    def __init__(
        self,
        plan: AttnResStagePlan,
        *,
        group=None,
        local_stage: Optional[Tuple[int, int]] = None,
        use_isend: Optional[bool] = None,
        tag: int = 7900,
        logger=None,
        auto_group: bool = True,
    ) -> None:
        self.plan = plan
        self.group = group
        self.local_stage = tuple(local_stage or plan.local_stage)
        if use_isend is None:
            use_isend = not (
                str(os.environ.get("SHENSI_ATTN_RES_PP_NO_ISEND", "")).strip().lower() in ("1", "true", "yes", "on")
            )
        self.use_isend = bool(use_isend)
        self.tag = int(tag)
        self.logger = logger
        self.auto_group = bool(auto_group)
        self.pp_group_ranks: Dict[int, int] = {}
        self.records: List[dict] = []
        self.n_import = 0
        self.n_export = 0
        self.payload_numels: List[int] = []
        self._exported_payloads: List[torch.Tensor] = []
        self.n_graph_refs_released = 0
        _flush_registry.append(self)
        self.bind_group()

    def resolve_pp_group_ranks(self) -> Optional[Tuple[int, ...]]:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return None
        try:
            from megatron.core import parallel_state as ps

            pp_group = ps.get_pipeline_model_parallel_group()
            ranks = tuple(sorted(torch.distributed.get_process_group_ranks(pp_group)))
            if not ranks:
                raise RuntimeError("the pipeline group returned by Megatron-Core is empty")
            return ranks
        except Exception as exc:  # noqa: BLE001 - always re-raised, see the docstring
            raise RuntimeError(
                "[attn_res_pp] could not obtain the pipeline model parallel group ("
                f"{type(exc).__name__}: {exc}). The attention-residual handoff must run on the "
                "real pipeline group: only local ranks inside it are pipeline neighbours, while "
                "world rank+-1 is usually a tensor/data-parallel peer, so using it would send the "
                "state to the wrong process. Construct the handoff after "
                "ps.initialize_model_parallel(...) (bind_group caches the group), or pass "
                "group=... / auto_group=False in offline tests."
            ) from exc

    def bind_group(self) -> None:
        if self.group is not None or not self.auto_group:
            return
        if self.plan.pp_size <= 1 and self.plan.vpp_size <= 1:
            return
        ranks = self.resolve_pp_group_ranks()
        if ranks is None:
            return
        if ranks not in _GROUP_CACHE:
            _GROUP_CACHE[ranks] = torch.distributed.new_group(ranks=list(ranks))
        self.group = _GROUP_CACHE[ranks]
        self._record("bind_group", -1, group_ranks=list(ranks), pp_size=self.plan.pp_size)

    def resolve_group(self):
        if self.group is None and self.auto_group and (self.plan.pp_size > 1 or self.plan.vpp_size > 1):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                raise RuntimeError(
                    "[attn_res_pp] the handoff group was not built during binding (self.group=None). "
                    "Do not create it inside a layer forward: new_group is collective and would "
                    "deadlock against the pipeline receives; construct the handoff after "
                    "parallel_state.initialize_model_parallel, call handoff.bind_group() "
                    "(safe to call again), or pass group=..."
                )
        return self.group

    def _neighbor_global_rank(self, delta: int) -> Optional[int]:
        tgt = self.local_stage[0] + int(delta)
        if tgt < 0 or tgt >= self.plan.pp_size:
            return None
        group = self.resolve_group()
        if group is None:
            return None
        local_idx = self.pp_group_ranks.get(tgt, tgt)
        return int(torch.distributed.get_global_rank(group, local_idx))

    def _dst_rank(self) -> Optional[int]:
        return self._neighbor_global_rank(+1)

    def _src_rank(self) -> Optional[int]:
        return self._neighbor_global_rank(-1)

    def _record(self, kind: str, layer: int, payload: Optional[torch.Tensor] = None, **extra) -> None:
        rec = {
            "kind": kind,
            "stage": f"pp{self.local_stage[0]}.vp{self.local_stage[1]}",
            "global_layer": int(layer),
            "payload_numel": int(payload.numel()) if payload is not None else 0,
        }
        if payload is not None and _track_payloads():
            # The digest reads the payload back to the host, which stalls the pipeline, so it
            # belongs to the payload tracking mode rather than to every transfer.
            rec["payload_sha256"] = hashlib.sha256(
                payload.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()[:16]
        rec.update(extra)
        self.records.append(rec)
        if self.logger is not None:
            self.logger(
                f"[attn_res_pp][{kind}] stage={rec['stage']} layer={rec['global_layer']} "
                f"numel={rec['payload_numel']}"
                + (f" sha256={rec['payload_sha256']}" if "payload_sha256" in rec else "")
                + (" " + " ".join(f"{k}={v}" for k, v in extra.items()) if extra else "")
            )

    def export_layer(
        self,
        state,
        layer_idx: int,
        *,
        stage_output: Optional[torch.Tensor] = None,
        send: bool = True,
    ) -> Optional[torch.Tensor]:
        self.flush()
        if not self.plan.needs_export(layer_idx):
            return stage_output
        header, payload = state.state_for_pp()
        dst = self._dst_rank()
        out = payload if stage_output is None else stage_output
        if send and dst is not None:
            group = self.resolve_group()
            if stage_output is None:
                raise ValueError(
                    "export_layer needs stage_output to wire the state channel into the backward "
                    "pass; exporting only the payload gives autograd no gradient for it."
                )
            out = _ShensiAttnResPPBridge.apply(stage_output, header, payload, dst, group, self.tag, self.use_isend)
        self.n_export += 1
        self.payload_numels.append(int(payload.numel()))
        if _track_payloads():
            self._exported_payloads.append(payload)
        self._record(
            "export",
            layer_idx,
            payload,
            dst=dst,
            num_blocks=int(state.num_blocks),
            num_written=int(getattr(state, "num_written", -1)),
            block_index=self.plan.block_of(layer_idx),
            crossing=bool(self.plan.block_crosses(layer_idx)),
            payload_device=str(payload.device),
        )
        return out

    def import_layer(
        self,
        state,
        layer_idx: int,
        *,
        hidden_flat: Optional[torch.Tensor] = None,
        numel: Optional[int] = None,
        recv: bool = True,
    ) -> Optional[dict]:
        self.flush()
        if not self.plan.needs_import(layer_idx):
            return None
        if numel is None:
            if hidden_flat is None:
                raise ValueError("import_layer needs hidden_flat (to infer the payload size) or an explicit numel")
            numel = int(hidden_flat.numel()) * (1 + int(state.num_blocks))
        src = self._src_rank()
        header, payload = None, None
        if recv:
            group = self.resolve_group()
            header, payload = _recv_payload(int(numel), src, group, self.tag, device=payload_device(hidden_flat))
        if payload is None:
            raise RuntimeError(
                f"[attn_res_pp] layer {layer_idx} of stage pp{self.local_stage[0]}.vp{self.local_stage[1]} "
                f"needs an imported attention-residual block state but received no payload (src={src})"
            )
        info = state.load_from_pp(
            header,
            payload,
            dtype=None if hidden_flat is None else hidden_flat.dtype,
            expected_hidden_flat_numel=None if hidden_flat is None else int(hidden_flat.numel()),
        )
        self.n_import += 1
        self.payload_numels.append(int(payload.numel()))
        self._record(
            "import",
            layer_idx,
            payload,
            src=src,
            num_blocks=int(info["num_blocks"]),
            num_written=int(info["num_written"]),
            block_index=self.plan.block_of(layer_idx),
            crossing=bool(self.plan.block_crosses(layer_idx)),
            payload_device=str(payload.device),
        )
        return info

    def flush(self) -> None:
        _flush_all()

    def clear_graph_refs(self) -> None:
        n = len(self._exported_payloads)
        self._exported_payloads.clear()
        self.n_graph_refs_released += n
        return None

    def live_graph_refs(self) -> int:
        return len(self._exported_payloads)

    def pending_transfer_bytes(self) -> int:
        return pending_transfer_bytes()

    def payload_diagnostics(self) -> Dict[str, int]:
        stats = live_payload_stats()
        return {
            "container_refs": len(self._exported_payloads),
            "n_live_payloads": int(stats["n_live"]),
            "bytes_live_payloads": int(stats["bytes_live"]),
            "pending_bytes": pending_transfer_bytes(),
            "released_bytes": released_transfer_bytes(),
            "n_graph_refs_released": int(self.n_graph_refs_released),
        }

    def summary(self) -> dict:
        return {
            "stage": f"pp{self.local_stage[0]}.vp{self.local_stage[1]}",
            "pp_size": self.plan.pp_size,
            "vpp_size": self.plan.vpp_size,
            "local_layers": list(self.plan.local_layers),
            "import_layers": list(self.plan.local_import_layers()),
            "export_layers": list(self.plan.local_export_layers()),
            "n_import": self.n_import,
            "n_export": self.n_export,
            "payload_numels": self.payload_numels,
            "crossing_blocks": [
                {
                    "index": b.index,
                    "write_layer": b.write_layer,
                    "read_layers": list(b.read_layers),
                }
                for b, _ in self.plan.crossing_blocks()
            ],
            "records": self.records,
            "payload_diagnostics": self.payload_diagnostics(),
        }


# Mixture-of-experts layers
# ---------------------------


@dataclass
class ShensiMoESubmodules(MoESubmodules):
    """Submodules of the shensi MoE layer (low-rank routed experts + norm)."""

    routed_expert_norm: Optional[LayerNormBuilder] = None


class ShensiMoELayer(MoELayer):
    """Shensi MoE layer: low-rank routed experts, hash-MoE layers and the ERC path."""

    def __init__(
        self,
        config,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection=None,
        is_mtp_layer: bool = False,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
            name=name,
        )
        latent = config.moe_latent_size
        if not latent:
            raise ValueError(
                "ShensiMoELayer needs config.moe_latent_size = routed_expert_hidden_size "
                "(the routed-expert bottleneck); got None."
            )
        norm_spec = getattr(submodules, "routed_expert_norm", None)
        if norm_spec is None:
            raise ValueError(
                "ShensiMoESubmodules.routed_expert_norm is unset: routed experts produce their "
                "output in the rank space and must be normalized there before the up projection."
            )
        self.routed_expert_norm = build_module(
            norm_spec,
            config=config,
            hidden_size=int(latent),
            eps=config.layernorm_epsilon,
        )

    def combine(self, output: torch.Tensor) -> torch.Tensor:
        return self.routed_expert_norm(super().combine(output))

    def erc_weights(self):
        router_weight = self.router.weight
        down_proj_weight = self.fc1_latent_proj.weight
        gate_up = torch.stack(
            [getattr(self.experts.linear_fc1, f"weight{i}") for i in range(self.num_local_experts)],
            dim=0,
        )
        return router_weight, down_proj_weight, gate_up


def erc_gather_expert_weights(
    mlp,
    ep_group=None,
    *,
    grad_reduce: Optional[str] = None,
    verify_router_replica: Optional[bool] = None,
    verbose: bool = False,
):
    """Gather the expert weights used by the ERC (router / expert coupling) loss."""
    ep_size, ep_rank, ep_group = _resolve_parallel_group("expert_model_parallel", ep_group)
    etp_size, etp_rank, etp_group = _resolve_parallel_group("expert_tensor_parallel", None)
    if grad_reduce is None:
        grad_reduce = os.environ.get("SHENSI_ERC_EP_GRAD_REDUCE", "slice").strip().lower()
    if verify_router_replica is None:
        verify_router_replica = _env_flag("SHENSI_ERC_VERIFY_ROUTER_REPLICA", False)
    if _erc_local_expert_view():
        router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
        n_local = int(getattr(mlp, "num_local_experts", 0) or 0)
        if ep_size > 1 and n_local > 0:
            router_weight = router_weight[ep_rank * n_local : (ep_rank + 1) * n_local]
        if verbose:
            print_rank0(
                "[shensi][erc] erc_gather_expert_weights: SHENSI_ERC_EP_ALLOW_LOCAL=1 -> scoring "
                f"the {n_local} routed experts this rank holds"
            )
        return router_weight, down_proj_weight, gate_up_local
    router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
    if ep_size <= 1 and etp_size <= 1:
        return router_weight, down_proj_weight, gate_up_local
    if not hasattr(mlp, "fc1_latent_proj"):
        raise NotImplementedError(
            "erc_gather_expert_weights: the ERC term needs the down/up projections of the routed "
            "experts (exposed as fc1_latent_proj on the moe_latent_size path), but this MoE "
            "layer has no such module; configure moe_latent_size = routed_expert_hidden_size."
        )
    if verify_router_replica and ep_size > 1:
        _assert_replica(router_weight, ep_group, "router.weight")
        _assert_replica(down_proj_weight, ep_group, "fc1_latent_proj.weight")
    if ep_size > 1 and ep_group is None:
        raise RuntimeError(
            f"[shensi][erc] expert_model_parallel_size={ep_size} > 1 but no expert-parallel group "
            "is available (the parallel state is not initialized): the ERC term couples the router "
            "with the experts of the whole group, and a per-rank view scores a different "
            "objective. Build the model after initialize_model_parallel, or set "
            "SHENSI_ERC_EP_ALLOW_LOCAL=1 to score the local experts on purpose."
        )
    if etp_size > 1 and etp_group is None:
        raise RuntimeError(
            f"[shensi][erc] expert_tensor_parallel_size={etp_size} > 1 but no expert-tensor-parallel "
            "group is available: the ERC norms span the full intermediate dimension."
        )
    gate_up = gate_up_local
    if ep_size > 1:
        gate_up = _ShensiEPAllGather.apply(gate_up, ep_group, 0, grad_reduce)
    if etp_size > 1:
        gate_up = _ShensiEPAllGather.apply(gate_up, etp_group, 1, grad_reduce)
    if verbose:
        print_rank0(
            f"[shensi][erc] global view: ep={ep_size} (rank {ep_rank}, "
            f"{getattr(mlp, 'num_local_experts', '?')} local experts), etp={etp_size} "
            f"(rank {etp_rank}), gate_up_proj={tuple(gate_up.shape)} grad_reduce={grad_reduce}"
        )
    return router_weight, down_proj_weight, gate_up


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _erc_local_expert_view() -> bool:
    """Whether the ERC term scores the experts of this rank instead of the gathered group."""
    return _env_flag("SHENSI_ERC_EP_ALLOW_LOCAL", False)


def print_rank0(msg: str) -> None:
    """Print from rank 0 only."""
    try:
        from megatron.training import print_rank_0

        print_rank_0(msg)
    except Exception:  # noqa: BLE001
        rank = 0
        try:
            import torch.distributed as _dist

            if _dist.is_available() and _dist.is_initialized():
                rank = _dist.get_rank()
        except Exception:  # noqa: BLE001
            rank = 0
        if rank == 0:
            print(msg, flush=True)


def _resolve_parallel_group(kind: str, group):
    size, rank = 1, 0
    try:
        from megatron.core import parallel_state as _ps

        if not _ps.is_initialized():
            return size, rank, group
        if kind == "expert_model_parallel":
            size = int(_ps.get_expert_model_parallel_world_size())
            rank = int(_ps.get_expert_model_parallel_rank())
            if group is None and size > 1:
                group = _ps.get_expert_model_parallel_group()
        elif kind == "expert_tensor_parallel":
            size = int(_ps.get_expert_tensor_parallel_world_size())
            rank = int(_ps.get_expert_tensor_parallel_rank())
            if group is None and size > 1:
                group = _ps.get_expert_tensor_parallel_group()
    except Exception:  # noqa: BLE001
        return 1, 0, group
    return size, rank, group


def _group_size(group) -> int:
    import torch.distributed as dist

    return int(group.size()) if hasattr(group, "size") else int(dist.get_world_size(group))


def _all_gather_along_dim(t: torch.Tensor, group, dim: int) -> torch.Tensor:
    world = _group_size(group)
    if world <= 1:
        return t
    if dim == 0 and t.is_cuda:
        try:
            from megatron.core.tensor_parallel.mappings import _gather_along_first_dim

            return _gather_along_first_dim(t, group)
        except Exception:  # noqa: BLE001
            pass
    if dim != 0:
        order = (dim,) + tuple(i for i in range(t.dim()) if i != dim)
        inv = tuple(sorted(range(t.dim()), key=lambda i: order[i]))
        t_perm = t.permute(order).contiguous()
    else:
        order, inv, t_perm = None, None, t.contiguous()
    out_shape = list(t_perm.shape)
    out_shape[0] = out_shape[0] * world
    out = torch.empty(out_shape, dtype=t_perm.dtype, device=t_perm.device)
    torch.distributed.all_gather_into_tensor(out, t_perm, group=group)
    return out.permute(inv) if order is not None else out


def _slice_along_dim(t: torch.Tensor, group, dim: int, rank: Optional[int] = None) -> torch.Tensor:
    import torch.distributed as dist

    world = _group_size(group)
    if world <= 1:
        return t
    rank = dist.get_rank(group) if rank is None else rank
    n = t.shape[dim] // world
    return t.narrow(dim, rank * n, n).contiguous()


class _ShensiEPAllGather(torch.autograd.Function):
    """All-gather a tensor along `dim` from a group, and map the gradient back in the same layout.

    The forward concatenates the ranks' shards in rank order. The backward either hands this rank
    the shard it contributed (`grad_reduce="slice"`) or the mean over the group
    (`grad_reduce="reduce_scatter_mean"`).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, dim: int, grad_reduce: str):
        if grad_reduce not in ("slice", "reduce_scatter_mean"):
            raise NotImplementedError(f"unknown grad_reduce={grad_reduce!r} (slice / reduce_scatter_mean)")
        ctx.group, ctx.dim, ctx.grad_reduce = group, dim, grad_reduce
        return _all_gather_along_dim(tensor, group, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.distributed as dist

        if ctx.grad_reduce == "slice":
            return _slice_along_dim(grad_output, ctx.group, ctx.dim), None, None, None
        world = _group_size(ctx.group)
        dim = ctx.dim
        order = None
        g = grad_output
        if dim != 0:
            order = (dim,) + tuple(i for i in range(g.dim()) if i != dim)
            inv = tuple(sorted(range(g.dim()), key=lambda i: order[i]))
            g = g.permute(order).contiguous()
        out_shape = list(g.shape)
        out_shape[0] = out_shape[0] // world
        out = torch.empty(out_shape, dtype=g.dtype, device=g.device)
        try:
            dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group)
        except (RuntimeError, ValueError) as exc:  # noqa: BLE001
            print_rank0(
                f"[shensi][erc] [warn] this backend has no reduce_scatter_tensor ({exc!r}); "
                "falling back to all_reduce + narrow (identical numerics, more traffic)"
            )
            full = g.contiguous().clone()
            dist.all_reduce(full, group=ctx.group)
            out = full.narrow(0, 0, out_shape[0]).clone()
        out = out / world
        return (out.permute(inv) if order is not None else out), None, None, None


def _assert_replica(t: torch.Tensor, group, name: str) -> None:
    """Check that a tensor carries the same values on every rank of the group."""
    world = _group_size(group)
    if world <= 1:
        return
    flat = t.reshape(-1)
    gathered = _all_gather_along_dim(flat, group, 0)
    chunk = int(flat.numel())
    ref = gathered[:chunk]
    for rank in range(1, world):
        other = gathered[rank * chunk : (rank + 1) * chunk]
        if not torch.equal(other, ref):
            raise RuntimeError(
                f"[shensi][erc] {name} differs between rank0 and rank{rank} of the expert-parallel "
                f"group (max|delta|={float((other.float() - ref.float()).abs().max()):.3e}). The ERC "
                "global view requires these tensors to be replicas (the router and latent "
                "projections are built as 'duplicated'); a mismatch usually means the per-rank "
                "initialization RNG diverged. Load the weights from a checkpoint, which makes them "
                "replicas, or set SHENSI_ERC_EP_ALLOW_LOCAL=1 to score the local experts."
            )


class ShensiHashMLP(MLP):
    """Hash-MoE MLP: a deterministic token-to-expert table instead of a learned router."""

    def __init__(
        self,
        config,
        submodules,
        ffn_hidden_size: Optional[int] = None,
        name: str | None = None,
        **kwargs,
    ):
        vocab_size = getattr(config, "actual_vocab_size", None) or config.vocab_size
        hidden_intermediate = int(ffn_hidden_size or config.routed_expert_hidden_size)
        inner_config = copy.copy(config)
        inner_config.ffn_hidden_size = hidden_intermediate
        super().__init__(
            config=inner_config,
            submodules=submodules,
            ffn_hidden_size=hidden_intermediate,
            name=name,
        )
        self.deepemb = nn.Embedding(int(vocab_size), int(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        if input_ids is None:
            raise ValueError(
                "ShensiHashMLP needs input_ids for its embedding gate; make sure "
                "--moe-n-hash-layers (the number of hash layers) is set so input_ids reaches "
                "the decoder and its layers."
            )
        output_with_bias = super().forward(hidden_states)
        output, bias = output_with_bias[0], output_with_bias[1]
        ids = input_ids.transpose(0, 1) if input_ids.dim() == 2 else input_ids
        return output * self.deepemb(ids), bias

    def init_shensi_weights(self, std: float = 0.02, hidden_size=None) -> None:
        with torch.no_grad():
            nn.init.normal_(self.deepemb.weight, mean=0.0, std=std)


def _layer_is_hash(layer_idx: int, n_hash_layers: int) -> bool:
    return layer_idx < n_hash_layers


def tie_moe_groups(
    block, n_hash_layers: int, block_size: int, num_layers: Optional[int] = None
) -> Dict[int, ShensiMoELayer]:
    """Tie the per-layer MoE groups (router, experts, latent projection) of a model."""

    layers = block.layers
    global_idx = [int(getattr(layer, "layer_number", i + 1)) - 1 for i, layer in enumerate(layers)]
    n_total = int(num_layers) if num_layers else (max(global_idx) + 1 if global_idx else 0)
    types = attn_res_block_layer_types(n_total, n_hash_layers, block_size)
    n_blocks_total = num_attn_res_blocks(n_total, n_hash_layers, block_size)
    write_layers = [i for i, t in enumerate(types) if t == "block_write_layer"]
    owners: Dict[int, ShensiMoELayer] = {}
    groups: Dict[int, List[int]] = {}
    skipped_hash: List[int] = []
    cross_stage: List[int] = []
    for idx, layer in enumerate(layers):
        g = global_idx[idx]
        if _layer_is_hash(g, n_hash_layers):
            skipped_hash.append(g)
            continue
        block_id = max((k for k, w in enumerate(write_layers) if w <= g), default=None)
        if block_id is None:
            raise RuntimeError(
                f"[shensi][moe] global layer {g} precedes every attention-residual block "
                f"(writers={write_layers}); check attn_res_block_size / moe_n_hash_layers / layer_number."
            )
        mlp = layer.mlp
        if block_id not in owners:
            owners[block_id] = mlp
            groups[block_id] = [g]
            if write_layers[block_id] not in global_idx:
                cross_stage.append(block_id)
        else:
            owner = owners[block_id]
            groups[block_id].append(g)
            if mlp is not owner:
                mlp.router = owner.router
                mlp.experts = owner.experts
    for block_id, owner in owners.items():
        owner.sharing_layers = list(groups[block_id])
        owner.shensi_tie_cross_stage = block_id in cross_stage
    if cross_stage:
        print_rank0(
            f"[shensi][moe][warn] {len(cross_stage)} attention-residual blocks are split by a "
            f"pipeline boundary (block_id={cross_stage}): layers on different stages cannot "
            "share gate or expert parameters, because they have no common parent module and "
            "pipeline sends tensors only. Recorded on owner.shensi_tie_cross_stage."
        )
    if skipped_hash:
        print_rank0(
            f"[shensi][moe] tie_moe_groups: this stage skips the hash layers (global ids "
            f"{skipped_hash}), {len(owners)} shared groups, member global layer ids="
            f"{ {k: v for k, v in groups.items()} }"
            f" (total layers={n_total}, total blocks={n_blocks_total})"
        )
    return owners


# Decoder layer
# ---------------


@dataclass
class ShensiTransformerLayerSubmodules(TransformerLayerSubmodules):
    """Submodules of a shensi transformer layer."""

    self_attention_attn_res: Union[ModuleSpec, type] = IdentityOp
    mlp_attn_res: Union[ModuleSpec, type] = IdentityOp


def attn_res_to_streams(hidden_states: Tensor, *, hc_mult: int, hidden_size: int, is_first_in_block: bool) -> Tensor:
    """Expand a flat activation into the HC streams expected by the mHC modules."""
    s, b = hidden_states.shape[0], hidden_states.shape[1]
    if hidden_states.dim() == 4:
        return hidden_states
    if is_first_in_block:
        if hidden_states.dim() == 3 and hidden_states.shape[-1] == hidden_size:
            return hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
        if hidden_states.dim() == 3 and hidden_states.shape[-1] == hc_mult * hidden_size:
            return hidden_states.view(s, b, hc_mult, hidden_size)
        raise ValueError(
            f"the first layer of a block takes 1 stream [s, b, {hidden_size}], a flat "
            f"n-stream [s, b, {hc_mult * hidden_size}] or an expanded 4D n-stream tensor, "
            f"got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dim() == 4:
        return hidden_states
    if hidden_states.dim() == 3 and hidden_states.shape[-1] == hc_mult * hidden_size:
        return hidden_states.view(s, b, hc_mult, hidden_size)
    raise ValueError(
        f"layer input must be a flat n-stream [s, b, {hc_mult * hidden_size}], got {tuple(hidden_states.shape)}"
    )


def attn_res_to_flat(streams: Tensor) -> Tensor:
    """Collapse HC streams back into a flat activation."""
    return streams.reshape(streams.shape[0], streams.shape[1], -1)


def shensi_attn_res_layer_forward(
    *,
    state,
    hidden_states: Tensor,
    layer_number: int,
    hc_mult: int,
    hidden_size: int,
    is_first_in_block: bool,
    is_block_write_layer: bool,
    prev_valid_blocks: int,
    attn_res,
    mlp_attn_res,
    attn_hc,
    ffn_hc,
    input_norm_weight: Tensor,
    pre_mlp_norm_weight: Tensor,
    attn_fn,
    mlp_fn,
    handoff=None,
    packed_seq_params=None,
) -> Tensor:
    """Layer forward with AttnRes context, mHC streams and the MoE sublayer."""
    layer_idx = int(layer_number) - 1
    streams = attn_res_to_streams(
        hidden_states,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        is_first_in_block=is_first_in_block,
    )
    if handoff is not None:
        handoff.import_layer(state, layer_idx, hidden_flat=attn_res_to_flat(streams))
    if is_first_in_block:
        state.begin(streams)
        delta = None
    else:
        state.check_ready(layer_number)
        delta = streams - state.prefix_sum
    if is_block_write_layer:
        state.write_block(prev_valid_blocks, state.prefix_sum)
    attn_in, prefix_sum = attn_res(
        state.prefix_sum,
        delta,
        state.residual,
        output_norm_weight=input_norm_weight,
        num_blocks=prev_valid_blocks,
    )
    if is_block_write_layer:
        prefix_sum = None
    collapsed = attn_hc(attn_in)
    attn_output = attn_fn(collapsed)
    streams = attn_hc.write_back(attn_in, attn_output)
    prefix_sum = streams if prefix_sum is None else prefix_sum + streams
    mlp_in, prefix_sum = mlp_attn_res(
        prefix_sum,
        prefix_sum,
        state.residual,
        output_norm_weight=pre_mlp_norm_weight,
        num_blocks=prev_valid_blocks + int(is_block_write_layer),
    )
    collapsed = ffn_hc(mlp_in)
    mlp_output = mlp_fn(collapsed)
    streams = ffn_hc.write_back(mlp_in, mlp_output, packed_seq_params)
    state.prefix_sum = prefix_sum + streams
    flat = attn_res_to_flat(streams)
    if handoff is not None:
        flat = handoff.export_layer(state, layer_idx, stage_output=flat)
    return flat


# Hash layers gate their MLP with a token-embedding lookup. GPTModel passes input_ids to the
# embedding and the MTP layers but not to every decoder layer, so ShensiModel publishes the
# current micro-batch ids through a ContextVar that hash layers read in place.
HASH_INPUT_IDS: ContextVar[Optional[Tensor]] = ContextVar("shensi_hash_input_ids", default=None)


def publish_hash_input_ids(input_ids: Optional[Tensor]):
    """Publish the input_ids of the current forward pass; the returned token resets it."""
    return HASH_INPUT_IDS.set(input_ids)


def reset_hash_input_ids(token) -> None:
    """Undo the publication done by `publish_hash_input_ids`."""
    HASH_INPUT_IDS.reset(token)


class ShensiTransformerLayer(TransformerLayer):
    """Shensi decoder layer: mHC around attention and MoE, AttnRes in between."""

    def __init__(
        self,
        config,
        submodules: ShensiTransformerLayerSubmodules,
        layer_number: int = 1,
        is_mtp_layer: bool = False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            is_mtp_layer=is_mtp_layer,
            **kwargs,
        )
        self.hc_mult = int(config.num_residual_streams)
        self.n_hash_layers = int(config.moe_n_hash_layers)
        self.attn_res_block_size = int(config.attn_res_block_size)
        layer_idx = self.layer_number - 1
        if not 0 <= layer_idx < config.num_layers:
            raise ValueError(
                f"layer_number={self.layer_number} is outside the decoder (num_layers={config.num_layers})"
            )
        types = attn_res_block_layer_types(config.num_layers, self.n_hash_layers, self.attn_res_block_size)
        self.is_mtp_layer = bool(is_mtp_layer)
        if self.is_mtp_layer:
            self.is_block_write_layer = True
            self.prev_valid_blocks = 0
            self.is_first_in_block = True
            self.is_hash = False
        else:
            self.is_block_write_layer = types[layer_idx] == "block_write_layer"
            self.prev_valid_blocks = prev_valid_blocks(
                layer_idx,
                config.num_layers,
                self.n_hash_layers,
                self.attn_res_block_size,
            )
            self.is_first_in_block = self.layer_number == 1
            self.is_hash = layer_idx < self.n_hash_layers
        for name, norm in (
            ("input_layernorm", self.input_layernorm),
            ("pre_mlp_layernorm", self.pre_mlp_layernorm),
        ):
            if not hasattr(norm, "weight"):
                raise RuntimeError(
                    f"ShensiTransformerLayer needs {name} to expose .weight for the "
                    "attention-residual output norm; got "
                    f"{type(norm).__name__}. Do not fuse this layernorm into an IdentityOp."
                )
        self.attn_hc = build_module(
            submodules.self_attention_hyper_connection,
            config=config,
            layer_number=self.layer_number,
        )
        self.ffn_hc = build_module(
            submodules.mlp_hyper_connection,
            config=config,
            layer_number=self.layer_number,
        )
        if not isinstance(self.attn_hc, torch.nn.Module) or self.attn_hc.__class__ is ShensiAttentionResidual:
            raise TypeError("self_attention_hyper_connection must be a ShensiHyperConnection")
        self.self_attention_attn_res = build_module(submodules.self_attention_attn_res, config=config)
        self.mlp_attn_res = build_module(submodules.mlp_attn_res, config=config)
        self.attn_res_state = None
        self._attn_res_recompute_plan = None
        self.attn_res_handoff = None

    def _to_streams(self, hidden_states: Tensor) -> Tensor:
        return attn_res_to_streams(
            hidden_states,
            hc_mult=self.hc_mult,
            hidden_size=int(self.config.hidden_size),
            is_first_in_block=self.is_first_in_block,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager: Optional[Any] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        if getattr(self, "_attn_res_recompute_plan", None) is None:
            self._attn_res_recompute_plan = build_attn_res_recompute_plan(self.config)
        if not self._attn_res_recompute_plan.safe:
            raise NotImplementedError(
                "[shensi][attn_res] this activation-recompute shape is incompatible with the "
                f"attention-residual state machine: {self._attn_res_recompute_plan.describe()}. "
                "See check_attn_res_recompute_support for the supported shapes."
            )
        if mhc_recompute_manager is not None:
            raise NotImplementedError(
                "ShensiTransformerLayer does not support mhc recompute: the attention-residual "
                "state lives on the block; do not add 'mhc' to recompute_modules."
            )
        state = self.attn_res_state
        if state is None:
            raise RuntimeError(
                "attention-residual state is not bound: call "
                "attn_res.bind_attn_res_state(model.decoder, config, num_blocks)."
            )
        if input_ids is None and self.is_hash:
            input_ids = HASH_INPUT_IDS.get()

        def _attn_fn(collapsed):
            attention_output_with_bias = self.self_attention(
                collapsed,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            attn_output, attn_bias = (
                attention_output_with_bias
                if isinstance(attention_output_with_bias, tuple)
                else (attention_output_with_bias, None)
            )
            if attn_bias is not None:
                raise NotImplementedError("Shensi attention carries no bias (add_bias_linear=False)")
            return attn_output

        def _mlp_fn(collapsed):
            mlp_output_with_bias = self.mlp(collapsed, input_ids=input_ids) if self.is_hash else self.mlp(collapsed)
            mlp_output, mlp_bias = (
                mlp_output_with_bias if isinstance(mlp_output_with_bias, tuple) else (mlp_output_with_bias, None)
            )
            if mlp_bias is not None:
                raise NotImplementedError("Shensi MLP/MoE carries no bias (add_bias_linear=False)")
            return mlp_output

        flat = shensi_attn_res_layer_forward(
            state=state,
            hidden_states=hidden_states,
            layer_number=self.layer_number,
            hc_mult=self.hc_mult,
            hidden_size=int(self.config.hidden_size),
            is_first_in_block=self.is_first_in_block,
            is_block_write_layer=self.is_block_write_layer,
            prev_valid_blocks=self.prev_valid_blocks,
            attn_res=self.self_attention_attn_res,
            mlp_attn_res=self.mlp_attn_res,
            attn_hc=self.attn_hc,
            ffn_hc=self.ffn_hc,
            input_norm_weight=self.input_layernorm.weight,
            pre_mlp_norm_weight=self.pre_mlp_layernorm.weight,
            attn_fn=_attn_fn,
            mlp_fn=_mlp_fn,
            handoff=self.attn_res_handoff,
            packed_seq_params=packed_seq_params,
        )
        return flat, context


# Layer specs
# -------------

SHENSI_SPEC_OVERRIDES = {
    "module": (
        ShensiTransformerLayer,
        "ShensiTransformerLayer: attention-residual stack plus in-layer mHC stream mixing;"
        "the layer keeps the (output, context) return signature.",
    ),
    "self_attention_hyper_connection": (
        ShensiHyperConnection,
        "ShensiHyperConnection: fixed streams plus a top-k routed stream set, written back"
        "in place over the stream pool; the MLP side adds causal depth convolutions.",
    ),
    "mlp_hyper_connection": (
        ShensiHyperConnection,
        "The same module with is_mlp=True: causal depthwise convolutions plus their",
    ),
    "mlp": (
        ShensiMoELayer,
        "ShensiMoELayer: routed experts run through the low-rank bottleneck carried by"
        "moe_latent_size, with the rank-space routed_expert_norm applied before combine().",
    ),
    "mlp(hash)": (
        ShensiHashMLP,
        "ShensiHashMLP: a dense SwiGLU MLP gated by a token-embedding lookup rather than a"
        "router; hash layers carry no experts, so moe_n_hash_layers only routes input_ids.",
    ),
    "self_attention_attn_res": (
        ShensiAttentionResidual,
        "extra slot: the attention-side attention-residual module.",
    ),
    "mlp_attn_res": (ShensiAttentionResidual, "extra slot: the MLP-side attention-residual module"),
    "layer_norm(block)": (
        None,
        "Kept as None instead of IdentityOp so has_final_layernorm_in_this_stage() is false"
        "and no output contraction runs inside the block: the final norm lives on ShensiModel"
        "and runs after the stream contraction, where tensors are still n-stream.",
    ),
}


def _hash_mlp_builder(submodules: MLPSubmodules, ffn_hidden_size: int) -> Callable:
    def build(config, pg_collection=None, is_mtp_layer=False, name=None, **_ignored):
        return ShensiHashMLP(config, submodules=submodules, ffn_hidden_size=ffn_hidden_size, name=name)

    build.__name__ = "shensi_hash_mlp_builder"
    return build


def _moe_mlp_builder(experts_spec, norm_spec, layer_number: int) -> Callable:
    submodules = ShensiMoESubmodules(experts=experts_spec, routed_expert_norm=norm_spec)
    return functools.partial(ShensiMoELayer, submodules=submodules, layer_number=layer_number)


def get_shensi_layer_spec(
    use_te: bool,
    config,
    layer_number: int = 1,
    is_hash_layer: bool = False,
    build_engram: bool = False,
) -> ModuleSpec:
    """Module spec of one shensi decoder layer (MLA + mHC + MoE)."""
    if build_engram:
        raise NotImplementedError("Shensi has no engram component")
    backend = get_backend_from_config(config=config)
    attention_spec = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=backend)
    rms_norm = config.normalization == "RMSNorm"
    mlp_submodules = MLPSubmodules(
        linear_fc1=backend.column_parallel_linear(),
        linear_fc2=backend.row_parallel_linear(),
        activation_func=backend.activation_func(),
    )
    if is_hash_layer:
        mlp = _hash_mlp_builder(mlp_submodules, hmac_intermediate(config))
    else:
        mlp = _moe_mlp_builder(
            experts_spec=backend.grouped_mlp_modules(bool(config.moe_grouped_gemm)),
            norm_spec=backend.layer_norm(rms_norm=True, for_qk=False),
            layer_number=layer_number,
        )
    submodules = ShensiTransformerLayerSubmodules(
        input_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
        self_attention=attention_spec,
        self_attention_hyper_connection=ModuleSpec(module=ShensiHyperConnection, params={"is_mlp": False}),
        pre_mlp_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
        mlp=mlp,
        mlp_hyper_connection=ModuleSpec(module=ShensiHyperConnection, params={"is_mlp": True}),
        self_attention_attn_res=ModuleSpec(module=ShensiAttentionResidual),
        mlp_attn_res=ModuleSpec(module=ShensiAttentionResidual),
    )
    return ModuleSpec(module=ShensiTransformerLayer, submodules=submodules)


def get_shensi_mtp_layer_spec(config, use_transformer_engine: bool = True) -> ModuleSpec:
    """Module spec of the shensi MTP layer."""
    n_hash = int(config.moe_n_hash_layers)
    num_layers = int(config.num_layers)
    last_layer_is_hash = (num_layers - 1) < n_hash
    return get_shensi_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        layer_number=num_layers + 1,
        is_hash_layer=last_layer_is_hash,
    )


def get_shensi_mtp_block_spec(
    config,
    *,
    use_transformer_engine: bool = True,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """MTP block spec built from :func:`get_shensi_mtp_layer_spec`.

    ``get_gpt_mtp_block_spec`` accepts either a ``TransformerBlockSubmodules`` or a ``ModuleSpec``
    whose ``module`` is exactly ``TransformerLayer``; the single-layer spec is therefore wrapped
    into a block spec here.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    layer_spec = get_shensi_mtp_layer_spec(config=config, use_transformer_engine=use_transformer_engine)
    block_spec = get_gpt_mtp_block_spec(
        config,
        TransformerBlockSubmodules(layer_specs=[layer_spec], layer_norm=None),
        use_transformer_engine=use_transformer_engine,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )
    for spec in getattr(block_spec, "layer_specs", None) or []:
        spec.module = ShensiMultiTokenPredictionLayer
    return block_spec


def hmac_intermediate(config) -> int:
    """Intermediate size of the hash-MoE expert MLP."""
    return int(config.routed_expert_hidden_size)


def layer_ids_from_layout(layout, *, vp_stage=None, pp_rank=None) -> list[int]:
    """Layer ids of this rank, taken from an explicit pipeline layout."""
    from megatron.core.transformer.enums import LayerType

    return list(layout.get_layer_id_list(layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank))


def local_layer_ids(config, *, vp_stage=None, pp_rank=None) -> list[int]:
    """Layer ids built by this pipeline / virtual-pipeline stage."""
    if getattr(config, "pipeline_model_parallel_layout", None) is not None:
        return layer_ids_from_layout(config.pipeline_model_parallel_layout, vp_stage=vp_stage, pp_rank=pp_rank)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return list(range(offset, offset + num_layers_to_build))


def get_shensi_decoder_block_spec(
    config,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
    use_moe: Optional[bool] = True,
):
    """Decoder block spec for the current stage (local layers only)."""
    n_hash = int(config.moe_n_hash_layers)
    layer_specs: List[ModuleSpec] = []
    for layer_idx in range(config.num_layers):
        layer_specs.append(
            get_shensi_layer_spec(
                use_te=use_transformer_engine,
                config=config,
                layer_number=layer_idx + 1,
                is_hash_layer=layer_idx < n_hash,
            )
        )
    local_layer_specs = [
        layer_specs[layer_id] for layer_id in local_layer_ids(config, vp_stage=vp_stage, pp_rank=pp_rank)
    ]
    return TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=None)


def _spec_name(obj) -> str:
    if isinstance(obj, ModuleSpec):
        inner = obj.module
        if isinstance(inner, type):
            return inner.__name__
        func = getattr(inner, "func", None)
        if func is not None:
            return f"partial({getattr(func, '__name__', func)})"
        return f"ModuleSpec({type(inner).__name__})"
    if isinstance(obj, type):
        return obj.__name__
    func = getattr(obj, "func", None)
    if func is not None:
        return f"partial({getattr(func, '__name__', func)})"
    name = getattr(obj, "__name__", None)
    if isinstance(name, str):
        return name
    return type(obj).__name__


def describe_block_spec(block_spec) -> list:
    """Human readable view of a block spec (used by tests / logs)."""
    rows = []
    for idx, layer_spec in enumerate(block_spec.layer_specs):
        sub = layer_spec.submodules
        rows.append(
            (
                idx,
                _spec_name(sub.self_attention),
                _spec_name(sub.mlp),
                _spec_name(layer_spec.module),
                _spec_name(getattr(sub, "self_attention_hyper_connection", None)),
                _spec_name(getattr(sub, "mlp_hyper_connection", None)),
                _spec_name(getattr(sub, "self_attention_attn_res", None)),
                _spec_name(getattr(sub, "mlp_attn_res", None)),
            )
        )
    return rows


def build_shensi_transformer_layer(config, layer_number: int = 1, is_hash_layer: bool = False):
    """Build a single shensi transformer layer (tests / smoke runs)."""
    spec = get_shensi_layer_spec(
        use_te=True,
        config=config,
        layer_number=layer_number,
        is_hash_layer=is_hash_layer,
    )
    return build_module(spec, config=config, layer_number=layer_number)


ShensiTransformerLayerSubmodules = ShensiTransformerLayerSubmodules

# Model
# -------


def _only_trailing_padding(attention_mask: torch.Tensor) -> bool:
    """True when every row is right-padded (trailing zeros); an all-ones mask also qualifies.

    With causal attention, right padding is only visible to later pad positions, so valid tokens
    are unaffected; left padding or holes put pads before valid tokens and are rejected.
    """
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if attention_mask.dim() != 2:
        return False
    valid = attention_mask.sum(-1, keepdim=True)
    positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
    expected = (positions.unsqueeze(0) < valid).to(attention_mask.dtype)
    return bool(torch.equal(attention_mask.to(expected.dtype), expected))


class ShensiModel(DeepSeekModel):
    """Shensi model: DeepSeek-V4 style stack with mHC head, AttnRes state and MTP."""

    pending_shensi_overrides = tuple(SHENSI_PENDING_FIELDS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.config, ShensiTransformerConfig):
            raise TypeError(
                f"ShensiModel needs a ShensiTransformerConfig, got {type(self.config).__name__}; "
                "build it through ShensiModelProvider / build_shensi_transformer_config."
            )
        cfg = self.config
        self.hc_mult = int(cfg.num_residual_streams)
        self.n_hash_layers = int(cfg.moe_n_hash_layers)
        self.num_blocks = num_attn_res_blocks(cfg.num_layers, self.n_hash_layers, int(cfg.attn_res_block_size))
        self.output_attn_res = ShensiAttentionResidual(cfg)
        self.hc_head = ShensiHyperHead(cfg)
        backend = get_backend_from_config(config=cfg)
        self.output_norm = build_module(
            backend.layer_norm(rms_norm=cfg.normalization == "RMSNorm", for_qk=False),
            config=cfg,
            hidden_size=cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )
        if not hasattr(self.decoder, "layers"):
            raise RuntimeError(f"ShensiModel needs a decoder with layers, got {type(self.decoder).__name__}")
        self.attn_res_stage_plan = None
        self.attn_res_handoff = None
        pp_size = int(cfg.pipeline_model_parallel_size or 1)
        vpp_size = int(getattr(cfg, "virtual_pipeline_model_parallel_size", None) or 1)
        if pp_size > 1 or vpp_size > 1:
            report = check_attn_res_pp_support(cfg)
            plan = AttnResStagePlan.from_config(cfg)
            pp_rank, vp_stage = 0, 0
            try:
                from megatron.core import parallel_state as ps

                if ps.is_initialized():
                    pp_rank = int(ps.get_pipeline_model_parallel_rank())
                    vp_stage = int(ps.get_virtual_pipeline_model_parallel_rank() or 0)
            except Exception:  # noqa: BLE001
                pass
            plan.local_stage = (pp_rank, vp_stage)
            self.attn_res_stage_plan = plan
            verbose = str(os.environ.get("SHENSI_ATTN_RES_PP_VERBOSE", "0")).strip().lower()
            verbose_all = str(os.environ.get("SHENSI_ATTN_RES_PP_VERBOSE_ALL", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            verbose_on = verbose in ("1", "true", "yes", "on") or verbose == "all"
            if verbose_on:
                logger = print if (verbose_all or verbose == "all") else print_rank_0
            else:
                logger = None
            self.attn_res_handoff = ShensiAttnResPPHandoff(plan, local_stage=(pp_rank, vp_stage), logger=logger)
            print_rank_0(
                f"[shensi][attn_res_pp] pp={pp_size} vpp={vpp_size} "
                f"local layers={list(plan.local_layers)} import={list(plan.local_import_layers())} "
                f"export={list(plan.local_export_layers())} blocks crossing a stage boundary="
                f"{[b.index for b, _ in plan.crossing_blocks()]} | "
                f"layout check={'PASS' if report.ok else 'FAIL'}"
            )
        self.attn_res_state = bind_attn_res_state(
            self.decoder,
            cfg,
            self.num_blocks,
            handoff=self.attn_res_handoff,
            plan=self.attn_res_stage_plan,
        )
        self.mtp_attn_res_states = {}
        mtp_block = getattr(self, "mtp", None)
        if mtp_block is not None:
            self.mtp_attn_res_states = bind_mtp_attn_res_state(mtp_block, cfg)
            bound = bind_mtp_stream_contract(
                mtp_block,
                states=self.mtp_attn_res_states,
                output_attn_res=self.output_attn_res,
                hc_head=self.hc_head,
                output_norm=self.output_norm,
                hc_mult=self.hc_mult,
                hidden_size=int(cfg.hidden_size),
            )
            print_rank_0(
                f"[shensi][mtp] attention-residual state bound for {len(self.mtp_attn_res_states)} MTP stages"
                f" (one slot, num_blocks=1) and the stream contract bound on {bound} of them | "
                "is_hash=False / block_write=True / prev_valid_blocks=0 (matches the MTP semantics "
                "of an inference-side decoder layer)"
            )
        self.moe_group_owners = tie_moe_groups(
            self.decoder,
            self.n_hash_layers,
            int(cfg.attn_res_block_size),
            num_layers=int(cfg.num_layers),
        )
        if not self.moe_group_owners:
            self.moe_group_owners = {}
        self._init_shensi_only_weights()

        groups = fp32_keep_groups(cfg)
        self.hc_fp32_keep_groups = tuple(groups)
        self.hc_fp32_keep = bool(groups)
        if groups:
            self.fp32_keep_params = apply_fp32_keep(self, groups=groups, verbose=True)
            counts = ", ".join(
                f"{key}: {count} tensors / {numel} elements"
                for key, (count, numel) in sorted(dtype_counts(self).items())
            )
            print_rank_0(
                f"[shensi][fp32keep] ON: groups={','.join(groups)}, {len(self.fp32_keep_params)} "
                f"parameters kept in fp32 | model dtype counts: {counts}"
            )
        else:
            self.fp32_keep_params = []
            print_rank_0(
                "[shensi][fp32keep] OFF (hc_fp32_keep=False / SHENSI_HC_FP32_KEEP=0 / "
                "SHENSI_FP32_KEEP_GROUPS is empty): parameters follow the model dtype, which does "
                "not match the HF checkpoint and is for comparison only"
            )

    def _init_shensi_only_weights(self) -> None:
        std = float(self.config.init_method_std)
        hidden = int(self.config.hidden_size)
        for module in self.modules():
            init = getattr(module, "init_shensi_weights", None)
            if callable(init):
                init(std=std, hidden_size=hidden)

    def forward(self, input_ids, position_ids, attention_mask, *args, **kwargs):
        """GPTModel forward, plus publishing this micro-batch's input_ids for the hash layers.

        The attention mask is handled here too: the sparse attention accepts only an implicit
        causal mask, so an all-valid mask is dropped and padding or document boundaries raise.
        """
        if attention_mask is not None:
            if _only_trailing_padding(attention_mask):
                # All-valid, or right-padded rows: with causal attention the valid tokens never
                # see the pads, so dropping the mask is exact (rollout replies are right-padded).
                attention_mask = None
            else:
                valid = attention_mask.sum(-1).tolist() if attention_mask.dim() > 1 else None
                raise ValueError(
                    "the sparse attention supports only an implicit causal mask, but the received "
                    f"mask is not right-padded (shape={tuple(attention_mask.shape)}, valid per row={valid}). "
                    "Left padding or document boundaries place pads before valid tokens and corrupt "
                    "the numerics; use right padding, or no padding and no packing."
                )
        token = publish_hash_input_ids(input_ids)
        try:
            return super().forward(input_ids, position_ids, attention_mask, *args, **kwargs)
        finally:
            reset_hash_input_ids(token)

    def _postprocess(self, hidden_states: torch.Tensor, *args, **kwargs):
        from megatron.core.inference.utils import InferenceMode
        from megatron.core.utils import make_viewless_tensor

        if not self.post_process:
            return make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        state = self.attn_res_state
        mtp_in_postprocess = bool(kwargs.get("mtp_in_postprocess"))
        in_inference_mode = InferenceMode.is_active()
        inference_context = kwargs.get("inference_context")
        is_spec_decode = (
            in_inference_mode
            and inference_context is not None
            and inference_context.is_dynamic_batching()
            and inference_context.num_speculative_tokens > 0
        )
        if mtp_in_postprocess and not (in_inference_mode or is_spec_decode):
            if state is None or state.prefix_sum is None:
                raise RuntimeError("attention-residual state is empty: the decoder has not run forward yet")
            # The MTP layer takes contracted [s, b, h] states and expands the streams inside the
            # layer, so the decoder output is contracted before it is handed over.
            contracted = self.shensi_output_contract(hidden_states)
            contracted = self.mtp(
                input_ids=kwargs.get("input_ids"),
                position_ids=kwargs.get("position_ids"),
                hidden_states=contracted,
                attention_mask=kwargs.get("attention_mask"),
                inference_params=None,
                rotary_pos_emb=kwargs.get("rotary_pos_emb"),
                rotary_pos_cos=kwargs.get("rotary_pos_cos"),
                rotary_pos_sin=kwargs.get("rotary_pos_sin"),
                packed_seq_params=kwargs.get("packed_seq_params"),
                sequence_len_offset=kwargs.get("sequence_len_offset"),
                padding_mask=kwargs.get("padding_mask"),
                embedding=self.embedding,
                **(kwargs.get("extra_block_kwargs") or {}),
            )
            kwargs["mtp_in_postprocess"] = False
            return super()._postprocess(contracted, *args, **kwargs)
        hidden_states = self.shensi_output_contract(hidden_states)
        return super()._postprocess(hidden_states, *args, **kwargs)

    def shensi_output_contract(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return contract_output_streams(
            hidden_states,
            state=self.attn_res_state,
            output_attn_res=self.output_attn_res,
            hc_head=self.hc_head,
            output_norm=self.output_norm,
            hc_mult=self.hc_mult,
            hidden_size=int(self.config.hidden_size),
            num_blocks=self.num_blocks,
        )

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        ssd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        self.shensi_tied_shard_audit = self._audit_tied_shards(ssd)
        return ssd

    def _audit_tied_shards(self, ssd) -> dict:
        owners = getattr(self, "moe_group_owners", None) or {}
        audit = {"groups": len(owners), "tied_groups": 0, "expected": 0, "found": 0}
        if not owners:
            return audit
        seen: dict = {}
        for key, val in ssd.items():
            data = getattr(val, "data", val)
            if torch.is_tensor(data):
                seen.setdefault(id(data), []).append(key)
        by_id: dict = {}
        for name, param in self.named_parameters(remove_duplicate=False):
            by_id.setdefault(id(param), []).append(name)
        problems = []
        for block_id, owner in sorted(owners.items()):
            sharing = list(getattr(owner, "sharing_layers", []) or [])
            if len(sharing) <= 1:
                continue
            audit["tied_groups"] += 1
            tied = [
                p
                for mod in (
                    getattr(owner, "router", None),
                    getattr(owner, "experts", None),
                )
                if mod is not None
                for p in mod.parameters()
            ]
            for param in tied:
                names = by_id.get(id(param), [])
                if len(names) != len(sharing):
                    problems.append(
                        f"block {block_id}: shared parameters appear under only {len(names)} names "
                        f"(group has {len(sharing)} layers), so the sharing did not take effect"
                    )
                    continue
                keys = seen.get(id(param), [])
                audit["expected"] += len(names)
                audit["found"] += len(keys)
                if len(keys) != len(names):
                    problems.append(
                        f"block {block_id}: {len(names)} aliases (for example {names[0]}) map to "
                        f"only {len(keys)} keys in the sharded state dict"
                    )
        if problems:
            raise RuntimeError(
                "[shensi][dist_ckpt] the router and expert parameters of a tied MoE group are "
                "missing alias keys in the sharded state dict, so a distributed checkpoint stores "
                "them under fewer names than the layers that share them and loading it leaves the "
                "missing aliases at their initial values. Details: " + "; ".join(problems[:8]) + "."
            )
        return audit
