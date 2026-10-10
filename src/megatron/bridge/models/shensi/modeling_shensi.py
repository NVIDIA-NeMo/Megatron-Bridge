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
import math
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
from megatron.core.models.hybrid.hybrid_model import HybridModel
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
# to the compute dtype for the matmul and casts the gradient back. Which parameters are kept is
# controlled by the ``hc_fp32_keep`` config field.

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


def fp32_keep_enabled(config=None) -> bool:
    """Whether fp32 keeping is active for this config."""
    return bool(fp32_keep_groups(config))


def fp32_keep_groups(config=None) -> Tuple[str, ...]:
    """Parameter groups that stay in fp32 (the ``hc_fp32_keep`` config field toggles them)."""
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


def _attn_res_block_slots(blocks, num_blocks: int) -> Tuple[torch.Tensor, ...]:
    """The used block slots, one tensor per block.

    ``blocks`` is either a stacked ``[..., num_blocks, H]`` tensor (any leading dims) or the
    per-slot sequence a :class:`ShensiAttnResState` carries; both describe the same values.
    """
    if torch.is_tensor(blocks):
        slots = blocks[..., :num_blocks, :].unbind(-2)
    else:
        slots = tuple(blocks)[:num_blocks]
        if len(slots) < num_blocks:
            raise ValueError(
                f"the attention-residual block list holds {len(slots)} slots but {num_blocks} block(s) "
                "are being read: the state was built for a different block count"
            )
    for i, slot in enumerate(slots):
        if slot is None:
            raise RuntimeError(
                f"attention-residual block {i} has not been written yet: its block-writing layer "
                "has not run on this stage"
            )
    return slots


def _attn_res_sample_rows(slots, stacked, ti, ji, M: int, H: int) -> torch.Tensor:
    """Sample ``(row, block)`` value rows, in draw order.

    A stacked input gathers them with one advanced index; the per-slot layout gathers block by block
    and scatters the rows back into draw order, so every later reduction and the SVD see the same
    order.
    """
    if stacked is not None:
        return stacked.detach()[(*torch.unravel_index(ti, stacked.shape[:-2]), ji)].float()
    rows = torch.empty((int(ti.numel()), H), dtype=torch.float32, device=ti.device)
    order = torch.argsort(ji, stable=True)
    by_block = ji[order]
    for k, slot in enumerate(slots):
        picked = order[by_block == k]
        if picked.numel():
            rows[picked] = slot.detach().reshape(M, H)[ti[picked]].float()
    return rows


def _attn_res_block_scores(slots, stacked, q_dt: torch.Tensor) -> torch.Tensor:
    """``block_k . q`` for every block, one slot at a time."""
    if stacked is not None:
        return (stacked @ q_dt.unsqueeze(-1)).squeeze(-1).float()
    return torch.stack([torch.einsum("...d,...d->...", slot, q_dt) for slot in slots], dim=-1).float()


def _attn_res_block_value(slots, stacked, w: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``sum_k w_k * block_k`` over the block axis, one slot at a time.

    The sum multiplies the operands in block order and rounds once at the end, like a matmul over
    the block axis.
    """
    if stacked is not None:
        return (w.to(out_dtype).unsqueeze(-2) @ stacked).squeeze(-2).float()
    value_dtype = torch.promote_types(out_dtype, slots[0].dtype)
    acc: Optional[torch.Tensor] = None
    for k, slot in enumerate(slots):
        # The singleton dim is the block axis the equivalent matmul contracts.
        term = w[..., k].unsqueeze(-1).to(value_dtype).float() * slot.to(value_dtype).float()
        acc = term if acc is None else acc + term
    return acc.to(value_dtype).float()


class ShensiAttentionResidual(ShensiFp32KeepMixin, MegatronModule):
    """Attention residual: mixes a block's stream context into the layer input."""

    SHENSI_FP32_KEEP_GROUP = GROUP_ATTN_RES

    def __init__(self, config, layer_number: int = 1) -> None:
        super().__init__(config)
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.eps = float(config.layernorm_epsilon)
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
        blocks: Union[torch.Tensor, Sequence[torch.Tensor]],
        output_norm_weight: Optional[torch.Tensor],
        num_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the block context into the layer input.

        ``blocks`` is either the reference layout (``[..., nb, H]``) or the per-slot sequence of the
        state; both hold the same per-stream values and the same math runs on either.
        """
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self._norm(prefix + (delta if delta is not None else 0.0))

        # ---- write path ----
        g0, g1 = self.g_a_proj, self.g_b_proj
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()
        # keep decay <= 1 without freezing the gate: clamp the forward value, pass the
        # gradient through (a plain clamp() has zero grad at the boundary)
        decay_scale = decay_scale + (decay_scale.clamp(min=0.0) - decay_scale).detach()
        r_decay, r_erase, r_write = (
            F.linear(F.linear(state, g0.weight.float(), g0.bias.float()), g1.weight.float(), g1.bias.float())
            .reshape(*state.shape[:-1], 3, -1)
            .unbind(-2)
        )
        decay = torch.exp(F.softplus(r_decay) * (-decay_scale * self.t.exp()))
        erase = F.softplus(r_erase) * erase_scale
        write = 1.0 + torch.tanh(r_write) * write_scale
        k0, k1 = self.k_a_proj, self.k_b_proj
        khat = F.normalize(
            F.linear(F.linear(delta if delta is not None else state, k0.weight.float()), k1.weight.float()),
            dim=-1,
        )
        m = torch.addcmul(decay * prefix, write, delta) if delta is not None else decay * prefix
        lam = erase.mean(dim=-1, keepdim=True).clamp(min=-0.5)
        updated = torch.addcmul(
            m, khat, torch.einsum("...d,...d->...", khat, m).unsqueeze(-1) * lam / (1.0 + lam), value=-1.0
        )

        # ---- read path: diag + above-MP-edge spike metric, rank decided by the data ----
        if num_blocks > 0:
            slots = _attn_res_block_slots(blocks, num_blocks)
            # A stacked view of the same slots, for the caller that passes one tensor.
            stacked = blocks[..., :num_blocks, :] if torch.is_tensor(blocks) else None
            H = updated.shape[-1]
            M = updated.numel() // H
            q0, q1 = self.q_a_proj, self.q_b_proj
            query = F.linear(F.linear(state, q0.weight.float()), q1.weight.float())

            # deterministic-shape budget: K = 2H sampled value rows
            K = min(2 * H, M * num_blocks)
            ti = torch.randint(M, (K,), device=updated.device)
            ji = torch.randint(num_blocks, (K,), device=updated.device)
            rows = _attn_res_sample_rows(slots, stacked, ti, ji, M, H)

            # per-coordinate second moments, shrunk toward 1 by evidence: H/(K+H);
            # floor keeps dead coordinates finite, no epsilon tuning
            m2 = torch.einsum("kd,kd->d", rows, rows) / K
            s = torch.lerp(m2, m2.new_ones(()), H / (K + H)).clamp_min(1e-20).rsqrt()
            z = rows * s

            # one SVD + Marchenko-Pastur edge: keep only statistically estimable
            # directions; BBP debiasing recovers the population spike strength
            _, S, V = torch.svd_lowrank(z, q=min(64, K, H), niter=4)
            ell = S.square() / K
            keep = ell > (1.0 + math.sqrt(H / K)) ** 2
            b = ell[keep] - 1.0 - H / K
            theta = 0.5 * (b + (b.square() - 4.0 * H / K).clamp_min(0.0).sqrt())
            U = V[:, keep] * (theta / (1.0 + theta)).unsqueeze(0) * s.unsqueeze(-1)

            # metric folded entirely into the query: C^-1 ~ s^2 (I - U U^T),
            # values are only scored and retrieved, never touched by the metric
            q = s * (query - (query @ U) @ U.T)

            # scoring and retrieval: the only two passes over all values
            logits = torch.cat(
                (
                    _attn_res_block_scores(slots, stacked, q.to(out_dtype)),
                    torch.einsum("...d,...d->...", updated, q).unsqueeze(-1),
                ),
                dim=-1,
            )
            # exp(logit - softplus(lse)) == sigmoid(lse) * softmax(logit);
            # score sum stays below 1, the residual always dominates the readout
            w = torch.softmax(logits, dim=-1)
            g = torch.sigmoid(torch.logsumexp(logits, dim=-1, keepdim=True)) * read_scale
            routed = torch.addcmul(
                _attn_res_block_value(slots, stacked, w[..., :num_blocks], out_dtype),
                w[..., num_blocks:] * g,
                updated,
            )
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
    """Per-block AttnRes bookkeeping shared between the layers of one block.

    The block history is kept as one tensor per slot (``blocks``), each holding its block's
    per-stream prefix sum. A stacked ``[s, b, hc, nb, H]`` buffer holds the same values but must be
    copied at every block boundary and keeps one buffer generation per copy alive for backward;
    keeping the tensors stores the identical values without either. ``residual`` reassembles the
    stacked layout on demand for tooling and tests.
    """

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
        self.blocks: List[Optional[torch.Tensor]] = []
        self.block_dtype: Optional[torch.dtype] = None
        self.num_written = 0
        self.pp_imported = False

    @property
    def residual(self) -> Optional[torch.Tensor]:
        """The block history as one stacked ``[s, b, hc, nb, H]`` tensor (allocates a copy).

        Unwritten slots read as zeros. The read path does not use this; it consumes ``blocks`` slot
        by slot.
        """
        reference = next((b for b in self.blocks if b is not None), self.prefix_sum)
        if reference is None:
            return None
        zeros = torch.zeros_like(reference)
        return torch.stack([zeros if b is None else b for b in self.blocks], dim=-2)

    def begin(self, stream: torch.Tensor) -> None:
        if stream.dim() != 4:
            raise ValueError(f"ShensiAttnResState.begin expects a 4D n-stream input, got {stream.shape}")
        self.prefix_sum = stream
        s, b, hc, hidden = stream.shape
        self.blocks = [None] * self.num_blocks
        self.block_dtype = stream.dtype
        self.hc_mult, self.hidden_size = int(hc), int(hidden)
        self.num_written = 0
        self.pp_imported = False

    def write_block(self, slot: int, prefix_sum: torch.Tensor) -> None:
        """Record one block's per-stream prefix sum by keeping the tensor in its slot.

        The slot is assigned, never written into: the read path puts the block history into the
        autograd graph, so an in-place write would trip autograd's "modified by an inplace
        operation" check at the next block boundary (only reported during backward), and the
        functional copy-on-write that replaces it costs one full-buffer copy per boundary.
        """
        if not self.blocks:
            raise RuntimeError("attention-residual state is unset: the first layer of a block did not call begin()")
        index = int(slot)
        if not 0 <= index < len(self.blocks):
            raise ValueError(f"block slot {index} is outside the state's {len(self.blocks)} slots")
        value = prefix_sum if self.block_dtype is None else prefix_sum.to(self.block_dtype)
        self.blocks[index] = value
        self.num_written = max(self.num_written, index + 1)

    def check_ready(self, layer_number: int) -> None:
        if self.prefix_sum is None or not self.blocks:
            raise RuntimeError(
                f"attention-residual state is unset: layer {layer_number} is not the first layer "
                "of the stack, no layer called begin(), and nothing was imported from the "
                "previous pipeline stage."
            )


def attn_res_payload_rows(seq_len: int, hc_mult: int, num_blocks: int) -> int:
    """Rows the attention-residual state appends to one stage payload."""
    return int(seq_len) * int(hc_mult) * (1 + int(num_blocks))


def pack_attn_res_payload(stage_output: torch.Tensor, state: "ShensiAttnResState") -> torch.Tensor:
    """Append the attention-residual state to a stage output as extra sequence rows.

    The pipeline schedule ships whatever the model returns, so a block that spans a stage boundary
    rides the activation payload: the state (`prefix_sum` plus one slot per block,
    `[s, b, hc, 1 + nb, h]`) flattens into `s * hc * (1 + nb)` rows of `hidden_size`, concatenated
    below the activations. The receiving stage recovers the row count from the shape
    (``unpack_attn_res_payload``); the state's gradients return through the same tensor.
    """
    if state.prefix_sum is None:
        raise RuntimeError("cannot pack an attention-residual payload: the block state is empty")
    ps = state.prefix_sum
    s, b, hc, h = ps.shape
    zeros = torch.zeros_like(ps)
    slots = [zeros if slot is None else slot for slot in state.blocks]
    stacked = torch.cat([ps.unsqueeze(-2)] + [v.unsqueeze(-2) for v in slots], dim=-2)  # [s, b, hc, 1+nb, h]
    # Flatten (s, hc, 1+nb) into one axis; `b` sits between them, so permute before reshaping.
    rows = stacked.permute(0, 2, 3, 1, 4).reshape(s * hc * (1 + len(slots)), b, h)
    return torch.cat([stage_output, rows.to(stage_output.dtype)], dim=0)


def unpack_attn_res_payload(
    payload: torch.Tensor,
    state: "ShensiAttnResState",
    *,
    hc_mult: int,
    num_blocks: int,
) -> torch.Tensor:
    """Split a stage payload into the activations and the attention-residual state.

    Inverse of ``pack_attn_res_payload``. Prefix sum and slots are views of the payload, so the
    gradient sent for the state arrives as the gradient of this tensor.
    """
    state_rows_per_token = int(hc_mult) * (1 + int(num_blocks))
    rows_per_token = 1 + state_rows_per_token  # one activation row plus the state rows, per token
    total = int(payload.shape[0])
    if rows_per_token <= 2 or total % rows_per_token:
        raise ValueError(
            f"stage payload with {total} rows does not carry an attention-residual state for "
            f"hc_mult={hc_mult}, num_blocks={num_blocks} (expected a multiple of {rows_per_token} rows); "
            "check that both stages agree on the layer plan and on attn_res_pp_state_transfer"
        )
    seq = total // rows_per_token
    batch, hidden = int(payload.shape[1]), int(payload.shape[2])
    region = payload[seq:].reshape(seq, hc_mult, 1 + int(num_blocks), batch, hidden).permute(0, 3, 1, 2, 4)
    state.prefix_sum = region[:, :, :, 0, :]
    state.blocks = [region[:, :, :, 1 + i, :] for i in range(int(num_blocks))]
    state.block_dtype = state.prefix_sum.dtype
    state.hc_mult, state.hidden_size = int(hc_mult), hidden
    state.num_written = int(num_blocks)
    state.pp_imported = True
    return payload[:seq]


def bind_attn_res_state(block, config, num_blocks: int, *, plan=None) -> ShensiAttnResState:
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
    block.shensi_attn_res_state = state
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
        # The state belongs on the decoder-shaped layers *inside* the wrapper (that is where
        # `ShensiTransformerLayer` reads it) and on the inner stack itself, mirroring
        # `bind_attn_res_state` for the main stack.
        inner_layers = getattr(inner, "layers", None)
        if inner_layers is None:
            raise AttributeError(f"{type(inner).__name__} has no layers, cannot bind attention-residual state")
        for layer in inner_layers:
            layer.attn_res_state = state
        # The wrapper carries the state as well: `ShensiMultiTokenPredictionLayer` reads it there,
        # while the inner layers read the same object in forward.
        inner.attn_res_state = state
        inner.shensi_attn_res_state = state
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
        state.blocks,
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

    Shensi contracts the streams in the model post-process, so every MTP depth performs that step
    before the shared head. Megatron-Core does it internally only on its ``enable_mhc_connections``
    path, which shensi does not use.
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
    state_transfer_required: bool = False
    state_transfer_enabled: Optional[bool] = None
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
            f"  boundaries (adjacent stage pairs) = {self.boundaries}; cross-stage state needed = "
            f"{self.state_transfer_required} (carried in the stage payload = {self.state_transfer_enabled})",
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
        return int(seq_len) * int(micro_batch) * int(hc_mult) * int(hidden_size) * (1 + nb)

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

    @property
    def carries_state(self) -> bool:
        """True when this stage's output payload has to carry the attention-residual state."""
        return bool(self.local_export_layers())

    @property
    def imports_state(self) -> bool:
        """True when this stage's input payload carries the attention-residual state."""
        return bool(self.local_import_layers())

    def local_import_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.local_layers if self.needs_import(i))

    def local_export_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in self.local_layers if self.needs_export(i))

    def report(self, *, state_transfer_enabled: Optional[bool] = None) -> LayoutReport:
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
            if state_transfer_enabled is False:
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
                    f"{len(crossing)} blocks cross a stage boundary; their state rides the stage payload:"
                    + "; ".join(f"block{b.index}" for b, _ in crossing)
                )
            for b, st in crossing:
                vps = {v for _, v in st}
                if len(vps) > 1:
                    problems.append(
                        f"block{b.index} crosses virtual stages (vps {sorted(vps)}): with "
                        "interleaved scheduling the carrier only covers adjacent pipeline stages"
                    )
                    suggestions.append(
                        "align block boundaries with virtual stage boundaries (see "
                        "recommended_layout_str) or verify with vpp=1 first"
                    )
                elif self.vpp_size > 1:
                    notes.append(f"block{b.index} crosses a pp stage but stays inside one vp stage: pp carriage")
        state_transfer_required = self.pp_size > 1 or self.vpp_size > 1
        if state_transfer_required and state_transfer_enabled is False:
            problems.append(
                f"pipeline parallelism (pp={self.pp_size}, vpp={self.vpp_size}) requires carrying the block state: "
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
            state_transfer_required=state_transfer_required,
            state_transfer_enabled=state_transfer_enabled,
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
    state_transfer_enabled: Optional[bool] = None,
    raise_on_error: bool = True,
) -> LayoutReport:
    """Validate pipeline splits against the AttnRes block boundaries."""
    if raise_on_error:
        check_attn_res_recompute_support(config)
    if state_transfer_enabled is None:
        state_transfer_enabled = bool(getattr(config, "attn_res_pp_state_transfer", True))
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
    report = plan.report(state_transfer_enabled=state_transfer_enabled)
    if state_transfer_enabled and plan.crossing_blocks():
        # The state rides the stage payload, whose shape is then larger than the pipeline's static
        # shape: the schedule has to exchange shapes per microbatch for the move to work.
        handshake = bool(getattr(config, "variable_seq_lengths", False)) or bool(
            getattr(config, "mtp_standalone", False)
        )
        if not handshake and raise_on_error:
            raise AttnResPPLayoutError(
                f"a block spans a pipeline stage boundary (pp={plan.pp_size}, vpp={plan.vpp_size}) and its "
                "state rides the stage payload, so the payload shape differs from the pipeline's static "
                "shape. Set `variable_seq_lengths: true` (or `mtp_standalone`) so the schedule exchanges "
                "shapes per microbatch, or align the split with the block boundaries."
            )
    if report.ok or not raise_on_error:
        return report
    header = f"attention-residual layout check failed (pp={report.pp_size}, vpp={report.vpp_size}):\n" + str(report)
    transfer_disabled = any("attn_res_pp_state_transfer" in p for p in report.problems)
    if transfer_disabled:
        raise NotImplementedError(
            header + "\n\nfix: set config.attn_res_pp_state_transfer to True so the block state rides the "
            "stage payload, or use one of the block-aligned layouts suggested above."
        )
    raise AttnResPPLayoutError(header)


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
    grad_reduce: str = "slice",
    verify_router_replica: bool = False,
    local_view: bool = False,
    verbose: bool = False,
):
    """Gather the expert weights used by the ERC (router / expert coupling) loss.

    Args:
        mlp: The routed-expert layer to read the weights from.
        ep_group: Expert-parallel group override (resolved from the parallel state by default).
        grad_reduce: How the gather reduces the gradient over the group: ``"slice"`` keeps the
            contribution a rank's own loss produced, ``"reduce_scatter_mean"`` averages it.
        verify_router_replica: Assert the router/latent weights are replica-identical across the
            expert-parallel group before gathering.
        local_view: Score only the experts this rank holds instead of gathering the group's experts;
            intended for debugging on multi-EP runs, where a per-rank view is a different objective.
        verbose: Log the resolved view on rank 0.
    """
    ep_size, ep_rank, ep_group = _resolve_parallel_group("expert_model_parallel", ep_group)
    etp_size, etp_rank, etp_group = _resolve_parallel_group("expert_tensor_parallel", None)
    grad_reduce = (grad_reduce or "slice").strip().lower()
    if local_view:
        router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
        n_local = int(getattr(mlp, "num_local_experts", 0) or 0)
        if ep_size > 1 and n_local > 0:
            router_weight = router_weight[ep_rank * n_local : (ep_rank + 1) * n_local]
        if verbose:
            print_rank0(
                "[shensi][erc] erc_gather_expert_weights: local_view=True -> scoring "
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
            "objective. Build the model after initialize_model_parallel, or pass local_view=True to "
            "score the local experts on purpose."
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
                "replicas, or pass local_view=True to score the local experts."
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
    packed_seq_params=None,
) -> Tensor:
    """Layer forward with AttnRes context, mHC streams and the MoE sublayer."""
    streams = attn_res_to_streams(
        hidden_states,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        is_first_in_block=is_first_in_block,
    )
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
        state.blocks,
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
        state.blocks,
        output_norm_weight=pre_mlp_norm_weight,
        num_blocks=prev_valid_blocks + int(is_block_write_layer),
    )
    collapsed = ffn_hc(mlp_in)
    mlp_output = mlp_fn(collapsed)
    streams = ffn_hc.write_back(mlp_in, mlp_output, packed_seq_params)
    state.prefix_sum = prefix_sum + streams
    return attn_res_to_flat(streams)


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


class ShensiModel(HybridModel):
    """Shensi model: DeepSeek-V4 style stack with mHC head, AttnRes state and MTP.

    The container is Megatron-Core's ``HybridModel``; the decoder itself is the bridge-side
    ``ShensiDecoderStack`` (see ``shensi_hybrid``), which owns the shensi layer stack and applies the
    attention-residual output contract at its end. MTP heads ride the hybrid pattern and are built
    from the shensi MTP block spec; the attention-residual state and stream contract are bound onto
    the decoder and the MTP block right after construction.
    """

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
        # Whether this stage packs its state into the output payload / unpacks it from the input
        # payload (see ``pack_attn_res_payload``); both stay False for a single-stage model.
        self.attn_res_payload_export = False
        self.attn_res_payload_import = False
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
            transfer = bool(getattr(cfg, "attn_res_pp_state_transfer", True))
            self.attn_res_payload_export = bool(plan.carries_state) and transfer
            self.attn_res_payload_import = bool(plan.imports_state) and transfer
            print_rank_0(
                f"[shensi][attn_res_pp] pp={pp_size} vpp={vpp_size} "
                f"local layers={list(plan.local_layers)} import={list(plan.local_import_layers())} "
                f"export={list(plan.local_export_layers())} blocks crossing a stage boundary="
                f"{[b.index for b, _ in plan.crossing_blocks()]} | "
                f"layout check={'PASS' if report.ok else 'FAIL'}"
            )
            crossing = [b.index for b, _ in plan.crossing_blocks()]
            if crossing:
                # A crossing block adds hc_mult * (1 + num_blocks) * hidden elements per token to
                # the stage payload. Splits aligned with the block boundaries carry nothing.
                per_token = int(cfg.num_residual_streams) * (1 + int(self.num_blocks)) * int(cfg.hidden_size)
                print_rank_0(
                    f"[shensi][attn_res_pp] {len(crossing)} attention-residual block(s) span a "
                    f"pipeline boundary (blocks {crossing}): the stage payload carries ~{per_token} "
                    "extra elements per token per transition (hc_mult x (1 + num_blocks) x hidden). "
                    "Align the pipeline split with the block boundaries so nothing has to cross."
                )
        self.attn_res_state = bind_attn_res_state(
            self.decoder,
            cfg,
            self.num_blocks,
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
                "[shensi][fp32keep] OFF (hc_fp32_keep=False): parameters follow the model dtype, "
                "which does not match the HF checkpoint and is for comparison only"
            )
        # The decoder stack applies the attention-residual output contract at its end (it replaces the
        # GPT-style postprocess hook, which HybridModel does not have).
        set_contract = getattr(self.decoder, "set_output_contract", None)
        if callable(set_contract):
            set_contract(self.shensi_output_contract)

    def _init_shensi_only_weights(self) -> None:
        std = float(self.config.init_method_std)
        hidden = int(self.config.hidden_size)
        for module in self.modules():
            init = getattr(module, "init_shensi_weights", None)
            if callable(init):
                init(std=std, hidden_size=hidden)

    def forward(self, input_ids, position_ids, attention_mask, *args, **kwargs):
        """HybridModel forward, plus publishing this micro-batch's input_ids for the hash layers.

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
        if self.attn_res_payload_import:
            self._load_attn_res_payload()
        token = publish_hash_input_ids(input_ids)
        try:
            hidden_states = super().forward(input_ids, position_ids, attention_mask, *args, **kwargs)
        finally:
            reset_hash_input_ids(token)
        if self.attn_res_payload_export:
            hidden_states = pack_attn_res_payload(hidden_states, self.attn_res_state)
        return hidden_states

    def _load_attn_res_payload(self) -> None:
        """Load the block state carried by the received stage payload.

        The schedule delivers the previous stage's output through ``set_input_tensor``, with the
        block state appended below the activations (``pack_attn_res_payload``); the state is loaded
        into the bound state and the stack receives the plain activation slice.
        """
        decoder = self.decoder
        payload = getattr(decoder, "input_tensor", None)
        if payload is None:
            raise RuntimeError(
                "[shensi][attn_res_pp] this stage carries a block state in its input payload but no "
                "input tensor was set: the schedule must deliver the previous stage's output through "
                "set_input_tensor (a block that spans the boundary would otherwise read an empty state)"
            )
        plain = unpack_attn_res_payload(
            payload,
            self.attn_res_state,
            hc_mult=self.hc_mult,
            num_blocks=self.num_blocks,
        )
        decoder.set_input_tensor(plain)

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
