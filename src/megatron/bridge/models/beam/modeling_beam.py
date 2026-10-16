# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 Reflection AI. All rights reserved.
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

"""Beam modules on top of an unmodified Megatron-Core ``HybridModel``.

Each Beam decoder layer becomes two ``HybridStack`` layers: an attention-only layer
(``*``) followed by a dense MLP (``-``) or MoE (``E``) layer. Megatron-Core's layers
cover Beam except for:

* SSMax query scaling on NoPE layers,
* a learnable per-head bias on the attention output gate,
* sandwich norms (a norm on each sublayer output before the residual add),
* depth-scaled residual adds,
* fp32 router parameters (and two router-GEMM fixes they need),
* fp32 final-logit soft-capping.

Everything except the last item is swapped into the stack spec by
:func:`beam_hybrid_stack_spec`, which builds the hybrid layers from Megatron-Core's
Transformer Engine GPT layer submodules with the subclasses below. The pinned Megatron-Core
distributed optimizer supports mixed FP32/BF16 parameter groups directly.
"""

import dataclasses
import functools
import math
from typing import Optional, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention, TENorm
from megatron.core.jit import jit_fuser
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from megatron.core.utils import divide, get_tensor_model_parallel_group_if_none


# ---------------------------------------------------------------------------
# Final-logit soft-capping
# ---------------------------------------------------------------------------


def beam_logit_softcapping(logits: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    """``cap * tanh(logits / cap)`` computed and returned in fp32, exactly as ``BeamForCausalLM`` does.

    ``None`` / ``0`` disables the cap and returns ``logits`` unchanged (in their own dtype).
    """
    if not cap:
        return logits
    return torch.tanh(logits.float() / cap) * cap


class BeamOutputLayer(ColumnParallelLinear):
    """``ColumnParallelLinear`` whose output is soft-capped in fp32.

    Applied to the output layer by BeamModelBuilder.
    """

    def forward(self, *args, **kwargs):
        """Project, then soft-cap in fp32."""
        output, bias = super().forward(*args, **kwargs)
        return beam_logit_softcapping(output, self.config.final_logit_softcapping), bias


# ---------------------------------------------------------------------------
# Config validation for the Beam fields (declared on BeamTransformerConfig)
# ---------------------------------------------------------------------------

BEAM_FEATURE_FIELDS = (
    "attention_output_gate_bias",
    "post_self_attn_layernorm",
    "post_mlp_layernorm",
    "depth_scaled_residual",
    "ssmax",
    "moe_router_fp32_params",
)


def validate_beam_config(config) -> None:
    """Reject Beam features on code paths that would silently skip them.

    SSMax runs in core attention, sandwich norms and depth-scaled residuals replace the
    layer's bias-dropout-add; paths that bypass either would build a different model.
    """
    if config.attention_output_gate_bias and not config.attention_output_gate:
        raise ValueError("attention_output_gate_bias requires attention_output_gate=True.")

    if config.moe_router_fp32_params and config.moe_router_dtype != "fp32":
        raise ValueError(
            "moe_router_fp32_params requires moe_router_dtype='fp32': fp32 router parameters "
            "are only meaningful when the routing GEMM itself runs in fp32."
        )

    if config.ssmax:
        if not config.no_rope_freq:
            raise ValueError("ssmax requires no_rope_freq: scaling only applies to NoPE layers.")
        if config.flash_decode:
            raise ValueError(
                "ssmax cannot be used with flash_decode: flash decoding bypasses core attention, "
                "where SSMax is applied."
            )
        if config.multi_latent_attention:
            raise ValueError("ssmax is not implemented for multi_latent_attention.")
        if config.experimental_attention_variant is not None:
            raise ValueError(
                f"ssmax is not implemented for experimental_attention_variant {config.experimental_attention_variant!r}."
            )

    if config.post_self_attn_layernorm or config.post_mlp_layernorm or config.depth_scaled_residual:
        what = "post_self_attn_layernorm / post_mlp_layernorm / depth_scaled_residual"
        if config.inference_fuse_tp_communication:
            raise ValueError(
                f"{what} cannot be used with inference_fuse_tp_communication: that path adds the "
                "residual inside linear_proj/linear_fc2, so it would be added twice."
            )
        if config.transformer_impl == "inference_optimized":
            raise ValueError(f"{what} are not supported by transformer_impl='inference_optimized'.")
        if config.enable_mhc_connections:
            raise ValueError(
                f"{what} are not supported with enable_mhc_connections (HyperConnection layers use "
                "their own residual kernel)."
            )
        if config.overlap_moe_expert_parallel_comm:
            raise ValueError(
                f"{what} are not supported with overlap_moe_expert_parallel_comm: the fine-grained "
                "MoE schedule runs the MLP residual outside TransformerLayer."
            )
        if config.depth_scaled_residual and config.mtp_num_layers:
            raise ValueError(
                "depth_scaled_residual is not supported with mtp_num_layers: MTP layers restart "
                "layer_number at 1, so the depth index would be wrong."
            )


# ---------------------------------------------------------------------------
# SSMax (scalable softmax) on NoPE layers
# ---------------------------------------------------------------------------


def ssmax_absolute_positions(
    local_seq_len: int, position_offset: int = 0, cp_rank: int = 0, cp_size: int = 1, device=None
) -> torch.Tensor:
    """Absolute sequence positions for the tokens this rank actually holds.

    Without CP, positions are contiguous. Otherwise, ``get_batch_on_this_cp_rank``
    splits the sequence into ``2 * cp_size`` chunks and gives rank ``r`` chunks
    ``(r, 2 * cp_size - r - 1)``. For CP=2 with 16 tokens::

        rank 0 local -> [0 1 2 3][12 13 14 15]
        rank 1 local -> [4 5 6 7][ 8  9 10 11]
    """
    if cp_size == 1:
        return torch.arange(position_offset, position_offset + local_seq_len, device=device, dtype=torch.float32)

    if local_seq_len % 2 != 0:
        raise ValueError(
            f"Context parallelism gives each rank two chunks, so the local sequence length "
            f"must be even; got {local_seq_len}."
        )
    chunk = local_seq_len // 2
    first_start = cp_rank * chunk
    second_start = (2 * cp_size - 1 - cp_rank) * chunk
    first = torch.arange(first_start, first_start + chunk, device=device, dtype=torch.float32)
    second = torch.arange(second_start, second_start + chunk, device=device, dtype=torch.float32)
    return torch.cat([first, second]) + position_offset


def ssmax_packed_positions(
    cu_seqlens: torch.Tensor,
    position_offset: int = 0,
    cp_rank: int = 0,
    cp_size: int = 1,
    total_tokens: int | None = None,
) -> torch.Tensor:
    """Within-sequence positions for every token of a packed (``thd``) buffer.

    As in RoPE's ``thd`` path, use global cumulative lengths (the padded variant
    for padded batches). Apply :func:`ssmax_absolute_positions`' zigzag layout
    independently to each sequence under CP.

    Fully vectorised; passing ``total_tokens`` (the local buffer length) avoids a
    host sync. Tokens past the last real token (tail padding) are clamped onto the
    final sequence; they are masked out of attention anyway.
    """
    cu = cu_seqlens.to(torch.int64)
    global_lens = cu[1:] - cu[:-1]
    local_lens = global_lens // cp_size if cp_size > 1 else global_lens
    local_cu = torch.zeros_like(cu)
    local_cu[1:] = torch.cumsum(local_lens, dim=0)
    if total_tokens is None:
        total_tokens = int(local_cu[-1].item())
    token = torch.arange(total_tokens, device=cu.device, dtype=torch.int64)
    seq_idx = (torch.searchsorted(local_cu, token, right=True) - 1).clamp(min=0, max=cu.numel() - 2)
    local_pos = token - local_cu[seq_idx]
    if cp_size > 1:
        seg = local_lens[seq_idx] // 2  # each rank holds two chunks of this size per sequence
        second = local_pos >= seg
        pos = torch.where(second, (2 * cp_size - 1 - cp_rank) * seg + (local_pos - seg), cp_rank * seg + local_pos)
    else:
        pos = local_pos
    return pos.to(torch.float32) + position_offset


def apply_ssmax_to_query(
    query: torch.Tensor,
    position_offset: int = 0,
    cp_rank: int = 0,
    cp_size: int = 1,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale ``query`` by ``(1 + (position + 1) / 2048) ** 0.1`` for each of its positions.

    Args:
        query: ``[sq, b, np, hn]`` normally, ``[t, np, hn]`` for packed ``thd`` batches.
        position_offset: Absolute position of ``query[0]`` (non-zero during incremental decoding).
        cp_rank: This rank's index within the context-parallel group.
        cp_size: Context-parallel world size; see :func:`ssmax_absolute_positions`.
        cu_seqlens: Global cumulative sequence lengths of a packed batch; when given,
            positions restart at every sequence (see :func:`ssmax_packed_positions`).

    Returns:
        The scaled query, same shape and dtype as the input.
    """
    seq_len = query.shape[0]
    if cu_seqlens is None:
        positions = ssmax_absolute_positions(seq_len, position_offset, cp_rank, cp_size, device=query.device)
    else:
        positions = ssmax_packed_positions(cu_seqlens, position_offset, cp_rank, cp_size, total_tokens=seq_len)
    # Multiply in fp32 before rounding once: a bf16 scale rounds to 1 near the sequence start.
    scale = torch.exp(0.1 * torch.log1p((positions + 1) / 2048))
    scale = scale.view(seq_len, *([1] * (query.dim() - 1)))
    return (query.float() * scale).to(query.dtype)


def _is_ssmax_layer(config, layer_number: int) -> bool:
    """SSMax applies to the NoPE layers (``no_rope_freq`` is 0-indexed, ``layer_number`` 1-indexed)."""
    return bool(config.ssmax and config.no_rope_freq and config.no_rope_freq[layer_number - 1])


class _BeamSSMaxCoreAttentionMixin:
    """Core attention that applies SSMax to the query of NoPE layers.

    ``Attention`` calls core attention right after RoPE, on both the plain and the
    activation-checkpointed path, so scaling here is equivalent to scaling in
    ``Attention.forward``. Paths that bypass core attention (flash decoding, dynamic
    batching) are rejected by :func:`validate_beam_config` and :class:`BeamSelfAttention`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs.get("config", args[0] if args else None)
        layer_number = kwargs.get("layer_number", args[1] if len(args) > 1 else 1)
        self._beam_ssmax = _is_ssmax_layer(config, layer_number)
        pg_collection = kwargs.get("pg_collection")
        if pg_collection is not None and hasattr(pg_collection, "cp"):
            self._beam_cp_group = pg_collection.cp
        else:
            self._beam_cp_group = parallel_state.get_context_parallel_group(check_initialized=False)

    def forward(self, query, key, value, attention_mask, *args, **kwargs):
        """Scale the query on NoPE layers, then run the wrapped core attention."""
        if self._beam_ssmax:
            query = self._beam_apply_ssmax(
                query, key, kwargs.get("packed_seq_params", args[2] if len(args) > 2 else None)
            )
        return super().forward(query, key, value, attention_mask, *args, **kwargs)

    def _beam_apply_ssmax(self, query, key, packed_seq_params):
        cp_group = self._beam_cp_group
        cp_size = cp_group.size() if cp_group is not None else 1
        cp_rank = cp_group.rank() if cp_size > 1 else 0
        cu_seqlens = None
        if packed_seq_params is not None:
            if packed_seq_params.qkv_format != "thd":
                raise NotImplementedError(
                    "SSMax with packed sequences is implemented for the 'thd' format only, "
                    f"got qkv_format={packed_seq_params.qkv_format!r}."
                )
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        position_offset = 0
        if key.shape[0] != query.shape[0]:
            # Static-batching decode: the key holds the cached prefix, the query only new tokens.
            if cu_seqlens is not None:
                raise NotImplementedError("SSMax with packed sequences is implemented for the training forward only.")
            if cp_size > 1:
                raise NotImplementedError(
                    "SSMax with context parallelism is only implemented for the full-sequence forward, "
                    "not incremental decoding."
                )
            position_offset = key.shape[0] - query.shape[0]
        return apply_ssmax_to_query(query, position_offset, cp_rank=cp_rank, cp_size=cp_size, cu_seqlens=cu_seqlens)


class BeamTEDotProductAttention(_BeamSSMaxCoreAttentionMixin, TEDotProductAttention):
    """Transformer Engine ``TEDotProductAttention`` with SSMax on NoPE layers."""


# ---------------------------------------------------------------------------
# Attention output gate with a per-head bias
# ---------------------------------------------------------------------------


def apply_output_gate(x: torch.Tensor, gate: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gate ``x`` with ``sigmoid(gate [+ bias per head])``.

    The sigmoid (and the bias add) run in fp32; the result is cast back to ``x.dtype``.

    Args:
        x: attention output, ``[..., num_heads * head_dim]`` with heads major.
        gate: gate pre-activations with as many elements as ``x``.
        bias: optional ``[num_heads]`` bias, one scalar per (local) head.
    """
    x_dtype = x.dtype
    gate = gate.contiguous().view(*x.shape)
    if bias is None:
        return (x * torch.sigmoid(gate.float())).to(x_dtype)

    num_heads = bias.numel()
    if x.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Gate bias has {num_heads} heads but the attention output width {x.shape[-1]} is not divisible by it."
        )
    gate = gate.view(*x.shape[:-1], num_heads, -1).float()
    gate = gate + bias.float().view(*([1] * (x.dim() - 1)), num_heads, 1)
    return (x * torch.sigmoid(gate).view(*x.shape)).to(x_dtype)


class AttentionOutputGateBias(MegatronModule):
    """Zero-initialized, TP-sharded per-head bias for the attention output gate.

    A dedicated module owns the bias so checkpoint converters can identify it by class name.
    """

    def __init__(
        self, config, num_heads_per_partition: int, tp_group: Optional[torch.distributed.ProcessGroup] = None
    ):
        super().__init__(config=config)
        self.tp_group = tp_group
        if config.use_cpu_initialization or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.cuda.current_device()
        self.bias = torch.nn.Parameter(torch.empty(num_heads_per_partition, dtype=config.params_dtype, device=device))
        if config.perform_initialization:
            self.reset_parameters()
        set_tensor_model_parallel_attributes(self.bias, is_parallel=True, dim=0, stride=1)

    def reset_parameters(self) -> None:
        """Initialize the bias to zero."""
        torch.nn.init.zeros_(self.bias)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ):
        """Shard ``bias`` along dim 0 across tensor-parallel ranks."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"bias": 0},
            sharded_offsets,
            tp_group=get_tensor_model_parallel_group_if_none(self.tp_group),
            dp_cp_group=metadata["dp_cp_group"],
        )


class BeamSelfAttention(SelfAttention):
    """``SelfAttention`` whose output gate carries a per-head bias.

    The bias follows the local query heads, including under KV replication (TP > KV heads).
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.output_gate_bias = None
        if config.attention_output_gate and config.attention_output_gate_bias:
            self.output_gate_bias = AttentionOutputGateBias(
                config,
                divide(config.num_attention_heads, self.world_size),
                tp_group=self.pg_collection.tp,
            )

    @jit_fuser
    def _apply_output_gate(self, x, gate):
        # Fused like Megatron-Core's Attention._apply_output_gate, so the gate rounds identically.
        bias = self.output_gate_bias.bias if self.output_gate_bias is not None else None
        return apply_output_gate(x, gate, bias)

    def forward_pre_attn_and_core_attn(self, *args, **kwargs):
        """Reject dynamic-batching inference on SSMax layers: it bypasses core attention."""
        if _is_ssmax_layer(self.config, self.layer_number):
            inference_context = kwargs.get("inference_context")
            if inference_context is None:
                inference_context = kwargs.get("inference_params")
            if inference_context is None and len(args) > 3:
                inference_context = args[3]
            if inference_context is not None and not inference_context.is_static_batching():
                raise NotImplementedError("SSMax is not implemented for dynamic-batching inference.")
        return super().forward_pre_attn_and_core_attn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Sandwich norms and depth-scaled residuals
# ---------------------------------------------------------------------------


def depth_scaled_residual_coefs(layer_number: int) -> Tuple[float, float]:
    """``(sublayer_scale, residual_scale)`` for the residual add of one hybrid layer; the squares sum to one.

    ``residual_add_idx = layer_number + 1`` with ``HybridStack``'s 1-based ``layer_number``,
    which counts attention and MLP layers separately: the embedding is the first term of the
    residual stream, so Beam layer ``i`` (0-based) adds at ``2i + 2`` (attention) and
    ``2i + 3`` (MLP).
    """
    residual_add_idx = layer_number + 1
    return 1.0 / math.sqrt(residual_add_idx), math.sqrt((residual_add_idx - 1) / residual_add_idx)


def scaled_residual_add(output: torch.Tensor, residual: torch.Tensor, coefs: Tuple[float, float]) -> torch.Tensor:
    """``output * coefs[0] + residual * coefs[1]``, in the residual's dtype.

    Scalar coefficients avoid saving the residual for backward, allowing activation
    offload to release its storage after the add.
    """
    sublayer_scale, residual_scale = coefs
    return torch.add(residual * residual_scale, output.to(residual.dtype), alpha=sublayer_scale)


@dataclasses.dataclass
class BeamTransformerLayerSubmodules(TransformerLayerSubmodules):
    """``TransformerLayerSubmodules`` plus the two sandwich norms."""

    post_self_attn_layernorm: type = IdentityOp
    post_mlp_layernorm: type = IdentityOp


class BeamTransformerLayerMixin:
    """Sandwich norms and depth-scaled residual adds for an attention-only or MLP-only hybrid layer.

    The bias-dropout-add of the sublayer the layer owns is replaced by :meth:`_beam_bda`,
    which computes ``bias -> post norm -> dropout -> (depth-scaled) residual add``. The
    absent sublayer keeps its identity pass-through. Everything else of the base layer
    (offload commits, recompute hooks, viewless outputs) stays in place.
    """

    def __init__(self, config, submodules, layer_number: int = 1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        has_attention = not isinstance(self.self_attention, IdentityOp)
        has_mlp = not isinstance(self.mlp, IdentityOp)
        dsr = config.depth_scaled_residual
        if (config.post_self_attn_layernorm or config.post_mlp_layernorm or dsr) and not isinstance(
            self.cross_attention, IdentityOp
        ):
            raise NotImplementedError(
                "post_self_attn_layernorm / post_mlp_layernorm / depth_scaled_residual are implemented for "
                "decoder-only layers; cross-attention adds a residual that the depth index and post-norms "
                "do not account for."
            )
        if dsr and has_attention and has_mlp:
            raise NotImplementedError(
                "depth_scaled_residual expects split hybrid layers (attention '*' and MLP '-'/'E' as separate "
                "layers); a layer holding both would need two residual indices."
            )

        self.post_self_attn_layernorm = (
            self._beam_build_post_norm(getattr(submodules, "post_self_attn_layernorm", IdentityOp))
            if has_attention and config.post_self_attn_layernorm
            else None
        )
        self.post_mlp_layernorm = (
            self._beam_build_post_norm(getattr(submodules, "post_mlp_layernorm", IdentityOp))
            if has_mlp and config.post_mlp_layernorm
            else None
        )
        self._dsr_coefs = depth_scaled_residual_coefs(self.layer_number) if dsr else None

        # The spec's bias-dropout-add may be a function or a module (IdentityFuncOp); drop a
        # registered module before replacing it with a plain callable.
        if has_attention and (self.post_self_attn_layernorm is not None or dsr):
            self._modules.pop("self_attn_bda", None)
            self.self_attn_bda = functools.partial(self._beam_bda, is_attention=True)
        if has_mlp and (self.post_mlp_layernorm is not None or dsr):
            self._modules.pop("mlp_bda", None)
            self.mlp_bda = functools.partial(self._beam_bda, is_attention=False)

    def _beam_build_post_norm(self, norm):
        if norm is IdentityOp:
            raise ValueError("Enabled post-layer norms require a normalization module in the spec.")
        return norm(config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon)

    def _beam_bda(self, training: bool, fused: bool, *, is_attention: bool):
        """Same call contract as ``get_bias_dropout_add``: returns ``f(x_with_bias, residual, prob)``."""
        norm = self.post_self_attn_layernorm if is_attention else self.post_mlp_layernorm
        coefs = self._dsr_coefs

        def _sandwich_residual_add(output_with_bias, residual, prob):
            # Order is bias -> norm -> dropout -> add. Normalizing after dropout would undo
            # its 1/(1-p) rescale.
            output, bias = output_with_bias
            if bias is not None:
                output = output + bias
            if norm is not None:
                output = norm(output)
            if training and prob > 0.0:
                output = torch.nn.functional.dropout(output, p=prob)
            if coefs is not None:
                return scaled_residual_add(output, residual, coefs)
            return residual + output.to(residual.dtype)

        return _sandwich_residual_add


class BeamTransformerLayer(BeamTransformerLayerMixin, TransformerLayer):
    """Attention-only hybrid layer (``*``) with Beam's sandwich norm and depth-scaled residual."""


class BeamMLPLayer(BeamTransformerLayerMixin, MLPLayer):
    """Dense MLP hybrid layer (``-``) with Beam's sandwich norm and depth-scaled residual."""


class BeamMoETransformerLayer(BeamTransformerLayerMixin, MoETransformerLayer):
    """MoE hybrid layer (``E``) with Beam's sandwich norm and depth-scaled residual."""


# ---------------------------------------------------------------------------
# fp32 router
# ---------------------------------------------------------------------------


class _BeamRouterGatingLinear(torch.autograd.Function):
    """``RouterGatingLinearFunction`` for routers whose weight dtype differs from the input dtype.

    Two changes to Megatron-Core's version: cuBLASLt requires both GEMM operands in
    ``router_dtype`` (fp32 router weights meet bf16 activations; saved activations keep their
    input dtype), and the TE GEMM reads raw storage and ignores strides, so an expanded
    (stride-0) ``grad_output`` such as ``logits.sum().backward()`` produces is made contiguous.
    """

    @staticmethod
    def forward(ctx, inp, weight, bias, router_dtype):
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype
        ctx.weight_dtype = weight.dtype
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])

        te_general_gemm = moe_utils.te_general_gemm
        if te_general_gemm is not None and router_dtype != torch.float64:
            gemm_bias = bias.to(router_dtype) if bias is not None else None
            gemm_weight, gemm_inp = weight, inp
            if weight.dtype != inp.dtype:
                gemm_weight, gemm_inp = weight.to(router_dtype), inp.to(router_dtype)
            output = te_general_gemm(gemm_weight, gemm_inp, router_dtype, layout="TN", bias=gemm_bias)[0]
        elif bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t())
        return output.view(*inp_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        grad_shape = grad_output.shape
        inp = inp.view(-1, inp_shape[-1])
        grad_output = grad_output.contiguous().view(-1, grad_shape[-1])

        te_general_gemm = moe_utils.te_general_gemm
        if te_general_gemm is not None and ctx.router_dtype != torch.float64:
            grad_input = te_general_gemm(
                weight.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NN", grad=True
            )
            grad_weight = te_general_gemm(
                inp.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NT", grad=True
            )
            grad_input = grad_input[0].to(ctx.input_dtype)
            grad_weight = grad_weight[0].to(ctx.weight_dtype)
        else:
            grad_input = torch.mm(grad_output, weight.to(ctx.router_dtype)).to(ctx.input_dtype)
            grad_weight = torch.mm(grad_output.t(), inp.to(ctx.router_dtype)).to(ctx.weight_dtype)

        grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        return grad_input.view(*inp_shape), grad_weight, grad_bias, None


def beam_router_gating_linear(inp, weight, bias, router_dtype):
    """Router logits ``inp @ weight.T (+ bias)`` in ``router_dtype``; see :class:`_BeamRouterGatingLinear`."""
    return _BeamRouterGatingLinear.apply(inp, weight, bias, router_dtype)


class BeamTopKRouter(TopKRouter):
    """``TopKRouter`` that keeps its weight (and bias) in fp32 when ``moe_router_fp32_params`` is set.

    Parameters marked ``keep_in_fp32`` survive ``Float16Module`` and checkpoint loads, so the
    router reproduces HF's fp32 expert selection.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Fixed checkpoint biases still affect routing, but need no token-count collective.
        self.frozen_expert_bias = self.enable_expert_bias and config.moe_router_bias_update_rate == 0.0

    def reset_parameters(self):
        """Initialize like ``Router.reset_parameters``, casting to fp32 instead of ``params_dtype``."""
        if not self.config.moe_router_fp32_params:
            return super().reset_parameters()
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
            if self.bias is not None:
                self.config.init_method(self.bias)
        self.weight.data = self.weight.data.to(dtype=torch.float32)
        setattr(self.weight, "sequence_parallel", self.config.sequence_parallel)
        mark_keep_in_fp32(self.weight)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=torch.float32)
            setattr(self.bias, "sequence_parallel", self.config.sequence_parallel)
            mark_keep_in_fp32(self.bias)

    def gating(self, input: torch.Tensor):
        """``Router.gating`` through :func:`beam_router_gating_linear`."""
        if self.weight.device.type == "cpu":
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == "cpu":
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        router_dtype = input.dtype
        if self.config.moe_router_dtype == "fp32":
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == "fp64":
            router_dtype = torch.float64
        return beam_router_gating_linear(input, self.weight, self.bias, router_dtype)


# ---------------------------------------------------------------------------
# Hybrid stack spec
# ---------------------------------------------------------------------------


def _beam_attention_spec(attention: ModuleSpec) -> ModuleSpec:
    """Rewrite Megatron-Core's TE ``SelfAttention`` spec to the Beam gate and SSMax core attention."""
    if not (isinstance(attention, ModuleSpec) and attention.module is SelfAttention):
        raise TypeError(f"Beam expects SelfAttention in the layer spec, got {attention!r}.")
    core_attention = attention.submodules.core_attention
    if core_attention is not TEDotProductAttention:
        raise TypeError(f"Beam expects TEDotProductAttention in the layer spec, got {core_attention!r}.")
    return dataclasses.replace(
        attention,
        module=BeamSelfAttention,
        submodules=dataclasses.replace(attention.submodules, core_attention=BeamTEDotProductAttention),
    )


def _beam_moe_spec(mlp):
    """Swap ``BeamTopKRouter`` into Megatron-Core's MoE spec."""
    if not (isinstance(mlp, functools.partial) and isinstance(mlp.func, type) and issubclass(mlp.func, MoELayer)):
        raise TypeError(f"Beam expects an MoELayer in the MoE layer spec, got {mlp!r}.")
    moe_submodules = mlp.keywords["submodules"]
    if moe_submodules.router is not TopKRouter:
        raise TypeError(f"Beam expects TopKRouter in the MoE spec, got {moe_submodules.router!r}.")
    return functools.partial(
        mlp.func,
        *mlp.args,
        **{**mlp.keywords, "submodules": dataclasses.replace(moe_submodules, router=BeamTopKRouter)},
    )


def beam_hybrid_stack_spec(config) -> ModuleSpec:
    """``HybridStack`` spec with Beam's attention (``*``), dense MLP (``-``) and MoE (``E``) layers.

    The layers reuse the attention, MLP and MoE submodules of Megatron-Core's Transformer
    Engine GPT layer spec, split into one sublayer per hybrid layer, with sandwich norms
    as TE norms. Accepts a ``TransformerConfig`` or a model config with a nested ``transformer``.
    """
    transformer_config = getattr(config, "transformer", config)
    validate_beam_config(transformer_config)
    if transformer_config.transformer_impl != "transformer_engine":
        raise NotImplementedError(
            f"Beam's hybrid stack is built from Transformer Engine layers; got "
            f"transformer_impl={transformer_config.transformer_impl!r}."
        )

    def _te_submodules(num_experts):
        return get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_experts,
            moe_grouped_gemm=transformer_config.moe_grouped_gemm if num_experts else False,
            qk_layernorm=transformer_config.qk_layernorm,
            qk_l2_norm=transformer_config.qk_l2_norm,
            use_te_activation_func=transformer_config.use_te_activation_func,
        )

    dense = _te_submodules(None)
    submodules = HybridStackSubmodules(
        attention_layer=ModuleSpec(
            module=BeamTransformerLayer,
            submodules=BeamTransformerLayerSubmodules(
                input_layernorm=dense.input_layernorm,
                self_attention=_beam_attention_spec(dense.self_attention),
                self_attn_bda=dense.self_attn_bda,
                post_self_attn_layernorm=TENorm,
            ),
        ),
        mlp_layer=ModuleSpec(
            module=BeamMLPLayer,
            submodules=BeamTransformerLayerSubmodules(
                pre_mlp_layernorm=dense.pre_mlp_layernorm,
                mlp=dense.mlp,
                mlp_bda=dense.mlp_bda,
                post_mlp_layernorm=TENorm,
                sharded_state_dict_keys_map=dense.sharded_state_dict_keys_map,
            ),
        ),
    )
    if transformer_config.num_moe_experts:
        moe = _te_submodules(transformer_config.num_moe_experts)
        submodules.moe_layer = ModuleSpec(
            module=BeamMoETransformerLayer,
            submodules=BeamTransformerLayerSubmodules(
                pre_mlp_layernorm=moe.pre_mlp_layernorm,
                mlp=_beam_moe_spec(moe.mlp),
                mlp_bda=moe.mlp_bda,
                post_mlp_layernorm=TENorm,
            ),
        )
    return ModuleSpec(module=HybridStack, submodules=submodules)
