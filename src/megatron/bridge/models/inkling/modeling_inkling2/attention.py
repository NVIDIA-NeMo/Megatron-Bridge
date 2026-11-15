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

"""Inkling2 attention: learned relative logits, a query-conditioned sink, and K/V short convolutions.

Megatron Core's attention modules cannot express this layer: scores add a
query-dependent relative logit per causal distance, a per-query sink logit
absorbs probability mass, Q/K/R are L2-clipped instead of normalized, K and V
pass through short convolutions (with per-token dynamic taps in full-attention
layers), and full-attention layers read Q, K, and V from three different
residual mixtures. The core attention runs on PyTorch FlexAttention without the
sink; the sink then scales each output row through the log-normalizer that
FlexAttention returns, which keeps the sink out of the attention kernels.

Under context parallelism each rank keeps its own queries and gathers K and V
from the whole CP group (see ``context_parallel``), so ``cp_comm_type`` does not
apply to this layer.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from torch import nn

from megatron.bridge.models.inkling.modeling_inkling2.context_parallel import (
    ContextParallelLayout,
    context_parallel_layout,
)
from megatron.bridge.models.inkling.modeling_inkling2.ops import (
    document_ids,
    document_positions,
    dynamic_short_conv,
    l2_clip,
    length_scaling_tau,
    static_short_conv,
)


_compiled_flex_attention = None
_block_mask_cache: "OrderedDict[tuple, object]" = OrderedDict()


def _flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex_attention


def _attend_with_sink(query, key, value, sink, *, score_mod, block_mask, out_dtype):
    """Run FlexAttention without the sink, then add the sink through the log-normalizer.

    A zero-value sink adds ``exp(sink)`` to each softmax denominator, which scales the
    output row by ``exp(lse) / (exp(lse) + exp(sink)) = sigmoid(lse - sink)``. Keeping the
    sink out of the kernel matters for speed: as an extra key block that every query block
    attends to, it serializes the backward pass over all query blocks.

    Args:
        query: ``[S_q, B, Hq, D]``.
        key: ``[S_kv, B, Hkv, D]``.
        value: ``[S_kv, B, Hkv, D]``.
        sink: Sink logits ``[B, Hq, S_q]``.
        score_mod: FlexAttention score modification (the relative logits).
        block_mask: FlexAttention block mask over ``S_q`` queries and ``S_kv`` keys.
        out_dtype: Dtype of the returned output.

    Returns:
        Attention output ``[S_q, B, Hq, D]``.
    """
    head_dim = query.shape[-1]
    q = query.permute(1, 2, 0, 3).contiguous()
    k = key.permute(1, 2, 0, 3).contiguous()
    v = value.permute(1, 2, 0, 3).contiguous()
    output, lse = _flex_attention()(
        q, k, v, score_mod=score_mod, block_mask=block_mask, enable_gqa=True, scale=head_dim**-0.5, return_lse=True
    )
    # Multiplying the kernel's output by the FP32 scale promotes to FP32 without keeping an
    # FP32 copy of the output for the backward pass; autograd saves the kernel's own output.
    output = output * torch.sigmoid(lse - sink.float()).unsqueeze(-1)
    return output.permute(2, 0, 1, 3).to(out_dtype)


def packed_cu_seqlens(packed_seq_params: PackedSeqParams | None) -> torch.Tensor | None:
    """Return the token-layout document boundaries of a packed batch, or ``None``."""
    if packed_seq_params is None:
        return None
    if packed_seq_params.qkv_format != "thd":
        raise ValueError(f"Inkling2 supports packed sequences only in THD format, got {packed_seq_params.qkv_format}")
    padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    return padded if padded is not None else packed_seq_params.cu_seqlens_q


def _block_mask(seq_len: int, window: int | None, docs: torch.Tensor, cache_key: tuple):
    from torch.nn.attention.flex_attention import create_block_mask

    key = (seq_len, window, str(docs.device), cache_key)
    mask = _block_mask_cache.get(key)
    if mask is not None:
        _block_mask_cache.move_to_end(key)
        return mask

    def mask_mod(b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        visible = (distance >= 0) & (docs[q_idx] == docs[torch.clamp(kv_idx, max=seq_len - 1)]) & (kv_idx < seq_len)
        if window is not None:
            visible = visible & (distance < window)
        return visible

    # The mask is identical for every batch row and head, so it broadcasts over both.
    mask = create_block_mask(mask_mod, None, None, seq_len, seq_len, device=docs.device)
    _block_mask_cache[key] = mask
    while len(_block_mask_cache) > 8:
        _block_mask_cache.popitem(last=False)
    return mask


def relative_sink_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rel_logits: torch.Tensor,
    sink: torch.Tensor,
    *,
    window: int | None,
    cu_seqlens: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Causal attention with relative logits and a zero-value sink.

    ``score(i, j, h) = q_i . k_j / sqrt(D) + rel_logits[h, i, i - j]`` for
    ``0 <= i - j < E``, and the sink contributes ``exp(sink[h, i])`` to the
    softmax denominator with a zero value, which equals scaling the ordinary
    output by ``sigmoid(lse - sink)``.

    Args:
        query: ``[S, B, Hq, D]``.
        key: ``[S, B, Hkv, D]``.
        value: ``[S, B, Hkv, D]``.
        rel_logits: Relative logits ``[B, Hq, S, E]``.
        sink: Sink logits ``[B, Hq, S]``.
        window: Local attention span including the current token, or ``None``.
        cu_seqlens: Document boundaries of a packed batch (``B == 1``).
        out_dtype: Dtype of the output; the dtype of ``query`` when ``None``.

    Returns:
        Attention output ``[S, B, Hq, D]``.
    """
    seq_len = query.shape[0]
    extent = rel_logits.shape[-1]
    out_dtype = query.dtype if out_dtype is None else out_dtype
    docs = document_ids(seq_len, query.device, cu_seqlens)
    if not query.is_cuda:
        return _eager_relative_sink_attention(
            query, key, value, rel_logits, sink, window=window, docs=docs, out_dtype=out_dtype
        )

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        bias = rel_logits[b, h, q_idx, torch.clamp(distance, 0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias.to(score.dtype), 0.0)

    cache_key = () if cu_seqlens is None else tuple(cu_seqlens.tolist())
    block_mask = _block_mask(seq_len, window, docs, cache_key)
    return _attend_with_sink(query, key, value, sink, score_mod=score_mod, block_mask=block_mask, out_dtype=out_dtype)


def _context_parallel_block_mask(layout: ContextParallelLayout, window: int | None, query_len: int):
    from torch.nn.attention.flex_attention import create_block_mask

    mask = layout.block_masks.get(window)
    if mask is not None:
        return mask
    kv_len = layout.total_tokens
    query_positions, query_docs, docs = layout.local_index, layout.local_documents, layout.documents

    def mask_mod(b, h, q_idx, kv_idx):
        distance = query_positions[q_idx] - kv_idx
        same_doc = query_docs[q_idx] == docs[torch.clamp(kv_idx, max=kv_len - 1)]
        visible = (distance >= 0) & same_doc & (kv_idx < kv_len)
        if window is not None:
            visible = visible & (distance < window)
        return visible

    mask = create_block_mask(mask_mod, None, None, query_len, kv_len, device=docs.device)
    layout.block_masks[window] = mask
    return mask


def context_parallel_relative_sink_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rel_logits: torch.Tensor,
    sink: torch.Tensor,
    *,
    window: int | None,
    layout: ContextParallelLayout,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """:func:`relative_sink_attention` for one context-parallel rank's queries over the full sequence.

    Args:
        query: This rank's queries ``[S_local, B, Hq, D]``, in local order.
        key: Keys of the full sequence ``[S, B, Hkv, D]``, in global order.
        value: Values of the full sequence ``[S, B, Hkv, D]``, in global order.
        rel_logits: Relative logits of this rank's queries ``[B, Hq, S_local, E]``.
        sink: Sink logits of this rank's queries ``[B, Hq, S_local]``.
        window: Local attention span including the current token, or ``None``.
        layout: This rank's token layout.
        out_dtype: Dtype of the output; the dtype of ``query`` when ``None``.

    Returns:
        Attention output for this rank's queries ``[S_local, B, Hq, D]``.
    """
    query_len = query.shape[0]
    kv_len = key.shape[0]
    if kv_len != layout.total_tokens:
        raise ValueError(f"Keys cover {kv_len} tokens, but the layout covers {layout.total_tokens}")
    extent = rel_logits.shape[-1]
    out_dtype = query.dtype if out_dtype is None else out_dtype
    query_positions = layout.local_index
    if not query.is_cuda:
        return _eager_relative_sink_attention(
            query,
            key,
            value,
            rel_logits,
            sink,
            window=window,
            docs=layout.documents,
            query_positions=query_positions,
            out_dtype=out_dtype,
        )

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = query_positions[q_idx] - kv_idx
        bias = rel_logits[b, h, q_idx, torch.clamp(distance, 0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias.to(score.dtype), 0.0)

    block_mask = _context_parallel_block_mask(layout, window, query_len)
    return _attend_with_sink(query, key, value, sink, score_mod=score_mod, block_mask=block_mask, out_dtype=out_dtype)


def _eager_relative_sink_attention(
    query, key, value, rel_logits, sink, *, window, docs, query_positions=None, out_dtype=None
):
    """Reference implementation of the relative-sink attention for CPU inputs.

    ``docs`` labels every key; ``query_positions`` gives each query's index into the keys
    (all of them, in order, when ``None``). The output has ``out_dtype``, or the dtype of
    ``query`` when ``None``.
    """
    query_len, _, heads, head_dim = query.shape
    kv_len = key.shape[0]
    repeat = heads // key.shape[2]
    extent = rel_logits.shape[-1]
    q = query.float().permute(1, 2, 0, 3)
    k = key.float().repeat_interleave(repeat, dim=2).permute(1, 2, 0, 3)
    v = value.float().repeat_interleave(repeat, dim=2).permute(1, 2, 0, 3)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * head_dim**-0.5
    if query_positions is None:
        query_positions = torch.arange(query_len, device=query.device)
    distance = query_positions[:, None] - torch.arange(kv_len, device=query.device)[None, :]
    index = distance.clamp(0, extent - 1).expand(*rel_logits.shape[:2], query_len, kv_len)
    bias = torch.gather(rel_logits.float(), 3, index)
    scores = scores + torch.where((distance >= 0) & (distance < extent), bias, torch.zeros_like(bias))
    visible = (distance >= 0) & (docs[query_positions][:, None] == docs[None, :])
    if window is not None:
        visible &= distance < window
    scores = scores.masked_fill(~visible, float("-inf"))
    probabilities = torch.softmax(torch.cat((scores, sink.float().unsqueeze(-1)), dim=-1), dim=-1)[..., :kv_len]
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, v).permute(2, 0, 1, 3)
    return output.to(query.dtype if out_dtype is None else out_dtype)


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output, bias = module(inputs)
    if bias is not None:
        raise ValueError(f"Inkling2 projections are bias-free, got a bias from {type(module).__name__}")
    return output


def _tp_sharded_parameter(tensor: torch.Tensor, partition_dim: int) -> nn.Parameter:
    parameter = nn.Parameter(tensor)
    set_tensor_model_parallel_attributes(parameter, True, partition_dim, 1)
    return parameter


class Inkling2Attention(MegatronModule):
    """One Inkling2 attention sublayer, including its output short convolution.

    The module receives the normalized Q, K, and V sources (one shared tensor in
    local layers). Under sequence parallelism it gathers them, computes the
    projections for its tensor-parallel heads on the full sequence, and returns
    its sequence-parallel slice of the output.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config=config)
        del cp_comm_type, pp_layer_offset, name
        if is_mtp_layer:
            raise ValueError("Inkling2 attention does not support MTP layers")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        # Hybrid layer 2 * i holds the attention of decoder block i.
        self.layer_idx = (layer_number - 1) // 2
        self.is_local = self.layer_idx in config.inkling2_local_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        self.head_dim = config.kv_channels
        if self.num_heads % self.tp_size or self.num_kv_heads % self.tp_size:
            raise ValueError(
                f"TP size {self.tp_size} must divide {self.num_heads} heads and {self.num_kv_heads} KV heads"
            )
        self.local_heads = self.num_heads // self.tp_size
        self.local_kv_heads = self.num_kv_heads // self.tp_size
        self.rel_dim = config.inkling2_rel_dim
        self.window = config.inkling2_sliding_window if self.is_local else None
        self.rel_extent = config.inkling2_sliding_window if self.is_local else config.inkling2_rel_extent
        self.conv_width = config.inkling2_sconv_kernel_size

        hidden = config.hidden_size
        self.linear_q = self._column_linear(hidden, self.num_heads * self.head_dim)
        self.linear_k = self._column_linear(hidden, self.num_kv_heads * self.head_dim)
        self.linear_v = self._column_linear(hidden, self.num_kv_heads * self.head_dim)
        self.linear_r = self._column_linear(hidden, self.num_heads * self.rel_dim)
        self.linear_gate = self._column_linear(hidden, self.num_heads * self.head_dim)
        self.linear_sink = self._column_linear(hidden, self.num_heads)
        self.linear_proj = TERowParallelLinear(
            self.num_heads * self.head_dim,
            hidden,
            config=self.linear_config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

        device = torch.cuda.current_device()
        dtype = config.params_dtype
        kv_width = self.local_kv_heads * self.head_dim
        self.q_clip_scale = _tp_sharded_parameter(
            torch.ones(self.local_heads, self.head_dim, device=device, dtype=dtype), 0
        )
        self.r_clip_scale = _tp_sharded_parameter(
            torch.ones(self.local_heads, self.rel_dim, device=device, dtype=dtype), 0
        )
        # Every TP rank uses the full table for its own heads, so gradients add across ranks.
        self.rel_proj = nn.Parameter(torch.zeros(self.rel_dim, self.rel_extent, device=device, dtype=dtype))
        self.rel_proj.sum_gradients_across_tp_domain = True
        self._tp_axes = {"q_clip_scale": 0, "r_clip_scale": 0}
        if self.is_local:
            for prefix in ("k", "v"):
                setattr(
                    self,
                    f"{prefix}_conv_weight",
                    _tp_sharded_parameter(torch.zeros(kv_width, 1, self.conv_width, device=device, dtype=dtype), 0),
                )
                self._tp_axes[f"{prefix}_conv_weight"] = 0
        else:
            for prefix in ("k", "v"):
                setattr(
                    self,
                    f"{prefix}_conv_static",
                    _tp_sharded_parameter(torch.zeros(self.conv_width, kv_width, device=device, dtype=dtype), 1),
                )
                setattr(
                    self,
                    f"{prefix}_conv_proj",
                    _tp_sharded_parameter(
                        torch.zeros(self.local_kv_heads * self.conv_width, hidden, device=device, dtype=dtype), 0
                    ),
                )
                self._tp_axes[f"{prefix}_conv_static"] = 1
                self._tp_axes[f"{prefix}_conv_proj"] = 0
        # Every TP rank convolves the identical all-reduced output, so its gradients already agree.
        self.out_conv_weight = nn.Parameter(torch.zeros(hidden, 1, self.conv_width, device=device, dtype=dtype))

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Declare the tensor-parallel axes of the directly owned parameters."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        direct = {name: getattr(self, name) for name in self._tp_axes}
        sharded_state_dict.update(
            make_sharded_tensors_for_checkpoint(
                direct,
                prefix,
                dict(self._tp_axes),
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded_state_dict

    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.sequence_parallel:
            return tensor
        return gather_from_sequence_parallel_region(tensor, tensor_parallel_output_grad=False, group=self.tp_group)

    def forward(
        self,
        query_source: torch.Tensor,
        key_source: torch.Tensor | None = None,
        value_source: torch.Tensor | None = None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        """Compute the attention sublayer output, including its output short convolution.

        Args:
            query_source: Normalized Q-side source ``[S(/TP), B, H]``; it also
                feeds R, the output gate, and the sink.
            key_source: Normalized K-side source; defaults to ``query_source``.
            value_source: Normalized V-side source; defaults to ``query_source``.
            packed_seq_params: Document boundaries of a packed (THD) batch; under
                context parallelism, the boundaries of the full sequence.

        Returns:
            Output of shape ``[S(/TP), B, H]``.
        """
        key_source = query_source if key_source is None else key_source
        value_source = query_source if value_source is None else value_source
        gathered = {}
        for source in (query_source, key_source, value_source):
            if id(source) not in gathered:
                gathered[id(source)] = self._gather(source)
        xq, xk, xv = gathered[id(query_source)], gathered[id(key_source)], gathered[id(value_source)]
        seq_len, batch = xq.shape[:2]
        cu_seqlens = packed_cu_seqlens(packed_seq_params)
        layout = context_parallel_layout(
            self.cp_group,
            local_tokens=seq_len,
            batch=batch,
            device=xq.device,
            width=self.conv_width,
            cu_seqlens=cu_seqlens,
        )
        if layout is None:
            positions = document_positions(seq_len, batch, xq.device, cu_seqlens)
        else:
            positions = layout.local_positions

        compiled = self.config.inkling2_compile_ops
        query = _linear(self.linear_q, xq)
        key = _linear(self.linear_k, xk)
        value = _linear(self.linear_v, xv)
        rel = _linear(self.linear_r, xq)
        gate = _linear(self.linear_gate, xq)
        sink = _linear(self.linear_sink, xq)

        if self.is_local:
            key = static_short_conv(key, self.k_conv_weight.squeeze(1), positions, halo=layout, compiled=compiled)
            value = static_short_conv(value, self.v_conv_weight.squeeze(1), positions, halo=layout, compiled=compiled)
            tau = None
        else:
            key = self._dynamic_conv(key, xk, self.k_conv_static, self.k_conv_proj, positions, layout)
            value = self._dynamic_conv(value, xv, self.v_conv_static, self.v_conv_proj, positions, layout)
            tau = length_scaling_tau(
                positions, self.config.inkling2_log_scaling_n_floor, self.config.inkling2_log_scaling_alpha
            )

        query = l2_clip(
            query.view(seq_len, batch, self.local_heads, self.head_dim),
            self.config.inkling2_qk_clip_threshold,
            scale=self.q_clip_scale,
            output_scale=tau,
            compiled=compiled,
        )
        key = l2_clip(
            key.view(seq_len, batch, self.local_kv_heads, self.head_dim),
            self.config.inkling2_qk_clip_threshold,
            compiled=compiled,
        )
        value = value.view(seq_len, batch, self.local_kv_heads, self.head_dim)
        if self.config.inkling2_value_norm:
            value = F.rms_norm(value.float(), (self.head_dim,), eps=self.config.layernorm_epsilon).to(value.dtype)
        rel = l2_clip(
            rel.view(seq_len, batch, self.local_heads, self.rel_dim),
            self.config.inkling2_rel_clip_threshold,
            scale=self.r_clip_scale,
            output_scale=tau,
            compiled=compiled,
        )
        rel_logits = torch.einsum("sbhd,de->bhse", rel, self.rel_proj.to(rel.dtype))
        sink = sink.view(seq_len, batch, self.local_heads).permute(1, 2, 0).float()

        # FP32 attention output: the sink scale and the output gate round to the activation dtype once.
        if layout is None:
            output = relative_sink_attention(
                query, key, value, rel_logits, sink, window=self.window, cu_seqlens=cu_seqlens, out_dtype=torch.float32
            )
        else:
            output = context_parallel_relative_sink_attention(
                query,
                layout.gather(key),
                layout.gather(value),
                rel_logits,
                sink,
                window=self.window,
                layout=layout,
                out_dtype=torch.float32,
            )
        gate = gate.view(seq_len, batch, self.local_heads, self.head_dim)
        output = (output * torch.sigmoid(gate.float())).to(query.dtype).flatten(2)
        output = _linear(self.linear_proj, output)
        output = static_short_conv(output, self.out_conv_weight.squeeze(1), positions, halo=layout, compiled=compiled)
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output

    def _dynamic_conv(
        self,
        inputs: torch.Tensor,
        source: torch.Tensor,
        static_weight: torch.Tensor,
        projection: torch.Tensor,
        positions: torch.Tensor,
        layout: ContextParallelLayout | None,
    ) -> torch.Tensor:
        # The projection reads the full-width source for this rank's KV heads only,
        # so its input gradient must be all-reduced across the TP ranks.
        source = copy_to_tensor_model_parallel_region(source, group=self.tp_group)
        dynamic = F.linear(source, projection.to(source.dtype))
        dynamic = dynamic.view(*dynamic.shape[:-1], self.local_kv_heads, self.conv_width)
        return dynamic_short_conv(
            inputs, static_weight, dynamic, positions, halo=layout, compiled=self.config.inkling2_compile_ops
        )
