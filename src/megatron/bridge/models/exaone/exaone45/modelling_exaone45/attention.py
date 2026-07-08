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

from __future__ import annotations

import torch
from einops import rearrange
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    apply_rotary_pos_emb,
)
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer import attention as mcore_attention
from megatron.core.transformer.attention import (
    HAVE_FA3,
    HAVE_FA4,
    HAVE_FUSED_QKV_ROPE,
    BaseInferenceContext,
    IdentityOp,
    SelfAttention,
    deprecate_inference_params,
    is_fa_min_version,
    is_using_quantization_scales,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.typed_torch import apply_module
from torch import Tensor


def _apply_vision_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    *,
    config,
    cu_seqlens: Tensor | None,
    mscale: float,
    cp_group: torch.distributed.ProcessGroup | None,
) -> Tensor:
    """Apply Exaone4.5 vision RoPE.

    The vision encoder passes packed THD metadata for attention, but its rotary frequencies are
    already laid out per visual token. The old Megatron-Core patch handled this with the
    ``is_vision_encoder=True`` branch; keep that behavior local to Exaone4.5.
    """
    if cu_seqlens is None:
        return apply_rotary_pos_emb(
            t,
            freqs,
            config=config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=cp_group,
        )

    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1),
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        mscale=mscale,
    ).squeeze()


class Exaone45SelfAttention(SelfAttention):
    """Self-attention with Exaone4.5 vision-encoder RoPE handling."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Perform a forward pass through the attention module.

        Args:
            hidden_states: Hidden states.
            attention_mask: Attention mask.
            key_value_states: Key/value states for cross attention.
            inference_context: Inference context that manages KV cache.
            rotary_pos_emb: Rotary embedding tensor or query/key rotary embedding tensors.
            rotary_pos_cos: Rotary embedding cosine.
            rotary_pos_sin: Rotary embedding sine.
            rotary_pos_cos_sin: Combined rotary embedding cosine and sine, used by dynamic batching.
            attention_bias: Attention bias.
            packed_seq_params: Parameters used for THD format.
            sequence_len_offset: Sequence length offset used for inference CUDA graphs.
            inference_params: Deprecated alias for ``inference_context``.

        Returns:
            Attention output and optional bias.
        """
        original_cp_group = self.pg_collection.cp
        if packed_seq_params is not None and packed_seq_params.local_cp_size is not None:
            assert packed_seq_params.cp_group is not None, "cp_group must be set in dynamic-cp mode"
            self.pg_collection.cp = packed_seq_params.cp_group

        # Check if we need to skip RoPE. ``no_rope_freq`` is 0-indexed and layer_number is 1-indexed.
        no_rope = self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA4 or HAVE_FA3 or is_fa_min_version("2.7.3"), (
                "flash attn verion v2.7.3 and above is required for dynamic batching."
            )

        is_inference_mode = InferenceMode.is_active()
        is_using_flash_decode = is_inference_mode and self.config.flash_decode
        is_using_flashinfer_rope = (
            is_inference_mode
            and inference_context is not None
            and not inference_context.is_static_batching()
            and inference_context.use_flashinfer_fused_rope
        )
        if is_using_flash_decode or is_using_flashinfer_rope:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it is not already.
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        nvtx_range_push(suffix="qkv")
        split_qkv = (self.attention_type == "cross") or not all(
            [
                not self.config.test_mode,
                self.config.fused_single_qkv_rope,
                inference_context is None,
                packed_seq_params is None,
                (rotary_pos_emb is not None and rotary_pos_emb[0] is not None and rotary_pos_emb[1] is not None),
                not self.config.flash_decode,
                HAVE_FUSED_QKV_ROPE,
                self.q_layernorm is None or isinstance(self.q_layernorm, IdentityOp),
                self.k_layernorm is None or isinstance(self.k_layernorm, IdentityOp),
            ]
        )
        if self.attention_type != "cross":
            assert not (self.config.fused_single_qkv_rope and split_qkv), (
                "fused_single_qkv_rope requested but not available/supported for the config."
            )

        with off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear") as hidden_states:
            qkv_output = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                split_qkv=split_qkv,
                output_gate=self.config.attention_output_gate,
            )
        if self.offload_qkv_linear:
            qkv_output = off_interface.group_commit(qkv_output, name="qkv_linear", forced_released_tensors=[])
        attn_mask_type = self.attn_mask_type
        block_table = None
        gate = None
        if split_qkv:
            if self.config.attention_output_gate:
                query, key, value, gate = qkv_output
            else:
                query, key, value = qkv_output
            mixed_qkv = qkv_split_arg_list = None
        else:
            assert not self.config.attention_output_gate, (
                "attention_output_gate is not supported for unsplit mixed_qkv tensor."
            )
            mixed_qkv, qkv_split_arg_list = qkv_output
        nvtx_range_pop(suffix="qkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        in_decode_mode = (
            inference_context is not None and inference_context.is_decode_only() and InferenceMode.is_active()
        )

        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            if gate is not None:
                context_layer = self._apply_output_gate(context_layer, gate)
            output, bias = apply_module(self.linear_proj)(context_layer)
            self.pg_collection.cp = original_cp_group
            return output, bias

        if in_decode_mode and self.config.cuda_graph_impl == "local" and inference_context.is_static_batching():
            raise ValueError("CUDA graphs must use flash decode with static batching!")

        if split_qkv:
            query, key, value, rotary_pos_emb, attn_mask_type, block_table = self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                rotary_pos_cos_sin,
                sequence_len_offset,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and (not self.config.flash_decode or inference_context is None):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
                cu_seqlens_q = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None
                    else packed_seq_params.cu_seqlens_q
                )
                cu_seqlens_kv = (
                    packed_seq_params.cu_seqlens_kv_padded
                    if packed_seq_params.cu_seqlens_kv_padded is not None
                    else packed_seq_params.cu_seqlens_kv
                )
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            mscale = _yarn_get_concentration_factor_from_config(self.config)
            if split_qkv:
                if q_pos_emb is not None:
                    if inference_context is None or inference_context.is_static_batching():
                        query = _apply_vision_rotary_pos_emb(
                            query,
                            q_pos_emb,
                            config=self.config,
                            cu_seqlens=cu_seqlens_q,
                            mscale=mscale,
                            cp_group=self.pg_collection.cp,
                        )
                    else:
                        query = inference_context.apply_rotary_emb_query(
                            query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                        )
                if k_pos_emb is not None:
                    key = _apply_vision_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                    )
            else:
                query, key, value = mcore_attention.apply_fused_qkv_rotary_pos_emb(
                    mixed_qkv, q_pos_emb, k_pos_emb, qkv_split_arg_list
                )
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # core attention computation
        # ==================================
        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                with off_interface(self.offload_core_attention and self.training, query, "core_attn") as query:
                    core_attn_out = apply_module(self.core_attention)(
                        query,
                        key,
                        value,
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )
            else:
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                    inference_context.is_decode_only(),
                )
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")

                # Clear outputs for padding tokens when using quantization scales to avoid
                # corrupting amax calculations.
                if is_using_quantization_scales(self.config):
                    core_attn_out[inference_context.padding_slice] = 0.0

            if self.offload_core_attention and self.training:
                core_attn_out = off_interface.group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        if gate is not None:
            nvtx_range_push(suffix="output_gate")
            core_attn_out = self._apply_output_gate(core_attn_out, gate)
            nvtx_range_pop(suffix="output_gate")

        # =================
        # Output. [sq, b, h]
        # =================
        nvtx_range_push(suffix="linear_proj")
        with off_interface(self.offload_attn_proj, core_attn_out, "attn_proj") as core_attn_out:
            output, bias = apply_module(self.linear_proj)(core_attn_out)
        if self.offload_attn_proj:
            output = off_interface.group_commit(output, name="attn_proj", forced_released_tensors=[core_attn_out])
        nvtx_range_pop(suffix="linear_proj")

        self.pg_collection.cp = original_cp_group
        return output, bias
