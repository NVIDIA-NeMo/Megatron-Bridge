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

"""Builder-backed configuration for the Inkling2 HybridModel integration."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class Inkling2TransformerConfig(TransformerConfig):
    """Transformer settings read by the Inkling2 attention, AttnRes, and MoE layers."""

    variable_seq_lengths: bool = True
    """Each layer passes the AttnRes bank packed with its partial block, so the width of the
    tensor between pipeline stages depends on the boundary and pipeline communication exchanges shapes."""

    hetereogenous_dist_checkpoint: bool = True
    """Save each layer's parameters under its own checkpoint key. The layers differ: full-attention
    layers carry three AttnRes queries and key and value input norms, and the first layer has no
    attention query, so the parameters cannot be stacked along a common layer axis."""

    inkling2_local_layers: tuple[int, ...] = ()
    """0-based indices of the sliding-window decoder blocks; the remaining blocks use full attention.
    Block ``i`` is Hybrid layers ``2 * i`` (attention) and ``2 * i + 1`` (MoE)."""

    inkling2_sliding_window: int = 1024
    """Local attention span, including the current token; also the local relative extent."""

    inkling2_rel_dim: int = 16
    """Width of the per-head query feature that produces relative logits."""

    inkling2_rel_extent: int = 1024
    """Number of causal distances with a learned relative logit in full-attention layers."""

    inkling2_qk_clip_threshold: float = 22.627416997969522
    inkling2_rel_clip_threshold: float = 8.0
    inkling2_value_norm: bool = False
    """Whether values pass through a weightless RMSNorm over the head dimension."""

    inkling2_log_scaling_n_floor: float = 128000.0
    inkling2_log_scaling_alpha: float = 0.1
    inkling2_sconv_kernel_size: int = 4
    inkling2_attn_res_block_size: int = 6
    """Number of decoder blocks whose outputs form one AttnRes block."""

    inkling2_attn_res_normalize_values: bool = False
    inkling2_mlp_gate_softcap: float = 0.0
    inkling2_mlp_up_softcap: float = 0.0
    inkling2_final_logit_softcap: float | None = None
    inkling2_unpadded_vocab_size: int | None = None
    """Logits at and above this token ID are masked out of the softmax."""

    inkling2_compile_ops: bool = False
    """Run the AttnRes mixtures, short convolutions, L2 clips, SiTU activations, and residual additions through
    ``torch.compile`` as fused kernels. Dynamo compiles a graph for every new call signature, so this suits training
    at fixed sequence shapes; leave it off for inference over changing sequence lengths."""


@dataclass(kw_only=True)
class Inkling2ModelConfig(BridgeHybridModelConfig):
    """Inkling2 HybridModel with an optional image encoder on the embedding stage.

    Each decoder block is an attention layer followed by a MoE layer, so the
    Hybrid pattern repeats ``*E`` and ``num_layers`` is twice the block count.
    """

    builder: ClassVar[str] = "megatron.bridge.models.inkling.inkling2_builder.Inkling2ModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Inkling2TransformerConfig

    vision_config: dict[str, Any] = field(default_factory=dict)
    """HF ``vision_config`` of the checkpoint; empty for a text-only model."""

    image_token_id: int | None = None
    """Placeholder token ID replaced by image features: ``<|unused_200054|>`` (200054) in the release."""

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the media placeholder IDs for collators."""
        if self.hf_model_text_only or not self.vision_config or self.image_token_id is None:
            return {}
        return {"images": self.image_token_id}

    def finalize(self) -> None:
        """Reject execution paths that the Inkling2 layers do not implement."""
        config = self.transformer
        if config.hierarchical_context_parallel_sizes:
            raise ValueError("Inkling2 gathers K and V over one CP group; hierarchical CP is not supported.")
        if config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Inkling2 does not support virtual pipeline parallelism yet.")
        if config.expert_tensor_parallel_size != 1:
            raise ValueError(
                "Inkling2 currently requires expert_tensor_parallel_size=1; its interleaved gate/up experts "
                "are validated only without expert tensor parallelism."
            )
        if config.transformer_impl != "transformer_engine":
            raise ValueError("Inkling2 uses Transformer Engine projections and grouped experts.")
        if config.mtp_num_layers:
            raise ValueError("Inkling2's draft model is a separate DSpark network, not Megatron MTP layers.")
        if config.fp8 or config.fp4:
            raise ValueError("Validate the BF16 Inkling2 path before enabling FP8 or FP4 training.")
        if config.cuda_graph_impl != "none" or config.overlap_moe_expert_parallel_comm:
            raise ValueError("Inkling2 layers are not integrated with CUDA graphs or EP overlap yet.")
        if config.hidden_dropout or config.attention_dropout or config.add_bias_linear or config.add_qkv_bias:
            raise ValueError("Inkling2 requires bias-free projections and zero dropout.")
        if config.moe_shared_expert_intermediate_size:
            raise ValueError("Inkling2 shared-expert sink routing is not implemented.")
        if config.enable_mhc_connections or config.wide_residual is not None:
            raise ValueError("Inkling2 uses its own Block Attention Residuals.")
        if config.num_layers % 2 or (self.hybrid_layer_pattern or "").replace("|", "") != "*E" * (
            config.num_layers // 2
        ):
            raise ValueError("Inkling2 requires an alternating attention/MoE Hybrid pattern ('*E' per block).")
        super().finalize()

    def _get_num_floating_point_operations_with_runtime_stats(
        self,
        *,
        batch_size: int,
        seqlen_sum: int | None,
        seqlen_squared_sum: int | None,
        cross_seqlen_sum: int | None = None,
        cross_seqlen_product_sum: int | None = None,
    ) -> float:
        """Return the forward and backward FLOPs of one global batch of text.

        Counts the matrix multiplications with the conventions of the standard GPT
        estimate: the query, key, value, relative-query, gate, sink, and output
        projections; the dynamic-convolution projections of full-attention layers;
        the relative logits; causal attention over the full context or the sliding
        window; the router and the routed experts; and the output layer.
        Normalization, short convolutions, and the AttnRes mixtures are elementwise
        or per-source reductions and, like norms in the standard estimate, are not
        counted; neither is the image encoder.

        Args:
            batch_size: Number of sequences in the global batch.
            seqlen_sum: Total tokens in the global batch, or ``None`` for full-length sequences.
            seqlen_squared_sum: Sum of squared subsequence lengths, or ``None`` for full-length sequences.
            cross_seqlen_sum: Unused; Inkling2 has no cross-attention.
            cross_seqlen_product_sum: Unused; Inkling2 has no cross-attention.

        Returns:
            Floating-point operations of one training step.
        """
        del cross_seqlen_sum, cross_seqlen_product_sum
        if seqlen_sum is None and seqlen_squared_sum is None:
            seqlen_sum = batch_size * self.seq_length
            seqlen_squared_sum = batch_size * self.seq_length**2
        elif seqlen_sum is None or seqlen_squared_sum is None:
            raise ValueError("Inkling2 FLOPs need both sequence-length sums or neither.")
        if seqlen_sum == 0:
            return 0.0

        t = self.transformer
        hidden = t.hidden_size
        heads = t.num_attention_heads
        kv_heads = t.num_query_groups or heads
        head_dim = t.kv_channels or hidden // heads
        query_width = heads * head_dim
        kv_width = kv_heads * head_dim
        num_blocks = t.num_layers // 2
        num_local = len(t.inkling2_local_layers)
        num_full = num_blocks - num_local
        window = t.inkling2_sliding_window

        # Average attended keys per token. Subsequence lengths enter through their
        # length-weighted mean, which is exact when the subsequences are equally long.
        mean_length = seqlen_squared_sum / seqlen_sum
        full_context = mean_length / 2
        if window < mean_length:
            local_context = window - window * (window - 1) / (2 * mean_length)
        else:
            local_context = full_context

        projections = hidden * (2 * query_width + 2 * kv_width + heads * t.inkling2_rel_dim + heads)
        projections += query_width * hidden
        relative_logits = heads * t.inkling2_rel_dim * (num_local * window + num_full * t.inkling2_rel_extent)
        dynamic_convolution = num_full * 2 * hidden * kv_heads * t.inkling2_sconv_kernel_size
        core_attention = 2 * query_width * (num_full * full_context + num_local * local_context)
        attention = num_blocks * projections + relative_logits + dynamic_convolution + core_attention

        # The SiTU experts have gate, up, and down projections.
        experts = num_blocks * (hidden * t.num_moe_experts + 3 * hidden * t.moe_ffn_hidden_size * t.moe_router_topk)
        logits = hidden * self.vocab_size
        # Two FLOPs per multiply-accumulate; the backward pass costs twice the forward.
        return 3 * 2 * seqlen_sum * (attention + experts + logits)
