# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Native, conversion-free DiffusionGemma BF16 text SFT provider."""

from dataclasses import dataclass

import torch

from megatron.bridge.diffusion.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaModel,
    DiffusionGemmaSelfConditioning,
)
from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.gemma.modeling_gemma4 import HAVE_TE
from megatron.bridge.models.gemma.modules import extend_instance


@dataclass
class DiffusionGemmaModelProvider(Gemma4ModelProvider):
    """Shared encoder/decoder Gemma4 MoE; BF16 and TP=EP=PP=CP=1 only.

    Checkpoints use native Megatron sharded state. This provider does not load
    or convert Hugging Face checkpoints. The self-conditioning projection uses
    the shared-expert intermediate dimension (HF text_config.intermediate_size).
    """

    num_layers: int = 30
    hidden_size: int = 2816
    num_attention_heads: int = 16
    ffn_hidden_size: int = 2112
    vocab_size: int = 262144
    attention_k_eq_v: bool = True
    moe_aux_loss_coeff: float = 0.0
    moe_router_load_balancing_type: str = "none"
    scatter_embedding_sequence_parallel: bool = False
    freeze_router: bool = True

    def validate_diffusion_config(self) -> None:
        """Fail before allocating for execution modes bypassed by explicit passes."""
        unsupported = []
        for name in (
            "tensor_model_parallel_size",
            "expert_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_tensor_parallel_size",
        ):
            if getattr(self, name, 1) not in (None, 1):
                unsupported.append(f"{name}=1 required")
        for name in (
            "apply_query_key_layer_scaling",
            "qk_clip",
            "log_max_attention_logit",
            "use_inference_optimized_layers",
            "inference_fuse_tp_communication",
            "flash_decode",
        ):
            if getattr(self, name, False):
                unsupported.append(name)
        if self.fp32_residual_connection:
            unsupported.append("FP32 residual connections")
        if getattr(self, "mtp_num_layers", None):
            unsupported.append("multi-token prediction")
        if self.num_moe_experts is None:
            unsupported.append("dense layers")
        if self.sequence_parallel:
            unsupported.append("sequence_parallel")
        if self.recompute_granularity is not None:
            unsupported.append("activation recompute")
        if self.cpu_offloading or self.fine_grained_activation_offloading:
            unsupported.append("activation offloading")
        if self.transformer_impl != "transformer_engine" or not HAVE_TE:
            unsupported.append("Transformer Engine required for Gemma4 post-attention norm")
        if self.cuda_graph_impl != "none":
            unsupported.append("CUDA graphs")
        if not self.bf16 or self.fp16 or self.params_dtype != torch.bfloat16:
            unsupported.append("BF16 required")
        if self.fp8 or self.fp4:
            unsupported.append("FP8/FP4")
        if self.apply_rope_fusion:
            unsupported.append("fused RoPE with per-example positions")
        if self.attention_dropout or self.hidden_dropout:
            unsupported.append("nonzero dropout")
        if self.moe_aux_loss_coeff != 0 or self.moe_router_load_balancing_type != "none":
            unsupported.append("MoE auxiliary loss")
        if not self.share_embeddings_and_output_weights:
            unsupported.append("tied embeddings/output required")
        if self.softmax_scale != 1.0:
            unsupported.append("softmax_scale=1.0 required")
        if unsupported:
            raise ValueError("DiffusionGemma SFT does not support: " + ", ".join(unsupported))

    def finalize(self) -> None:
        """Validate explicit-pass constraints before Core computes derived fields."""
        self.validate_diffusion_config()
        super().finalize()

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> DiffusionGemmaModel:
        """Build one native stack and attach self-conditioning and encoder scalars."""
        self.validate_diffusion_config()
        if pre_process is False or post_process is False or vp_stage is not None:
            raise ValueError("DiffusionGemma requires a complete PP=1 model")
        self.finalize()
        model = super().provide(pre_process=True, post_process=True)
        extend_instance(model, DiffusionGemmaModel)
        device = model.embedding.word_embeddings.weight.device
        model.self_conditioning = DiffusionGemmaSelfConditioning(
            self.hidden_size, self.moe_shared_expert_intermediate_size, self.layernorm_epsilon, self.params_dtype
        ).to(device)
        with torch.no_grad():
            for projection in (
                model.self_conditioning.gate_proj,
                model.self_conditioning.up_proj,
                model.self_conditioning.down_proj,
            ):
                self.init_method(projection.weight)
        for layer in model.decoder.layers:
            layer.register_buffer("encoder_layer_scalar", torch.ones_like(layer.layer_scalar))
            if self.freeze_router:
                layer.mlp.router.requires_grad_(False)
        return model
