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

"""Gemma 4 text-only model providers.

Gemma4DenseProvider: Dense (E2B, E4B, and 31B) — builds GPTModel with local spec,
    dual RoPE, PLE, and shared KV.
Gemma4ModelProvider: MoE (26B-A4B and similar) — extends GPTModelProvider
    with TE-based layer spec, dual RoPE, and softcapped output layer.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from megatron.core.activations import fast_gelu
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gemma.gemma3_provider import Gemma3LanguageModelEmbedding
from megatron.bridge.models.gemma.modeling_gemma4 import (
    HAVE_TE,
    Gemma4DenseRotaryEmbedding,
    Gemma4OutputLayer,
    Gemma4RotaryEmbedding,
    _attach_ple_modules,
    _install_ple_forward,
    _install_tied_kv,
    gemma4_block_spec,
    get_gemma4_layer_spec,
    wire_gemma4_kv_sharing,
)
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


def _validate_gemma4_moe_orchestration(provider: GPTModelProvider) -> None:
    """Reject MCore execution modes bypassed by Gemma 4's custom MoE forward."""
    unsupported = []
    if provider.transformer_impl == "inference_optimized":
        unsupported.append("transformer_impl='inference_optimized'")
    if provider.cuda_graph_impl != "none":
        unsupported.append(f"cuda_graph_impl={provider.cuda_graph_impl!r}")
    if provider.moe_shared_expert_overlap:
        unsupported.append("moe_shared_expert_overlap=True")
    if provider.mlp_chunks_for_prefill > 1:
        unsupported.append("mlp_chunks_for_prefill > 1")
    if provider.mlp_chunks_for_training > 1:
        unsupported.append("mlp_chunks_for_training > 1")
    if provider.inference_fuse_tp_communication:
        unsupported.append("inference_fuse_tp_communication=True")
    recompute_modules = set(provider.recompute_modules or [])
    if provider.recompute_granularity == "selective" and "layernorm" in recompute_modules:
        unsupported.append("selective layernorm recompute")
    offload_modules = set(provider.offload_modules or [])
    if provider.fine_grained_activation_offloading and "mlp_norm" in offload_modules:
        unsupported.append("MLP norm activation offloading")
    if unsupported:
        raise ValueError(
            "Gemma 4 MoE's separate router/shared/routed inputs do not yet support: " + ", ".join(unsupported)
        )


def _install_gemma4_dense_load_state_aliases(model: torch.nn.Module) -> None:
    """Translate Gemma4 Dense checkpoint attention aliases before load_state_dict.

    Gemma4 Dense saves sliding/global attention tensors under separate names in
    dist-checkpoints because the two layer types have different sharded shapes.
    After dist-checkpoint load materializes a regular state_dict, PyTorch module
    loading expects the real module attribute name, ``self_attention``.
    """

    if getattr(model, "_gemma4_dense_load_state_aliases_installed", False):
        return

    def _load_state_dict_pre_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        for key in list(state_dict.keys()):
            if prefix and not key.startswith(prefix):
                continue

            canonical_key = None
            if ".self_attention_sliding." in key:
                canonical_key = key.replace(".self_attention_sliding.", ".self_attention.")
            elif ".self_attention_global." in key:
                canonical_key = key.replace(".self_attention_global.", ".self_attention.")

            if canonical_key is None:
                continue

            state_dict.setdefault(canonical_key, state_dict[key])
            state_dict.pop(key)

    model._register_load_state_dict_pre_hook(_load_state_dict_pre_hook)
    model._gemma4_dense_load_state_aliases_installed = True


# ---------------------------------------------------------------------------
# Dense provider
# ---------------------------------------------------------------------------


@dataclass
class Gemma4DenseProvider(GPTModelProvider):
    """Gemma 4 dense E2B, E4B, and 31B provider for clean Megatron-Core.

    All Gemma4-specific settings are encoded here as dataclass fields so that
    no Gemma4-specific CLI arguments are required.
    """

    num_layers: int = 42
    hidden_size: int = 2560
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 8
    num_query_groups: int = 2
    kv_channels: int = 256
    seq_length: int = 131072
    vocab_size: int = 262143
    make_vocab_size_divisible_by: int = 128

    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    # fast_gelu == gelu(x, approximate='tanh'), already registered in ACTIVATION_FUNC_MAP
    # as "gelu_pytorch_tanh" — required for HF export to recognise the activation.
    activation_func: Callable = field(default_factory=lambda: fast_gelu)

    scale_embeddings_by_hidden_size: bool = True
    share_embeddings_and_output_weights: bool = True
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    window_size: Optional[Tuple[int, int]] = (511, 0)
    window_attn_skip_freq: Union[int, List[int]] = 6

    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16
    use_cpu_initialization: bool = False

    global_kv_channels: int = 512
    num_global_query_groups: int = 2
    attention_k_eq_v: bool = False
    sliding_window_rope_base: float = 10000.0
    full_attention_rope_base: float = 1000000.0
    full_attention_rope_partial_factor: float = 0.25
    num_kv_shared_layers: int = 18
    use_double_wide_mlp: bool = False
    per_layer_embed_vocab_size: int = 262144
    per_layer_embed_dim: int = 256
    final_logit_softcapping: float | None = 30.0

    num_moe_experts: Optional[int] = None
    moe_router_topk: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None

    def finalize(self) -> None:
        if self.use_double_wide_mlp:
            self.hetereogenous_dist_checkpoint = True
        super().finalize()
        self._gemma4_dense_finalized = True

    def _ensure_finalized(self) -> None:
        if not getattr(self, "_gemma4_dense_finalized", False):
            self.finalize()

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> "torch.nn.Module":
        if vp_stage is not None or getattr(self, "pipeline_model_parallel_size", 1) != 1:
            raise NotImplementedError("Gemma4DenseProvider currently supports PP=1 only.")

        return self.build(
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
        )

    def build(
        self,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> "torch.nn.Module":
        """Build a Gemma-4 Dense GPTModel and attach Bridge-specific components."""
        from megatron.core.models.gpt import GPTModel

        self._ensure_finalized()
        config = self

        padded_vocab = (
            (self.vocab_size + self.make_vocab_size_divisible_by - 1)
            // self.make_vocab_size_divisible_by
            * self.make_vocab_size_divisible_by
        )

        dual_rope_attrs = {
            "sliding_window_rope_base": self.sliding_window_rope_base,
            "full_attention_rope_base": self.full_attention_rope_base,
            "full_attention_rope_partial_factor": self.full_attention_rope_partial_factor,
        }
        for attr in dual_rope_attrs:
            setattr(config, attr, None)
        try:
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gemma4_layer_spec(config),
                vocab_size=padded_vocab,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=getattr(self, "_pg_collection", None),
            )
        finally:
            for attr, value in dual_rope_attrs.items():
                setattr(config, attr, value)

        model.rotary_pos_emb = Gemma4DenseRotaryEmbedding(config)

        if pre_process:
            _attach_ple_modules(model, config, self)
        wire_gemma4_kv_sharing(model)
        _install_ple_forward(model)
        _install_gemma4_dense_load_state_aliases(model)

        if (
            hasattr(model, "output_layer")
            and self.final_logit_softcapping is not None
            and not isinstance(model.output_layer, Gemma4OutputLayer)
        ):
            extend_instance(model.output_layer, Gemma4OutputLayer)

        return model


# ---------------------------------------------------------------------------
# MoE provider
# ---------------------------------------------------------------------------


@dataclass
class Gemma4ModelProvider(GPTModelProvider):
    """Configuration and provider for Megatron Core Gemma 4 MoE models."""

    seq_length: int = 262_144

    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)
    share_embeddings_and_output_weights: bool = True

    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1e-6

    kv_channels: int = 256
    num_query_groups: int = 8
    window_size: int = 1024
    interleaved_attn_pattern: tuple = (5, 1)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_backend: AttnBackend = AttnBackend.auto
    softmax_scale: float = 1.0
    qk_layernorm: bool = True
    attention_k_eq_v: bool = False

    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    global_rotary_percent: float = 0.25

    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = fast_gelu

    num_moe_experts: Optional[int] = 128
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 704
    moe_shared_expert_intermediate_size: int = 2112
    moe_shared_expert_overlap: bool = False
    moe_shared_expert_gate: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_pre_softmax: bool = True
    moe_router_dtype: str = "fp32"
    moe_aux_loss_coeff: float = 0.001
    moe_permute_fusion: bool = True
    moe_layer_freq: int = 1

    final_logit_softcapping: float | None = 30.0

    flash_decode: bool = False
    transformer_layer_spec: Union[Callable, object] = field(
        default_factory=lambda: partial(gemma4_block_spec, use_transformer_engine=HAVE_TE)
    )
    scatter_embedding_sequence_parallel: bool = True

    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def finalize(self) -> None:
        _validate_gemma4_moe_orchestration(self)
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma 4 MoE model."""
        rotary_base_local, rotary_base_global = self.rotary_base
        self.rotary_base = rotary_base_local
        try:
            model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        finally:
            self.rotary_base = (rotary_base_local, rotary_base_global)

        if hasattr(model, "embedding"):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )

        model.rotary_pos_emb = Gemma4RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
            global_kv_channels=self.global_head_dim,
            global_rotary_percent=self.global_rotary_percent,
        )

        if (
            hasattr(model, "output_layer")
            and self.final_logit_softcapping is not None
            and not isinstance(model.output_layer, Gemma4OutputLayer)
        ):
            extend_instance(model.output_layer, Gemma4OutputLayer)

        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()

        _install_tied_kv(model, self)

        return model
