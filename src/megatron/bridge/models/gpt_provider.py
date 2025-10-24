# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import inspect
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.dot_product_attention import DotProductAttention as MCoreDotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils import fusions
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


logger = logging.getLogger(__name__)


def transformer_engine_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Create a Transformer Engine layer specification based on the provided config."""
    if "use_te_op_fuser" in inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters:
        kwargs = {"use_te_op_fuser": config.use_transformer_engine_op_fuser}
    else:
        kwargs = {}
    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        fp8=bool(config.num_moe_experts and (config.fp8 is not None)),
        **kwargs,
    )


def transformer_engine_full_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Create a full Transformer Engine layer specification with autocast support.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for full TE layers
    """
    from megatron.bridge.models.gpt_full_te_layer_autocast_spec import get_gpt_full_te_layer_autocast_spec

    return get_gpt_full_te_layer_autocast_spec(transformer_config=config)


def local_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Create a local layer specification without Transformer Engine.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for local implementation layers
    """
    return get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )


def quantization_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer specification for quantization with ModelOpt."""
    return get_gpt_modelopt_spec(
        config=config,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=True,
    )


def default_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability."""
    if config.restore_modelopt_state:
        return quantization_layer_spec(config)
    elif config.use_transformer_engine_full_layer_spec:
        return transformer_engine_full_layer_spec(config)
    else:
        return transformer_engine_layer_spec(config)


@dataclass
class GPTModelProvider(TransformerConfig, ModelProviderMixin[MCoreGPTModel]):
    """Configuration and provider for Megatron Core GPT models.

    This class extends TransformerConfig with GPT-specific parameters and
    provides a method to instantiate configured GPT models.
    """

    # Model configuration
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = True
    tp_only_amax_red: bool = False
    tp_comm_overlap_cfg: Optional[Union[str, dict[str, Any]]] = None
    """Config file when tp_comm_overlap is enabled."""

    use_transformer_engine_full_layer_spec: bool = False
    use_transformer_engine_op_fuser: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = default_layer_spec

    generation_config: Optional[Any] = None

    # This represents the unpadded vocab size
    # The padded vocab size is automatically calculated in the provide() method.
    vocab_size: Optional[int] = None
    # Set if the tokenizer provides the vocab size. In this case, the vocab size will be padded
    # Controls whether vocab size should be padded for tensor parallelism
    should_pad_vocab: bool = False

    # MoE / FP8
    num_moe_experts: Optional[int] = None
    moe_grouped_gemm: bool = False
    qk_layernorm: bool = False
    fp8: Optional[str] = None
    normalization: str = "LayerNorm"

    # Multi-token prediction
    mtp_enabled: bool = False

    # Additional parameters that might be needed
    init_model_with_meta_device: bool = False
    use_te_rng_tracker: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # Fusions
    masked_softmax_fusion: bool = field(default_factory=fusions.can_enable_masked_softmax_fusion)
    cross_entropy_loss_fusion: bool = True  # Generally beneficial, no specific dependencies
    gradient_accumulation_fusion: bool = field(default_factory=fusions.can_enable_gradient_accumulation_fusion)
    bias_activation_fusion: bool = False  # Disabled by default as it can interfere with certain architectures
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = field(default_factory=fusions.can_enable_bias_dropout_fusion)
    apply_rope_fusion: bool = field(default_factory=fusions.can_enable_apply_rope_fusion)

    # If True, restore the modelopt_state that contains quantization, sparsity, speculative decoding transformation state.
    # When resuming modelopt_state, we also change the transformer_layer_spec to `megatron.core.post_training.modelopt.gpt.model_specs` which is a combination of local spec + TEDotProductAttention.

    restore_modelopt_state: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Configure and instantiate a Megatron Core GPT model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        # Validate fusion configurations
        if not fusions.validate_rope_fusion_compatibility(self):
            self.apply_rope_fusion = False

        if self.cuda_graph_impl != "none":
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_layers_in_first_pipeline_stage", None)
            or getattr(self, "num_layers_in_last_pipeline_stage", None)
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (
            getattr(self, "pipeline_model_parallel_layout", None) is not None
        )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(torch.device, device="meta")

        # Check if mtp_block_spec parameter is supported
        kwargs = {}
        if "mtp_block_spec" in inspect.signature(MCoreGPTModel.__init__).parameters:
            kwargs["mtp_block_spec"] = mtp_block_spec(self, vp_stage=vp_stage)
        if self.attention_backend == AttnBackend.local:
            if hasattr(transformer_layer_spec, "submodules"):
                transformer_layer_spec.submodules.self_attention.submodules.core_attention = MCoreDotProductAttention
        with model_init_device_context():
            model = MCoreGPTModel(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=padded_vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process
                or parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=post_process
                or parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                vp_stage=vp_stage,
                **kwargs,
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if self.use_transformer_engine_full_layer_spec:
            # Copied from:
            # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        child.set_tensor_parallel_group(tp_group)

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_stream = torch.cuda.Stream()
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_context_parallel_group"):
                        child.set_context_parallel_group(
                            parallel_state.get_context_parallel_group(),
                            parallel_state.get_context_parallel_global_ranks(),
                            cp_stream,
                        )

        return model


def mtp_block_spec(config: "GPTModelProvider", vp_stage: Optional[int] = None) -> Optional[ModuleSpec]:
    """Pass in the MTP block spec if model has MTP layers.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: The MTP module specification
    """
    if getattr(config, "mtp_num_layers", None):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if isinstance(config.transformer_layer_spec, Callable):
            if "vp_stage" in inspect.signature(config.transformer_layer_spec).parameters:
                spec = config.transformer_layer_spec(config, vp_stage=vp_stage)
            else:
                spec = config.transformer_layer_spec(config)
        else:
            spec = config.transformer_layer_spec
        if hasattr(spec, "layer_specs") and len(spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            spec = default_layer_spec(config)
        return get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True, vp_stage=vp_stage)
    else:
        return None


@dataclass
class GPTProvider126M(GPTModelProvider):
    """Configuration for a 126M parameter GPT model.

    Predefined configuration for a small GPT model with 12 layers,
    768 hidden size, and 12 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True


@dataclass
class GPTProvider5B(GPTModelProvider):
    """Configuration for a 5B parameter GPT model.

    Predefined configuration for a medium-sized GPT model with 24 layers,
    4096 hidden size, and 32 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 24
    hidden_size: int = 4096
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True


@dataclass
class GPTProvider7B(GPTModelProvider):
    """Configuration for a 7B parameter GPT model.

    Predefined configuration for a medium-sized GPT model with 32 layers,
    4096 hidden size, and 32 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 10880
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True


@dataclass
class GPTProvider20B(GPTModelProvider):
    """Configuration for a 20B parameter GPT model.

    Predefined configuration for a large GPT model with 44 layers,
    6144 hidden size, and 48 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 44
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True


@dataclass
class GPTProvider40B(GPTModelProvider):
    """Configuration for a 40B parameter GPT model.

    Predefined configuration for a large GPT model with 48 layers,
    8192 hidden size, and 64 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 48
    hidden_size: int = 8192
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 64
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True


@dataclass
class GPTProvider175B(GPTModelProvider):
    """Configuration for a 175B parameter GPT model.

    Predefined configuration for a massive GPT model with 96 layers,
    12288 hidden size, and 96 attention heads.
    """

    seq_length: int = 2048
    num_layers: int = 96
    hidden_size: int = 12288
    ffn_hidden_size: int = 49152
    num_attention_heads: int = 96
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    layernorm_zero_centered_gamma: bool = True
