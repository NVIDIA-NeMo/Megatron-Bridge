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

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import torch
from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as default_mamba_stack_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.bridge.utils.common_utils import print_rank_0

logger = logging.getLogger(__name__)


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Return the default Mamba stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Mamba stack specification from megatron.core
    """
    return default_mamba_stack_spec


def modelopt_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
    """Mamba stack specification for quantization with ModelOpt.

    Uses Norm instead of TENorm and ColumnParallelLinear/RowParallelLinear
    instead of TE layers to enable proper quantizer insertion by ModelOpt.

    Args:
        config: Mamba configuration object

    Returns:
        ModuleSpec: Module specification for quantization-ready Mamba stack
    """
    return get_mamba_stack_modelopt_spec(
        local_core_attention=False,
        remap_te_layernorm=False,
    )


def get_default_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
    """Determine the most appropriate Mamba stack specification based on configuration.

    Args:
        config: Mamba configuration object

    Returns:
        ModuleSpec: Appropriate module specification based on config
    """
    if config.restore_modelopt_state:
        return modelopt_mamba_stack_spec(config)
    else:
        return transformer_engine_mamba_stack_spec()


@dataclass
class MambaModelProvider(TransformerConfig, ModelProviderMixin[MCoreMambaModel]):
    """Configuration and provider for Megatron Core Mamba models.

    This class extends TransformerConfig with Mamba-specific parameters and
    provides a method to instantiate configured Mamba models.
    """

    # Model configuration
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = 2
    mamba_num_groups: int = 8
    num_attention_heads: int = 1
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: Optional[str] = None
    seq_length: int = 8192
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    attention_backend: AttnBackend = AttnBackend.flash
    deallocate_pipeline_outputs: bool = True
    bias_dropout_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    mamba_stack_spec: Union[ModuleSpec, Callable[[], ModuleSpec], Callable[["MambaModelProvider"], ModuleSpec]] = (
        get_default_mamba_stack_spec
    )
    mtp_mamba_stack_spec: Union[ModuleSpec, Callable[[], ModuleSpec], Callable[["MambaModelProvider"], ModuleSpec]] = get_default_mamba_stack_spec
    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False
    hf_model_id: Optional[str] = None
    _pg_collection: Optional[ProcessGroupCollection] = None
    
    # MTP
    mtp_num_layers: int = 0
    mtp_hybrid_override_pattern: Optional[str] = None
    keep_mtp_spec_in_bf16: bool = False

    # Additional parameters that might be needed
    # TODO(liding): double check these
    use_te_rng_tracker: bool = False
    enable_cuda_graph: bool = False
    cuda_graph_impl: str = "none"
    cuda_graph_scope: list[str] = field(default_factory=list)
    # TODO(liding): does not exist in MLM training branch. added here
    embedding_init_method_std: Optional[float] = None
    overlap_moe_expert_parallel_comm: bool = False

    """Optional HuggingFace model identifier associated with this provider."""

    # If True, restore the modelopt_state that contains quantization, sparsity, speculative decoding transformation state.
    # When resuming modelopt_state, we also change the mamba_stack_spec to use quantization-ready layers.
    restore_modelopt_state: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreMambaModel:
        """Configure and instantiate a Megatron Core Mamba model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreMambaModel: Configured Megatron Core Mamba model instance
        """
        mamba_stack_spec = self.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            # Check if the function accepts config parameter
            import inspect

            if len(inspect.signature(mamba_stack_spec).parameters) > 0:
                mamba_stack_spec = mamba_stack_spec(self)
            else:
                mamba_stack_spec = mamba_stack_spec()

        sep = Symbols.MTP_SEPARATOR
        # Derive the hybrid override pattern from the saved hybrid_override_pattern, 
        # mtp_hybrid_override_pattern, and mtp_num_layers.
        # This allows us to use a different mtp_num_layers than the one saved in the checkpoint.
        main_pattern = self.hybrid_override_pattern.split(sep)[0]
        self.hybrid_override_pattern = main_pattern + sep + sep.join([self.mtp_hybrid_override_pattern] * self.mtp_num_layers)

        if self.hybrid_override_pattern and sep in self.hybrid_override_pattern:
            parsed = parse_hybrid_pattern(self.hybrid_override_pattern)
            if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
                inferred_mtp_num_layers = parsed.mtp_num_depths
                if self.mtp_num_layers is None:
                    self.mtp_num_layers = inferred_mtp_num_layers
                elif self.mtp_num_layers != inferred_mtp_num_layers:
                    print(
                        f"--mtp-num-layers ({self.mtp_num_layers}) conflicts with "
                        f"MTP depth count ({inferred_mtp_num_layers}) in pattern '{self.hybrid_override_pattern}'. "
                        f"Using the inferred value ({inferred_mtp_num_layers})."
                    )
                    self.mtp_num_layers = inferred_mtp_num_layers

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in SSM/Mamaba "
            "models due to upstream MCore MambaModel API dependency"
        )

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            # mtp_hybrid_override_pattern=self.mtp_hybrid_override_pattern,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or is_pp_first_stage(self._pg_collection.pp),
            post_process=post_process or is_pp_last_stage(self._pg_collection.pp),
            pg_collection=self._pg_collection,
        )
