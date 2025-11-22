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
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Model, FalconH1Config
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer_specs import falconh1_stack_spec as default_falconh1_stack_spec
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


logger = logging.getLogger(__name__)


def get_default_falconh1_stack_spec():
    """Return the default FalconH1 stack spec.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default FalconH1 stack specification
    """
    return default_falconh1_stack_spec


@dataclass
class FalconH1ModelProvider(FalconH1Config, ModelProviderMixin[FalconH1Model]):
    """Configuration and provider for FalconH1 hybrid models.

    This class extends FalconH1Config with model instantiation capabilities
    and provides a method to create configured FalconH1 models.
    """

    # Model configuration
    seq_length: int = 4096
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True

    # Layer configuration - defaults to uniform FalconH1 layers
    hybrid_attention_ratio: float = 0.0  # Not used when falconh1_ratio = 1.0
    hybrid_mlp_ratio: float = 0.0  # Not used when falconh1_ratio = 1.0
    falconh1_ratio: float = 1.0  # Use uniform FalconH1 layers by default
    hybrid_override_pattern: Optional[str] = None

    # Position embeddings - RoPE for FalconH1
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "rope"
    rotary_percent: float = 1.0
    rotary_base: int = 100000000000  # FalconH1 uses 1e11 base
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = False

    # Vocabulary configuration
    make_vocab_size_divisible_by: int = 128
    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False

    # Training configuration
    gated_linear_unit: bool = True  # FalconH1 uses SwiGLU
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    attention_backend: AttnBackend = AttnBackend.unfused
    deallocate_pipeline_outputs: bool = True
    bias_dropout_fusion: bool = False
    cross_entropy_loss_fusion: bool = False
    transformer_impl: str = "local"

    #Falcon H1 Mup Fwd Multpliers
    embedding_multiplier: float = 1.0
    lm_head_multiplier: float = 1.0
    key_multiplier: float = 1.0
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 1.0
    ssm_in_multiplier: float = 1.0
    ssm_out_multiplier: float = 1.0
    mlp_multipliers: tuple = (1.0, 1.0)
    ssm_multipliers: tuple = (1.0, 1.0, 1.0, 0.5, 1.0)


    # Stack specification
    falconh1_stack_spec: Union[ModuleSpec, Callable[[], ModuleSpec]] = get_default_falconh1_stack_spec

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> FalconH1Model:
        """Configure and instantiate a FalconH1 model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage (currently unsupported)

        Returns:
            FalconH1Model: Configured FalconH1 model instance
        """
        falconh1_stack_spec = self.falconh1_stack_spec
        if not isinstance(falconh1_stack_spec, ModuleSpec):
            falconh1_stack_spec = falconh1_stack_spec()

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in FalconH1 models"
        )

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"

        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        return FalconH1Model(
            config=self,
            falconh1_stack_spec=falconh1_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            falconh1_ratio=self.falconh1_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
        )

    def finalize(self) -> None:
        # Call parent class finalize if it exists
        if hasattr(super(), 'finalize'):
            super().finalize()

@dataclass
class FalconH1ModelProvider500M(FalconH1ModelProvider):
    """Configuration for FalconH1 0.5B model.
    Based on: https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct
    """
    # Model architecture from config.json
    num_layers: int = 36
    hidden_size: int = 1024
    ffn_hidden_size: int = 2048
    num_attention_heads: int = 8
    num_query_groups: int = 2
    seq_length: int = 16384

    # Mamba-specific parameters
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_heads: int = 24
    mamba_num_groups: int = 1
    expand: int = 2
    d_conv: int = 4
    chunk_size: int = 128
    rmsnorm: bool = False

    # Model settings
    vocab_size: int = 32784
    tie_word_embeddings: bool = False
    make_vocab_size_divisible_by: int = 16

    # All layers are FalconH1 layers
    falconh1_ratio: float = 1.0
    use_mamba: bool = True
    use_attention: bool = True
    use_mlp: bool = True

    # MuP multipliers for 0.5B
    embedding_multiplier: float = 5.656854249492381
    lm_head_multiplier: float = 0.0390625
    key_multiplier: float = 0.39062499999999994
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.9375
    ssm_in_multiplier: float = 1.25
    ssm_out_multiplier: float = 0.23570226039551587
    mlp_multipliers: tuple = (0.8838834764831844, 0.5859375)
    ssm_multipliers: tuple = (0.3535533905932738, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738)


@dataclass
class FalconH1ModelProvider1P5BDeep(FalconH1ModelProvider):
    """Configuration for FalconH1 1.5B Deep model.
    Based on: https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct
    """
    # Model architecture from config.json
    num_layers: int = 66
    hidden_size: int = 1280
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 6
    num_query_groups: int = 2
    seq_length: int = 131072

    # Mamba-specific parameters
    mamba_state_dim: int = 256
    mamba_head_dim: int = 64
    mamba_num_heads: int = 24
    mamba_num_groups: int = 1
    expand: int = 2
    d_conv: int = 4
    chunk_size: int = 128
    rmsnorm: bool = True

    # Model settings
    vocab_size: int = 65537
    tie_word_embeddings: bool = False
    make_vocab_size_divisible_by: int = 1

    # All layers are FalconH1 layers
    falconh1_ratio: float = 1.0
    use_mamba: bool = True
    use_attention: bool = True
    use_mlp: bool = True

    # MuP multipliers for 1.5B Deep
    embedding_multiplier: float = 5.656854249492381
    lm_head_multiplier: float = 0.03125
    key_multiplier: float = 0.17677669529663687
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.5
    ssm_in_multiplier: float = 1.0
    ssm_out_multiplier: float = 0.23570226039551587
    mlp_multipliers: tuple = (0.7071067811865476, 0.3125)
    ssm_multipliers: tuple = (0.3535533905932738, 0.25, 0.1767766952966369, 0.5, 0.3535533905932738)


@dataclass
class FalconH1ModelProvider7B(FalconH1ModelProvider):
    """Configuration for FalconH1 7B model.
    Based on: https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct
    """
    # Model architecture from config.json
    num_layers: int = 44
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 12
    num_query_groups: int = 2
    seq_length: int = 262144

    # Mamba-specific parameters
    mamba_state_dim: int = 256
    mamba_head_dim: int = 128
    mamba_num_heads: int = 24
    mamba_num_groups: int = 1
    expand: int = 2
    d_conv: int = 4
    chunk_size: int = 256
    rmsnorm: bool = True

    # Model settings
    vocab_size: int = 130049
    tie_word_embeddings: bool = False
    make_vocab_size_divisible_by: int = 1

    # All layers are FalconH1 layers
    falconh1_ratio: float = 1.0
    use_mamba: bool = True
    use_attention: bool = True
    use_mlp: bool = True

    # MuP multipliers for 7B
    embedding_multiplier: float = 5.656854249492381
    lm_head_multiplier: float = 0.013020833333333334
    key_multiplier: float = 0.030690398488999456
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.10416666666666667
    ssm_in_multiplier: float = 0.4166666666666667
    ssm_out_multiplier: float = 0.11785113019775793
    mlp_multipliers: tuple = (0.2946278254943948, 0.032552083333333336)
    ssm_multipliers: tuple = (0.3535533905932738, 0.25, 0.1767766952966369, 0.5, 0.3535533905932738)


@dataclass
class FalconH1ModelProvider34B(FalconH1ModelProvider):
    """Configuration for FalconH1 34B model.
    Based on: https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct
    """
    # Model architecture from config.json
    num_layers: int = 72
    hidden_size: int = 5120
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 20
    num_query_groups: int = 4
    seq_length: int = 262144

    # Mamba-specific parameters
    mamba_state_dim: int = 256
    mamba_head_dim: int = 128
    mamba_num_heads: int = 32
    mamba_num_groups: int = 2
    expand: int = 2
    d_conv: int = 4
    chunk_size: int = 128
    rmsnorm: bool = True

    # Model settings
    vocab_size: int = 261120
    tie_word_embeddings: bool = False
    make_vocab_size_divisible_by: int = 128

    # All layers are FalconH1 layers
    falconh1_ratio: float = 1.0
    use_mamba: bool = True
    use_attention: bool = True
    use_mlp: bool = True

    # MuP multipliers for 34B
    embedding_multiplier: float = 5.656854249492381
    lm_head_multiplier: float = 0.0078125
    key_multiplier: float = 0.011048543456039804
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.0375
    ssm_in_multiplier: float = 0.25
    ssm_out_multiplier: float = 0.08838834764831845
    mlp_multipliers: tuple = (0.1767766952966369, 0.011160714285714284)
    ssm_multipliers: tuple = (0.3535533905932738, 0.25, 0.1767766952966369, 0.5, 0.3535533905932738)