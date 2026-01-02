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
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as default_mamba_stack_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


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


def quantization_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
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
        return quantization_mamba_stack_spec(config)
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
    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False
    hf_model_id: Optional[str] = None
    _pg_collection: Optional[ProcessGroupCollection] = None
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


@dataclass
class MambaModelProvider130M(MambaModelProvider):
    """Configuration for a 130M parameter Mamba model.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M" * 24
    num_layers: int = 24
    seq_length: int = 2048
    hidden_size: int = 768
    mamba_num_groups: int = 1
    ffn_hidden_size: int = 768
    make_vocab_size_divisible_by: int = 16

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("MambaModelProvider130M")


@dataclass
class MambaModelProvider370M(MambaModelProvider):
    """Configuration for a 370M parameter Mamba model.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1024
    mamba_num_groups: int = 1
    ffn_hidden_size: int = 1024
    make_vocab_size_divisible_by: int = 16

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("MambaModelProvider370M")


@dataclass
class MambaModelProvider780M(MambaModelProvider):
    """Configuration for a 780M parameter Mamba model.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1536
    mamba_num_groups: int = 1
    ffn_hidden_size: int = 1536
    make_vocab_size_divisible_by: int = 16

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("MambaModelProvider780M")


@dataclass
class MambaModelProvider1P3B(MambaModelProvider):
    """Configuration for a 1.3B parameter Mamba model.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 2048
    mamba_num_groups: int = 1
    ffn_hidden_size: int = 2048
    make_vocab_size_divisible_by: int = 16

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("MambaModelProvider1P3B")


@dataclass
class MambaModelProvider2P7B(MambaModelProvider):
    """Configuration for a 2.7B parameter Mamba model.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M" * 64
    num_layers: int = 64
    seq_length: int = 2048
    hidden_size: int = 2560
    mamba_num_groups: int = 1
    ffn_hidden_size: int = 2560
    make_vocab_size_divisible_by: int = 16

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("MambaModelProvider2P7B")


@dataclass
class NVIDIAMambaModelProvider8B(MambaModelProvider):
    """Configuration for a 8B parameter Mamba model used in NVIDIA research.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M" * 56
    num_attention_heads: int = 32
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_num_groups: int = 8
    ffn_hidden_size: int = 4096
    make_vocab_size_divisible_by: int = 128

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("NVIDIAMambaModelProvider8B")


@dataclass
class NVIDIAMambaHybridModelProvider8B(MambaModelProvider):
    """Configuration for a 8B parameter hybrid Mamba model used in NVIDIA research.

    Deprecated:
        This class is deprecated and will be removed in a future release.
    """

    hybrid_override_pattern: str = "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_num_groups: int = 8
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128

    def finalize(self) -> None:
        super().finalize()
        _warn_class_deprecated("NVIDIAMambaHybridModelProvider8B")


# -----------------------------------------------------------------------------
# Deprecated aliases (to be removed in a future release)
# -----------------------------------------------------------------------------


def _warn_deprecated(old_cls: str, new_cls: str) -> None:
    if get_rank_safe() == 0:
        warnings.warn(
            f"{old_cls} is deprecated and will be removed in a future release. Use {new_cls} instead.",
            DeprecationWarning,
            stacklevel=2,
        )


def _warn_class_deprecated(cls_name: str) -> None:
    """Log a deprecation warning for a class.

    Args:
        cls_name: The name of the deprecated class.
    """
    if get_rank_safe() == 0:
        warnings.warn(
            f"{cls_name} is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=4,
        )


@dataclass
class MambaProvider(MambaModelProvider):
    """Deprecated alias for ``MambaModelProvider``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MambaModelProvider`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MambaProvider", "MambaModelProvider")
        super().__post_init__()


@dataclass
class MambaProvider130M(MambaModelProvider130M):
    """Deprecated alias for ``MambaModelProvider130M``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MambaModelProvider130M`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MambaProvider130M", "MambaModelProvider130M")
        super().__post_init__()


@dataclass
class MambaProvider370M(MambaModelProvider370M):
    """Deprecated alias for ``MambaModelProvider370M``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MambaModelProvider370M`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MambaProvider370M", "MambaModelProvider370M")
        super().__post_init__()


@dataclass
class MambaProvider780M(MambaModelProvider780M):
    """Deprecated alias for ``MambaModelProvider780M``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MambaModelProvider780M`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MambaProvider780M", "MambaModelProvider780M")
        super().__post_init__()


@dataclass
class MambaProvider1_3B(MambaModelProvider1P3B):
    """Deprecated alias for ``MambaModelProvider1P3B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MambaModelProvider1P3B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MambaProvider1_3B", "MambaModelProvider1P3B")
        super().__post_init__()


@dataclass
class MambaProvider2_7B(MambaModelProvider2P7B):
    """Deprecated alias for ``MambaModelProvider2P7B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MambaModelProvider2P7B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MambaProvider2_7B", "MambaModelProvider2P7B")
        super().__post_init__()


@dataclass
class NVIDIAMambaProvider8B(NVIDIAMambaModelProvider8B):
    """Deprecated alias for ``NVIDIAMambaModelProvider8B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NVIDIAMambaModelProvider8B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NVIDIAMambaProvider8B", "NVIDIAMambaModelProvider8B")
        super().__post_init__()


@dataclass
class NVIDIAMambaHybridProvider8B(NVIDIAMambaHybridModelProvider8B):
    """Deprecated alias for ``NVIDIAMambaHybridModelProvider8B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``NVIDIAMambaHybridModelProvider8B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("NVIDIAMambaHybridProvider8B", "NVIDIAMambaHybridModelProvider8B")
        super().__post_init__()
