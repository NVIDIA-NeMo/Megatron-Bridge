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

import inspect
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import torch
from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as default_mamba_stack_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils import fusions
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


try:
    from megatron.core.ssm.mamba_hybrid_layer_allocation import (
        get_hybrid_total_layer_count as _mcore_get_hybrid_total_layer_count,
    )
except ImportError:
    # TODO(yuya): remove fallback once MCore pin includes get_hybrid_total_layer_count
    _mcore_get_hybrid_total_layer_count = None

# MCore renamed `hybrid_override_pattern` → `hybrid_layer_pattern` in the dev branch.
# Support both main and dev branch submodule by detecting which parameter is present at import time.
# TODO: remove fallback once the dev rename lands in main and Bridge pins the new main commit.
_MCORE_MAMBA_INIT_PARAMS = set(inspect.signature(MCoreMambaModel.__init__).parameters)
_HYBRID_LAYER_PATTERN_KWARG = (
    "hybrid_layer_pattern" if "hybrid_layer_pattern" in _MCORE_MAMBA_INIT_PARAMS else "hybrid_override_pattern"
)


logger = logging.getLogger(__name__)

_HYBRID_MAIN_PATTERN_SYMBOLS = frozenset({"M", "*", "-", "E", "|"})


def _fallback_get_hybrid_total_layer_count(pattern: str) -> int:
    """Count main-decoder layers for older MCore branches.

    Older MCore revisions predate ``get_hybrid_total_layer_count`` and do not
    understand pipe-delimited fVPP layouts. Bridge still needs to derive
    ``num_layers`` correctly for both legacy and newer hybrid patterns.
    """

    main_pattern = pattern.split("/")[0]
    invalid_chars = sorted({char for char in main_pattern if char not in _HYBRID_MAIN_PATTERN_SYMBOLS})
    if invalid_chars:
        raise ValueError(
            f"In main pattern, '{invalid_chars[0]}' is not a valid layer symbol. "
            f"Valid symbols are: {_HYBRID_MAIN_PATTERN_SYMBOLS}"
        )
    return len(main_pattern.replace("|", ""))


def _get_hybrid_total_layer_count(pattern: str) -> int:
    if _mcore_get_hybrid_total_layer_count is not None:
        return _mcore_get_hybrid_total_layer_count(pattern)
    return _fallback_get_hybrid_total_layer_count(pattern)


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


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Return the default Mamba stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Mamba stack specification from megatron.core
    """
    return default_mamba_stack_spec


def get_default_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
    """Determine the most appropriate Mamba stack specification based on configuration.

    Args:
        config: Mamba configuration object

    Returns:
        ModuleSpec: Appropriate module specification based on config
    """
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
    num_layers: int = None
    mamba_num_groups: int = 8
    num_attention_heads: int = 1
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: Optional[str] = None
    hybrid_layer_pattern: Optional[str] = None
    seq_length: int = 8192
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal["learned_absolute", "rope", "yarn", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    # YaRN parameters — only applied when position_embedding_type == "yarn"
    yarn_rotary_scaling_factor: float = 8.0
    yarn_original_max_position_embeddings: Optional[int] = None
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0
    yarn_correction_range_round_to_int: bool = True
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
    gradient_accumulation_fusion: bool = field(default_factory=fusions.can_enable_gradient_accumulation_fusion)
    mamba_stack_spec: Union[ModuleSpec, Callable[[], ModuleSpec], Callable[["MambaModelProvider"], ModuleSpec]] = (
        get_default_mamba_stack_spec
    )
    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False
    hf_model_id: Optional[str] = None
    _pg_collection: Optional[ProcessGroupCollection] = None

    # MTP
    mtp_num_layers: int = 0
    mtp_hybrid_override_pattern: Optional[str] = None
    keep_mtp_spec_in_bf16: bool = False

    """Optional HuggingFace model identifier associated with this provider."""

    # If True, restore the modelopt_state that contains quantization, sparsity, speculative decoding transformation state.
    restore_modelopt_state: bool = False

    def finalize(self) -> None:
        """Finalize the Mamba model provider.
        Calculates the number of layers from the hybrid_layer_pattern.
        Executes the deferred MCore post-init logic.
        """
        # Check if hybrid_override_pattern is specified and throw deprecation warning
        used_hybrid_override_pattern = False
        if self.hybrid_override_pattern is not None:
            assert self.hybrid_layer_pattern is None, (
                "hybrid_override_pattern and hybrid_layer_pattern cannot both be specified. "
                "hybrid_override_pattern is deprecated; use hybrid_layer_pattern instead."
            )
            if get_rank_safe() == 0:
                warnings.warn(
                    "hybrid_override_pattern is deprecated. Use hybrid_layer_pattern instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.hybrid_layer_pattern = self.hybrid_override_pattern
            self.hybrid_override_pattern = None
            used_hybrid_override_pattern = True

        # --- MTP pattern construction ---
        # Combine hybrid_layer_pattern (main decoder) with mtp_hybrid_override_pattern
        # into a single unified pattern that MCoreMambaModel can parse.
        # Format: "MAIN_PATTERN/MTP_BLOCK/MTP_BLOCK/..."
        # This must happen before num_layers derivation so the count reflects
        # only the main decoder layers (get_hybrid_total_layer_count strips MTP).
        if self.hybrid_layer_pattern is not None and self.mtp_hybrid_override_pattern:
            sep = Symbols.MTP_SEPARATOR
            main_pattern = self.hybrid_layer_pattern.split(sep)[0]
            # When mtp_use_repeated_layer=True, the shared MTP layer always exists
            # in the model and mtp_num_layers controls forward pass repetitions.
            # Include the pattern at least once so the MTP block (and its weights)
            # are created even when mtp_num_layers=0.
            if self.mtp_use_repeated_layer:
                num_pattern_copies = max(1, self.mtp_num_layers)
            else:
                num_pattern_copies = self.mtp_num_layers
            self.hybrid_layer_pattern = (
                main_pattern + sep + sep.join([self.mtp_hybrid_override_pattern] * num_pattern_copies)
            )

            # Validate mtp_num_layers against the constructed pattern
            if sep in self.hybrid_layer_pattern:
                parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
                if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
                    inferred_mtp_num_layers = parsed.mtp_num_depths
                    if self.mtp_num_layers is None:
                        self.mtp_num_layers = inferred_mtp_num_layers
                    elif self.mtp_use_repeated_layer:
                        # With repeated layers, pattern count reflects architecture
                        # (always 1 shared layer) while mtp_num_layers controls
                        # forward pass repetitions. They are intentionally decoupled.
                        pass
                    elif self.mtp_num_layers != inferred_mtp_num_layers:
                        logger.warning(
                            f"mtp_num_layers ({self.mtp_num_layers}) conflicts with "
                            f"MTP depth count ({inferred_mtp_num_layers}) in pattern "
                            f"'{self.hybrid_layer_pattern}'. "
                            f"Using the inferred value ({inferred_mtp_num_layers})."
                        )
                        self.mtp_num_layers = inferred_mtp_num_layers

        # Check if hybrid_layer_pattern is specified and derive num_layers from pattern
        if self.hybrid_layer_pattern is not None:
            # Derive num_layers from pattern
            num_layers_in_pattern = _get_hybrid_total_layer_count(self.hybrid_layer_pattern)
            if self.num_layers is not None:
                if used_hybrid_override_pattern:
                    assert self.num_layers == num_layers_in_pattern, (
                        f"num_layers ({self.num_layers}) does not match the number of layers "
                        f"derived from hybrid_override_pattern ({num_layers_in_pattern}). "
                        f"Please correct num_layers or the pattern."
                    )
                else:
                    assert self.num_layers == num_layers_in_pattern, (
                        f"num_layers ({self.num_layers}) does not match the number of layers "
                        f"derived from hybrid_layer_pattern ({num_layers_in_pattern}). "
                        f"Please correct num_layers or the pattern."
                    )
            self.num_layers = num_layers_in_pattern

        super().finalize()

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

        if self.position_embedding_type == "yarn" and self.yarn_original_max_position_embeddings is None:
            self.yarn_original_max_position_embeddings = int(self.seq_length / self.yarn_rotary_scaling_factor)

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
            **{_HYBRID_LAYER_PATTERN_KWARG: self.hybrid_layer_pattern},
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
