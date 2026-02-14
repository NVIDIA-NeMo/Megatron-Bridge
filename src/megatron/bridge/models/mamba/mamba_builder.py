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

from dataclasses import dataclass
from typing import Callable, ClassVar, Literal

from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as default_mamba_stack_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.common import ModelBuildConfig, ModelBuilder
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Return the default Mamba stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Mamba stack specification from megatron.core
    """
    return default_mamba_stack_spec


def modelopt_mamba_stack_spec(config: "MambaExtraConfig") -> ModuleSpec:
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


def get_default_mamba_stack_spec(config: "MambaExtraConfig") -> ModuleSpec:
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
class MambaExtraConfig(ModelBuildConfig):
    """Configuration for Mamba model building that is NOT part of TransformerConfig.

    This is a pure data container with no behavior - just configuration values.
    All the logic for using these values lives in the MambaModelBuilder.
    """

    builder: ClassVar[str] = "megatron.bridge.models.mamba.MambaModelBuilder"
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str | None = None
    seq_length: int = 8192
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: float | None = None
    make_vocab_size_divisible_by: int = 128
    mamba_stack_spec: ModuleSpec | Callable[[], ModuleSpec] | Callable[["MambaExtraConfig"], ModuleSpec] = (
        get_default_mamba_stack_spec
    )
    vocab_size: int | None = None
    should_pad_vocab: bool = False


class MambaModelBuilder(ModelBuilder[MCoreMambaModel, MambaExtraConfig]):
    """Builder to construct Megatron Core Mamba models.

    Example:
        >>> # model_config is MCore TransformerConfig for GPT
        >>> model_cfg = TransformerConfig(num_layers=32, hidden_size=4096, ...)
        >>> build_cfg = MambaExtraConfig(vocab_size=32000, seq_length=2048, ...)
        >>>
        >>> # Build model
        >>> model = MambaModelBuilder(model_cfg, build_cfg).build_model(pg_collection)
    """

    def __init__(self, transformer_config: TransformerConfig, mamba_config: MambaExtraConfig):
        super.__init__(transformer_config, mamba_config)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MCoreMambaModel:
        mamba_stack_spec = self.build_config.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            # Check if the function accepts config parameter
            import inspect

            if len(inspect.signature(mamba_stack_spec).parameters) > 0:
                mamba_stack_spec = mamba_stack_spec(self.build_config)
            else:
                mamba_stack_spec = mamba_stack_spec()

        assert getattr(self.model_config, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in SSM/Mamba "
            "models due to upstream MCore MambaModel API dependency"
        )

        assert self.build_config.vocab_size is not None, "vocab_size must be configured before calling build_model()"
        if self.build_config.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.build_config.vocab_size,
                self.build_config.make_vocab_size_divisible_by,
                self.model_config.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = self.build_config.vocab_size

        return MCoreMambaModel(
            config=self.model_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.build_config.seq_length,
            hybrid_attention_ratio=self.build_config.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.build_config.hybrid_mlp_ratio,
            hybrid_override_pattern=self.build_config.hybrid_override_pattern,
            fp16_lm_cross_entropy=self.build_config.fp16_lm_cross_entropy,
            parallel_output=self.build_config.parallel_output,
            share_embeddings_and_output_weights=self.build_config.share_embeddings_and_output_weights,
            position_embedding_type=self.build_config.position_embedding_type,
            rotary_percent=self.build_config.rotary_percent,
            rotary_base=self.build_config.rotary_base,
            seq_len_interpolation_factor=self.build_config.seq_len_interpolation_factor,
            pre_process=pre_process or is_pp_first_stage(pg_collection.pp),
            post_process=post_process or is_pp_last_stage(pg_collection.pp),
            pg_collection=pg_collection,
        )
