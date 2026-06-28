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

"""Pure builder configuration for Falcon-H1 models."""

from dataclasses import dataclass
from typing import ClassVar

from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer_specs import falconh1_stack_spec
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Model
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class FalconH1ModelConfig(BridgeGPTModelConfig):
    """Serializable Falcon-H1 model build configuration."""

    builder: ClassVar[str] = "megatron.bridge.models.falcon_h1.model_config.FalconH1ModelBuilder"
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    falconh1_ratio: float = 1.0
    hybrid_override_pattern: str | None = None
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_groups: int = 1
    mamba_num_heads: int | None = None
    use_mamba_mem_eff_path: bool = True
    A_init_dist: str = "uniform"
    d_conv: int = 4
    conv_init: float | None = 1.0
    expand: int = 2
    A_init_range: tuple[float, float] = (1, 16)
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    chunk_size: int = 128
    use_mamba: bool = True
    use_attention: bool = True
    use_mlp: bool = True
    embedding_multiplier: float = 1.0
    lm_head_multiplier: float = 1.0
    key_multiplier: float = 1.0
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 1.0
    ssm_in_multiplier: float = 1.0
    ssm_out_multiplier: float = 1.0
    mlp_multipliers: tuple[float, float] = (1.0, 1.0)
    ssm_multipliers: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)


class FalconH1ModelBuilder(GPTModelBuilder):
    """Build Falcon-H1 without a provider object."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> FalconH1Model:
        """Build one Falcon-H1 pipeline stage.

        Args:
            pg_collection: Process groups for distributed model construction.
            pre_process: Whether the stage owns the embedding layer.
            post_process: Whether the stage owns the output layer.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Falcon-H1 model stage.

        Raises:
            ValueError: If virtual pipeline parallelism is enabled or vocabulary
                size is missing.
        """
        config = self._model_config
        if vp_stage is not None or config.transformer.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Virtual pipeline parallelism is not supported for Falcon-H1.")
        if config.vocab_size is None:
            raise ValueError("vocab_size must be configured before building Falcon-H1.")
        vocab_size = config.vocab_size
        if config.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size,
                config.make_vocab_size_divisible_by,
                config.transformer.tensor_model_parallel_size,
            )
        return FalconH1Model(
            config=config.transformer,
            model_config=config,
            falconh1_stack_spec=falconh1_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=config.seq_length,
            hybrid_attention_ratio=config.hybrid_attention_ratio,
            hybrid_mlp_ratio=config.hybrid_mlp_ratio,
            falconh1_ratio=config.falconh1_ratio,
            hybrid_override_pattern=config.hybrid_override_pattern,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            scatter_embedding_sequence_parallel=config.scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            pre_process=is_pp_first_stage(pg_collection.pp) if pre_process is None else pre_process,
            post_process=is_pp_last_stage(pg_collection.pp) if post_process is None else post_process,
            pg_collection=pg_collection,
        )


__all__ = ["FalconH1ModelBuilder", "FalconH1ModelConfig"]
