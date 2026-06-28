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

"""YaRN construction helpers that keep family fields off TransformerConfig."""

from dataclasses import replace
from typing import Protocol

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
    _yarn_get_concentration_factor,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig


class YarnBuildConfig(Protocol):
    """Outer model-config fields needed to construct YaRN explicitly."""

    transformer: object
    rotary_percent: float
    rotary_base: int
    seq_len_interpolation_factor: float | None
    yarn_rotary_scaling_factor: float
    yarn_original_max_position_embeddings: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_mscale: float | None
    yarn_mscale_all_dim: float | None
    yarn_correction_range_round_to_int: bool


def build_gpt_with_yarn(
    config: GPTModelConfig,
    *,
    pg_collection: ProcessGroupCollection,
    pre_process: bool | None,
    post_process: bool | None,
    vp_stage: int | None,
) -> GPTModel:
    """Build GPT with an exact MCore config and install YaRN from outer data.

    MCore's current GPT constructor reads non-dataclass YaRN attributes from
    ``TransformerConfig``. Bridge instead constructs the base model with RoPE,
    then installs the equivalent YaRN embedding and attention concentration
    factor without ever adding phantom fields to the nested config.
    """
    yarn = config
    runtime_transformer = config.transformer
    runtime_config = replace(config, position_embedding_type="rope")
    model = GPTModelBuilder(runtime_config).build_model(
        pg_collection,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )

    model.position_embedding_type = "yarn"
    model.rotary_pos_emb = YarnRotaryEmbedding(
        kv_channels=runtime_transformer.kv_channels,
        rotary_percent=yarn.rotary_percent,
        rotary_interleaved=runtime_transformer.rotary_interleaved,
        seq_len_interpolation_factor=yarn.seq_len_interpolation_factor,
        rotary_base=yarn.rotary_base,
        scaling_factor=yarn.yarn_rotary_scaling_factor,
        original_max_position_embeddings=yarn.yarn_original_max_position_embeddings,
        beta_fast=yarn.yarn_beta_fast,
        beta_slow=yarn.yarn_beta_slow,
        mscale=yarn.yarn_mscale,
        mscale_all_dim=yarn.yarn_mscale_all_dim,
        correction_range_round_to_int=yarn.yarn_correction_range_round_to_int,
        use_cpu_initialization=runtime_transformer.use_cpu_initialization,
    )
    model.rotary_pos_emb_cache = {}

    concentration_factor = _yarn_get_concentration_factor(
        yarn.yarn_rotary_scaling_factor,
        yarn.yarn_mscale,
        yarn.yarn_mscale_all_dim,
    )
    for module in model.modules():
        if hasattr(module, "_yarn_concentration_factor"):
            module._yarn_concentration_factor = concentration_factor
    return model


__all__ = ["build_gpt_with_yarn"]
