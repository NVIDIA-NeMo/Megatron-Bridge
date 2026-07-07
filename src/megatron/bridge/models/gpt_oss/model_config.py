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

"""Provider-neutral GPT-OSS model config and builder."""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
    _yarn_get_concentration_factor,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig

from megatron.bridge.models.config_proxy import FlatTransformerConfigMixin
from megatron.bridge.models.gpt.model_config import ACTIVATION_FUNC_METADATA_KEY
from megatron.bridge.utils.activation_map import callable_to_str


@dataclass(kw_only=True)
class GPTOSSModelConfig(FlatTransformerConfigMixin, HybridModelConfig):
    """GPT-OSS HybridModel config preserving all YaRN construction parameters."""

    builder: ClassVar[str] = "megatron.bridge.models.gpt_oss.model_config.GPTOSSModelBuilder"
    yarn_rotary_scaling_factor: float = 32.0
    yarn_original_max_position_embeddings: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_correction_range_round_to_int: bool = False
    extra_checkpoint_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the nested activation as stable metadata."""
        data = super().as_dict()
        activation_name = callable_to_str(self.transformer.activation_func)
        if activation_name is None:
            raise ValueError("Cannot serialize an unregistered GPT-OSS activation callable.")
        metadata = dict(data.get("extra_checkpoint_metadata") or {})
        metadata[ACTIVATION_FUNC_METADATA_KEY] = activation_name
        data["extra_checkpoint_metadata"] = metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTOSSModelConfig":
        """Deserialize through Bridge's validated ModelConfig loader."""
        from megatron.bridge.models.common.base import ModelConfig

        result = ModelConfig.from_dict(data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result


def build_hybrid_with_yarn(
    config: GPTOSSModelConfig,
    *,
    pg_collection: ProcessGroupCollection,
    pre_process: bool | None,
    post_process: bool | None,
    vp_stage: int | None,
) -> HybridModel:
    """Build HybridModel with YaRN sourced only from the outer config."""
    runtime_transformer = config.transformer
    runtime_config = replace(config, position_embedding_type="rope")
    model = HybridModelBuilder(runtime_config).build_model(
        pg_collection,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )

    model.position_embedding_type = "yarn"
    model.rotary_pos_emb = YarnRotaryEmbedding(
        kv_channels=runtime_transformer.kv_channels,
        rotary_percent=config.rotary_percent,
        seq_len_interpolation_factor=config.seq_len_interpolation_factor,
        rotary_base=config.rotary_base,
        scaling_factor=config.yarn_rotary_scaling_factor,
        original_max_position_embeddings=config.yarn_original_max_position_embeddings,
        beta_fast=config.yarn_beta_fast,
        beta_slow=config.yarn_beta_slow,
        mscale=config.yarn_mscale,
        mscale_all_dim=config.yarn_mscale_all_dim,
        correction_range_round_to_int=config.yarn_correction_range_round_to_int,
        use_cpu_initialization=runtime_transformer.use_cpu_initialization,
        cp_group=pg_collection.cp,
    )
    model.rotary_pos_emb_cache = {}

    concentration_factor = _yarn_get_concentration_factor(
        config.yarn_rotary_scaling_factor,
        config.yarn_mscale,
        config.yarn_mscale_all_dim,
    )
    for module in model.modules():
        if hasattr(module, "_yarn_concentration_factor"):
            module._yarn_concentration_factor = concentration_factor
    return model


class GPTOSSModelBuilder(HybridModelBuilder):
    """Hybrid builder that binds outer YaRN config to MCore during construction."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build GPT-OSS HybridModel with YaRN sourced only from the outer config."""
        config = self._model_config
        assert isinstance(config, GPTOSSModelConfig)
        return build_hybrid_with_yarn(
            config,
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )


__all__ = ["GPTOSSModelBuilder", "GPTOSSModelConfig", "build_hybrid_with_yarn"]
