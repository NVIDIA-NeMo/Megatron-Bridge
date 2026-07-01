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

"""Pure ERNIE 4.5 VL model config and standalone builder."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Callable, ClassVar, cast

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.training.models.gpt import GPTModelBuilder
from transformers import PretrainedConfig

from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_decoder_layer_spec import (
    get_ernie45_vl_decoder_block_spec,
)
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.model import Ernie45VLModel
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.vision_transformer_config import _quick_gelu
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def _namespace(data: Mapping[object, object]) -> SimpleNamespace | dict[object, object]:
    converted = {key: _namespace(value) if isinstance(value, dict) else value for key, value in data.items()}
    if not all(isinstance(key, str) for key in converted):
        return converted
    return SimpleNamespace(**cast(dict[str, object], converted))


@dataclass(kw_only=True)
class Ernie45VLModelConfig(BridgeGPTModelConfig):
    """Serializable ERNIE VL config with exact text and vision MCore configs."""

    builder: ClassVar[str] = "megatron.bridge.models.ernie_vl.model_config.Ernie45VLModelBuilder"
    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(
        default_factory=lambda: get_ernie45_vl_decoder_block_spec
    )
    vision_transformer: TransformerConfig
    vision_config: dict[str, object] = field(default_factory=dict)
    hf_config: dict[str, object] = field(default_factory=dict)
    patch_size: int = 14
    in_channels: int = 3
    spatial_merge_size: int = 2
    mrope_section: list[int] = field(default_factory=lambda: [22, 22, 20])
    moe_intermediate_size: tuple[int, int] = (1536, 512)
    image_start_token_id: int = 101304
    image_end_token_id: int = 101305
    image_token_id: int = 100295
    video_start_token_id: int = 101306
    video_end_token_id: int = 101307
    video_token_id: int = 103367
    use_mg_vit: bool = False
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


class Ernie45VLModelBuilder(GPTModelBuilder):
    """Build ERNIE VL from exact configs without a provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Ernie45VLModel:
        """Build one ERNIE VL pipeline stage."""
        config = self._model_config
        language_model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        vision_transformer = replace(
            config.vision_transformer,
            activation_func=_quick_gelu,
            tensor_model_parallel_size=config.transformer.tensor_model_parallel_size,
        )
        runtime_config = PretrainedConfig.from_dict(config.hf_config)
        if isinstance(getattr(runtime_config, "text_config", None), dict):
            runtime_config.text_config = _namespace(runtime_config.text_config)
        runtime_config.hf_config = runtime_config
        runtime_config.vision_config = PretrainedConfig.from_dict(config.vision_config)
        runtime_config.return_dict = True
        for name in (
            "mrope_section",
            "image_start_token_id",
            "image_end_token_id",
            "image_token_id",
            "video_start_token_id",
            "video_end_token_id",
            "video_token_id",
            "use_mg_vit",
        ):
            setattr(runtime_config, name, getattr(config, name))
        runtime_config.sequence_parallel = config.transformer.sequence_parallel
        # Conversion task global-name resolution reads the composite model's
        # runtime config when translating EP-local expert indices. ERNIE owns
        # two pools with this many experts each, so expose the per-pool count
        # without polluting the serialized Hugging Face config.
        runtime_config.num_moe_experts = config.transformer.num_moe_experts
        runtime_config.kv_channels = config.transformer.kv_channels
        runtime_config.rotary_interleaved = config.transformer.rotary_interleaved
        runtime_config.rotary_base = config.rotary_base
        runtime_config.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        model = Ernie45VLModel(
            config=runtime_config,
            language_transformer_config=config.transformer,
            vision_transformer_config=vision_transformer,
            language_model=language_model,
            pg_collection=pg_collection,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            spatial_merge_size=config.spatial_merge_size,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                config.freeze_language_model,
                config.freeze_vision_model,
                config.freeze_vision_projection,
            )
        return model


__all__ = ["Ernie45VLModelBuilder", "Ernie45VLModelConfig"]
