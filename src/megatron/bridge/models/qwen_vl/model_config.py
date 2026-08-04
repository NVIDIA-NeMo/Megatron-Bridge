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

"""Serializable model configurations and standalone builders for Qwen VL."""

import importlib
import inspect
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.dot_product_attention import DotProductAttention as MCoreDotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.training.models.gpt import GPTModelBuilder, default_layer_spec, mtp_block_spec
from megatron.training.vocab_utils import calculate_padded_vocab_size
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.qwen.model_config import qwen_hybrid_mtp_block_spec
from megatron.bridge.models.qwen_vl.modeling_qwen25_vl import Qwen25VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel


def _pipeline_flags(
    config: BridgeGPTModelConfig, pg_collection: ProcessGroupCollection, pre_process, post_process, vp_stage
):
    vp_size = config.transformer.virtual_pipeline_model_parallel_size
    if pre_process is None:
        pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
    if post_process is None:
        post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)
    return pre_process, post_process


def _language_transformer(config: BridgeGPTModelConfig):
    """Return an exact MCore config with its declared M-RoPE field populated."""
    return replace(config.transformer, mrope_section=config.mrope_section)


def _vision_config_from_dict(data: dict[str, Any], target: str):
    module_name, class_name = target.rsplit(".", 1)
    config_class = getattr(importlib.import_module(module_name), class_name)
    return config_class(**data)


def _language_layer_spec(config: BridgeGPTModelConfig, vp_stage: int | None) -> ModuleSpec:
    """Resolve the language layer spec without constructing the language model."""
    transformer_layer_spec = config.transformer_layer_spec
    if transformer_layer_spec is None:
        transformer_layer_spec = default_layer_spec(config, vp_stage)
    elif not isinstance(transformer_layer_spec, ModuleSpec) and callable(transformer_layer_spec):
        if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
            transformer_layer_spec = transformer_layer_spec(config, vp_stage=vp_stage)
        else:
            transformer_layer_spec = transformer_layer_spec(config)

    if config.attention_backend == AttnBackend.local and hasattr(transformer_layer_spec, "submodules"):
        transformer_layer_spec.submodules.self_attention.submodules.core_attention = MCoreDotProductAttention
    return transformer_layer_spec


def _language_vocab_size(config: BridgeGPTModelConfig) -> int:
    """Return the vocabulary size used to construct the language model."""
    if config.vocab_size is None:
        raise ValueError("vocab_size must be configured before building Qwen2.5-VL")
    if not config.should_pad_vocab:
        return config.vocab_size
    return calculate_padded_vocab_size(
        config.vocab_size,
        config.make_vocab_size_divisible_by,
        config.transformer.tensor_model_parallel_size,
    )


@dataclass(kw_only=True)
class Qwen25VLModelConfig(BridgeGPTModelConfig):
    """Pure builder input for Qwen2.5-VL."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen_vl.model_config.Qwen25VLModelBuilder"
    vision_config: dict[str, Any] = field(default_factory=dict)
    language_max_sequence_length: int = 2048
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


class Qwen25VLModelBuilder(GPTModelBuilder):
    """Build Qwen2.5-VL without a model provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen25VLModel:
        """Build one Qwen2.5-VL pipeline stage.

        Args:
            pg_collection: Process groups for distributed construction.
            pre_process: Whether this stage owns input processing.
            post_process: Whether this stage owns output processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Qwen2.5-VL stage.
        """
        config = self._model_config
        pre_process, post_process = _pipeline_flags(config, pg_collection, pre_process, post_process, vp_stage)
        transformer = _language_transformer(config)
        runtime_config = replace(config, transformer=transformer)
        layer_spec = _language_layer_spec(runtime_config, vp_stage)
        model = Qwen25VLModel(
            language_transformer_config=transformer,
            language_transformer_layer_spec=layer_spec,
            vision_transformer_config=Qwen2_5_VLVisionConfig(**config.vision_config),
            model_config=config,
            language_vocab_size=_language_vocab_size(config),
            parallel_output=config.parallel_output,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            mtp_block_spec=mtp_block_spec(runtime_config, layer_spec, vp_stage=vp_stage),
            vp_stage=vp_stage,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(config.freeze_language_model, config.freeze_vision_model, config.freeze_vision_projection)
        return model


@dataclass
class DistTrainConfig:
    """Distributed training settings shared by builder and legacy provider paths."""

    use_dist_train: bool = False
    vision_to_llm_dp_ratio: int | None = None
    vision_world_size: int | None = None
    language_world_size: int | None = None
    vision_tensor_model_parallel_size: int | None = None
    vision_pipeline_model_parallel_size: int | None = None
    vision_context_parallel_size: int | None = None
    vision_expert_tensor_parallel_size: int | None = None
    vision_expert_model_parallel_size: int | None = None
    has_language_module: bool = True


@dataclass(kw_only=True)
class Qwen3VLModelConfig(BridgeGPTModelConfig):
    """Pure builder input shared by dense and MoE Qwen3-VL."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen_vl.model_config.Qwen3VLModelBuilder"
    vision_config: dict[str, Any] = field(default_factory=dict)
    vision_config_target: str = "transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLVisionConfig"
    language_max_sequence_length: int = 2048
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
    use_hf_vision_model: bool = False
    vision_dp_when_cp: bool = False
    add_encoder: bool = True
    add_decoder: bool = True
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    dist_train: DistTrainConfig = field(default_factory=DistTrainConfig)
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)


class Qwen3VLModelBuilder(GPTModelBuilder):
    """Build dense or MoE Qwen3-VL without a model provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen3VLModel:
        """Build one Qwen3-VL pipeline stage.

        Args:
            pg_collection: Process groups for distributed construction.
            pre_process: Whether this stage owns input processing.
            post_process: Whether this stage owns output processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Qwen3-VL stage.
        """
        config = self._model_config
        pre_process, post_process = _pipeline_flags(config, pg_collection, pre_process, post_process, vp_stage)
        transformer = _language_transformer(config)
        vision_config = _vision_config_from_dict(config.vision_config, config.vision_config_target)
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=transformer.num_moe_experts,
            moe_grouped_gemm=True,
            qk_layernorm=transformer.qk_layernorm,
            fp8=False,
        )
        model = Qwen3VLModel(
            language_transformer_config=transformer,
            language_transformer_layer_spec=layer_spec,
            vision_transformer_config=vision_config,
            parallel_output=config.parallel_output,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=config.add_encoder,
            add_decoder=config.add_decoder,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            model_config=config,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(config.freeze_language_model, config.freeze_vision_model, config.freeze_vision_projection)
        return model


@dataclass(kw_only=True)
class Qwen35VLModelConfig(Qwen3VLModelConfig):
    """Pure builder input shared by dense and MoE Qwen3.5-VL."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen_vl.model_config.Qwen35VLModelBuilder"


def _patch_attention(spec) -> None:
    if hasattr(spec, "layer_specs"):
        for layer_spec in spec.layer_specs:
            _patch_attention(layer_spec)
        return
    submodules = getattr(spec, "submodules", None)
    attention = getattr(submodules, "self_attention", None)
    if attention is not None and hasattr(attention, "submodules") and hasattr(attention.submodules, "linear_qkv"):
        attention.module = Qwen3VLSelfAttention


class Qwen35VLModelBuilder(Qwen3VLModelBuilder):
    """Build dense or MoE Qwen3.5-VL hybrid models without a provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen3VLModel:
        """Build one Qwen3.5-VL pipeline stage.

        Args:
            pg_collection: Process groups for distributed construction.
            pre_process: Whether this stage owns input processing.
            post_process: Whether this stage owns output processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Qwen3.5-VL stage.
        """
        config = self._model_config
        pre_process, post_process = _pipeline_flags(config, pg_collection, pre_process, post_process, vp_stage)
        transformer = _language_transformer(config)
        vision_config = _vision_config_from_dict(config.vision_config, config.vision_config_target)
        layer_spec = get_transformer_block_with_experimental_attention_variant_spec(transformer, vp_stage=vp_stage)
        _patch_attention(layer_spec)
        mtp_spec = qwen_hybrid_mtp_block_spec(config, layer_spec, vp_stage=vp_stage)
        _patch_attention(mtp_spec)
        model = Qwen3VLModel(
            language_transformer_config=transformer,
            language_transformer_layer_spec=layer_spec,
            vision_transformer_config=vision_config,
            parallel_output=config.parallel_output,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=config.add_encoder,
            add_decoder=config.add_decoder,
            pg_collection=pg_collection,
            mtp_block_spec=mtp_spec,
            vp_stage=vp_stage,
            model_config=config,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(config.freeze_language_model, config.freeze_vision_model, config.freeze_vision_projection)
        return model


__all__ = [
    "DistTrainConfig",
    "Qwen25VLModelBuilder",
    "Qwen25VLModelConfig",
    "Qwen35VLModelBuilder",
    "Qwen35VLModelConfig",
    "Qwen3VLModelBuilder",
    "Qwen3VLModelConfig",
]
