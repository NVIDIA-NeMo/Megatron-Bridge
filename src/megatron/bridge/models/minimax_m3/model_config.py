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

"""Serializable MiniMax-M3 model configs and standalone builders."""

from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Any, Callable, ClassVar

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.gpt.model_builder import LayerSpecGPTModelBuilder
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VLModel


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

try:
    from megatron.core.fusions.fused_bias_geglu import quick_gelu
except ImportError:
    quick_gelu = torch.nn.functional.gelu


class MiniMaxM3TopKRouter(TopKRouter):
    """MiniMax-M3 router that computes its projection in the weight dtype."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """Match HF by widening router inputs to the FP32 router weight dtype."""
        return super().gating(input.to(dtype=self.weight.dtype))


def minimax_m3_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool = True,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
    **kwargs: object,
) -> TransformerBlockSubmodules:
    """Build a GPT block spec that uses MiniMax-M3's FP32 router projection."""
    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
        **kwargs,
    )

    for layer_spec in block_spec.layer_specs:
        mlp_spec = layer_spec.submodules.mlp
        if isinstance(mlp_spec, partial) and isinstance(mlp_spec.func, type) and issubclass(mlp_spec.func, MoELayer):
            mlp_kwargs = dict(mlp_spec.keywords or {})
            mlp_submodules = mlp_kwargs["submodules"]
            if mlp_submodules.router is not TopKRouter:
                continue
            mlp_kwargs["submodules"] = replace(mlp_submodules, router=MiniMaxM3TopKRouter)
            layer_spec.submodules.mlp = partial(mlp_spec.func, *mlp_spec.args, **mlp_kwargs)

    return block_spec


def _promote_router_weights_to_float32(model: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Keep MiniMax-M3 router parameters in FP32 for every load path."""
    for model_chunk in model:
        for module in model_chunk.modules():
            if isinstance(module, TopKRouter) and module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.float()
            if isinstance(module, TopKRouter):
                module._keep_in_float32_parameter_names = ("weight",)
    return model


@dataclass(kw_only=True)
class MiniMaxM3TextModelConfig(BridgeGPTModelConfig):
    """Builder config for MiniMax-M3's checkpoint-compatible text backbone."""

    builder: ClassVar[str] = "megatron.bridge.models.minimax_m3.model_config.MiniMaxM3TextModelBuilder"
    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(
        default_factory=lambda: partial(minimax_m3_block_spec, use_transformer_engine=HAVE_TE)
    )

    def __post_init__(self) -> None:
        """Install the FP32-router hook on fresh and deserialized configs."""
        if _promote_router_weights_to_float32 not in self.pre_wrap_hooks:
            self.pre_wrap_hooks.insert(0, _promote_router_weights_to_float32)


@dataclass(kw_only=True)
class MiniMaxM3VLModelConfig(MiniMaxM3TextModelConfig):
    """Pure-data config for the complete MiniMax-M3 vision-language model."""

    builder: ClassVar[str] = "megatron.bridge.models.minimax_m3.model_config.MiniMaxM3VLModelBuilder"
    scatter_embedding_sequence_parallel: bool = False
    vision_config: dict[str, Any] = field(default_factory=dict)
    hf_config_dict: dict[str, Any] = field(default_factory=dict)
    image_token_id: int = 200025
    video_token_id: int = 200026
    projector_hidden_size: int = 6144
    multimodal_projector_bias: bool = True
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    lightning_indexer_layers: list[int] = field(default_factory=list)
    index_n_heads: int = 4
    index_head_dim: int = 128
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the token IDs used by multimodal data pipelines."""
        return {"images": self.image_token_id}

    def to_text_config(self) -> MiniMaxM3TextModelConfig:
        """Return the text-only config used by pretraining and SFT recipes."""
        text_field_names = {
            config_field.name for config_field in fields(MiniMaxM3TextModelConfig) if config_field.init
        }
        values = {name: getattr(self, name) for name in text_field_names}
        values["transformer"] = replace(self.transformer)
        values["pre_wrap_hooks"] = list(self.pre_wrap_hooks)
        values["post_wrap_hooks"] = list(self.post_wrap_hooks)
        return MiniMaxM3TextModelConfig(**values)


class MiniMaxM3TextModelBuilder(LayerSpecGPTModelBuilder):
    """Build the MiniMax-M3 text backbone with its FP32 router spec."""


class MiniMaxM3VLModelBuilder(MiniMaxM3TextModelBuilder):
    """Build the MiniMax-M3 language stage and inject it into the VLM."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MiniMaxM3VLModel:
        """Build one MiniMax-M3 VLM pipeline stage."""
        config = self._model_config
        assert isinstance(config, MiniMaxM3VLModelConfig)
        language_model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        runtime_config = replace(config, transformer=replace(config.transformer))
        object.__setattr__(runtime_config, "_pg_collection", pg_collection)
        model = MiniMaxM3VLModel(
            runtime_config,
            language_model=language_model,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        return model


__all__ = [
    "MiniMaxM3TextModelBuilder",
    "MiniMaxM3TextModelConfig",
    "MiniMaxM3TopKRouter",
    "MiniMaxM3VLModelBuilder",
    "MiniMaxM3VLModelConfig",
    "_promote_router_weights_to_float32",
    "minimax_m3_block_spec",
    "quick_gelu",
]
