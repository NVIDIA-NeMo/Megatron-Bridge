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

"""Provider-neutral EXAONE model configurations."""

from copy import copy
from dataclasses import dataclass, field
from typing import Callable, ClassVar

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_decoder_layer_specs
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.exaone.layer_specs import exaone4_layer_spec
from megatron.bridge.models.gpt.model_builder import LayerSpecGPTModelBuilder
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


class _MTPDenseLayerSpecsList(list):
    """Return a dense layer spec when MCore asks which spec to use for MTP."""

    def __init__(self, data: list[ModuleSpec], dense_mtp_spec: ModuleSpec) -> None:
        super().__init__(data)
        self._dense_mtp_spec = dense_mtp_spec

    def __getitem__(self, index):
        if isinstance(index, int) and index < 0:
            return self._dense_mtp_spec
        return super().__getitem__(index)


def build_exaone_moe_layer_spec(
    config: BridgeGPTModelConfig,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    """Build EXAONE MoE decoder specs while keeping MTP sub-layers dense."""
    transformer = config.transformer
    block = get_gpt_decoder_block_spec(
        transformer,
        use_transformer_engine=HAVE_TE,
        vp_stage=vp_stage,
    )
    if transformer.mtp_num_layers:
        dense_config = copy(transformer)
        dense_config.moe_layer_freq = [0] * transformer.num_layers
        dense_config.num_moe_experts = None
        dense_config.moe_grouped_gemm = False
        dense_mtp_spec = get_gpt_decoder_layer_specs(
            dense_config,
            use_transformer_engine=HAVE_TE,
            vp_stage=vp_stage,
        )[-1]
        block.layer_specs = _MTPDenseLayerSpecsList(block.layer_specs, dense_mtp_spec)
    return block


@dataclass(kw_only=True)
class Exaone4ModelConfig(BridgeGPTModelConfig):
    """GPT build config preserving EXAONE's post-LN and Llama 3 RoPE settings."""

    builder: ClassVar[str] = "megatron.bridge.models.exaone.model_config.Exaone4ModelBuilder"
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: exaone4_layer_spec)
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 4.0
    rope_scaling_original_max_position_embeddings: int = 8192


class Exaone4ModelBuilder(GPTModelBuilder):
    """GPT builder applying all configurable Llama 3 frequency bands."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build EXAONE and replace the default fixed-band Llama 3 frequencies."""
        model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        if not self._model_config.rope_scaling or not hasattr(model, "rotary_pos_emb"):
            return model

        rotary = model.rotary_pos_emb
        dim = self._model_config.kv_channels
        if self._model_config.rotary_percent < 1.0:
            dim = int(dim * self._model_config.rotary_percent)
        unscaled_inv_freq = 1.0 / (
            self._model_config.rotary_base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=rotary.inv_freq.device) / dim)
        )
        rotary.inv_freq = rotary._apply_scaling(
            unscaled_inv_freq,
            factor=self._model_config.rope_scaling_factor,
            low_freq_factor=self._model_config.rope_scaling_low_freq_factor,
            high_freq_factor=self._model_config.rope_scaling_high_freq_factor,
            original_max_position_embeddings=self._model_config.rope_scaling_original_max_position_embeddings,
        )
        return model


@dataclass(kw_only=True)
class ExaoneMoeModelConfig(BridgeGPTModelConfig):
    """Builder config preserving EXAONE MoE's hybrid attention and dense MTP."""

    builder: ClassVar[str] = "megatron.bridge.models.exaone.model_config.ExaoneMoeModelBuilder"
    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(
        default_factory=lambda: build_exaone_moe_layer_spec
    )


class ExaoneMoeModelBuilder(LayerSpecGPTModelBuilder):
    """Build EXAONE MoE with the configured hybrid decoder spec."""


__all__ = [
    "Exaone4ModelBuilder",
    "Exaone4ModelConfig",
    "ExaoneMoeModelBuilder",
    "ExaoneMoeModelConfig",
    "build_exaone_moe_layer_spec",
]
