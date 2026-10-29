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

"""Step3.7 model provider.

Mirrors ``qwen_vl/qwen3_vl_provider.py``: extends the text-decoder provider
with multimodal fields (vision config, image token id, projector knobs) and
returns a :class:`Step37Model` instance instead of a bare ``GPTModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.modelling_step37.transformer_block import (
    get_step37_text_layer_spec,
)
from megatron.bridge.models.stepfun.step35_provider import Step35ModelProvider


@dataclass
class Step37ModelProvider(Step35ModelProvider):
    """Model provider for Step3.7.

    Inherits every Step-3.5 text-decoder field from
    :class:`Step35ModelProvider` (per-layer ``layer_types`` /
    ``rotary_percents`` / ``swiglu_limits``, ``head_wise_attn_gate``, MoE
    settings, MTP layers, sliding-attention overrides) and adds the multimodal
    fields needed to build :class:`Step37Model`.
    """

    # Override the GPTModelProvider default ("learned_absolute") so any path
    # that constructs Step37ModelProvider directly (recipe-only, no bridge;
    # bridge that forgets to assign) still gets a legal pairing with
    # ``rotary_base_per_layer``. Mirrors ``Qwen3VLModelProvider`` which sets
    # ``position_embedding_type: str = "mrope"`` for the same reason.
    position_embedding_type: str = "rope"

    # Vision / multimodal fields surfaced through Step37Bridge.provider_bridge.
    vision_config: Optional[Any] = None
    image_token_id: int = 128001
    understand_projector_stride: int = 2
    projector_bias: bool = False
    language_max_sequence_length: int = 262144

    # Freeze knobs (mirroring Qwen3VLModelProvider).
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # PP / encoder/decoder gating (mirroring Qwen3VLModelProvider).
    add_encoder: bool = True
    add_decoder: bool = True

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> Step37Model:
        """Build a :class:`Step37Model` for the current PP/VP stage."""
        # Reuse Step-3.5's hybrid layer spec — Step3.7 keeps that decoder
        # verbatim. ``transformer_layer_spec`` is set on the provider by
        # Step35Bridge.provider_bridge when moe_layers_enum is present, so
        # we honour that override; otherwise we fall back to the helper.
        layer_spec = self.transformer_layer_spec
        if not callable(layer_spec) and layer_spec is None:
            layer_spec = get_step37_text_layer_spec
        if callable(layer_spec):
            language_transformer_layer_spec = layer_spec(self, vp_stage=vp_stage)
        else:
            language_transformer_layer_spec = layer_spec

        model = Step37Model(
            language_transformer_config=self,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=self.vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
            add_encoder=self.add_encoder,
            add_decoder=self.add_decoder,
            vp_stage=vp_stage,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> GPTModel:
        """Provide just the text decoder (no vision tower)."""
        return GPTModelProvider.provide(
            self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )


__all__ = ["Step37ModelProvider"]
