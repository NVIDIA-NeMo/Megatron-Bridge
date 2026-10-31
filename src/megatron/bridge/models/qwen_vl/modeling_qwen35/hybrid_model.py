# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen3.5 vision-language composition with Core's HybridModel decoder."""

from contextvars import ContextVar
from copy import copy
from typing import Any

import torch
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from transformers import PretrainedConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import configure_qwen35_hybrid, qwen35_hybrid_stack_spec
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.token_classification import Qwen3VLForTokenClassification


_POSITION_IDS: ContextVar[torch.Tensor | None] = ContextVar("qwen35_hybrid_position_ids", default=None)


class _HybridMultimodalRotaryEmbedding(Qwen3VLMultimodalRotaryEmbedding):
    def forward(self, max_seq_len: int, *, packed_seq: bool = False) -> torch.Tensor:
        position_ids = _POSITION_IDS.get()
        if position_ids is None:
            raise RuntimeError("Qwen3.5 Hybrid mRoPE requires the current forward's position_ids.")
        return super().forward(position_ids, self.mrope_section)


def qwen35_pipeline_pattern(config: GPTModelProvider, pattern: str) -> str:
    """Place each HF block on the PP/VPP stage that the GPT decoder assigned it to.

    The provider keeps pipeline settings in HF block units. Each block owns two
    Hybrid positions, so the GPT stage boundaries become pipe separators.
    """
    pp_size = config.pipeline_model_parallel_size
    vp_size = config.virtual_pipeline_model_parallel_size
    if pp_size * (vp_size or 1) == 1:
        return pattern

    blocks = [pattern[index : index + 2] for index in range(0, len(pattern), 2)]
    layout = config.pipeline_model_parallel_layout
    segments, offset = [], 0
    # Hybrid numbers segments as vp_stage * pp_size + pp_rank, which is also the
    # order in which the GPT decoder assigns consecutive layers to stages.
    for vp_stage in range(vp_size or 1):
        for pp_rank in range(pp_size):
            if layout is not None:
                count = layout.layout[pp_rank][vp_stage].count(LayerType.decoder)
            else:
                count = get_num_layers_to_build(config, vp_stage if vp_size else None, pp_rank)
            segments.append("".join(blocks[offset : offset + count]))
            offset += count
    if offset != len(blocks):
        raise ValueError(f"Pipeline stages hold {offset} layers, but the language model has {len(blocks)}.")
    return "|".join(segments)


def qwen35_vl_hybrid_stack_spec() -> ModuleSpec:
    """Use Qwen's absolute multimodal rotary attention in the Hybrid stack."""
    spec = qwen35_hybrid_stack_spec()
    spec.submodules.attention_layer.submodules.self_attention.module = Qwen3VLSelfAttention
    return spec


class Qwen35VLHybridLanguageModel(HybridModel):
    """Adapt the existing VL provider's HF block coordinates and mRoPE inputs."""

    def __init__(
        self,
        config: GPTModelProvider,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        *,
        position_embedding_type: str = "mrope",
        mtp_block_spec: ModuleSpec | None = None,
        **kwargs: Any,
    ) -> None:
        text_config = PretrainedConfig(
            num_hidden_layers=config.num_layers,
            num_experts=config.num_moe_experts,
            full_attention_interval=config.linear_attention_freq,
        )
        hybrid_kwargs = {}
        configure_qwen35_hybrid(hybrid_kwargs, text_config)
        pattern = qwen35_pipeline_pattern(config, hybrid_kwargs.pop("hybrid_layer_pattern"))
        hybrid_kwargs.pop("hybrid_stack_spec")
        hybrid_config = copy(config)
        for name, value in hybrid_kwargs.items():
            setattr(hybrid_config, name, value)
        # The stage boundaries now live in the pattern's pipe separators.
        hybrid_config.num_layers_in_first_pipeline_stage = None
        hybrid_config.num_layers_in_last_pipeline_stage = None
        hybrid_config.pipeline_model_parallel_layout = None
        if config.mtp_num_layers:
            pattern += "/" + "/".join([hybrid_config.mtp_hybrid_override_pattern] * config.mtp_num_layers)

        super().__init__(
            config=hybrid_config,
            hybrid_stack_spec=transformer_layer_spec,
            hybrid_layer_pattern=pattern,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type="rope",
            **kwargs,
        )
        self.rotary_pos_emb = _HybridMultimodalRotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=kwargs.get("rotary_percent", config.rotary_percent),
            rotary_interleaved=config.rotary_interleaved,
            rotary_base=kwargs.get("rotary_base", 10000),
            seq_len_interpolation_factor=kwargs.get("seq_len_interpolation_factor"),
            cp_group=self.pg_collection.cp,
        )
        self.rotary_pos_emb.mrope_section = config.mrope_section

        if self.mtp_process:
            self.mtp.register_forward_pre_hook(self._mtp_token_embedding_inputs, with_kwargs=True)

    @staticmethod
    def _mtp_token_embedding_inputs(
        _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # The GPT VLM conditions MTP on shifted token embeddings. HybridModel
        # otherwise reuses the composed vision-language embeddings, changing
        # image positions and boundary values used by the MoE auxiliary loss.
        kwargs["decoder_input"] = None
        return args, kwargs

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Pass explicit multimodal positions to Core's rotary preparation and MTP path."""
        if deepstack_visual_embeds:
            raise ValueError("Qwen3.5 does not use DeepStack visual embeddings.")
        token = _POSITION_IDS.set(position_ids)
        try:
            return super().forward(input_ids, position_ids, attention_mask, **kwargs)
        finally:
            _POSITION_IDS.reset(token)


class Qwen35VLHybridModel(Qwen3VLModel):
    """Reuse Qwen vision encoding and modality insertion with a Hybrid language model."""

    language_model_class = Qwen35VLHybridLanguageModel

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        super()._expose_language_model_for_cuda_graph_helper()
        self.position_embedding_type = "mrope"


class Qwen35HybridForTokenClassification(Qwen3VLForTokenClassification):
    """Qwen3.5's existing classifier head over the Hybrid language model."""

    language_model_class = Qwen35VLHybridLanguageModel

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        super()._expose_language_model_for_cuda_graph_helper()
        self.position_embedding_type = "mrope"
