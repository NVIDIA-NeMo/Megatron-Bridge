# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU interface tests for Qwen3-VL DeepStack across the MCore MIMO forward paths."""

from unittest.mock import Mock

import pytest
import torch
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import ModuleLayout, ModuleStageInfo, RankRole
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import (
    Qwen3VLMIMOModel,
    Qwen3VLMIMOVisionModel,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider


pytestmark = pytest.mark.unit


class _TextEmbedding(torch.nn.Embedding):
    def forward(self, input_ids, position_ids):
        return super().forward(input_ids).transpose(0, 1)


class _LanguageModel(torch.nn.Module):
    def __init__(self, config, max_sequence_length):
        super().__init__()
        self.config = config
        self.embedding = _TextEmbedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input,
        labels,
        loss_mask,
        mtp_input_mask,
        packed_seq_params,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        output = decoder_input.transpose(0, 1).clone()
        for features in deepstack_visual_embeds:
            output[visual_pos_masks] += features
        return output.transpose(0, 1)


class _VisionModel(Qwen3VLMIMOVisionModel):
    def __init__(self):
        # Keep the real Qwen MIMO packing forward without constructing GPU vision layers.
        torch.nn.Module.__init__(self)


@pytest.fixture
def mimo_model(monkeypatch):
    def vision_forward(self, hidden_states, grid_thw, inference_params=None, extra_block_kwargs=None):
        return hidden_states[0], list(hidden_states[1:])

    monkeypatch.setattr(Qwen3VLVisionModel, "forward", vision_forward)
    config = Qwen3VLModelProvider(
        num_layers=3,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=4,
        vocab_size=32,
        vision_config=Qwen3VLVisionConfig(
            depth=3,
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            out_hidden_size=8,
            deepstack_visual_indexes=[0, 1, 2],
        ),
    )
    return Qwen3VLMIMOModel(
        MimoModelConfig(
            language_model_spec=ModuleSpec(module=_LanguageModel, params={"config": config, "max_sequence_length": 4}),
            modality_submodules_spec={
                "images": ModuleSpec(
                    module=VisionModalitySubmodules,
                    submodules={"encoders": {"qwen_visual": ModuleSpec(module=_VisionModel)}},
                )
            },
            special_token_ids={"images": 31},
        )
    )


class TestQwen3VLMIMOModel:
    @pytest.mark.parametrize("colocated", [True, False], ids=["forward_all_modules", "forward_language_module"])
    def test_deepstack_features_reach_language_forward(self, mimo_model, monkeypatch, colocated):
        # Use distinct values for every image row and DeepStack layer to detect swaps.
        input_ids = torch.tensor([[1, 31, 2, 31], [31, 3, 31, 4]])
        image_mask = input_ids == 31
        hidden_size = mimo_model.config.hidden_size
        deepstack_count = len(mimo_model.config.vision_config.deepstack_visual_indexes)
        features = torch.arange((deepstack_count + 1) * 4 * hidden_size, dtype=torch.float32)
        features = features.reshape(deepstack_count + 1, 4, hidden_size).requires_grad_()
        modality_inputs = {"images": {"qwen_visual": {"hidden_states": features, "grid_thw": None}}}
        payloads = []
        vision = mimo_model.modality_submodules["images"].encoders["qwen_visual"]
        vision.register_forward_hook(lambda module, args, output: payloads.append(output))
        lm_forward = Mock(wraps=mimo_model.language_model.forward)
        monkeypatch.setattr(mimo_model.language_model, "forward", lm_forward)

        # Exercise stock MCore dispatch and input preparation; only encoder/LM math is stubbed.
        if not colocated:
            payload = mimo_model.modality_submodules["images"](encoder_inputs=modality_inputs["images"])
            mimo_model.role = RankRole(
                modules={"language": ModuleStageInfo(is_first_stage=True, is_last_stage=True)},
                mode=ModuleLayout.NON_COLOCATED,
            )
            mimo_model.set_input_tensor({"images": payload})
        output, _ = mimo_model(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            modality_inputs=modality_inputs if colocated else None,
        )

        assert len(payloads) == 1
        assert payloads[0].shape == (4, hidden_size * (deepstack_count + 1))
        torch.testing.assert_close(payloads[0], torch.cat(list(features.unbind()), dim=-1), rtol=0, atol=0)
        lm_forward.assert_called_once()
        inputs = lm_forward.call_args.kwargs
        torch.testing.assert_close(inputs["visual_pos_masks"], image_mask)
        decoder_input = inputs["decoder_input"].transpose(0, 1)
        assert decoder_input.shape == (*input_ids.shape, hidden_size)
        torch.testing.assert_close(decoder_input[image_mask], features[0], rtol=0, atol=0)
        text_embeddings = mimo_model.language_model.embedding.weight[input_ids[~image_mask]]
        torch.testing.assert_close(decoder_input[~image_mask], text_embeddings, rtol=0, atol=0)
        assert len(inputs["deepstack_visual_embeds"]) == deepstack_count
        for actual, expected in zip(inputs["deepstack_visual_embeds"], features[1:], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Verify that the packing/splitting boundary preserves the auxiliary gradient paths.
        output.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features), rtol=0, atol=0)
