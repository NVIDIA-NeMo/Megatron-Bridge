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

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig

import megatron.bridge.models.qwen_vl.modeling_qwen25_vl as qwen25_modeling
from megatron.bridge.models.qwen_vl.model_config import Qwen25VLModelConfig


def test_qwen25_model_constructs_vision_and_language_modules(monkeypatch):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    model_config = Qwen25VLModelConfig(
        transformer=transformer,
        vocab_size=32,
        vision_config={"hidden_size": 16},
    )
    vision_config = SimpleNamespace()
    vision_model = SimpleNamespace()
    language_model = SimpleNamespace(shared_embedding_or_output_weight="shared")
    captured = {}

    def build_vision_model(config):
        captured["vision_config"] = config
        return vision_model

    monkeypatch.setattr(qwen25_modeling.Qwen2_5_VisionTransformerPretrainedModel, "_from_config", build_vision_model)
    monkeypatch.setattr(qwen25_modeling, "hook_hf_module_setattr_for_tp_grad_sync", lambda model: None)

    def build_language_model(**kwargs):
        captured["language_kwargs"] = kwargs
        return language_model

    monkeypatch.setattr(qwen25_modeling, "GPTModel", build_language_model)

    model = qwen25_modeling.Qwen25VLModel(
        language_transformer_config=transformer,
        language_transformer_layer_spec=SimpleNamespace(),
        vision_transformer_config=vision_config,
        model_config=model_config,
        language_vocab_size=32,
        pg_collection=SimpleNamespace(tp=object()),
    )

    assert captured["vision_config"] is vision_config
    assert model.visual is vision_model
    assert model.language_model is language_model
    assert captured["language_kwargs"]["config"] is transformer
    assert captured["language_kwargs"]["pg_collection"] is model.pg_collection
    assert model.shared_embedding_or_output_weight == "shared"


def test_qwen25_packed_forward_resets_mrope_per_sequence(monkeypatch):
    model = Mock()
    model.pre_process = True
    model.config = SimpleNamespace(sequence_parallel=False, image_token_id=91, video_token_id=92)
    model.language_model = Mock()
    model.language_model.embedding.return_value = torch.randn(8, 1, 4)
    expected_row_positions = torch.tensor(
        [
            [[0, 1, 0], [0, 1, 2]],
            [[0, 1, 0], [10, 11, 12]],
            [[0, 1, 0], [20, 21, 22]],
        ]
    )
    model.get_rope_index = Mock(return_value=(expected_row_positions, torch.zeros(2, 1)))
    model.language_model.forward.return_value = torch.tensor(1.0)
    monkeypatch.setattr(qwen25_modeling, "is_transformers_min_version", lambda version: True)

    input_ids = torch.tensor([[10, 11, 0, 0, 20, 21, 22, 0]])
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )

    qwen25_modeling.Qwen25VLModel.forward(
        model,
        input_ids=input_ids,
        packed_seq_params=packed_seq_params,
    )

    rope_input_ids = model.get_rope_index.call_args.args[0]
    rope_attention_mask = model.get_rope_index.call_args.kwargs["attention_mask"]
    assert rope_input_ids.tolist() == [[10, 11, 0], [20, 21, 22]]
    assert rope_attention_mask.tolist() == [[True, True, False], [True, True, True]]
    packed_position_ids = model.language_model.forward.call_args.kwargs["position_ids"]
    assert packed_position_ids.tolist() == [
        [[0, 1, 0, 0, 0, 1, 2, 0]],
        [[0, 1, 0, 0, 10, 11, 12, 0]],
        [[0, 1, 0, 0, 20, 21, 22, 0]],
    ]
