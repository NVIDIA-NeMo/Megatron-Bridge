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

import megatron.bridge.models.qwen_vl.modeling_qwen25_vl as qwen25_modeling


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
