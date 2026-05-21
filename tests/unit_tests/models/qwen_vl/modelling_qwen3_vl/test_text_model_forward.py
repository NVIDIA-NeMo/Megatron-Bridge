# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for Qwen3VL text model forward behavior."""

import torch

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel


class _DummyDecoder:
    def __init__(self):
        self.called_with = None

    def __call__(self, **kwargs):
        self.called_with = kwargs
        return torch.zeros(1, 1, 1)


class _DummyModel:
    def __init__(self):
        self.decoder = _DummyDecoder()
        self.mtp_process = False
        self.preprocess_output = None
        self.postprocess_args = None

    def _preprocess(self, **_):
        self.preprocess_output = (
            torch.randn(1, 1, 1),
            torch.randn(1, 1),
            torch.randn(1, 1),
            torch.randn(1, 1),
            torch.tensor([0]),
            torch.randn(1, 1),
        )
        return self.preprocess_output

    def _postprocess(self, **kwargs):
        self.postprocess_args = kwargs
        return "ok"


def test_forward_accepts_extra_preprocess_output():
    """Ensure forward ignores extra values returned by _preprocess."""
    dummy = _DummyModel()
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    position_ids = torch.zeros((1, 4), dtype=torch.long)
    attention_mask = torch.ones((1, 4), dtype=torch.long)

    output = Qwen3VLGPTModel.forward(
        dummy,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    preproc = dummy.preprocess_output
    assert output == "ok"
    assert dummy.decoder.called_with["hidden_states"] is preproc[0]
    assert dummy.decoder.called_with["rotary_pos_emb"] is preproc[1]
    assert dummy.decoder.called_with["rotary_pos_cos"] is preproc[2]
    assert dummy.decoder.called_with["rotary_pos_sin"] is preproc[3]
    assert dummy.decoder.called_with["sequence_len_offset"] is preproc[4]
    assert not any(value is preproc[5] for value in dummy.decoder.called_with.values())
    assert dummy.postprocess_args["decoder_input"] is preproc[0]
