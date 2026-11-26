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

"""LoRA export of Inkling2's interleaved gate/up experts: lora_B rows in the HF weight's row order."""

import pytest
import torch


pytest.importorskip("transformers", minversion="5.23")

from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.inkling.inkling2_bridge import _InterleavedGatedExpertMapping


pytestmark = pytest.mark.unit


def test_lora_out_rows_follow_the_interleaved_expert_layout():
    """A LoRA delta on interleaved gate/up experts must land on the HF rows its base weight exports to."""
    mapping = _InterleavedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight3", "model.llm.layers.0.mlp.experts.w13_weight"
    )
    rank, inter, hidden = 2, 3, 4
    lora_b = torch.randn(2 * inter, rank)
    lora_a = torch.randn(rank, hidden)
    delta_megatron = lora_b @ lora_a  # [gate; up] rows
    delta_hf = mapping.hf_lora_out_rows(lora_b) @ lora_a
    assert torch.equal(delta_hf[0::2], delta_megatron[:inter]) and torch.equal(delta_hf[1::2], delta_megatron[inter:])
    # Packed adapters (a leading expert dim) are reordered per expert.
    stacked = torch.stack([lora_b, 2 * lora_b])
    assert torch.equal(mapping.hf_lora_out_rows(stacked)[1], 2 * mapping.hf_lora_out_rows(lora_b))
    # Mappings that keep Megatron's row order leave lora_B unchanged.
    plain = AutoMapping("decoder.layers.0.self_attention.linear_q.weight", "model.llm.layers.0.attn.wq_du.weight")
    assert plain.hf_lora_out_rows(lora_b) is lora_b
