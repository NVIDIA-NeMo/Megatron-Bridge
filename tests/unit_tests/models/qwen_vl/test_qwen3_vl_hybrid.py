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

import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
    Qwen3VLHybridModel,
    Qwen3VLHybridStack,
    get_qwen3_vl_hybrid_stack_spec,
)


def test_hybrid_stack_spec_uses_qwen_attention():
    config = SimpleNamespace(transformer_impl="transformer_engine", restore_modelopt_state=False)

    spec = get_qwen3_vl_hybrid_stack_spec(config)

    assert spec.module is Qwen3VLHybridStack
    assert spec.submodules.attention_layer.submodules.self_attention.module is Qwen3VLSelfAttention


def test_deepstack_embedding_is_added_after_logical_mlp_layer():
    stack = Qwen3VLHybridStack.__new__(Qwen3VLHybridStack)
    torch.nn.Module.__init__(stack)
    stack.layer_type_list = [Symbols.ATTENTION, Symbols.MLP]
    attention_layer = torch.nn.Identity()
    attention_layer.layer_number = 1
    mlp_layer = torch.nn.Identity()
    mlp_layer.layer_number = 2
    stack.layers = torch.nn.ModuleList([attention_layer, mlp_layer])
    hidden_states = torch.zeros(3, 1, 2)
    visual_pos_masks = torch.tensor([[False, True, False]])
    visual_embeds = (torch.tensor([[2.0, 3.0]]),)

    after_attention = stack._maybe_add_deepstack_embedding(
        hidden_states,
        0,
        visual_pos_masks,
        visual_embeds,
    )
    after_mlp = stack._maybe_add_deepstack_embedding(
        hidden_states,
        1,
        visual_pos_masks,
        visual_embeds,
    )

    torch.testing.assert_close(after_attention, hidden_states)
    torch.testing.assert_close(after_mlp[1, 0], visual_embeds[0][0])


def test_multimodal_context_does_not_replace_hybrid_rotary_embedding():
    stack = Qwen3VLHybridStack.__new__(Qwen3VLHybridStack)
    torch.nn.Module.__init__(stack)
    visual_pos_masks = torch.tensor([[True]])
    deepstack_visual_embeds = [torch.ones(1, 2)]
    stack.set_multimodal_context(visual_pos_masks, deepstack_visual_embeds)

    assert not hasattr(stack, "_qwen_rotary_pos_emb")
    assert stack._qwen_visual_pos_masks is visual_pos_masks
    assert stack._qwen_deepstack_visual_embeds is deepstack_visual_embeds
    stack.clear_multimodal_context()
    assert stack._qwen_visual_pos_masks is None
    assert stack._qwen_deepstack_visual_embeds is None


def test_mrope_adapter_uses_explicit_positions_for_hybrid_call():
    rope = Qwen3VLMultimodalRotaryEmbedding.__new__(Qwen3VLMultimodalRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.inv_freq = torch.ones(4)
    rope.rotary_interleaved = False
    rope.seq_len_interpolation_factor = None
    rope.mrope_section = [1, 1, 2]
    rope.is_thd_format = False
    rope.cp_group = SimpleNamespace(size=lambda: 1)
    rope._position_ids_context = None
    rope._mrope_section_context = None
    position_ids = torch.tensor([[[0, 1]], [[0, 2]], [[0, 3]]])
    rope.set_forward_context(position_ids, [1, 1, 2])

    expected = rope(position_ids, [1, 1, 2])
    actual = rope(3, packed_seq=False)

    torch.testing.assert_close(actual, expected)


def test_hybrid_model_temporarily_enables_rope_path_for_parent_forward(monkeypatch):
    class _Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context = None

        def set_forward_context(self, *context):
            self.context = context

        def clear_forward_context(self):
            self.context = None

    class _Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context = None

        def set_multimodal_context(self, *context):
            self.context = context

        def clear_multimodal_context(self):
            self.context = None

    model = Qwen3VLHybridModel.__new__(Qwen3VLHybridModel)
    torch.nn.Module.__init__(model)
    model.position_embedding_type = "mrope"
    model.mrope_section = [1, 1, 2]
    model.rotary_pos_emb = _Rotary()
    model.decoder = _Decoder()
    observed = {}

    def _parent_forward(instance, **kwargs):
        observed["position_embedding_type"] = instance.position_embedding_type
        return kwargs["decoder_input"]

    monkeypatch.setattr(HybridModel, "forward", _parent_forward)
    decoder_input = torch.randn(2, 1, 4)
    result = model.forward(
        input_ids=torch.ones(1, 2, dtype=torch.long),
        position_ids=torch.zeros(3, 1, 2, dtype=torch.long),
        attention_mask=None,
        decoder_input=decoder_input,
    )

    assert result is decoder_input
    assert observed["position_embedding_type"] == "rope"
    assert model.position_embedding_type == "mrope"
    assert model.rotary_pos_emb.context is None
    assert model.decoder.context is None
