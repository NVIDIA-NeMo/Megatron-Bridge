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

from types import SimpleNamespace

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import roll_tensor

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
    Qwen3VLGPTModel,
    _get_mtp_packed_seq_params,
)


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


def test_forward_forwards_output_processor_hook():
    """Forward accepts and passes output-processor arguments to postprocessing."""
    dummy = _DummyModel()

    def output_processor(**kwargs):
        return kwargs

    output_processor_context = object()

    output = Qwen3VLGPTModel.forward(
        dummy,
        input_ids=torch.zeros((1, 4), dtype=torch.long),
        position_ids=torch.zeros((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        output_processor=output_processor,
        output_processor_context=output_processor_context,
    )

    assert output == "ok"
    assert dummy.postprocess_args["output_processor"] is output_processor
    assert dummy.postprocess_args["output_processor_context"] is output_processor_context


def test_mtp_sequence_parallel_embedding_scatter_uses_tp_group(monkeypatch):
    """The MTP embedding wrapper must not fall back to global tensor-parallel state."""
    expected_group = object()
    calls = {"group": None}

    def _identity_scatter(x, *, group=None):
        calls["group"] = group
        return x

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model.tensor_parallel.scatter_to_sequence_parallel_region",
        _identity_scatter,
    )

    class _DummyEmbedding:
        word_embeddings = object()

        def __call__(self, *, input_ids, position_ids):  # noqa: ARG002
            return torch.ones(1, 1, 1)

    class _DummyMTPModel(_DummyModel):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(sequence_parallel=True)
            self.embedding = _DummyEmbedding()
            self.mtp_process = True
            self.pg_collection = SimpleNamespace(tp=expected_group)

        def _postprocess(self, **kwargs):
            self.embedding(input_ids=kwargs["input_ids"], position_ids=kwargs["position_ids"])
            return "ok"

    dummy = _DummyMTPModel()
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    position_ids = torch.zeros((1, 4), dtype=torch.long)
    attention_mask = torch.ones((1, 4), dtype=torch.long)

    output = Qwen3VLGPTModel.forward(
        dummy,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    assert output == "ok"
    assert calls["group"] is expected_group


def test_mtp_uses_padded_boundaries_for_packed_token_rolling():
    """MTP rolls within physical padded segments instead of crossing alignment gaps."""
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )

    mtp_packed_seq_params = _get_mtp_packed_seq_params(packed_seq_params)
    tokens = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 0]])
    rolled_tokens, _ = roll_tensor(tokens, packed_seq_params=mtp_packed_seq_params)

    assert mtp_packed_seq_params is not packed_seq_params
    assert mtp_packed_seq_params.cu_seqlens_q is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_q is cu_seqlens
    assert rolled_tokens.tolist() == [[2, 3, 0, 0, 5, 6, 0, 0]]


def test_mtp_postprocess_receives_padded_boundaries():
    """Qwen keeps original metadata for attention and padded offsets for MTP postprocessing."""

    class _DummyMTPModel(_DummyModel):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(sequence_parallel=False)
            self.mtp_process = True

    dummy = _DummyMTPModel()
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )

    output = Qwen3VLGPTModel.forward(
        dummy,
        input_ids=torch.zeros((1, 8), dtype=torch.long),
        position_ids=torch.zeros((3, 1, 8), dtype=torch.long),
        attention_mask=None,
        packed_seq_params=packed_seq_params,
    )

    assert output == "ok"
    assert dummy.decoder.called_with["packed_seq_params"] is packed_seq_params
    assert dummy.postprocess_args["packed_seq_params"] is not packed_seq_params
    assert dummy.postprocess_args["packed_seq_params"].cu_seqlens_q is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_q is cu_seqlens


def test_tied_mtp_state_dict_drops_redundant_output_weight():
    """MTP ranks keep the duplicated embedding as the canonical tied weight."""
    dummy = Qwen3VLGPTModel.__new__(Qwen3VLGPTModel)
    dummy.mtp_process = True
    dummy.pre_process = False
    sharded_state_dict = {
        "language_model.output_layer.weight": object(),
        "language_model.embedding.word_embeddings.weight": object(),
    }

    Qwen3VLGPTModel.tie_embeddings_and_output_weights_state_dict(
        dummy,
        sharded_state_dict,
        "language_model.output_layer.weight",
        "language_model.embedding.word_embeddings.weight",
        {},
    )

    assert "language_model.output_layer.weight" not in sharded_state_dict
    assert "language_model.embedding.word_embeddings.weight" in sharded_state_dict


def test_tied_non_mtp_state_dict_delegates_to_gpt_model(monkeypatch):
    """Non-MTP tied-output handling stays on the upstream GPTModel path."""
    calls = {}

    def fake_tie_embeddings(
        self,
        sharded_state_dict,
        output_layer_weight_key,
        first_stage_word_emb_key,
        metadata,
    ):
        calls["args"] = (self, sharded_state_dict, output_layer_weight_key, first_stage_word_emb_key, metadata)

    monkeypatch.setattr(GPTModel, "tie_embeddings_and_output_weights_state_dict", fake_tie_embeddings)

    dummy = Qwen3VLGPTModel.__new__(Qwen3VLGPTModel)
    dummy.mtp_process = False
    dummy.pre_process = False
    sharded_state_dict = {"language_model.output_layer.weight": object()}
    metadata = {"dp_cp_group": object()}

    Qwen3VLGPTModel.tie_embeddings_and_output_weights_state_dict(
        dummy,
        sharded_state_dict,
        "language_model.output_layer.weight",
        "language_model.embedding.word_embeddings.weight",
        metadata,
    )

    assert calls["args"] == (
        dummy,
        sharded_state_dict,
        "language_model.output_layer.weight",
        "language_model.embedding.word_embeddings.weight",
        metadata,
    )
