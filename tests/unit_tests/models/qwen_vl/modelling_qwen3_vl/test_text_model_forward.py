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

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import text_model as text_model_mod
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel


class _DummyCP:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _DummyEmbedding:
    word_embeddings = object()

    def __init__(self, output=None):
        self.output = output if output is not None else torch.arange(8, dtype=torch.float32).view(8, 1, 1)
        self.calls = []

    def __call__(self, input_ids, position_ids):
        self.calls.append((input_ids, position_ids))
        return self.output


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
        self.config = SimpleNamespace(sequence_parallel=False)
        self.pg_collection = SimpleNamespace(cp=_DummyCP())
        self._embedding = _DummyEmbedding()
        self.call_embedding_in_postprocess = False
        self.mtp_embedding_output = None
        self.mtp_embedding_word_embeddings = None
        self.preprocess_output = None
        self.postprocess_args = None

    def __getattr__(self, name):
        if name == "embedding":
            return self._embedding
        raise AttributeError(name)

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
        if self.call_embedding_in_postprocess:
            self.mtp_embedding_output = self.embedding(kwargs["input_ids"], kwargs["position_ids"])
            self.mtp_embedding_word_embeddings = self.embedding.word_embeddings
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


def test_mtp_embedding_wrapper_applies_cp_shard_before_sp_scatter(monkeypatch):
    """MTP embedding path should mirror main path ordering: CP shard, then SP scatter."""
    dummy = _DummyModel()
    dummy.mtp_process = True
    dummy.config.sequence_parallel = True
    dummy.pg_collection = SimpleNamespace(cp=_DummyCP(size=2, rank=1))
    dummy.call_embedding_in_postprocess = True
    calls = []

    def fake_split_data_cp_rank(tensor, cp_size, seq_dim, cp_rank):
        calls.append(("cp", tensor.clone(), cp_size, seq_dim, cp_rank))
        return tensor[:4] + 100

    def fake_scatter_to_sequence_parallel_region(tensor):
        calls.append(("sp", tensor.clone()))
        return tensor[::2] + 1000

    monkeypatch.setattr(text_model_mod, "split_data_cp_rank", fake_split_data_cp_rank)
    monkeypatch.setattr(
        text_model_mod.tensor_parallel,
        "scatter_to_sequence_parallel_region",
        fake_scatter_to_sequence_parallel_region,
    )

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
    assert [call[0] for call in calls] == ["cp", "sp"]
    assert calls[0][2:] == (2, 0, 1)
    torch.testing.assert_close(calls[0][1], dummy._embedding.output)
    torch.testing.assert_close(calls[1][1], dummy._embedding.output[:4] + 100)
    torch.testing.assert_close(dummy.mtp_embedding_output, (dummy._embedding.output[:4] + 100)[::2] + 1000)
    assert dummy.mtp_embedding_word_embeddings is dummy._embedding.word_embeddings
    assert "embedding" not in dummy.__dict__
    assert dummy.embedding is dummy._embedding


def test_mtp_embedding_wrapper_runs_for_context_parallel_without_sequence_parallel(monkeypatch):
    """CP-only MTP still needs the wrapper even when sequence_parallel is false."""
    dummy = _DummyModel()
    dummy.mtp_process = True
    dummy.pg_collection = SimpleNamespace(cp=_DummyCP(size=2, rank=0))
    dummy.call_embedding_in_postprocess = True
    calls = []

    def fake_split_data_cp_rank(tensor, cp_size, seq_dim, cp_rank):
        calls.append(("cp", cp_size, seq_dim, cp_rank))
        return tensor[:4]

    def fake_scatter_to_sequence_parallel_region(_):
        raise AssertionError("sequence-parallel scatter should not run when sequence_parallel is false")

    monkeypatch.setattr(text_model_mod, "split_data_cp_rank", fake_split_data_cp_rank)
    monkeypatch.setattr(
        text_model_mod.tensor_parallel,
        "scatter_to_sequence_parallel_region",
        fake_scatter_to_sequence_parallel_region,
    )

    Qwen3VLGPTModel.forward(
        dummy,
        input_ids=torch.zeros((1, 4), dtype=torch.long),
        position_ids=torch.zeros((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
    )

    assert calls == [("cp", 2, 0, 0)]
    torch.testing.assert_close(dummy.mtp_embedding_output, dummy._embedding.output[:4])


def test_mtp_embedding_wrapper_skips_cp_shard_for_packed_sequence_params(monkeypatch):
    """Packed/THD paths handle CP separately, matching the guard in model.py."""
    dummy = _DummyModel()
    dummy.mtp_process = True
    dummy.config.sequence_parallel = True
    dummy.pg_collection = SimpleNamespace(cp=_DummyCP(size=2, rank=0))
    dummy.call_embedding_in_postprocess = True
    calls = []

    def fake_split_data_cp_rank(*_):
        raise AssertionError("CP shard should not run for packed sequence params")

    def fake_scatter_to_sequence_parallel_region(tensor):
        calls.append("sp")
        return tensor + 1000

    monkeypatch.setattr(text_model_mod, "split_data_cp_rank", fake_split_data_cp_rank)
    monkeypatch.setattr(
        text_model_mod.tensor_parallel,
        "scatter_to_sequence_parallel_region",
        fake_scatter_to_sequence_parallel_region,
    )

    Qwen3VLGPTModel.forward(
        dummy,
        input_ids=torch.zeros((1, 4), dtype=torch.long),
        position_ids=torch.zeros((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        packed_seq_params=object(),
    )

    assert calls == ["sp"]
    torch.testing.assert_close(dummy.mtp_embedding_output, dummy._embedding.output + 1000)
