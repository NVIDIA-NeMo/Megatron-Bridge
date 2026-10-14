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

"""Input preparation for the Bridge-local Qwen4-Exp PLE module."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.qwen.modeling_qwen4_exp import per_layer_embedding
from megatron.bridge.models.qwen.modeling_qwen4_exp.model import Qwen4ExpHybridModel
from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpTransformerConfig


@pytest.fixture
def ple_module(monkeypatch: pytest.MonkeyPatch) -> per_layer_embedding.PerLayerEmbedding:
    # Keep the actual PLE constructor and prepare path; omit the distributed table allocation.
    embedding = torch.nn.Module()
    embedding.compute_ngram_ids = Mock(side_effect=lambda tokens, cu: tokens.unsqueeze(-1))
    monkeypatch.setattr(per_layer_embedding, "NGramEmbedding", lambda *args, **kwargs: embedding)
    groups = ProcessGroupCollection()
    groups.tp = Mock(spec=torch.distributed.ProcessGroup)
    groups.cp = Mock(spec=torch.distributed.ProcessGroup)
    groups.tp.size.return_value = 1
    groups.cp.size.return_value = 1
    config = Qwen4ExpTransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        mhc_num_residual_streams=4,
        mhc_gated_residual_rank=2,
        ple_layer_ids=[2],
        ple_embed_dim=8,
        params_dtype=torch.float32,
        perform_initialization=False,
    )
    return per_layer_embedding.PerLayerEmbedding(config, 2, pg_collection=groups)


@pytest.mark.unit
@pytest.mark.parametrize(
    "boundaries,expected_positions",
    [(None, [0, 1, 2, 3, 4, 5]), ([0, 3, 6], [0, 1, 2, 0, 1, 2])],
)
def test_prepare_retains_context_group(
    ple_module: per_layer_embedding.PerLayerEmbedding,
    boundaries: list[int] | None,
    expected_positions: list[int],
) -> None:
    tokens = torch.arange(6).reshape(1, 6)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32) if boundaries is not None else None
    assert ple_module.cp_group.size() == 1

    ple_module.prepare(tokens, cu_seqlens)

    ple_module.ple_embedding.compute_ngram_ids.assert_called_once_with(tokens, cu_seqlens)
    torch.testing.assert_close(ple_module._ngram_ids, tokens.unsqueeze(-1))
    torch.testing.assert_close(ple_module._local_positions(6), torch.tensor(expected_positions).reshape(6, 1))


@pytest.mark.unit
def test_prepare_rejects_context_parallelism(
    ple_module: per_layer_embedding.PerLayerEmbedding, monkeypatch: pytest.MonkeyPatch
) -> None:
    ple_module.cp_group.size.return_value = 2
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(AssertionError, match="does not support context parallelism"):
        ple_module.prepare(torch.arange(6).reshape(1, 6))

    ple_module.ple_embedding.compute_ngram_ids.assert_not_called()


@pytest.mark.unit
def test_hybrid_forward_prepares_ple_with_raw_ids_and_external_embeddings(monkeypatch):
    model = Qwen4ExpHybridModel.__new__(Qwen4ExpHybridModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(ple_layer_ids=[1], mhc_num_residual_streams=2)
    prepare = Mock()
    model.decoder = SimpleNamespace(layers=[SimpleNamespace(per_layer_embedding=SimpleNamespace(prepare=prepare))])
    model.rotary_pos_emb = torch.nn.Identity()
    parent_forward = Mock(side_effect=lambda **kwargs: kwargs["decoder_input"])
    monkeypatch.setattr(HybridModel, "forward", parent_forward)
    tokens = torch.tensor([[1, 2, 3, 4]])
    embeddings = torch.randn(4, 1, 8, requires_grad=True)
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries,
        max_seqlen_q=2,
        max_seqlen_kv=2,
    )

    output = model.forward(
        tokens,
        torch.arange(4).unsqueeze(0),
        None,
        decoder_input=embeddings,
        packed_seq_params=packed,
    )

    prepare.assert_called_once_with(tokens, boundaries)
    assert parent_forward.call_args.kwargs["input_ids"] is tokens
    torch.testing.assert_close(output, embeddings.repeat(1, 1, 2))
    output.sum().backward()
    torch.testing.assert_close(embeddings.grad, torch.full_like(embeddings, 2.0))
