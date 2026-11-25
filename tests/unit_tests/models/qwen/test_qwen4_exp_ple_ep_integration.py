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

"""PLE conversion and hash/SP integration; collectives are recorded, not GPU-qualified."""

from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from torch import nn

from megatron.bridge.models.qwen import qwen4_exp_bridge as bridge
from megatron.bridge.models.qwen.modeling_qwen4_exp import per_layer_embedding as ple
from megatron.bridge.models.qwen.modeling_qwen4_exp.ep_ngram_embedding import FrozenEPNgramEmbedding
from megatron.bridge.models.qwen.modeling_qwen4_exp.model import Qwen4ExpHybridModel


pytestmark = pytest.mark.unit


class _Source(Mapping):
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.tensors[key]

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self):
        return len(self.tensors)


class _Mapping(bridge.PLENGramEmbeddingMapping):
    # Deliberately inconsistent TP and EP ownership detects accidental TP fallback.
    tp_rank = 0
    tp_size = 2
    tp_group = object()

    def broadcast_from_pp_rank(self, tensor, **kwargs):
        return tensor


def _destination(rank, size, rows, group):
    module = FrozenEPNgramEmbedding.__new__(FrozenEPNgramEmbedding)
    nn.Module.__init__(module)
    module.ep_rank, module.ep_size, module.sharding_group = rank, size, group
    module.weight = nn.Parameter(rows.clone(), requires_grad=False)
    return module


def _aliases(mapping, source):
    # Exercise import preprocessing, not only the hook: it must remain lazy too.
    owner = bridge.Qwen4ExpBridge()
    owner.hf_config = SimpleNamespace(quantization_config={"weight_block_size": [2, 2]})
    result = owner._load_hf_weights(mapping.hf_param, source)
    assert source.reads == []
    assert list(result) == list(mapping.hf_param)
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_lazy_ep_import_uses_destination_and_only_overlapping_shards(dtype):
    full = (torch.arange(512 * 2, dtype=torch.float32).view(512, 2) % 16).to(dtype)
    mapping = _Mapping("decoder.layers.*.ple.weight", "model.layers.*.shard_{}.weight", 128).resolve(("3",))
    assert mapping.megatron_param == "decoder.layers.3.ple.weight"
    assert not mapping.is_expert
    tensors = {mapping.hf_param[f"shard_{i}"]: full[i * 4 : (i + 1) * 4] for i in range(128)}
    if dtype == torch.float8_e4m3fn:
        tensors.update({name + "_scale_inv": torch.tensor([[0.5], [2.0]]) for name in list(tensors)})
    source = _Source(tensors)
    lazy = _aliases(mapping, source)
    module = _destination(1, 32, torch.empty(16, 2, dtype=torch.bfloat16), object())
    actual = mapping.hf_to_megatron(lazy, module)
    expected = full[16:32].float()
    if dtype == torch.float8_e4m3fn:
        expected *= torch.tensor([0.5, 0.5, 2.0, 2.0]).repeat(4)[:, None]
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    names = [mapping.hf_param[f"shard_{i}"] for i in (0, 127, 4, 5, 6, 7)]
    expected_reads = (
        [key for name in names for key in (name, name + "_scale_inv")] if dtype == torch.float8_e4m3fn else names
    )
    assert source.reads == expected_reads


@pytest.mark.parametrize("mapped", [False, True])
def test_non_ple_import_keeps_fp8_sidecar_preprocessing(mapped):
    owner = bridge.Qwen4ExpBridge()
    owner.hf_config = SimpleNamespace(quantization_config={"weight_block_size": [2, 2]})
    codes = torch.ones(4, 2, dtype=torch.float8_e4m3fn)
    source = {"linear.weight": codes, "linear.weight_scale_inv": torch.tensor([[0.5], [2.0]])}
    result = owner._load_hf_weights({"weight": "linear.weight"} if mapped else "linear.weight", source)
    torch.testing.assert_close(
        result["weight"] if mapped else result, torch.tensor([[0.5, 0.5]] * 2 + [[2.0, 2.0]] * 2)
    )


@pytest.mark.parametrize("rank", range(4))
def test_single_hf_tensor_retains_original_key_and_ep_rows(rank):
    full = torch.arange(48, dtype=torch.float32).view(24, 2)
    name = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    mapping = _Mapping("decoder.layers.0.per_layer_embedding.ple_embedding.ngram_embedding.weight", {"shard_0": name})
    source = _Source({name: full})
    module = _destination(rank, 4, torch.empty(6, 2), object())
    actual = mapping.hf_to_megatron(_aliases(mapping, source), module)
    torch.testing.assert_close(actual, full[rank * 6 : (rank + 1) * 6], rtol=0, atol=0)
    assert source.reads == [name]
    assert list(mapping.megatron_to_hf(actual, module)) == [name]


@pytest.mark.parametrize(
    "ep_size,shards,dtype", [(1, 3, torch.float32), (2, 3, torch.bfloat16), (4, 5, torch.float32)]
)
def test_lazy_export_uses_ep_group_and_exact_owner_intersections(monkeypatch, ep_size, shards, dtype):
    full = (torch.arange(48, dtype=torch.float32).view(24, 2) - 19).to(dtype)
    full[0, 0] = -0.0
    full[8, 1] = -0.0
    words = full.view(torch.int16 if dtype == torch.bfloat16 else torch.int32)
    words[15, 0] = 0x7FC1 if dtype == torch.bfloat16 else 0x7FC12345
    words[16, 1] = 0x7FE3 if dtype == torch.bfloat16 else 0x7FE54321
    mapping = _Mapping("ple.weight", "hf.shard_{}", shards)
    ep_group = object()
    global_ranks = [13, 4, 29, 8][:ep_size]
    calls = []
    local_rows = full.shape[0] // ep_size

    def global_rank(group, owner):
        assert group is ep_group
        return global_ranks[owner]

    def broadcast(tensor, src, group):
        assert group is ep_group and tensor.is_contiguous()
        owner = global_ranks.index(src)
        lo, hi = max(start, owner * local_rows), min(stop, (owner + 1) * local_rows)
        expected = full[lo:hi]
        # Verify the sender really supplied its slice before the simulated broadcast.
        before = expected if rank == owner else torch.zeros_like(expected)
        assert torch.equal(tensor.view(torch.uint8), before.view(torch.uint8))
        tensor.copy_(expected)
        calls.append((src, lo, hi))

    def forbidden_reduce(*args, **kwargs):
        pytest.fail("EP export must not reduce signed-zero/NaN payload bytes")

    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_reduce)
    monkeypatch.setattr(torch.distributed, "get_global_rank", global_rank)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    exports = []
    for rank in range(ep_size):
        local = full[rank * local_rows : (rank + 1) * local_rows]
        exports.append(mapping.megatron_to_hf(local, _destination(rank, ep_size, local, ep_group)))
    assert calls == []
    assert all(list(export) == list(exports[0]) for export in exports)
    assert calls == []
    width = (len(full) + shards - 1) // shards
    for i, name in enumerate(exports[0]):
        start, stop = i * width, min((i + 1) * width, len(full))
        expected = full[start:stop]
        owners = range(start // local_rows, (stop - 1) // local_rows + 1)
        for rank, export in enumerate(exports):
            calls.clear()
            actual = export[name]
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert [call[0] for call in calls] == ([] if ep_size == 1 else [global_ranks[o] for o in owners])


def test_tp_import_and_export_remain_on_tp_group(monkeypatch):
    full = torch.arange(48, dtype=torch.float32).view(24, 2)
    mapping = _Mapping("ple.weight", "hf.shard_{}", 4)
    module = nn.Linear(2, 12, bias=False)
    source = _Source({f"hf.shard_{i}": full[i * 6 : (i + 1) * 6] for i in range(4)})
    actual = mapping.hf_to_megatron(_aliases(mapping, source), module)
    torch.testing.assert_close(actual, full[:12], rtol=0, atol=0)
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group: calls.append(group))
    mapping.megatron_to_hf(actual, module)["hf.shard_0"]
    assert calls == [mapping.tp_group]


def _ngram_config(**overrides):
    values = dict(
        ple_ngram_size=3,
        ple_heads_per_ngram=1,
        ple_eos_token_id=0,
        ple_ngram_vocab_size_base=17,
        ple_ngram_vocab_divisible_by=4,
        ple_unigram_vocab_size=32,
        ple_seed=42,
        ple_embedding_sharding="ep",
        ple_freeze_ngram_embeddings=True,
        sequence_parallel=True,
        init_method=lambda x: None,
    )
    return SimpleNamespace(**(values | overrides))


class _Lookup(nn.Module):
    def __init__(self, config, rows, width, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) / 100, requires_grad=False
        )
        self.ids = None

    def forward(self, ids):
        self.ids = ids.clone()
        return self.weight[ids]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_ngram_hashes_full_sequence_before_sp_lookup(monkeypatch, tp_size):
    monkeypatch.setattr(ple, "FrozenEPNgramEmbedding", _Lookup)
    monkeypatch.setattr(ple.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ple, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(ple, "get_pg_rank", lambda group: group.rank())
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 7]])
    full_outputs = []
    for rank in range(tp_size):
        group = SimpleNamespace(size=lambda: tp_size, rank=lambda: rank)
        model = ple.NGramEmbedding(
            _ngram_config(), 4, 0, pg_collection=SimpleNamespace(tp=group, ep=object(), expt_dp=object())
        )
        ids = model.compute_ngram_ids(tokens)
        actual = model(ids)
        width = tokens.shape[1] // tp_size
        expected_ids = ids[:, rank * width : (rank + 1) * width]
        assert torch.equal(model.ngram_embedding.ids, expected_ids)
        expected = model.ngram_embedding.weight[expected_ids].transpose(0, 1).flatten(-2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if rank:
            local_hash = model.compute_ngram_ids(tokens[:, rank * width : (rank + 1) * width])
            assert not torch.equal(local_hash[:, 0], expected_ids[:, 0])
        full_outputs.append(actual)
    torch.testing.assert_close(
        torch.cat(full_outputs), model.ngram_embedding.weight[ids].transpose(0, 1).flatten(-2), rtol=0, atol=0
    )


def test_real_ple_prepare_and_dense_backward_keep_only_table_frozen(monkeypatch):
    monkeypatch.setattr(ple, "FrozenEPNgramEmbedding", _Lookup)
    monkeypatch.setattr(ple.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ple, "get_pg_size", lambda group: group.size())
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    config = _ngram_config(
        sequence_parallel=False,
        ple_layer_ids=[1],
        mhc_num_residual_streams=2,
        hidden_size=4,
        layernorm_epsilon=1e-6,
        ple_embed_dim=4,
        params_dtype=torch.float32,
        ple_conv_kernel_size=1,
        perform_initialization=True,
        init_method_std=0.1,
    )
    torch.manual_seed(23)
    model = ple.PerLayerEmbedding(
        config, 1, pg_collection=SimpleNamespace(tp=group, cp=group, ep=object(), expt_dp=object())
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    model.cp_group = SimpleNamespace(size=lambda: 2)
    with monkeypatch.context() as cp_patch:
        cp_patch.setattr(torch.distributed, "is_initialized", lambda: True)
        with pytest.raises(AssertionError, match="does not support context parallelism"):
            model.prepare(tokens)
    model.cp_group = group
    model.prepare(tokens)
    hidden = torch.randn(4, 1, 8, requires_grad=True)
    model(hidden).square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, param in model.named_parameters():
        if name == "ple_embedding.ngram_embedding.weight":
            assert not param.requires_grad and param.grad is None
        else:
            assert param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all(), name
    assert model.key_proj.weight.grad.abs().sum() > 0
    assert model.value_proj.weight.grad.abs().sum() > 0


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
