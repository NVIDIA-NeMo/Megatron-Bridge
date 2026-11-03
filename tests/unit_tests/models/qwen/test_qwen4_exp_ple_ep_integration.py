"""PLE conversion and hash/SP integration; collectives are recorded, not GPU-qualified."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.bridge.models.qwen import qwen4_exp_bridge as bridge
from megatron.bridge.models.qwen.modeling_qwen4_exp import per_layer_embedding as ple
from megatron.bridge.models.qwen.modeling_qwen4_exp.ep_ngram_embedding import FrozenEPNgramEmbedding
from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpModelConfig


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
    # Actual bridge hook must not read any tensors before the destination is known.
    result = bridge.Qwen4ExpBridge.maybe_modify_loaded_hf_weight(None, mapping.hf_param, source)
    assert source.reads == []
    assert list(result) == list(mapping.hf_param)
    return result


def test_lazy_ep_import_uses_destination_and_only_overlapping_shards():
    full = torch.arange(48, dtype=torch.float32).view(24, 2)
    mapping = _Mapping("decoder.layers.*.ple.weight", "model.layers.*.shard_{}.weight", 8).resolve(("3",))
    assert mapping.megatron_param == "decoder.layers.3.ple.weight"
    assert not mapping.is_expert
    source = _Source({mapping.hf_param[f"shard_{i}"]: full[i * 3 : (i + 1) * 3] for i in range(8)})
    lazy = _aliases(mapping, source)
    module = _destination(1, 4, torch.empty(6, 2, dtype=torch.bfloat16), object())
    actual = mapping.hf_to_megatron(lazy, module)
    assert actual.dtype == module.weight.dtype
    torch.testing.assert_close(actual, full[6:12].bfloat16(), rtol=0, atol=0)
    assert source.reads == [mapping.hf_param[f"shard_{i}"] for i in (0, 7, 2, 3)]


@pytest.mark.parametrize("num_shards", [0, 4])
def test_registry_selects_destination_aware_mapping_for_split_and_single_table(num_shards):
    mappings = bridge.Qwen4ExpBridge.get_lm_mappings("model.", False, SimpleNamespace(ple_layer_ids=[1]), num_shards)
    tables = [mapping for mapping in mappings if mapping.megatron_param.endswith("ngram_embedding.weight")]
    assert len(tables) == 1 and isinstance(tables[0], bridge.PLENGramEmbeddingMapping)
    resolved = tables[0].resolve(("0",))
    assert not resolved.is_expert
    assert resolved.num_shards == (num_shards or 1)
    suffix = "shard_0.weight" if num_shards else "weight"
    assert resolved.hf_param["shard_0"] == "model.layers.0.ple.ple_embedding.ngram_embedding." + suffix


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


@pytest.mark.parametrize("ep_size,shards", [(2, 3), (4, 5)])
def test_lazy_export_uses_ep_group_and_exact_owner_intersections(monkeypatch, ep_size, shards):
    full = torch.arange(48, dtype=torch.float32).view(24, 2) - 19
    mapping = _Mapping("ple.weight", "hf.shard_{}", shards)
    ep_group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group: calls.append((group, tensor.clone())))
    local_rows = full.shape[0] // ep_size
    exports = []
    for rank in range(ep_size):
        local = full[rank * local_rows : (rank + 1) * local_rows]
        exports.append(mapping.megatron_to_hf(local, _destination(rank, ep_size, local, ep_group)))
    assert calls == []
    assert all(list(export) == list(exports[0]) for export in exports)
    assert calls == []
    width = (len(full) + shards - 1) // shards
    for i, name in enumerate(exports[0]):
        calls.clear()
        pieces = [export[name] for export in exports]
        assert len(calls) == ep_size and all(group is ep_group for group, _ in calls)
        expected = full[i * width : min((i + 1) * width, len(full))]
        torch.testing.assert_close(torch.stack(pieces).sum(0), expected, rtol=0, atol=0)
        # The simulated reduction checks each owner's pre-collective buffer, not just its sum.
        for rank, (_, piece) in enumerate(calls):
            owned = torch.zeros_like(expected)
            start, stop = max(i * width, rank * local_rows), min((i + 1) * width, (rank + 1) * local_rows, len(full))
            if stop > start:
                owned[start - i * width : stop - i * width] = full[start:stop]
            torch.testing.assert_close(piece, owned, rtol=0, atol=0)


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


@pytest.mark.parametrize(
    "sharding,freeze,etp,error,match",
    [
        ("ep", False, 1, ValueError, "frozen|freeze"),
        ("ep", True, 2, NotImplementedError, "expert_tensor"),
        ("invalid", True, 1, ValueError, "sharding"),
    ],
)
def test_config_rejects_unsupported_ep_modes(sharding, freeze, etp, error, match):
    config = SimpleNamespace(
        ple_embedding_sharding=sharding, ple_freeze_ngram_embeddings=freeze, expert_tensor_parallel_size=etp
    )
    with pytest.raises(error, match=match):
        Qwen4ExpModelConfig._validate_qwen4_exp(config)


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
    model.prepare(torch.tensor([[1, 2, 3, 4]]))
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
