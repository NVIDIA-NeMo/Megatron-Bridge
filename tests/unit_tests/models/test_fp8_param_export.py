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

"""Unit tests for FP8 export behavior."""

import gc
import logging
import sys
import types
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    MegatronModelBridge,
    WeightConversionTask,
    _HFNameSuffixMapping,
)
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    LocalMXFP8Param,
    MegatronParamMapping,
    QKVMapping,
    RowParallelMapping,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.quant_bridge import (
    _extract_native_mxfp8_storage,
    _lookup_grouped_expert_mapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashQKVMapping
from megatron.bridge.models.stepfun.step35_bridge import (
    StackedExpertAutoMapping,
    StackedExpertGatedMLPMapping,
)


_QKV_GLOBAL = "decoder.layers.0.self_attention.linear_qkv.weight"
_MODEL_MB = "megatron.bridge.models.conversion.model_bridge"
_PARAM_MB = "megatron.bridge.models.conversion.param_mapping"
_QUANT_MB = "megatron.bridge.models.conversion.quant_bridge"


class _FakeNativeMXFP8Tensor:
    shape = torch.Size((5, 64))
    ndim = 2

    def __init__(self):
        self.data_bytes = torch.arange(5 * 64, dtype=torch.uint8).view(5, 64)
        self.scale_bytes = torch.zeros((128, 4), dtype=torch.uint8)

    def get_metadata(self):
        return {
            "rowwise_data": self.data_bytes,
            "rowwise_scale_inv": self.scale_bytes,
            "is_2D_scaled": False,
            "quantizer": SimpleNamespace(block_len=32),
        }


def test_native_mxfp8_storage_crops_only_documented_padding():
    param = _FakeNativeMXFP8Tensor()
    storage = _extract_native_mxfp8_storage(param, "decoder.linear.weight")
    assert storage.weight.dtype == torch.float8_e4m3fn
    assert storage.weight.shape == (5, 64)
    assert storage.weight_scale.dtype == torch.uint8
    assert storage.weight_scale.shape == (5, 2)
    assert storage.weight.untyped_storage().data_ptr() == param.data_bytes.untyped_storage().data_ptr()
    assert storage.weight_scale.untyped_storage().data_ptr() == param.scale_bytes.untyped_storage().data_ptr()


def test_native_mxfp8_storage_preserves_zero_scale_bytes():
    param = _FakeNativeMXFP8Tensor()
    storage = _extract_native_mxfp8_storage(param, "decoder.linear.weight")
    assert torch.count_nonzero(storage.weight_scale) == 0


def test_native_mxfp8_storage_crops_rank_generic_scale_padding():
    param = _FakeNativeMXFP8Tensor()
    param.shape = torch.Size((2, 5, 64))
    param.ndim = 3
    param.data_bytes = torch.arange(2 * 5 * 64, dtype=torch.uint8).view(2, 5, 64)
    param.scale_bytes = torch.zeros((4, 128, 4), dtype=torch.uint8)

    storage = _extract_native_mxfp8_storage(param, "decoder.linear.weight")

    assert storage.weight.shape == (2, 5, 64)
    assert storage.weight_scale.shape == (2, 5, 2)
    assert storage.weight.untyped_storage().data_ptr() == param.data_bytes.untyped_storage().data_ptr()
    assert storage.weight_scale.untyped_storage().data_ptr() == param.scale_bytes.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "configure",
    [
        pytest.param(lambda param: setattr(param, "get_metadata", lambda: None), id="missing-metadata"),
        pytest.param(
            lambda param: setattr(param, "data_bytes", param.data_bytes.float()),
            id="wrong-value-dtype",
        ),
        pytest.param(
            lambda param: setattr(param, "scale_bytes", param.scale_bytes.float()),
            id="wrong-scale-dtype",
        ),
        pytest.param(
            lambda param: setattr(
                param,
                "data_bytes",
                torch.zeros((64, 5), dtype=torch.uint8).transpose(0, 1),
            ),
            id="noncontiguous-value-storage",
        ),
        pytest.param(
            lambda param: setattr(
                param,
                "scale_bytes",
                torch.zeros((4, 128), dtype=torch.uint8).transpose(0, 1),
            ),
            id="noncontiguous-scale-storage",
        ),
        pytest.param(
            lambda param: (
                setattr(param, "shape", torch.Size((5, 63))),
                setattr(param, "ndim", 2),
            ),
            id="K-not-divisible-by-32",
        ),
        pytest.param(
            lambda param: setattr(param, "data_bytes", torch.zeros((5, 64, 1), dtype=torch.uint8)),
            id="value-rank-mismatch",
        ),
        pytest.param(
            lambda param: setattr(param, "scale_bytes", torch.zeros((128, 4, 1), dtype=torch.uint8)),
            id="scale-rank-mismatch",
        ),
        pytest.param(
            lambda param: setattr(param, "data_bytes", torch.zeros((4, 64), dtype=torch.uint8)),
            id="value-storage-too-small",
        ),
        pytest.param(
            lambda param: setattr(param, "scale_bytes", torch.zeros((4, 2), dtype=torch.uint8)),
            id="scale-storage-too-small",
        ),
        pytest.param(
            lambda param: setattr(
                param,
                "get_metadata",
                lambda: {
                    "rowwise_data": param.data_bytes,
                    "rowwise_scale_inv": param.scale_bytes,
                    "is_2D_scaled": False,
                    "quantizer": SimpleNamespace(block_len=16),
                },
            ),
            id="invalid-block-length",
        ),
        pytest.param(
            lambda param: setattr(
                param,
                "get_metadata",
                lambda: {
                    "rowwise_data": param.data_bytes,
                    "rowwise_scale_inv": param.scale_bytes,
                    "is_2D_scaled": True,
                    "quantizer": SimpleNamespace(block_len=32),
                },
            ),
            id="swizzled-scale-storage",
        ),
    ],
)
def test_native_mxfp8_storage_rejects_malformed_storage(configure):
    param = _FakeNativeMXFP8Tensor()
    configure(param)

    with pytest.raises(ValueError, match="decoder\\.linear\\.weight"):
        _extract_native_mxfp8_storage(param, "decoder.linear.weight")


def _make_qkv_mapping_type(global_name: str = _QKV_GLOBAL):
    class MegatronQkvMapping:
        hf_param = "hf.qkv.weight"
        megatron_param = global_name
        allow_hf_name_mismatch = False

        def resolve(self, _captures):
            return MegatronQkvMapping()

        def set_process_groups_from_pg_collection(self, _pg_collection):
            pass

        def hf_to_megatron(self, hf_weights, _module):
            return hf_weights

        def megatron_to_hf(self, megatron_weights, _module):
            return {"model.layers.0.self_attn.q_proj.weight": megatron_weights}

    return MegatronQkvMapping


def _patch_export_task_context(monkeypatch, bridge, global_name: str, **kwargs):
    """Common patches for build_export_fp8_tasks tests (single local rank, minimal PP)."""
    pp_rank = kwargs.get("pp_rank", 0)
    pp_size = kwargs.get("pp_size", 1)
    monkeypatch.setattr(bridge, "mapping_registry", kwargs["registry_factory"])
    monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda *_a, **_k: False)
    monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda *_a, **_k: [global_name])
    monkeypatch.setattr(bridge, "_detect_fp8_params", kwargs.get("detect_fp8", lambda *_a, **_k: {global_name: True}))
    monkeypatch.setattr(
        f"{_MODEL_MB}.unwrap_model",
        lambda models: models if isinstance(models, list) else [models],
    )
    monkeypatch.setattr(
        f"{_MODEL_MB}.parallel_state.get_pipeline_model_parallel_rank",
        lambda: pp_rank,
    )
    monkeypatch.setattr(
        f"{_MODEL_MB}.parallel_state.get_pipeline_model_parallel_group",
        lambda: SimpleNamespace(size=lambda: pp_size),
    )
    monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda *_a, **_k: [])
    monkeypatch.setattr(
        f"{_MODEL_MB}._megatron_local_name_to_global",
        lambda *_a, **_k: _a[2],
    )


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        return None

    def mapping_registry(self):
        return MegatronMappingRegistry()


class _IdentityMapping(MegatronParamMapping):
    def __init__(self, hf_param, megatron_param="dummy.megatron.weight", ep_rank=0):
        super().__init__(megatron_param, hf_param)
        self._test_ep_rank = ep_rank

    @property
    def ep_rank(self) -> int:
        return self._test_ep_rank

    def hf_to_megatron(self, hf_weights, _megatron_module):
        return hf_weights

    def megatron_to_hf(self, megatron_weights, _megatron_module):
        return {"model.weight": megatron_weights}

    def resolve(self, _captures):
        return _IdentityMapping(self.hf_param, self.megatron_param, self.ep_rank)


def test_local_mxfp8_param_keeps_value_and_scale_atomic():
    weight = torch.zeros((8, 64), dtype=torch.float8_e4m3fn)
    scale = torch.zeros((8, 2), dtype=torch.uint8)
    result = LocalMXFP8Param(
        name="model.layers.0.self_attn.o_proj.weight",
        weight=weight,
        weight_scale=scale,
        global_weight_shape=torch.Size((8, 128)),
        shard_group="tp",
        shard_dim=1,
    )
    assert result.weight is weight
    assert result.weight_scale is scale
    assert result.shard_group == "tp"
    assert result.shard_dim == 1


def test_base_mapping_rejects_native_mxfp8_projection():
    mapping = _IdentityMapping("hf.weight", "decoder.weight")
    with pytest.raises(ValueError, match="does not support direct native MXFP8 export"):
        mapping.local_mxfp8_params(
            torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 2), dtype=torch.uint8),
            global_param_name="decoder.weight",
            megatron_module=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "tp_size,packed_rows,expected_shapes",
    [
        pytest.param(1, 64, ((32, 64), (16, 64), (16, 64)), id="tp1"),
        pytest.param(2, 32, ((16, 64), (8, 64), (8, 64)), id="tp2"),
    ],
)
def test_qkv_native_mxfp8_gqa_preserves_row_order(monkeypatch, tp_size, packed_rows, expected_shapes):
    config = SimpleNamespace(
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=4,
        hidden_size=128,
        attention_output_gate=False,
    )
    mapping = QKVMapping("decoder.linear_qkv.weight", "hf.q", "hf.k", "hf.v")
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=tp_size))
    weight = torch.arange(packed_rows * 64, dtype=torch.uint8).view(packed_rows, 64).view(torch.float8_e4m3fn)
    scale = torch.arange(packed_rows * 2, dtype=torch.uint8).view(packed_rows, 2)

    params = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.linear_qkv.weight",
        megatron_module=SimpleNamespace(config=config),
    )

    assert [p.name for p in params] == ["hf.q", "hf.k", "hf.v"]
    assert [p.weight.shape for p in params] == list(expected_shapes)
    assert [p.weight_scale.shape for p in params] == [(shape[0], 2) for shape in expected_shapes]
    assert [p.global_weight_shape for p in params] == [(32, 64), (16, 64), (16, 64)]
    assert all(p.shard_group == "tp" and p.shard_dim == 0 for p in params)

    local_config = SimpleNamespace(**vars(config))
    local_config.num_attention_heads //= tp_size
    local_config.num_query_groups //= tp_size
    expected_weights = split_qkv_weights(local_config, weight, feature_dim=64)
    expected_scales = split_qkv_weights(local_config, scale, feature_dim=2)
    for param, expected_weight, expected_scale in zip(params, expected_weights, expected_scales):
        assert torch.equal(param.weight.view(torch.uint8), expected_weight.view(torch.uint8))
        assert torch.equal(param.weight_scale, expected_scale)


@pytest.mark.parametrize(
    "tp_size,packed_rows,expected_shapes",
    [
        pytest.param(1, 96, ((64, 64), (16, 64), (16, 64)), id="tp1"),
        pytest.param(2, 48, ((32, 64), (8, 64), (8, 64)), id="tp2"),
    ],
)
def test_qkv_native_mxfp8_output_gate_preserves_qz_row_order(monkeypatch, tp_size, packed_rows, expected_shapes):
    config = SimpleNamespace(
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=4,
        hidden_size=128,
        attention_output_gate=True,
    )
    mapping = QKVMapping("decoder.linear_qkv.weight", "hf.q", "hf.k", "hf.v")
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=tp_size))
    weight = torch.arange(packed_rows * 64, dtype=torch.uint8).view(packed_rows, 64).view(torch.float8_e4m3fn)
    scale = torch.arange(packed_rows * 2, dtype=torch.uint8).view(packed_rows, 2)

    params = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.linear_qkv.weight",
        megatron_module=SimpleNamespace(config=config),
    )

    assert [p.weight.shape for p in params] == list(expected_shapes)
    assert [p.weight_scale.shape for p in params] == [(shape[0], 2) for shape in expected_shapes]
    assert [p.global_weight_shape for p in params] == [(64, 64), (16, 64), (16, 64)]
    local_config = SimpleNamespace(**vars(config))
    local_config.num_attention_heads //= tp_size
    local_config.num_query_groups //= tp_size
    expected_weights = split_qkv_weights(local_config, weight, feature_dim=64)
    expected_scales = split_qkv_weights(local_config, scale, feature_dim=2)
    for param, expected_weight, expected_scale in zip(params, expected_weights, expected_scales):
        assert torch.equal(param.weight.view(torch.uint8), expected_weight.view(torch.uint8))
        assert torch.equal(param.weight_scale, expected_scale)


@pytest.mark.parametrize(
    "tp_size,packed_rows,expected_shapes",
    [
        pytest.param(1, 96, ((32, 64), (32, 64), (32, 64)), id="tp1"),
        pytest.param(2, 48, ((16, 64), (16, 64), (16, 64)), id="tp2"),
    ],
)
def test_qkv_native_mxfp8_mha_preserves_row_order(monkeypatch, tp_size, packed_rows, expected_shapes):
    config = SimpleNamespace(
        num_attention_heads=8,
        num_query_groups=8,
        kv_channels=4,
        hidden_size=128,
        attention_output_gate=False,
    )
    mapping = QKVMapping("decoder.linear_qkv.weight", "hf.q", "hf.k", "hf.v")
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=tp_size))
    weight = torch.arange(packed_rows * 64, dtype=torch.uint8).view(packed_rows, 64).view(torch.float8_e4m3fn)
    scale = torch.arange(packed_rows * 2, dtype=torch.uint8).view(packed_rows, 2)

    params = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.linear_qkv.weight",
        megatron_module=SimpleNamespace(config=config),
    )

    assert [p.name for p in params] == ["hf.q", "hf.k", "hf.v"]
    assert [p.weight.shape for p in params] == list(expected_shapes)
    assert [p.weight_scale.shape for p in params] == [(shape[0], 2) for shape in expected_shapes]
    assert [p.global_weight_shape for p in params] == [(32, 64), (32, 64), (32, 64)]
    assert all(p.shard_group == "tp" and p.shard_dim == 0 for p in params)

    local_config = SimpleNamespace(**vars(config))
    local_config.num_attention_heads //= tp_size
    local_config.num_query_groups //= tp_size
    expected_weights = split_qkv_weights(local_config, weight, feature_dim=64)
    expected_scales = split_qkv_weights(local_config, scale, feature_dim=2)
    for param, expected_weight, expected_scale in zip(params, expected_weights, expected_scales):
        assert torch.equal(param.weight.view(torch.uint8), expected_weight.view(torch.uint8))
        assert torch.equal(param.weight_scale, expected_scale)


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("num_attention_heads", 7, id="heads"),
        pytest.param("num_query_groups", 3, id="query-groups"),
    ],
)
def test_qkv_native_mxfp8_rejects_nondivisible_tp_counts(monkeypatch, field, value):
    config = SimpleNamespace(
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=4,
        hidden_size=128,
        attention_output_gate=False,
    )
    setattr(config, field, value)
    mapping = QKVMapping("decoder.linear_qkv.weight", "hf.q", "hf.k", "hf.v")
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=2))

    with pytest.raises(ValueError, match=r"decoder\.linear_qkv\.weight"):
        mapping.local_mxfp8_params(
            torch.zeros((32, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((32, 2), dtype=torch.uint8),
            global_param_name="decoder.linear_qkv.weight",
            megatron_module=SimpleNamespace(config=config),
        )


@pytest.mark.parametrize("tp_size,local_k", [pytest.param(1, 64, id="tp1"), pytest.param(2, 32, id="tp2")])
def test_output_native_mxfp8_row_parallel_preserves_views(monkeypatch, tp_size, local_k):
    mapping = RowParallelMapping("decoder.linear_proj.weight", "hf.o")
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=tp_size))
    weight = torch.arange(64 * local_k, dtype=torch.uint8).view(64, local_k).view(torch.float8_e4m3fn)
    scale = torch.arange(64 * local_k // 32, dtype=torch.uint8).view(64, local_k // 32)

    params = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.linear_proj.weight",
        megatron_module=SimpleNamespace(),
    )

    assert len(params) == 1
    assert params[0].name == "hf.o"
    assert params[0].weight is weight
    assert params[0].weight_scale is scale
    assert params[0].global_weight_shape == (64, 64)
    assert params[0].shard_group == "tp"
    assert params[0].shard_dim == 1


def test_output_native_mxfp8_rejects_mismatched_scale_shape():
    mapping = RowParallelMapping("decoder.linear_proj.weight", "hf.o")

    with pytest.raises(ValueError, match=r"decoder\.linear_proj\.weight"):
        mapping.local_mxfp8_params(
            torch.zeros((64, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((64, 2), dtype=torch.uint8),
            global_param_name="decoder.linear_proj.weight",
            megatron_module=SimpleNamespace(),
        )


@pytest.mark.parametrize("tp_size,local_k", [pytest.param(1, 64, id="tp1"), pytest.param(2, 32, id="tp2")])
def test_output_native_mxfp8_auto_mapping_delegates_only_to_row_parallel(monkeypatch, tp_size, local_k):
    mapping = AutoMapping("decoder.linear_proj.weight", "hf.o")
    monkeypatch.setattr(RowParallelMapping, "tp_size", PropertyMock(return_value=tp_size))
    module = type("RowParallelLinear", (), {})()
    weight = torch.zeros((64, local_k), dtype=torch.float8_e4m3fn)
    scale = torch.zeros((64, local_k // 32), dtype=torch.uint8)

    params = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.linear_proj.weight",
        megatron_module=module,
    )

    assert len(params) == 1
    assert params[0].weight is weight
    assert params[0].weight_scale is scale
    assert params[0].global_weight_shape == (64, 64)
    assert isinstance(mapping._mapping, RowParallelMapping)


@pytest.mark.parametrize(
    "module_type",
    [
        pytest.param("ColumnParallelLinear", id="column"),
        pytest.param("LayerNorm", id="replicated"),
    ],
)
def test_output_native_mxfp8_auto_mapping_rejects_unsupported_mapping(module_type):
    mapping = AutoMapping("decoder.linear_proj.weight", "hf.o")
    module = type(module_type, (), {})()

    with pytest.raises(ValueError, match=r"decoder\.linear_proj\.weight"):
        mapping.local_mxfp8_params(
            torch.zeros((64, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((64, 1), dtype=torch.uint8),
            global_param_name="decoder.linear_proj.weight",
            megatron_module=module,
        )


@pytest.mark.parametrize("transform", ["permute_dims", "transpose_on_export"])
def test_output_native_mxfp8_auto_mapping_rejects_export_transform(transform):
    mapping = AutoMapping("decoder.linear_proj.weight", "hf.o")
    setattr(mapping, transform, (1, 0) if transform == "permute_dims" else True)
    module = type("RowParallelLinear", (), {})()

    with pytest.raises(ValueError, match=r"decoder\.linear_proj\.weight"):
        mapping.local_mxfp8_params(
            torch.zeros((64, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((64, 1), dtype=torch.uint8),
            global_param_name="decoder.linear_proj.weight",
            megatron_module=module,
        )


def test_output_native_mxfp8_auto_mapping_rejects_fsdp(monkeypatch):
    mapping = AutoMapping("decoder.linear_proj.weight", "hf.o")
    module = type("RowParallelLinear", (), {})()
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.param_mapping._module_uses_fsdp",
        lambda _module: True,
    )

    with pytest.raises(ValueError, match=r"decoder\.linear_proj\.weight"):
        mapping.local_mxfp8_params(
            torch.zeros((64, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((64, 1), dtype=torch.uint8),
            global_param_name="decoder.linear_proj.weight",
            megatron_module=module,
        )


def test_fc1_native_mxfp8_projects_paired_expert_views(monkeypatch):
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.weight",
        gate="model.layers.0.mlp.experts.2.gate_proj.weight",
        up="model.layers.0.mlp.experts.2.up_proj.weight",
    )
    tp_group = object()
    etp_group = object()
    mapping.set_process_groups_from_pg_collection(SimpleNamespace(pp=None, ep=None, tp=tp_group, expt_tp=etp_group))
    monkeypatch.setattr(
        f"{_PARAM_MB}.get_pg_size",
        lambda group: {tp_group: 4, etp_group: 2}[group],
    )
    weight = torch.arange(8 * 64, dtype=torch.uint8).view(8, 64).view(torch.float8_e4m3fn)
    scale = torch.arange(8 * 2, dtype=torch.uint8).view(8, 2)

    projected = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.weight",
        megatron_module=SimpleNamespace(),
    )

    assert [param.name for param in projected] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.up_proj.weight",
    ]
    assert [param.global_weight_shape for param in projected] == [(8, 64), (8, 64)]
    assert all(param.shard_group == "etp" and param.shard_dim == 0 for param in projected)
    assert mapping._etp_group is etp_group
    torch.testing.assert_close(projected[0].weight.view(torch.uint8), weight[:4].view(torch.uint8))
    torch.testing.assert_close(projected[1].weight.view(torch.uint8), weight[4:].view(torch.uint8))
    torch.testing.assert_close(projected[0].weight_scale, scale[:4])
    torch.testing.assert_close(projected[1].weight_scale, scale[4:])


def test_fc2_native_mxfp8_projects_ordinary_expert_with_etp_metadata(monkeypatch):
    mapping = AutoMapping(
        "decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.weight",
        "model.layers.0.mlp.experts.2.down_proj.weight",
    )
    tp_group = object()
    etp_group = object()
    mapping.set_process_groups_from_pg_collection(SimpleNamespace(pp=None, ep=None, tp=tp_group, expt_tp=etp_group))
    monkeypatch.setattr(
        f"{_PARAM_MB}.get_pg_size",
        lambda group: {tp_group: 4, etp_group: 2}[group],
    )
    module = type("TERowParallelLinear", (), {})()
    weight = torch.arange(64 * 32, dtype=torch.uint8).view(64, 32).view(torch.float8_e4m3fn)
    scale = torch.arange(64, dtype=torch.uint8).view(64, 1)

    projected = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.weight",
        megatron_module=module,
    )

    assert len(projected) == 1
    assert projected[0].name == "model.layers.0.mlp.experts.2.down_proj.weight"
    assert projected[0].weight is weight
    assert projected[0].weight_scale is scale
    assert projected[0].global_weight_shape == (64, 64)
    assert projected[0].shard_group == "etp"
    assert projected[0].shard_dim == 1


def test_explicit_expert_row_mapping_uses_etp_metadata(monkeypatch):
    mapping = RowParallelMapping(
        "decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.weight",
        "model.layers.0.mlp.experts.2.down_proj.weight",
    )
    tp_group = object()
    etp_group = object()
    mapping.set_process_groups_from_pg_collection(SimpleNamespace(pp=None, ep=None, tp=tp_group, expt_tp=etp_group))
    monkeypatch.setattr(
        f"{_PARAM_MB}.get_pg_size",
        lambda group: {tp_group: 4, etp_group: 2}[group],
    )

    projected = mapping.local_mxfp8_params(
        torch.zeros((64, 32), dtype=torch.float8_e4m3fn),
        torch.zeros((64, 1), dtype=torch.uint8),
        global_param_name="decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.weight",
        megatron_module=SimpleNamespace(),
    )

    assert projected[0].global_weight_shape == (64, 64)
    assert projected[0].shard_group == "etp"
    assert projected[0].shard_dim == 1


@pytest.mark.parametrize(
    "mapping",
    [
        pytest.param(
            StackedExpertAutoMapping(
                "decoder.layers.0.mlp.experts.linear_fc2.weight2",
                "model.layers.0.moe.down_proj.weight",
            ),
            id="stacked-auto",
        ),
        pytest.param(
            StackedExpertGatedMLPMapping(
                "decoder.layers.0.mlp.experts.linear_fc1.weight2",
                gate="model.layers.0.moe.gate_proj.weight",
                up="model.layers.0.moe.up_proj.weight",
            ),
            id="stacked-gated",
        ),
    ],
)
def test_stacked_expert_mappings_reject_native_mxfp8_projection(mapping):
    module = type("TERowParallelLinear", (), {})()

    with pytest.raises(ValueError, match=r"decoder\.layers\.0.*canonical local HF views"):
        mapping.local_mxfp8_params(
            torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 2), dtype=torch.uint8),
            global_param_name=mapping.megatron_param,
            megatron_module=module,
        )


@pytest.mark.parametrize(
    ("mapping", "expected_names"),
    [
        pytest.param(
            FusedGatedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc1.weight2",
                "model.layers.0.mlp.experts.gate_up_proj.weight",
            ),
            [
                "model.layers.0.mlp.experts.2.gate_proj.weight",
                "model.layers.0.mlp.experts.2.up_proj.weight",
            ],
            id="fc1",
        ),
        pytest.param(
            FusedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc2.weight2",
                "model.layers.0.mlp.experts.down_proj.weight",
            ),
            ["model.layers.0.mlp.experts.2.down_proj.weight"],
            id="fc2",
        ),
    ],
)
def test_fused_expert_native_mxfp8_accepts_weight_suffixed_hf_names(mapping, expected_names, monkeypatch):
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=1))
    projected = mapping.local_mxfp8_params(
        torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
        torch.zeros((8, 2), dtype=torch.uint8),
        global_param_name=mapping.megatron_param,
        megatron_module=SimpleNamespace(),
    )

    assert [param.name for param in projected] == expected_names


def test_grouped_fc1_native_mxfp8_uses_global_expert_ids(monkeypatch):
    mapping = FusedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "model.layers.0.mlp.experts.gate_up_proj",
    )
    monkeypatch.setattr(type(mapping), "ep_rank", PropertyMock(return_value=1))
    monkeypatch.setattr(type(mapping), "ep_size", PropertyMock(return_value=2))
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=2))
    members = torch.arange(2 * 8 * 64, dtype=torch.uint8).view(2, 8, 64).view(torch.float8_e4m3fn)
    scales = torch.arange(2 * 8 * 2, dtype=torch.uint8).view(2, 8, 2)

    projected = []
    for local_expert_id in range(2):
        global_expert_id = mapping.ep_rank * 2 + local_expert_id
        projected.extend(
            mapping.local_mxfp8_params(
                members[local_expert_id],
                scales[local_expert_id],
                global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{global_expert_id}",
                megatron_module=SimpleNamespace(),
            )
        )

    assert [param.name for param in projected] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.up_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
    ]
    assert all(param.shard_group == "etp" and param.shard_dim == 0 for param in projected)
    assert all(param.weight.shape == (4, 64) and param.weight_scale.shape == (4, 2) for param in projected)
    assert all(param.global_weight_shape == (8, 64) for param in projected)


def test_grouped_fc2_native_mxfp8_uses_global_expert_ids(monkeypatch):
    mapping = FusedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.weight0",
        "model.layers.0.mlp.experts.down_proj",
    )
    monkeypatch.setattr(type(mapping), "ep_rank", PropertyMock(return_value=1))
    monkeypatch.setattr(type(mapping), "ep_size", PropertyMock(return_value=2))
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=2))
    members = torch.arange(2 * 64 * 32, dtype=torch.uint8).view(2, 64, 32).view(torch.float8_e4m3fn)
    scales = torch.arange(2 * 64, dtype=torch.uint8).view(2, 64, 1)

    projected = []
    for local_expert_id in range(2):
        global_expert_id = mapping.ep_rank * 2 + local_expert_id
        projected.extend(
            mapping.local_mxfp8_params(
                members[local_expert_id],
                scales[local_expert_id],
                global_param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{global_expert_id}",
                megatron_module=SimpleNamespace(),
            )
        )

    assert [param.name for param in projected] == [
        "model.layers.0.mlp.experts.2.down_proj.weight",
        "model.layers.0.mlp.experts.3.down_proj.weight",
    ]
    assert all(param.shard_group == "etp" and param.shard_dim == 1 for param in projected)
    assert all(param.weight.shape == (64, 32) and param.weight_scale.shape == (64, 1) for param in projected)
    assert all(param.global_weight_shape == (64, 64) for param in projected)


def test_fc1_native_mxfp8_rejects_odd_rows():
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        gate="model.layers.0.mlp.experts.0.gate_proj.weight",
        up="model.layers.0.mlp.experts.0.up_proj.weight",
    )

    with pytest.raises(ValueError, match=r"decoder\.layers\.0.*even"):
        mapping.local_mxfp8_params(
            torch.zeros((7, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((7, 2), dtype=torch.uint8),
            global_param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            megatron_module=SimpleNamespace(),
        )


def test_fc1_native_mxfp8_rejects_fsdp(monkeypatch):
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        gate="model.layers.0.mlp.experts.0.gate_proj.weight",
        up="model.layers.0.mlp.experts.0.up_proj.weight",
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.param_mapping._module_uses_fsdp",
        lambda _module: True,
    )

    with pytest.raises(ValueError, match=r"decoder\.layers\.0.*DTensor/FSDP"):
        mapping.local_mxfp8_params(
            torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 2), dtype=torch.uint8),
            global_param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            megatron_module=SimpleNamespace(),
        )


@pytest.mark.parametrize("mapping_cls", [FusedGatedExpertMapping, FusedExpertMapping])
@pytest.mark.parametrize("transform", ["permute_dims", "transpose_on_export"])
def test_grouped_native_mxfp8_rejects_export_transforms(mapping_cls, transform):
    kwargs = {transform: (1, 0) if transform == "permute_dims" else True}
    suffix = "linear_fc1.weight0" if mapping_cls is FusedGatedExpertMapping else "linear_fc2.weight0"
    hf_name = "gate_up_proj" if mapping_cls is FusedGatedExpertMapping else "down_proj"
    mapping = mapping_cls(
        f"decoder.layers.0.mlp.experts.{suffix}",
        f"model.layers.0.mlp.experts.{hf_name}",
        **kwargs,
    )

    with pytest.raises(ValueError, match=r"decoder\.layers\.0.*mapping transforms"):
        mapping.local_mxfp8_params(
            torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 2), dtype=torch.uint8),
            global_param_name=f"decoder.layers.0.mlp.experts.{suffix}",
            megatron_module=SimpleNamespace(),
        )


class TestHFNameSuffixMapping:
    def test_getattr(self):
        base = SimpleNamespace(megatron_param="m.w", hf_param="h.w", extra=123)
        w = _HFNameSuffixMapping(base, "_scale_inv")
        assert w.megatron_param == "m.w"
        assert w.hf_param == "h.w"
        assert w.extra == 123

    @pytest.mark.parametrize("has_resolve", [False, True])
    def test_resolve(self, has_resolve):
        if has_resolve:

            class Base:
                megatron_param = "m"

                def resolve(self, captures):
                    return SimpleNamespace(megatron_param="resolved", resolved=True)

            base = Base()
        else:
            base = SimpleNamespace(megatron_param="m")

        w = _HFNameSuffixMapping(base, "_s")
        r = w.resolve(("0",) if has_resolve else ())
        assert isinstance(r, _HFNameSuffixMapping) and r._suffix == "_s"
        if has_resolve:
            assert r._base_mapping.resolved is True
        else:
            assert r._base_mapping is base

    def test_hf_to_megatron(self):
        class Base:
            def hf_to_megatron(self, hf_weights, megatron_module):
                return hf_weights + 1

        w = _HFNameSuffixMapping(Base(), "_s")
        t = torch.tensor([1.0])
        torch.testing.assert_close(w.hf_to_megatron(t, None), torch.tensor([2.0]))

    @pytest.mark.parametrize("empty_out", [False, True])
    def test_megatron_to_hf(self, empty_out):
        class Base:
            def megatron_to_hf(self, megatron_weights, megatron_module):
                return {} if empty_out else {"model.a": megatron_weights}

        w = _HFNameSuffixMapping(Base(), "_scale_inv")
        t = torch.tensor([3.0])
        out = w.megatron_to_hf(t, None)
        assert out == ({}) if empty_out else {"model.a_scale_inv": t}


class TestFp8ParamExport:
    def test_native_mxfp8_materialization_releases_preflight_projections_between_tasks(self, monkeypatch):
        bridge = DummyBridge()
        projected_weight_refs = []
        parameters = []
        tasks = []

        for task_id in range(3):
            global_name = f"decoder.layers.{task_id}.self_attention.linear_proj.weight"
            parameter = _FakeNativeMXFP8Tensor()
            parameter.shape = torch.Size((8, 64))
            parameter.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
            parameter.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
            mapping = RowParallelMapping(global_name, f"hf.{task_id}")

            def project(weight, weight_scale, *, global_param_name, megatron_module, hf_name=f"hf.{task_id}"):
                del global_param_name, megatron_module
                projected_weight = weight.clone()
                projected_weight_refs.append(weakref.ref(projected_weight))
                return (
                    LocalMXFP8Param(
                        name=hf_name,
                        weight=projected_weight,
                        weight_scale=weight_scale.clone(),
                        global_weight_shape=torch.Size(projected_weight.shape),
                        shard_group="tp",
                        shard_dim=1,
                    ),
                )

            monkeypatch.setattr(mapping, "local_mxfp8_params", project)
            parameters.append(parameter)
            tasks.append(
                WeightConversionTask(
                    param_name=global_name,
                    global_param_name=global_name,
                    mapping=mapping,
                    megatron_module=SimpleNamespace(),
                    param_weight=parameter,
                )
            )

        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight in parameters)

        iterator = bridge.iter_local_native_mxfp8_params(tasks)
        first = next(iterator)
        gc.collect()

        assert len(projected_weight_refs) == len(tasks) + 1
        assert all(ref() is None for ref in projected_weight_refs[: len(tasks)])
        assert projected_weight_refs[-1]() is first.weight

    def test_native_mxfp8_materialization_validates_all_tasks_before_exposing_results(self, monkeypatch):
        bridge = DummyBridge()
        first_name = "decoder.layers.0.self_attention.linear_proj.weight"
        malformed_name = "decoder.layers.1.self_attention.linear_proj.weight"
        first = _FakeNativeMXFP8Tensor()
        first.shape = torch.Size((8, 64))
        first.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
        first.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
        malformed = _FakeNativeMXFP8Tensor()
        malformed.shape = torch.Size((8, 63))
        malformed.data_bytes = torch.zeros((8, 63), dtype=torch.uint8)
        malformed.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
        first_mapping = RowParallelMapping(first_name, "hf.first")
        malformed_mapping = RowParallelMapping(malformed_name, "hf.malformed")
        monkeypatch.setattr(type(first_mapping), "tp_size", PropertyMock(return_value=1))
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(
            f"{_QUANT_MB}.is_mxfp8tensor",
            lambda weight: weight in (first, malformed),
        )
        tasks = [
            WeightConversionTask(
                param_name=first_name,
                global_param_name=first_name,
                mapping=first_mapping,
                megatron_module=SimpleNamespace(),
                param_weight=first,
            ),
            WeightConversionTask(
                param_name=malformed_name,
                global_param_name=malformed_name,
                mapping=malformed_mapping,
                megatron_module=SimpleNamespace(),
                param_weight=malformed,
            ),
        ]
        exposed = []

        with pytest.raises(ValueError, match=malformed_name):
            exposed.extend(bridge.iter_local_native_mxfp8_params(tasks))

        assert exposed == []

    def test_native_mxfp8_structural_preflight_rejects_late_mapping_result_before_real_projection(self, monkeypatch):
        bridge = DummyBridge()
        first_name = "decoder.layers.0.self_attention.linear_proj.weight"
        malformed_name = "decoder.layers.1.self_attention.linear_proj.weight"
        projection_devices = []
        tasks = []
        parameters = []

        for global_name, malformed in ((first_name, False), (malformed_name, True)):
            parameter = _FakeNativeMXFP8Tensor()
            parameter.shape = torch.Size((8, 64))
            parameter.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
            parameter.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
            mapping = RowParallelMapping(global_name, f"hf.{len(tasks)}")

            def project(
                weight,
                weight_scale,
                *,
                global_param_name,
                megatron_module,
                malformed=malformed,
            ):
                del megatron_module
                projection_devices.append((global_param_name, weight.device.type))
                projected_scale = weight_scale[:, :1] if malformed else weight_scale
                return (
                    LocalMXFP8Param(
                        name=f"hf.{global_param_name}",
                        weight=weight,
                        weight_scale=projected_scale,
                        global_weight_shape=torch.Size(weight.shape),
                        shard_group="tp",
                        shard_dim=1,
                    ),
                )

            monkeypatch.setattr(mapping, "local_mxfp8_params", project)
            parameters.append(parameter)
            tasks.append(
                WeightConversionTask(
                    param_name=global_name,
                    global_param_name=global_name,
                    mapping=mapping,
                    megatron_module=SimpleNamespace(),
                    param_weight=parameter,
                )
            )

        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight in parameters)
        exposed = []

        with pytest.raises(ValueError, match=rf"{malformed_name}.*expected weight_scale shape"):
            exposed.extend(bridge.iter_local_native_mxfp8_params(tasks))

        assert exposed == []
        assert projection_devices == [(first_name, "meta"), (malformed_name, "meta")]

    def test_native_mxfp8_materialization_rejects_fsdp_before_native_classification(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_proj.weight"

        class FakeDTensor:
            pass

        def classification_called(_weight):
            raise AssertionError("native MXFP8 classification ran for a DTensor/FSDP task")

        monkeypatch.setattr(f"{_PARAM_MB}.DTensor", FakeDTensor)
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", classification_called)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", classification_called)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=RowParallelMapping(global_name, "hf.o"),
            megatron_module=SimpleNamespace(_parameters={"weight": FakeDTensor()}),
            param_weight=torch.zeros((8, 64), dtype=torch.bfloat16),
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*DTensor/FSDP"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_wraps_grouped_cache_errors(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.mlp.experts.linear_fc2.weight"
        grouped = torch.nn.Parameter(torch.zeros(2, 8, 64))
        mapping = FusedExpertMapping(f"{global_name}0", "model.layers.0.mlp.experts.down_proj")
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is grouped)

        def missing_cache(_weight, *, create_if_missing):
            assert create_if_missing is False
            raise RuntimeError("cached members are unavailable")

        monkeypatch.setattr(f"{_QUANT_MB}.get_grouped_quantized_members", missing_cache)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=2)),
            param_weight=grouped,
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*cached grouped MXFP8 members"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_rejects_inherited_specialized_mapping(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_qkv.weight"
        parameter = _FakeNativeMXFP8Tensor()
        parameter.shape = torch.Size((64, 64))
        parameter.data_bytes = torch.zeros((64, 64), dtype=torch.uint8)
        parameter.scale_bytes = torch.zeros((64, 2), dtype=torch.uint8)
        mapping = MiMoV2FlashQKVMapping(global_name, "hf.q", "hf.k", "hf.v")
        monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=1))
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(
                config=SimpleNamespace(
                    num_attention_heads=8,
                    num_query_groups=4,
                    kv_channels=4,
                    v_head_dim=2,
                    hidden_size=64,
                    attention_output_gate=False,
                )
            ),
            param_weight=parameter,
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*exact native MXFP8 projection"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(SimpleNamespace(), id="wrong-record-type"),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.uint8),
                    weight_scale=torch.zeros((8, 2), dtype=torch.uint8),
                    global_weight_shape=torch.Size((8, 64)),
                    shard_group="tp",
                    shard_dim=1,
                ),
                id="wrong-weight-dtype",
            ),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 2), dtype=torch.float32),
                    global_weight_shape=torch.Size((8, 64)),
                    shard_group="tp",
                    shard_dim=1,
                ),
                id="wrong-scale-dtype",
            ),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 1), dtype=torch.uint8),
                    global_weight_shape=torch.Size((8, 64)),
                    shard_group="tp",
                    shard_dim=1,
                ),
                id="wrong-scale-shape",
            ),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 2), dtype=torch.uint8),
                    global_weight_shape=(8, 64),
                    shard_group="tp",
                    shard_dim=1,
                ),
                id="wrong-global-shape-type",
            ),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 2), dtype=torch.uint8),
                    global_weight_shape=torch.Size((9, 64)),
                    shard_group="tp",
                    shard_dim=1,
                ),
                id="inconsistent-global-shape",
            ),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 2), dtype=torch.uint8),
                    global_weight_shape=torch.Size((8, 64)),
                    shard_group="dp",
                    shard_dim=1,
                ),
                id="wrong-shard-group",
            ),
            pytest.param(
                LocalMXFP8Param(
                    name="hf.o",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 2), dtype=torch.uint8),
                    global_weight_shape=torch.Size((8, 64)),
                    shard_group="tp",
                    shard_dim=None,
                ),
                id="missing-shard-dim",
            ),
        ],
    )
    def test_native_mxfp8_materialization_rejects_malformed_mapping_result(self, monkeypatch, result):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_proj.weight"
        parameter = _FakeNativeMXFP8Tensor()
        parameter.shape = torch.Size((8, 64))
        parameter.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
        parameter.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
        mapping = RowParallelMapping(global_name, "hf.o")
        monkeypatch.setattr(mapping, "local_mxfp8_params", lambda *_args, **_kwargs: (result,))
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(),
            param_weight=parameter,
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*invalid native MXFP8 mapping result"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_validates_grouped_mapping_results(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.mlp.experts.linear_fc2.weight"
        grouped = torch.nn.Parameter(torch.zeros(1, 8, 64))
        member = _FakeNativeMXFP8Tensor()
        member.shape = torch.Size((8, 64))
        member.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
        member.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
        mapping = FusedExpertMapping(f"{global_name}0", "model.layers.0.mlp.experts.down_proj")
        monkeypatch.setattr(type(mapping), "ep_rank", PropertyMock(return_value=0))
        monkeypatch.setattr(type(mapping), "ep_size", PropertyMock(return_value=1))
        monkeypatch.setattr(
            mapping,
            "local_mxfp8_params",
            lambda *_args, **_kwargs: (
                LocalMXFP8Param(
                    name="model.layers.0.mlp.experts.0.down_proj.weight",
                    weight=torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
                    weight_scale=torch.zeros((8, 1), dtype=torch.uint8),
                    global_weight_shape=torch.Size((8, 64)),
                    shard_group="etp",
                    shard_dim=1,
                ),
            ),
        )
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is grouped)
        monkeypatch.setattr(
            f"{_QUANT_MB}.get_grouped_quantized_members",
            lambda _weight, *, create_if_missing: [member],
        )
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=1)),
            param_weight=grouped,
        )

        with pytest.raises(ValueError, match=rf"{global_name}0.*invalid native MXFP8 mapping result"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_preserves_task_expert_and_mapping_order(self, monkeypatch):
        bridge = DummyBridge()
        qkv_name = "decoder.layers.0.self_attention.linear_qkv.weight"
        grouped_name = "decoder.layers.1.mlp.experts.linear_fc1.weight"
        output_name = "decoder.layers.2.self_attention.linear_proj.weight"
        qkv = _FakeNativeMXFP8Tensor()
        qkv.shape = torch.Size((64, 64))
        qkv.data_bytes = torch.arange(64 * 64, dtype=torch.uint8).view(64, 64)
        qkv.scale_bytes = torch.arange(64 * 2, dtype=torch.uint8).view(64, 2)
        grouped = torch.nn.Parameter(torch.zeros(2, 8, 64))
        members = []
        for expert_id in range(2):
            member = _FakeNativeMXFP8Tensor()
            member.shape = torch.Size((8, 64))
            member.data_bytes = torch.full((8, 64), expert_id, dtype=torch.uint8)
            member.scale_bytes = torch.full((8, 2), expert_id + 1, dtype=torch.uint8)
            members.append(member)
        output = _FakeNativeMXFP8Tensor()
        output.shape = torch.Size((8, 64))
        output.data_bytes = torch.full((8, 64), 7, dtype=torch.uint8)
        output.scale_bytes = torch.full((8, 2), 8, dtype=torch.uint8)

        qkv_mapping = QKVMapping(qkv_name, "hf.q", "hf.k", "hf.v")
        grouped_mapping = FusedGatedExpertMapping(
            f"{grouped_name}0",
            "model.layers.1.mlp.experts.gate_up_proj",
        )
        output_mapping = RowParallelMapping(output_name, "hf.o")
        qkv_projection_devices = []
        grouped_projection_devices = []
        qkv_projection = qkv_mapping.local_mxfp8_params
        grouped_projection = grouped_mapping.local_mxfp8_params

        def track_qkv_projection(weight, weight_scale, **kwargs):
            qkv_projection_devices.append(weight.device.type)
            return qkv_projection(weight, weight_scale, **kwargs)

        def track_grouped_projection(weight, weight_scale, **kwargs):
            grouped_projection_devices.append(weight.device.type)
            return grouped_projection(weight, weight_scale, **kwargs)

        monkeypatch.setattr(qkv_mapping, "local_mxfp8_params", track_qkv_projection)
        monkeypatch.setattr(grouped_mapping, "local_mxfp8_params", track_grouped_projection)
        monkeypatch.setattr(type(qkv_mapping), "tp_size", PropertyMock(return_value=1))
        monkeypatch.setattr(type(grouped_mapping), "ep_rank", PropertyMock(return_value=1))
        monkeypatch.setattr(type(grouped_mapping), "ep_size", PropertyMock(return_value=2))
        monkeypatch.setattr(type(grouped_mapping), "tp_size", PropertyMock(return_value=1))
        monkeypatch.setattr(type(output_mapping), "tp_size", PropertyMock(return_value=1))

        grouped_member_calls = []
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is grouped)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: isinstance(weight, _FakeNativeMXFP8Tensor))
        monkeypatch.setattr(
            f"{_QUANT_MB}.get_grouped_quantized_members",
            lambda weight, *, create_if_missing: grouped_member_calls.append((weight, create_if_missing)) or members,
        )

        tasks = [
            WeightConversionTask(
                param_name="remote.weight",
                global_param_name="remote.weight",
                mapping=_IdentityMapping("hf.remote", "remote.weight"),
                param_weight=None,
            ),
            WeightConversionTask(
                param_name=qkv_name,
                global_param_name=qkv_name,
                mapping=qkv_mapping,
                megatron_module=SimpleNamespace(
                    config=SimpleNamespace(
                        num_attention_heads=8,
                        num_query_groups=4,
                        kv_channels=4,
                        hidden_size=64,
                        attention_output_gate=False,
                    )
                ),
                param_weight=qkv,
            ),
            WeightConversionTask(
                param_name="decoder.layers.0.mlp.linear_fc1.weight",
                global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
                mapping=_IdentityMapping("hf.bf16"),
                param_weight=torch.zeros((8, 64), dtype=torch.bfloat16),
            ),
            WeightConversionTask(
                param_name=grouped_name,
                global_param_name=grouped_name,
                mapping=grouped_mapping,
                megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=4)),
                param_weight=grouped,
            ),
            WeightConversionTask(
                param_name=output_name,
                global_param_name=output_name,
                mapping=output_mapping,
                megatron_module=SimpleNamespace(),
                param_weight=output,
            ),
        ]

        params = list(bridge.iter_local_native_mxfp8_params(tasks))

        assert [param.name for param in params] == [
            "hf.q",
            "hf.k",
            "hf.v",
            "model.layers.1.mlp.experts.2.gate_proj.weight",
            "model.layers.1.mlp.experts.2.up_proj.weight",
            "model.layers.1.mlp.experts.3.gate_proj.weight",
            "model.layers.1.mlp.experts.3.up_proj.weight",
            "hf.o",
        ]
        assert grouped_member_calls == [(grouped, False), (grouped, False)]
        assert qkv_projection_devices == ["meta", "cpu"]
        assert grouped_projection_devices == ["meta", "meta", "cpu", "cpu"]

    def test_native_mxfp8_materialization_live_storage_refresh(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_proj.weight"
        parameter = _FakeNativeMXFP8Tensor()
        parameter.shape = torch.Size((8, 64))
        parameter.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
        parameter.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
        mapping = RowParallelMapping(global_name, "hf.o")
        monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=1))
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(),
            param_weight=parameter,
        )

        first = list(bridge.iter_local_native_mxfp8_params([task]))[0]
        parameter.data_bytes.fill_(3)
        parameter.scale_bytes.fill_(5)
        second = list(bridge.iter_local_native_mxfp8_params([task]))[0]

        assert torch.count_nonzero(first.weight.view(torch.uint8) != 3) == 0
        assert torch.count_nonzero(first.weight_scale != 5) == 0
        assert torch.count_nonzero(second.weight.view(torch.uint8) != 3) == 0
        assert torch.count_nonzero(second.weight_scale != 5) == 0
        assert second.weight.untyped_storage().data_ptr() == parameter.data_bytes.untyped_storage().data_ptr()
        assert second.weight_scale.untyped_storage().data_ptr() == parameter.scale_bytes.untyped_storage().data_ptr()

    def test_native_mxfp8_materialization_uses_no_payload_collectives(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_proj.weight"
        parameter = _FakeNativeMXFP8Tensor()
        parameter.shape = torch.Size((8, 64))
        parameter.data_bytes = torch.zeros((8, 64), dtype=torch.uint8)
        parameter.scale_bytes = torch.zeros((8, 2), dtype=torch.uint8)
        mapping = RowParallelMapping(global_name, "hf.o")
        monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=1))
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)

        def payload_collective_called(*_args, **_kwargs):
            raise AssertionError("native MXFP8 materialization called a payload collective")

        monkeypatch.setattr(torch.distributed, "broadcast", payload_collective_called)
        monkeypatch.setattr(torch.distributed, "all_gather", payload_collective_called)
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", payload_collective_called)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(),
            param_weight=parameter,
        )

        params = list(bridge.iter_local_native_mxfp8_params([task]))

        assert [param.name for param in params] == ["hf.o"]

    def test_native_mxfp8_materialization_rejects_unsupported_mapping(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.unsupported.weight"
        parameter = _FakeNativeMXFP8Tensor()
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=_IdentityMapping("hf.unsupported", global_name),
            megatron_module=SimpleNamespace(),
            param_weight=parameter,
        )

        with pytest.raises(
            ValueError, match=rf"{global_name}.*does not explicitly support exact native MXFP8 projection"
        ):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_rejects_mtp(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "mtp.decoder.layers.0.self_attention.linear_proj.weight"
        parameter = _FakeNativeMXFP8Tensor()
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=RowParallelMapping(global_name, "hf.o"),
            megatron_module=SimpleNamespace(),
            param_weight=parameter,
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*co-trained MTP"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_rejects_fsdp(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_proj.weight"
        parameter = _FakeNativeMXFP8Tensor()
        mapping = AutoMapping(global_name, "hf.o")
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        monkeypatch.setattr(f"{_PARAM_MB}._module_uses_fsdp", lambda _module: True)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=type("RowParallelLinear", (), {})(),
            param_weight=parameter,
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*DTensor/FSDP"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    @pytest.mark.parametrize(
        "configure",
        [
            pytest.param(
                lambda parameter: setattr(parameter, "shape", torch.Size((5, 63))),
                id="bad-k-alignment",
            ),
            pytest.param(
                lambda parameter: setattr(parameter, "scale_bytes", torch.zeros((5, 1), dtype=torch.uint8)),
                id="bad-scale-shape",
            ),
        ],
    )
    def test_native_mxfp8_materialization_rejects_malformed_storage(self, monkeypatch, configure):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_proj.weight"
        parameter = _FakeNativeMXFP8Tensor()
        configure(parameter)
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        monkeypatch.setattr(f"{_QUANT_MB}.is_mxfp8tensor", lambda weight: weight is parameter)
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=RowParallelMapping(global_name, "hf.o"),
            megatron_module=SimpleNamespace(),
            param_weight=parameter,
        )

        with pytest.raises(ValueError, match=global_name):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_native_mxfp8_materialization_rejects_bad_grouped_count(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.mlp.experts.linear_fc2.weight"
        grouped = torch.nn.Parameter(torch.zeros(1, 8, 64))
        mapping = FusedExpertMapping(f"{global_name}0", "model.layers.0.mlp.experts.down_proj")
        monkeypatch.setattr(type(mapping), "ep_size", PropertyMock(return_value=1))
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is grouped)
        monkeypatch.setattr(
            f"{_QUANT_MB}.get_grouped_quantized_members",
            lambda _weight, *, create_if_missing: [_FakeNativeMXFP8Tensor()],
        )
        task = WeightConversionTask(
            param_name=global_name,
            global_param_name=global_name,
            mapping=mapping,
            megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=2)),
            param_weight=grouped,
        )

        with pytest.raises(ValueError, match=rf"{global_name}.*1 local members.*expected 2"):
            list(bridge.iter_local_native_mxfp8_params([task]))

    def test_build_export_mxfp8_tasks_keeps_resolved_expert_mapping_ordinary(self):
        global_name = "decoder.layers.0.mlp.experts.local_experts.2.linear_fc1.weight"
        mapping = FusedGatedExpertMapping(global_name, "model.layers.0.mlp.experts.gate_up_proj")

        class Registry:
            def megatron_to_hf_lookup(self, name):
                return mapping if name == global_name else None

        assert _lookup_grouped_expert_mapping(Registry(), global_name) is None

    def test_build_export_mxfp8_tasks_classifies_grouped_experts_by_mapping(self, monkeypatch):
        bridge = DummyBridge()
        grouped = "decoder.layers.0.mlp.pool.experts.linear_fc1.weight"
        parameter = torch.nn.Parameter(torch.zeros(2, 8, 64))
        mapping = FusedGatedExpertMapping(
            f"{grouped}0",
            "model.layers.0.mlp.experts.gate_up_proj",
        )

        class Registry:
            def set_process_groups_from_pg_collection(self, _pg_collection):
                pass

            def megatron_to_hf_lookup(self, name):
                return mapping if name == f"{grouped}0" else None

        config = SimpleNamespace(
            expert_model_parallel_size=1,
            moe_single_grouped_weight=True,
            num_moe_experts=2,
            share_embeddings_and_output_weights=False,
        )
        model = SimpleNamespace(config=config, named_parameters=lambda: [(grouped, parameter)])
        grouped_member_calls = []
        monkeypatch.setattr(bridge, "mapping_registry", Registry)
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
        monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: [grouped])
        monkeypatch.setattr(bridge, "_validate_conversion_mappings", lambda _registry, names, _hf_keys: {})
        monkeypatch.setattr(f"{_MODEL_MB}._get_pp_rank", lambda _models: 0)
        monkeypatch.setattr(f"{_MODEL_MB}._get_pg_collection_from_model", lambda _models: None)
        monkeypatch.setattr(f"{_MODEL_MB}.unwrap_model", lambda models: models)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda _model: [])
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_args: grouped)
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_args: (SimpleNamespace(config=config), parameter),
        )
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is parameter)
        monkeypatch.setattr(
            f"{_QUANT_MB}.get_grouped_quantized_members",
            lambda weight, *, create_if_missing: (
                grouped_member_calls.append((weight, create_if_missing)) or list(weight.unbind(0))
            ),
        )

        tasks = bridge.build_export_mxfp8_tasks(SimpleNamespace(config=SimpleNamespace()), [model])

        assert [task.global_param_name for task in tasks] == [grouped]
        assert tasks[0].mapping is mapping
        assert tasks[0].param_weight is parameter
        assert grouped_member_calls == [(parameter, True)]

    def test_build_export_mxfp8_tasks_rejects_grouped_member_count_mismatch(self, monkeypatch):
        bridge = DummyBridge()
        grouped = "decoder.layers.0.mlp.pool.experts.linear_fc2.weight"
        parameter = torch.nn.Parameter(torch.zeros(2, 64, 32))
        mapping = FusedExpertMapping(f"{grouped}0", "model.layers.0.mlp.experts.down_proj")

        class Registry:
            def set_process_groups_from_pg_collection(self, _pg_collection):
                pass

            def megatron_to_hf_lookup(self, name):
                return mapping if name == f"{grouped}0" else None

        config = SimpleNamespace(
            expert_model_parallel_size=1,
            moe_single_grouped_weight=True,
            num_moe_experts=2,
            share_embeddings_and_output_weights=False,
        )
        model = SimpleNamespace(config=config, named_parameters=lambda: [(grouped, parameter)])
        monkeypatch.setattr(bridge, "mapping_registry", Registry)
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
        monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: [grouped])
        monkeypatch.setattr(f"{_MODEL_MB}._get_pp_rank", lambda _models: 0)
        monkeypatch.setattr(f"{_MODEL_MB}._get_pg_collection_from_model", lambda _models: None)
        monkeypatch.setattr(f"{_MODEL_MB}.unwrap_model", lambda models: models)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda _model: [])
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_args: grouped)
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_args: (SimpleNamespace(config=config), parameter),
        )
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is parameter)
        monkeypatch.setattr(
            f"{_QUANT_MB}.get_grouped_quantized_members",
            lambda _weight, *, create_if_missing: [parameter[0]],
        )

        with pytest.raises(ValueError, match=rf"{grouped}.*1 local members.*expected 2"):
            bridge.build_export_mxfp8_tasks(SimpleNamespace(config=SimpleNamespace()), [model])

    def test_grouped_native_mxfp8_rejects_missing_cached_members():
        bridge = DummyBridge()
        grouped = "decoder.layers.0.mlp.experts.linear_fc2.weight"
        task = WeightConversionTask(
            pp_rank=0,
            vp_stage=0,
            param_name=grouped,
            global_param_name=grouped,
            megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=2)),
            param_weight=torch.nn.Parameter(torch.zeros(2, 64, 32)),
            mapping=FusedExpertMapping(f"{grouped}0", "model.layers.0.mlp.experts.down_proj"),
        )

        with pytest.raises(ValueError, match=rf"{grouped}.*cached grouped MXFP8 members"):
            tuple(bridge._iter_grouped_native_mxfp8_params(task, None))

    def test_grouped_native_mxfp8_projection_computes_global_expert_ids(self, monkeypatch):
        bridge = DummyBridge()
        grouped = "decoder.layers.0.mlp.experts.linear_fc1.weight"
        mapping = FusedGatedExpertMapping(
            f"{grouped}0",
            "model.layers.0.mlp.experts.gate_up_proj",
        )
        monkeypatch.setattr(type(mapping), "ep_rank", PropertyMock(return_value=1))
        monkeypatch.setattr(type(mapping), "ep_size", PropertyMock(return_value=2))
        monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=2))
        members = []
        for expert_id in range(2):
            member = _FakeNativeMXFP8Tensor()
            member.shape = torch.Size((8, 64))
            member.data_bytes = torch.full((8, 64), expert_id, dtype=torch.uint8)
            member.scale_bytes = torch.full((8, 2), expert_id + 1, dtype=torch.uint8)
            members.append(member)
        task = WeightConversionTask(
            pp_rank=0,
            vp_stage=0,
            param_name=grouped,
            global_param_name=grouped,
            megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=4)),
            param_weight=torch.nn.Parameter(torch.zeros(2, 8, 64)),
            mapping=mapping,
        )

        projected = tuple(bridge._iter_grouped_native_mxfp8_params(task, members))

        assert [param.name for param in projected] == [
            "model.layers.0.mlp.experts.2.gate_proj.weight",
            "model.layers.0.mlp.experts.2.up_proj.weight",
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            "model.layers.0.mlp.experts.3.up_proj.weight",
        ]
        assert [torch.unique(param.weight.view(torch.uint8)).item() for param in projected] == [0, 0, 1, 1]
        assert [torch.unique(param.weight_scale).item() for param in projected] == [1, 1, 2, 2]

    def test_build_export_mxfp8_tasks_keeps_native_grouped_global_order_and_vp_stage(self, monkeypatch):
        bridge = DummyBridge()
        dense_first = "decoder.layers.0.self_attention.linear_qkv.weight"
        grouped = "decoder.layers.1.mlp.experts.linear_fc1.weight"
        dense_last = "decoder.layers.2.mlp.linear_fc2.weight"
        global_names = [dense_first, grouped, dense_last]
        native_grouped_weight = torch.nn.Parameter(torch.zeros(2, 8, 16))
        weights_by_vp = {
            (0, dense_first): torch.nn.Parameter(torch.zeros(8, 16)),
            (1, grouped): native_grouped_weight,
            (1, dense_last): torch.nn.Parameter(torch.zeros(8, 16)),
        }
        mappings = {
            dense_first: _IdentityMapping("hf.dense_first", dense_first),
            f"{grouped}0": FusedGatedExpertMapping(
                f"{grouped}0",
                "hf.grouped.gate_up_proj",
            ),
            dense_last: _IdentityMapping("hf.dense_last", dense_last),
        }

        class Registry:
            def set_process_groups_from_pg_collection(self, _pg_collection):
                pass

            def megatron_to_hf_lookup(self, name):
                return mappings.get(name)

        config = SimpleNamespace(
            expert_model_parallel_size=1,
            moe_single_grouped_weight=True,
            num_moe_experts=2,
            share_embeddings_and_output_weights=False,
        )
        models = [
            SimpleNamespace(config=config, named_parameters=lambda: [(dense_first, weights_by_vp[(0, dense_first)])]),
            SimpleNamespace(
                config=config,
                named_parameters=lambda: [
                    (grouped, weights_by_vp[(1, grouped)]),
                    (dense_last, weights_by_vp[(1, dense_last)]),
                ],
            ),
        ]
        grouped_member_calls = []
        monkeypatch.setattr(bridge, "mapping_registry", Registry)
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
        monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: global_names)
        monkeypatch.setattr(
            bridge,
            "_validate_conversion_mappings",
            lambda _registry, names, _hf_keys: {name: mappings[name] for name in names},
        )
        monkeypatch.setattr(f"{_MODEL_MB}._get_pp_rank", lambda _models: 0)
        monkeypatch.setattr(f"{_MODEL_MB}._get_pg_collection_from_model", lambda _models: None)
        monkeypatch.setattr(f"{_MODEL_MB}.unwrap_model", lambda value: value if isinstance(value, list) else [value])
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda _model: [])
        monkeypatch.setattr(
            f"{_MODEL_MB}._megatron_local_name_to_global",
            lambda _models, _config, local_name, _vp_stage: local_name,
        )
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda _models, local_name, vp_stage: (
                SimpleNamespace(config=config),
                weights_by_vp[(vp_stage, local_name)],
            ),
        )
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda weight: weight is native_grouped_weight)
        monkeypatch.setattr(
            f"{_QUANT_MB}.get_grouped_quantized_members",
            lambda weight, *, create_if_missing: (
                grouped_member_calls.append((weight, create_if_missing)) or list(weight.unbind(0))
            ),
        )
        hf_pretrained = SimpleNamespace(
            config=SimpleNamespace(),
            state=SimpleNamespace(
                source=SimpleNamespace(get_all_keys=lambda: ["hf.dense_first", "hf.dense_last"]),
            ),
        )

        tasks = bridge.build_export_mxfp8_tasks(hf_pretrained, models)

        assert [task.global_param_name for task in tasks] == global_names
        assert [task.vp_stage for task in tasks] == [0, 1, 1]
        assert tasks[1].param_weight is native_grouped_weight
        assert grouped_member_calls == [(native_grouped_weight, True)]

    def test_build_export_mxfp8_tasks_expands_bf16_grouped_members(self, monkeypatch):
        bridge = DummyBridge()
        grouped = "decoder.layers.0.mlp.experts.linear_fc1.weight"
        members = torch.arange(2 * 8 * 16, dtype=torch.bfloat16).view(2, 8, 16)
        parameter = torch.nn.Parameter(members.clone())
        mappings = {
            f"{grouped}{expert_id}": _IdentityMapping(f"hf.grouped.{expert_id}", f"{grouped}{expert_id}")
            for expert_id in range(2)
        }

        class Registry:
            def set_process_groups_from_pg_collection(self, _pg_collection):
                pass

            def megatron_to_hf_lookup(self, name):
                return mappings.get(name)

        config = SimpleNamespace(
            expert_model_parallel_size=1,
            moe_single_grouped_weight=True,
            num_moe_experts=2,
            share_embeddings_and_output_weights=False,
        )
        model = SimpleNamespace(config=config, named_parameters=lambda: [(grouped, parameter)])
        monkeypatch.setattr(bridge, "mapping_registry", Registry)
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
        monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: [grouped])
        monkeypatch.setattr(
            bridge,
            "_validate_conversion_mappings",
            lambda _registry, names, _hf_keys: {name: mappings[name] for name in names},
        )
        monkeypatch.setattr(f"{_MODEL_MB}._get_pp_rank", lambda _models: 0)
        monkeypatch.setattr(f"{_MODEL_MB}._get_pg_collection_from_model", lambda _models: None)
        monkeypatch.setattr(f"{_MODEL_MB}.unwrap_model", lambda models: models)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda _model: [])
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_args: grouped)
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_args: (SimpleNamespace(config=config), parameter),
        )
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        hf_pretrained = SimpleNamespace(config=SimpleNamespace())

        tasks = bridge.build_export_mxfp8_tasks(hf_pretrained, [model])

        assert [task.global_param_name for task in tasks] == [f"{grouped}0", f"{grouped}1"]
        torch.testing.assert_close(tasks[0].param_weight, members[0])
        torch.testing.assert_close(tasks[1].param_weight, members[1])

    def test_build_export_mxfp8_tasks_uses_global_expert_ids_for_bf16_members(self, monkeypatch):
        bridge = DummyBridge()
        grouped = "decoder.layers.0.mlp.experts.linear_fc1.weight"
        members = torch.arange(2 * 8 * 16, dtype=torch.bfloat16).view(2, 8, 16)
        parameter = torch.nn.Parameter(members.clone())
        mappings = {
            f"{grouped}0": _IdentityMapping("hf.grouped.0", f"{grouped}0", ep_rank=1),
            f"{grouped}2": _IdentityMapping("hf.grouped.2", f"{grouped}2", ep_rank=1),
            f"{grouped}3": _IdentityMapping("hf.grouped.3", f"{grouped}3", ep_rank=1),
        }

        class Registry:
            def set_process_groups_from_pg_collection(self, _pg_collection):
                pass

            def megatron_to_hf_lookup(self, name):
                return mappings.get(name)

        config = SimpleNamespace(
            expert_model_parallel_size=2,
            moe_single_grouped_weight=True,
            num_moe_experts=4,
            share_embeddings_and_output_weights=False,
        )
        model = SimpleNamespace(config=config, named_parameters=lambda: [(grouped, parameter)])
        monkeypatch.setattr(bridge, "mapping_registry", Registry)
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
        monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: [grouped])
        monkeypatch.setattr(
            bridge,
            "_validate_conversion_mappings",
            lambda _registry, names, _hf_keys: {name: mappings[name] for name in names},
        )
        monkeypatch.setattr(f"{_MODEL_MB}._get_pp_rank", lambda _models: 0)
        monkeypatch.setattr(f"{_MODEL_MB}._get_pg_collection_from_model", lambda _models: None)
        monkeypatch.setattr(f"{_MODEL_MB}.unwrap_model", lambda models: models)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda _model: [])
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_args: grouped)
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_args: (SimpleNamespace(config=config), parameter),
        )
        monkeypatch.setattr(f"{_QUANT_MB}.is_grouped_mxfp8tensor", lambda _weight: False)
        tasks = bridge.build_export_mxfp8_tasks(SimpleNamespace(config=SimpleNamespace()), [model])

        assert [task.global_param_name for task in tasks] == [f"{grouped}2", f"{grouped}3"]
        assert [task.param_name for task in tasks] == [f"{grouped}0", f"{grouped}1"]
        torch.testing.assert_close(tasks[0].param_weight, members[0])
        torch.testing.assert_close(tasks[1].param_weight, members[1])

    def test_get_export_mxfp8_tasks_uses_public_auto_bridge_api(self):
        mock_hf = Mock(spec=PreTrainedCausalLM)
        mock_model_bridge = Mock()
        model = Mock()
        expected_tasks = [Mock(spec=WeightConversionTask)]
        mock_model_bridge.build_export_mxfp8_tasks.return_value = expected_tasks

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf)
            tasks = bridge.get_export_mxfp8_tasks(model)

        assert tasks == expected_tasks
        mock_model_bridge.build_export_mxfp8_tasks.assert_called_once_with(mock_hf, [model])

    def test_iter_local_native_mxfp8_params_uses_public_auto_bridge_api(self):
        mock_hf = Mock(spec=PreTrainedCausalLM)
        mock_model_bridge = Mock()
        tasks = [Mock(spec=WeightConversionTask)]
        expected_params = [Mock(spec=LocalMXFP8Param)]
        mock_model_bridge.iter_local_native_mxfp8_params.return_value = iter(expected_params)

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf)
            params = list(bridge.iter_local_native_mxfp8_params(tasks))

        assert params == expected_params
        mock_model_bridge.iter_local_native_mxfp8_params.assert_called_once_with(tasks)

    def test_build_export_mxfp8_tasks_keeps_remote_placeholders_concrete(self, monkeypatch):
        bridge = DummyBridge()
        global_name = "decoder.layers.0.self_attention.linear_qkv.weight"
        mapping = _IdentityMapping("hf.qkv", global_name)

        class Registry:
            def set_process_groups_from_pg_collection(self, _pg_collection):
                pass

            def megatron_to_hf_lookup(self, _name):
                return None

        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [],
        )
        monkeypatch.setattr(bridge, "mapping_registry", Registry)
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda _config: False)
        monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _models: [global_name])
        monkeypatch.setattr(
            bridge,
            "_validate_conversion_mappings",
            lambda _registry, names, _hf_keys: {name: mapping for name in names},
        )
        monkeypatch.setattr(f"{_MODEL_MB}._get_pp_rank", lambda _models: 1)
        monkeypatch.setattr(f"{_MODEL_MB}._get_pg_collection_from_model", lambda _models: None)
        monkeypatch.setattr(f"{_MODEL_MB}.unwrap_model", lambda models: models)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda _model: [])

        tasks = bridge.build_export_mxfp8_tasks(SimpleNamespace(config=SimpleNamespace()), [model])

        assert len(tasks) == 1
        assert tasks[0].global_param_name == global_name
        assert tasks[0].param_weight is None
        assert tasks[0].megatron_module is None
        assert tasks[0].vp_stage is None

    @pytest.mark.parametrize(
        "export_weight_dtype, expect_unquantized",
        [("fp8", True), ("bf16", False)],
    )
    def test_load_weights_captures_unquantized(self, monkeypatch, export_weight_dtype, expect_unquantized):
        bridge = DummyBridge()
        bridge.export_weight_dtype = export_weight_dtype
        target_param = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=True)
        converted = torch.full((2, 2), 3.0)
        task = WeightConversionTask(
            param_name="decoder.layers.0.linear.weight",
            global_param_name="decoder.layers.0.linear.weight",
            mapping=_IdentityMapping("hf.w0", "decoder.layers.0.linear.weight"),
            pp_rank=0,
            vp_stage=0,
            megatron_module=Mock(),
            param_weight=target_param,
        )
        monkeypatch.setattr(DummyBridge, "build_conversion_tasks", lambda self, *_a, **_k: [task])
        monkeypatch.setattr(DummyBridge, "_with_progress_tracking", lambda self, tasks, *_a, **_k: tasks)
        monkeypatch.setattr(DummyBridge, "finalize_hf_import", lambda self, *_a, **_k: None)
        hf_pretrained = SimpleNamespace(state={"hf.w0": converted}, model_name_or_path="dummy")
        models = [SimpleNamespace()]
        assert bridge.load_weights_hf_to_megatron(hf_pretrained, models) is models
        torch.testing.assert_close(target_param.detach(), converted)
        if expect_unquantized:
            assert "decoder.layers.0.linear.weight" in bridge.unquantized_state_dict["model"]
        else:
            assert bridge.unquantized_state_dict is None

    @pytest.mark.parametrize(
        "export_dtype, cfg, expect_raise, n_fp8_build_calls",
        [
            ("fp8", {"fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": True}, False, 1),
            ("fp8", {"fp8": "e4m3", "fp8_recipe": "tensorwise", "fp8_param": True}, True, 0),
            ("fp8", {"fp8": None, "fp8_recipe": "blockwise", "fp8_param": True}, True, 0),
            ("bf16", {"fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": True}, False, 0),
        ],
    )
    def test_export_hf_weights_fp8(self, export_dtype, cfg, expect_raise, n_fp8_build_calls):
        mock_hf = Mock(spec=PreTrainedCausalLM)
        mock_hf.config = Mock(architectures=["LlamaForCausalLM"], auto_map=None)
        megatron = [SimpleNamespace(config=SimpleNamespace(**cfg))]
        mock_mb = Mock()
        fp8_tasks = [Mock(name="fp8_w"), Mock(name="fp8_scale")]
        mock_mb.build_export_fp8_tasks.return_value = fp8_tasks
        mock_mb.stream_weights_megatron_to_hf.return_value = iter(
            [("model.layers.0.self_attn.q_proj.weight", torch.ones(1))]
        )

        with patch.object(AutoBridge, "_model_bridge", mock_mb):
            with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as tf:
                tf.LlamaForCausalLM = arch = Mock()
                bridge = AutoBridge(mock_hf)
                bridge.export_weight_dtype = export_dtype
                with patch.object(AutoBridge, "_causal_lm_architecture", new_callable=PropertyMock) as arch_prop:
                    arch_prop.return_value = arch
                    if expect_raise:
                        with pytest.raises(ValueError, match="only supports blockwise FP8 parameter export"):
                            list(bridge.export_hf_weights(megatron, cpu=True))
                    else:
                        list(bridge.export_hf_weights(megatron, cpu=True))
        assert mock_mb.build_export_fp8_tasks.call_count == n_fp8_build_calls
        if export_dtype == "fp8" and not expect_raise:
            mock_mb.build_export_fp8_tasks.assert_called_once_with(mock_hf, megatron)
            assert mock_mb.stream_weights_megatron_to_hf.call_args.kwargs["conversion_tasks"] == fp8_tasks
        elif expect_raise:
            mock_mb.build_export_fp8_tasks.assert_not_called()
            mock_mb.stream_weights_megatron_to_hf.assert_not_called()
        else:
            assert mock_mb.stream_weights_megatron_to_hf.call_args.kwargs["conversion_tasks"] is None

    @pytest.mark.parametrize(
        "scale_shape, quantizer, is_2d, warn_trim, expect_shape",
        [
            pytest.param((2, 8), SimpleNamespace(block_len=128), True, False, (2, 2), id="trim"),
            pytest.param((2, 2), SimpleNamespace(block_len=128), True, False, (2, 2), id="no_trim"),
            pytest.param((2, 8), None, True, True, (2, 8), id="no_quantizer"),
            pytest.param((2, 8), SimpleNamespace(block_len=128), False, True, (2, 8), id="not_2d"),
        ],
    )
    def test_build_export_fp8_tasks_scale_inv_trim(
        self, monkeypatch, caplog, scale_shape, quantizer, is_2d, warn_trim, expect_shape
    ):
        caplog.set_level(logging.WARNING, logger="megatron.bridge.models.conversion.model_bridge")
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        MappingT = _make_qkv_mapping_type(gname)

        rowwise = torch.ones(scale_shape, dtype=torch.float32)
        metadata = {
            "rowwise_data": torch.zeros((2, 256), dtype=torch.uint8),
            "rowwise_scale_inv": rowwise,
            "quantizer": quantizer,
            "is_2D_scaled": is_2d,
        }
        fake_w = SimpleNamespace(get_metadata=lambda: metadata, shape=(2, 256))
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [(gname, torch.nn.Parameter(torch.zeros(1)))],
        )
        _patch_export_task_context(
            monkeypatch,
            bridge,
            gname,
            registry_factory=lambda: MegatronMappingRegistry(MappingT()),
        )
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_a, **_k: (SimpleNamespace(config=model.config), fake_w),
        )
        tasks = bridge.build_export_fp8_tasks(
            SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace())), [model]
        )
        assert len(tasks) == 2 and tasks[1].global_param_name == f"{gname}_scale_inv"
        assert tasks[0].param_weight.dtype == torch.float8_e4m3fn
        assert tasks[1].param_weight.shape == expect_shape
        assert torch.all(tasks[1].param_weight == 1.0)
        assert ("block_len or not is_2d_scaled" in caplog.text) is warn_trim
        if tasks[1].param_weight.shape == rowwise.shape:
            assert tasks[1].param_weight.data_ptr() == rowwise.data_ptr()

    def test_detect_fp8_params_without_top_level_te_class(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL

        class BlockwiseMetadataTensor:
            pass

        monkeypatch.setitem(
            sys.modules,
            "transformer_engine.pytorch",
            types.ModuleType("transformer_engine.pytorch"),
        )

        holder = BlockwiseMetadataTensor()
        holder.get_metadata = lambda: {"rowwise_scale_inv": torch.ones(1), "is_2D_scaled": False}
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [(gname, torch.nn.Parameter(torch.zeros(1)))],
        )
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_a, **_k: (SimpleNamespace(config=model.config), holder),
        )
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_a, **_k: gname)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda *_a, **_k: [])
        monkeypatch.setattr(f"{_MODEL_MB}.get_pg_size", lambda _g: 2)

        def ag(output_list, obj, group=None):
            output_list[0] = obj
            output_list[1] = {"decoder.layers.1.other.weight": True}

        monkeypatch.setattr(f"{_MODEL_MB}.torch.distributed.all_gather_object", ag)
        flags = bridge._detect_fp8_params(
            [model], model.config, [gname, "decoder.layers.1.other.weight"], None, "_rowwise_scale_inv"
        )
        assert flags[gname] and flags["decoder.layers.1.other.weight"]

    def test_detect_fp8_params_ignores_tensor_without_blockwise_metadata(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [(gname, torch.nn.Parameter(torch.zeros(1)))],
        )
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_a, **_k: (None, torch.nn.Parameter(torch.zeros(1))),
        )
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_a, **_k: gname)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda *_a, **_k: [])
        monkeypatch.setattr(f"{_MODEL_MB}.get_pg_size", lambda _g: 1)

        def _ag1(out, obj, group=None):
            out[0] = obj

        monkeypatch.setattr(f"{_MODEL_MB}.torch.distributed.all_gather_object", _ag1)
        assert bridge._detect_fp8_params([model], model.config, [gname], None, "_rowwise_scale_inv") == {}

    def test_build_export_fp8_tasks_remote_pp_tasks_are_concrete(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        MappingT = _make_qkv_mapping_type(gname)
        _patch_export_task_context(
            monkeypatch,
            bridge,
            gname,
            registry_factory=lambda: MegatronMappingRegistry(MappingT()),
            pp_rank=1,
            pp_size=2,
            detect_fp8=lambda *_a, **_k: {gname: 1},
        )

        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [],
        )
        tasks = bridge.build_export_fp8_tasks(
            SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace())), [model]
        )
        assert len(tasks) == 2
        assert tasks[0].megatron_module is None and isinstance(tasks[0].mapping, MappingT)
        assert tasks[1].megatron_module is None and isinstance(tasks[1].mapping, _HFNameSuffixMapping)
        assert tasks[1].mapping.scale_block_size == 1

    def test_build_export_fp8_tasks_rejects_missing_mapping_on_remote_pp_rank(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        _patch_export_task_context(
            monkeypatch,
            bridge,
            gname,
            registry_factory=MegatronMappingRegistry,
            pp_rank=1,
            pp_size=2,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [],
        )

        with pytest.raises(ValueError, match=gname.replace(".", r"\.")):
            bridge.build_export_fp8_tasks(SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace())), [model])

    @pytest.mark.parametrize(
        "hidden_size, last_dim, expected_shapes, expected_error",
        [
            (16, 4, ((4, 4), (2, 4), (2, 4)), None),
            (4096, 32, ((32, 32), (16, 32), (16, 32)), None),
            (10, 4, None, "Cannot infer block divisor"),
            (12, 3, None, "Cannot scale head_size"),
        ],
    )
    def test_split_qkv_compressed(self, hidden_size, last_dim, expected_shapes, expected_error):
        qkv_dim = 8
        provider = SimpleNamespace(
            num_attention_heads=4,
            num_query_groups=2,
            hidden_size=hidden_size,
            kv_channels=None,
            attention_output_gate=False,
        )
        if expected_error is None:
            hs = hidden_size // provider.num_attention_heads
            div = hidden_size // last_dim
            qkv = torch.randn(qkv_dim * (hs // div), last_dim)
        else:
            qkv = torch.randn(qkv_dim, last_dim)
        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                split_qkv_weights(provider, qkv)
            return
        q, k, v = split_qkv_weights(provider, qkv)
        assert q.shape == expected_shapes[0]
        assert k.shape == expected_shapes[1]
        assert v.shape == expected_shapes[2]
