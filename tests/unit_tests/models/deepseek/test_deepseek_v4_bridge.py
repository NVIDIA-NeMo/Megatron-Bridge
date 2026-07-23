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

"""Unit tests for the DeepSeek-V4 bridge mapping registry.

Locks in the MTP mapping layout: per-MTP-layer HC head, separate ``e_proj``
and ``h_proj`` mappings, and no deprecated concatenated ``eh_proj`` path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    DeepSeekV4Bridge,
    _dsv4_compress_ratios,
    _dsv4_num_hash_layers,
)


@pytest.fixture
def bridge_with_mtp():
    """A DSv4 bridge with hf_config stubbed for a single MTP layer."""
    bridge = DeepSeekV4Bridge()
    # mapping_registry only reads num_nextn_predict_layers from hf_config.
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    return bridge


@pytest.fixture
def bridge_without_mtp():
    """A DSv4 bridge with hf_config that has zero MTP layers."""
    bridge = DeepSeekV4Bridge()
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)
    return bridge


def _by_megatron(registry):
    """Index mappings by megatron_param for quick lookup in assertions."""
    return {m.megatron_param: m for m in registry.mappings}


def _dummy_task():
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask

    return WeightConversionTask(param_name="", global_param_name="", mapping=None)


def _deepseek_v4_hf_config():
    return SimpleNamespace(
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling={"factor": 16, "original_max_position_embeddings": 65536, "rope_theta": 10000},
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        num_hash_layers=3,
        compress_ratios=[0, 4, 128, 4, 0],
        sliding_window=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        scoring_func="sqrtsoftplus",
        num_experts_per_tok=6,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        vocab_size=129280,
        swiglu_limit=10.0,
        moe_intermediate_size=1024,
        n_shared_experts=1,
        tie_word_embeddings=False,
    )


class TestNativeDeepSeekV4ConfigTranslation:
    """Native Transformers DSv4 config fields must map back to MCore fields."""

    def test_compress_ratios_from_native_layer_types(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
        )

        assert _dsv4_compress_ratios(hf_config) == [0, 0, 4, 128, 0]

    def test_legacy_compress_ratios_still_work(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            compress_ratios=[0, 0, 4, 128, 0],
        )

        assert _dsv4_compress_ratios(hf_config) == [0, 0, 4, 128, 0]

    def test_hash_layers_from_native_mlp_layer_types(self):
        hf_config = SimpleNamespace(
            mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe", "moe"],
        )

        assert _dsv4_num_hash_layers(hf_config) == 3

    def test_hash_layers_must_be_prefix(self):
        hf_config = SimpleNamespace(mlp_layer_types=["hash_moe", "moe", "hash_moe"])

        with pytest.raises(ValueError, match="contiguous prefix"):
            _dsv4_num_hash_layers(hf_config)


class TestDeepSeekV4QuantizedExport:
    """DSv4 export must regenerate quantized weights and scale tensors."""

    def test_import_accepts_legacy_flat_indexer_weight_name(self):
        bridge = DeepSeekV4Bridge()
        flat_name = "layers.1.attn.indexer.weights_proj.weight"
        scorer_name = "layers.1.attn.indexer.scorer.weights_proj.weight"
        weight = torch.randn(4, 4, dtype=torch.bfloat16)

        result = bridge.maybe_modify_loaded_hf_weight(scorer_name, {flat_name: weight})

        assert result is weight

    def test_export_quantizes_fp8_weight_and_emits_scale(self):
        bridge = DeepSeekV4Bridge()
        hf_param = "layers.0.attn.wq_a.weight"
        scale_key = "layers.0.attn.wq_a.scale"
        weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
        source_state = {scale_key: torch.ones((1, 1), dtype=torch.float32)}

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {hf_param: weight}, source_state)

        assert set(result) == {hf_param, scale_key}
        assert result[hf_param].dtype == torch.float8_e4m3fn
        assert result[scale_key].shape == source_state[scale_key].shape
        assert result[scale_key].dtype == source_state[scale_key].dtype

        restored = bridge.maybe_modify_loaded_hf_weight(hf_param, result)
        assert restored.dtype == torch.bfloat16
        assert torch.allclose(restored.float(), weight.float())

    def test_export_preserves_e8m0_scale_dtype(self):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            pytest.skip("torch.float8_e8m0fnu is unavailable")
        try:
            source_scale = torch.ones((1, 1), dtype=e8m0_dtype)
        except RuntimeError as exc:
            pytest.skip(f"torch.float8_e8m0fnu tensor creation is unavailable: {exc}")

        bridge = DeepSeekV4Bridge()
        hf_param = "layers.0.attn.wq_a.weight"
        scale_key = "layers.0.attn.wq_a.scale"
        weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {hf_param: weight}, {scale_key: source_scale})

        assert result[hf_param].dtype == torch.float8_e4m3fn
        assert result[scale_key].dtype == e8m0_dtype
        restored = bridge.maybe_modify_loaded_hf_weight(hf_param, result)
        assert torch.allclose(restored.float(), weight.float())

    def test_export_quantizes_routed_expert_to_mxfp4_and_emits_scale(self):
        bridge = DeepSeekV4Bridge()
        hf_param = "layers.0.ffn.experts.0.w1.weight"
        scale_key = "layers.0.ffn.experts.0.w1.scale"
        values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
        ).repeat(2)
        weight = values.reshape(1, 32).to(torch.bfloat16)
        source_state = {scale_key: torch.ones((1, 1), dtype=torch.float32)}

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {hf_param: weight}, source_state)

        assert set(result) == {hf_param, scale_key}
        assert result[hf_param].dtype == torch.int8
        assert result[hf_param].shape == (1, 16)
        assert result[scale_key].shape == source_state[scale_key].shape
        assert result[scale_key].dtype == source_state[scale_key].dtype

        restored = quantization_utils.dequantize_mxfp4_e2m1_packed(result[hf_param], result[scale_key])
        assert torch.equal(restored.float(), weight.float())

    @pytest.mark.parametrize(
        "hf_param",
        [
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.experts.0.w1.weight",
        ],
    )
    def test_export_uses_fp8_for_non_mxfp4_expert_scale_geometry(self, hf_param):
        bridge = DeepSeekV4Bridge()
        scale_key = hf_param.removesuffix(".weight") + ".scale"
        weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(
            _dummy_task(), {hf_param: weight}, {scale_key: torch.ones(1, 1)}
        )

        assert result[hf_param].dtype == torch.float8_e4m3fn
        assert result[scale_key].shape == (1, 1)

    def test_export_leaves_unscaled_weight_unchanged(self):
        bridge = DeepSeekV4Bridge()
        weight = torch.ones(4, 4, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {"norm.weight": weight}, {})

        assert set(result) == {"norm.weight"}
        assert result["norm.weight"] is weight

    def test_export_roundtrips_mixed_quantized_hf_state(self):
        bridge = DeepSeekV4Bridge()
        fp8_param = "layers.0.attn.wq_a.weight"
        fp8_scale = "layers.0.attn.wq_a.scale"
        mxfp4_param = "layers.0.ffn.experts.0.w1.weight"
        mxfp4_scale = "layers.0.ffn.experts.0.w1.scale"
        norm_param = "layers.0.attn_norm.weight"

        fp8_weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
        mxfp4_values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
        ).repeat(2)
        mxfp4_weight = mxfp4_values.reshape(1, 32).to(torch.bfloat16)
        norm_weight = torch.arange(4, dtype=torch.float32).to(torch.bfloat16)

        stale_scale = torch.full((1, 1), 9.0, dtype=torch.float32)
        result = bridge.maybe_modify_converted_hf_weight(
            _dummy_task(),
            {
                fp8_param: fp8_weight,
                fp8_scale: stale_scale,
                mxfp4_param: mxfp4_weight,
                mxfp4_scale: stale_scale,
                norm_param: norm_weight,
            },
            {
                fp8_scale: torch.ones((1, 1), dtype=torch.float32),
                mxfp4_scale: torch.ones((1, 1), dtype=torch.float32),
            },
        )

        assert set(result) == {fp8_param, fp8_scale, mxfp4_param, mxfp4_scale, norm_param}
        assert result[fp8_param].dtype == torch.float8_e4m3fn
        assert result[mxfp4_param].dtype == torch.int8
        assert result[mxfp4_param].shape == (1, 16)
        assert result[fp8_scale].shape == (1, 1)
        assert result[mxfp4_scale].shape == (1, 1)
        assert not torch.equal(result[fp8_scale], stale_scale)
        assert not torch.equal(result[mxfp4_scale], stale_scale)
        assert result[norm_param] is norm_weight

        restored_fp8 = bridge.maybe_modify_loaded_hf_weight(fp8_param, result)
        restored_mxfp4 = bridge.maybe_modify_loaded_hf_weight(mxfp4_param, result)
        assert torch.allclose(restored_fp8.float(), fp8_weight.float())
        assert torch.equal(restored_mxfp4.float(), mxfp4_weight.float())


def test_sequential_expert_mappings_present(bridge_with_mtp):
    """Sequential (non-grouped) expert mappings exist for moe_grouped_gemm=False (ModelOpt pruning)."""
    params = _by_megatron(bridge_with_mtp.mapping_registry())
    assert "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight" in params
    assert "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight" in params


class TestDecoderHCHeadMappings:
    """The global decoder HC-head triplet must be replicated mappings."""

    @pytest.mark.parametrize(
        "name",
        ["decoder.hc_head_fn", "decoder.hc_head_base", "decoder.hc_head_scale"],
    )
    def test_decoder_hc_head_replicated(self, bridge_with_mtp, name):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(name)
        assert mapping is not None, f"missing decoder HC-head mapping: {name}"
        assert isinstance(mapping, ReplicatedMapping)
        # HF side drops the 'decoder.' prefix.
        assert mapping.hf_param == name.removeprefix("decoder.")


class TestMTPHCHeadMappings:
    """Per-MTP-layer HC head must mirror the decoder pattern."""

    @pytest.mark.parametrize(
        "suffix",
        ["hc_head_fn", "hc_head_base", "hc_head_scale"],
    )
    def test_mtp_hc_head_replicated(self, bridge_with_mtp, suffix):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(f"mtp.layers.0.{suffix}")
        assert mapping is not None, f"missing MTP HC-head mapping: mtp.layers.0.{suffix}"
        assert isinstance(mapping, ReplicatedMapping)
        assert mapping.hf_param == f"mtp.0.{suffix}"

    def test_mtp_hc_head_absent_when_no_mtp(self, bridge_without_mtp):
        registry = bridge_without_mtp.mapping_registry()
        names = _by_megatron(registry)
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            assert f"mtp.layers.0.{suffix}" not in names


class TestMTPEHProjSplit:
    """MTP e_proj and h_proj are separate ColumnParallelLinear modules.

    The bridge must use two AutoMappings (which auto-detect column parallelism),
    not the deprecated concatenated eh_proj path.
    """

    @pytest.mark.parametrize("name", ["e_proj", "h_proj"])
    def test_split_proj_automapping(self, bridge_with_mtp, name):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(f"mtp.layers.0.{name}.weight")
        assert mapping is not None, f"missing MTP projection: {name}"
        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == f"mtp.0.{name}.weight"

    def test_eh_proj_not_in_registry(self, bridge_with_mtp):
        registry = bridge_with_mtp.mapping_registry()
        for mapping in registry.mappings:
            assert "eh_proj" not in mapping.megatron_param, (
                f"deprecated eh_proj reference found in megatron_param: {mapping.megatron_param}"
            )
            hf_param = mapping.hf_param
            if isinstance(hf_param, str):
                assert "eh_proj" not in hf_param, f"deprecated eh_proj reference found in hf_param: {hf_param}"
            elif isinstance(hf_param, dict):
                for v in hf_param.values():
                    assert "eh_proj" not in v, f"deprecated eh_proj reference found in hf_param dict value: {v}"


class TestDeepSeekV4RotaryPercent:
    """Regression: HF partial_rotary_factor (relative to head_dim=512) must not shrink
    the Megatron rope cache — qk_pos_emb_head_dim (64) already encodes the rope split.
    rotary_percent=0.125 yields an 8-dim cos/sin cache: the unfused path silently
    rotates 8/64 dims and the fused MLA rope kernel reads cos/sin out of bounds (SFT NaN)."""

    def test_provider_bridge_forces_full_rotary_percent(self):
        hf_pretrained = MagicMock()
        hf_pretrained.config = _deepseek_v4_hf_config()
        provider = MagicMock()
        # what the generic partial_rotary_factor -> rotary_percent mapping produces
        provider.rotary_percent = 0.125

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        with patch.object(MegatronModelBridge, "provider_bridge", return_value=provider):
            out = bridge.provider_bridge(hf_pretrained)

        assert out.rotary_percent == 1.0
        assert out.csa_compress_rotary_base == 160000


class TestDeepSeekV4HardwareDefaults:
    """DSv4 Blackwell-only fused kernels must not default on for Hopper."""

    @pytest.mark.parametrize(
        ("capability", "expected"),
        [
            ((9, 0), False),
            ((10, 0), True),
        ],
    )
    def test_provider_bridge_gates_blackwell_only_fusions(self, capability, expected):
        hf_pretrained = MagicMock()
        hf_pretrained.config = _deepseek_v4_hf_config()
        provider = MagicMock()

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        with (
            patch.object(MegatronModelBridge, "provider_bridge", return_value=provider),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch(
                "megatron.bridge.models.deepseek.deepseek_v4_bridge.deepseek_v4_supports_fused_dsa_kernels",
                return_value=True,
            ),
        ):
            out = bridge.provider_bridge(hf_pretrained)

        assert out.apply_dsa_kernel_fusion is expected
        assert out.use_fused_mhc is expected

    def test_provider_bridge_disables_blackwell_only_fusions_without_cuda(self):
        hf_pretrained = MagicMock()
        hf_pretrained.config = _deepseek_v4_hf_config()
        provider = MagicMock()

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        with (
            patch.object(MegatronModelBridge, "provider_bridge", return_value=provider),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            out = bridge.provider_bridge(hf_pretrained)

        assert out.apply_dsa_kernel_fusion is False
        assert out.use_fused_mhc is False

    def test_provider_bridge_disables_dsa_fusion_when_optional_kernels_are_missing(self):
        hf_pretrained = MagicMock()
        hf_pretrained.config = _deepseek_v4_hf_config()
        provider = MagicMock()

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        with (
            patch.object(MegatronModelBridge, "provider_bridge", return_value=provider),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch(
                "megatron.bridge.models.deepseek.deepseek_v4_bridge.deepseek_v4_supports_fused_dsa_kernels",
                return_value=False,
            ),
        ):
            out = bridge.provider_bridge(hf_pretrained)

        assert out.apply_dsa_kernel_fusion is False
        assert out.use_fused_mhc is True


class TestDeepSeekV4ExportWeightDtype:
    def test_weight_dtype_set_skips_requantization(self, monkeypatch):
        from dataclasses import replace
        from unittest.mock import MagicMock

        from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        task = WeightConversionTask(param_name="w", global_param_name="w", mapping=MagicMock())
        task = replace(task, weight_dtype=torch.bfloat16)  # frozen: must be settable via replace

        def fail_requantize(*args, **kwargs):
            raise AssertionError("requantize must be skipped when weight_dtype is set")

        monkeypatch.setattr(quantization_utils, "requantize_hf_weight_scale_pairs", fail_requantize)
        weight = torch.randn(4, 4, dtype=torch.float32)
        converted = {"model.layers.0.mlp.weight": weight}
        hf_state = {"model.layers.0.mlp.weight": weight, "model.layers.0.mlp.scale": torch.ones(1)}

        out = bridge.maybe_modify_converted_hf_weight(task, converted, hf_state)

        assert out is converted  # returned unchanged; generic path casts the dtype

    def test_generic_export_cast_applies_plain_dtype(self):
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

        weights = {
            "model.layers.0.mlp.weight": torch.randn(4, 4, dtype=torch.float32),
            "model.layers.0.mlp.bias_idx": torch.ones(2, dtype=torch.int32),
        }
        out = MegatronModelBridge._cast_export_weight_dtype(weights, torch.bfloat16)
        assert out["model.layers.0.mlp.weight"].dtype == torch.bfloat16
        assert out["model.layers.0.mlp.bias_idx"].dtype == torch.int32  # int preserved
        assert MegatronModelBridge._cast_export_weight_dtype(weights, None) is weights

    def test_no_weight_dtype_requantizes_by_default(self, monkeypatch):
        from unittest.mock import MagicMock

        from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        task = WeightConversionTask(param_name="w", global_param_name="w", mapping=MagicMock())
        called = {}

        def fake_requantize(converted, hf_state, *, use_mxfp4=None):
            called["hit"] = True
            return {"quantized": torch.zeros(1)}

        monkeypatch.setattr(quantization_utils, "requantize_hf_weight_scale_pairs", fake_requantize)
        out = bridge.maybe_modify_converted_hf_weight(task, {"a.weight": torch.ones(1)}, {})

        assert called.get("hit") and "quantized" in out
