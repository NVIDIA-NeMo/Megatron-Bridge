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

import json
import sys
from dataclasses import dataclass, fields
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from transformers import DeepseekV4Config

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    _DSV4_FUSED_DSA_WRAPPERS,
    DeepSeekV4Bridge,
    _dsv4_compress_ratios,
    _dsv4_hybrid_csa_compress_ratios,
    _dsv4_hybrid_layer_pattern,
    _dsv4_num_hash_layers,
    deepseek_v4_supports_fused_dsa_kernels,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.models.deepseek.deepseek_v4_config import DeepSeekV4ModelConfig, DeepSeekV4TransformerConfig
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider
from megatron.bridge.models.hybrid import BridgeHybridModelBuilder
from megatron.bridge.models.hybrid.hybrid_provider import (
    HybridModelProvider,
    transformer_engine_hybrid_stack_spec,
)
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


@pytest.fixture
def bridge_with_mtp():
    """A DSv4 bridge with hf_config stubbed for a single MTP layer."""
    bridge = DeepSeekV4Bridge()
    # mapping_registry reads num_hidden_layers and num_nextn_predict_layers from hf_config.
    bridge.hf_config = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
    return bridge


@pytest.fixture
def bridge_without_mtp():
    """A DSv4 bridge with hf_config that has zero MTP layers."""
    bridge = DeepSeekV4Bridge()
    bridge.hf_config = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)
    return bridge


def _by_megatron(registry):
    """Index mappings by megatron_param for quick lookup in assertions."""
    return {m.megatron_param: m for m in registry.mappings}


def _dummy_task(*, weight_dtype=None):
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask

    return WeightConversionTask(param_name="", global_param_name="", mapping=None, weight_dtype=weight_dtype)


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
        torch_dtype="bfloat16",
    )


class TestDeepSeekV4OutputProjectionConfig:
    """HF grouped-output geometry must reach the current MCore field names."""

    def test_model_config_sets_output_projection_fields(self):
        hf_config = _deepseek_v4_hf_config()
        hf_config.o_groups = 4
        hf_config.o_lora_rank = 256

        model_config = DeepSeekV4Bridge().hf_config_to_model_config(hf_config)

        assert model_config.output_projection_groups == 4
        assert model_config.output_projection_lora_rank == 256


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


class TestDeepSeekV4RouterExpertBias:
    """DSv4 checkpoints may omit their runtime router-bias buffers."""

    @staticmethod
    def _mapping(bridge):
        return _by_megatron(bridge.mapping_registry())["decoder.layers.1.inner_layer.mlp.router.expert_bias"]

    def test_mapping_allows_missing_hf_bias(self, bridge_without_mtp):
        mapping = self._mapping(bridge_without_mtp)

        assert isinstance(mapping, AutoMapping)
        assert mapping.allow_hf_name_mismatch

    def test_import_synthesizes_zero_bias_when_hf_checkpoint_omits_it(self):
        bridge = DeepSeekV4Bridge()
        bridge.hf_config = SimpleNamespace(n_routed_experts=8)

        result = bridge.maybe_modify_loaded_hf_weight("layers.0.ffn.gate.bias", {})

        assert result.dtype == torch.float32
        assert torch.equal(result, torch.zeros(8))

    def test_import_preserves_hf_bias_when_present(self):
        bridge = DeepSeekV4Bridge()
        bias = torch.arange(8, dtype=torch.float32)

        result = bridge.maybe_modify_loaded_hf_weight("layers.0.ffn.gate.bias", {"layers.0.ffn.gate.bias": bias})

        assert result is bias

    def test_export_omits_bias_absent_from_source_checkpoint(self):
        bridge = DeepSeekV4Bridge()
        bias = torch.zeros(8)

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {"layers.0.ffn.gate.bias": bias}, {})

        assert result == {}


class TestDeepSeekV4QuantizedExport:
    """DSv4 export must regenerate quantized weights and scale tensors."""

    def test_import_accepts_legacy_flat_indexer_weight_name(self):
        bridge = DeepSeekV4Bridge()
        flat_name = "layers.1.attn.indexer.weights_proj.weight"
        scorer_name = "layers.1.attn.indexer.scorer.weights_proj.weight"
        weight = torch.randn(4, 4, dtype=torch.bfloat16)

        result = bridge.maybe_modify_loaded_hf_weight(scorer_name, {flat_name: weight})

        assert result is weight

    @pytest.mark.parametrize("weight_dtype", [None, torch.bfloat16])
    def test_export_uses_legacy_flat_indexer_weight_name_when_source_requires_it(self, weight_dtype):
        bridge = DeepSeekV4Bridge()
        scorer_name = "layers.1.attn.indexer.scorer.weights_proj.weight"
        flat_name = "layers.1.attn.indexer.weights_proj.weight"
        weight = torch.randn(4, 4, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(
            _dummy_task(weight_dtype=weight_dtype), {scorer_name: weight}, {flat_name: weight}
        )

        assert scorer_name not in result
        assert result[flat_name] is weight

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
    """Sequential (non-grouped) expert mappings exist for moe_grouped_gemm=False (ModelOpt pruning).

    On the hybrid layout the MoE for logical layer 0 lives at hybrid layer index 1, under
    the HyperConnectionHybridLayer ``inner_layer`` wrapper.
    """
    params = _by_megatron(bridge_with_mtp.mapping_registry())
    assert "decoder.layers.1.inner_layer.mlp.experts.local_experts.*.linear_fc1.weight" in params
    assert "decoder.layers.1.inner_layer.mlp.experts.local_experts.*.linear_fc2.weight" in params


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

    def test_model_config_forces_full_rotary_percent(self):
        hf_config = _deepseek_v4_hf_config()
        # The generic partial_rotary_factor -> rotary_percent mapping would produce 0.125.
        hf_config.partial_rotary_factor = 0.125

        model_config = DeepSeekV4Bridge().hf_config_to_model_config(hf_config)

        assert model_config.rotary_percent == model_config.transformer.rotary_percent == 1.0
        assert model_config.csa_compress_rotary_base == 160000


class TestDeepSeekV4MoEDispatcher:
    """DeepSeek-V4 providers use the inference-validated HybridEP path."""

    def test_model_config_uses_hybridep(self):
        out = DeepSeekV4Bridge().hf_config_to_model_config(_deepseek_v4_hf_config())

        assert out.moe_token_dispatcher_type == "flex"
        assert out.moe_flex_dispatcher_backend == "hybridep"
        assert out.moe_flex_dispatcher_num_sms == 16
        assert out.moe_permute_fusion_into_hybridep is False


class TestDeepSeekV4HardwareCapabilities:
    def test_fused_dsa_accepts_current_cudnn_frontend_api(self):
        wrappers = {name: (lambda: None) for name in _DSV4_FUSED_DSA_WRAPPERS}
        modules = {
            "cudnn": SimpleNamespace(DSA=SimpleNamespace(**wrappers)),
            "flash_mla": SimpleNamespace(flash_mla_sparse_fwd=lambda: None),
        }
        with patch.dict(sys.modules, modules):
            assert deepseek_v4_supports_fused_dsa_kernels() is True

    def test_fused_dsa_rejects_obsolete_combined_indexer_api(self):
        modules = {
            "cudnn": SimpleNamespace(DSA=SimpleNamespace(indexer_forward_top_k_wrapper=lambda: None)),
            "flash_mla": SimpleNamespace(flash_mla_sparse_fwd=lambda: None),
        }
        with patch.dict(sys.modules, modules):
            assert deepseek_v4_supports_fused_dsa_kernels() is False


class TestDeepSeekV4HardwareDefaults:
    """DSv4 defaults cuDNN CSA on Hopper+ while fused mHC remains Blackwell-only."""

    @pytest.mark.parametrize(
        ("capability", "expected_dsa_backend", "expected_fused_mhc"),
        [
            ((8, 0), "none", False),
            ((9, 0), "cudnn", False),
            ((10, 0), "cudnn", True),
        ],
    )
    def test_model_config_gates_fusions_by_hardware(self, capability, expected_dsa_backend, expected_fused_mhc):
        hf_config = _deepseek_v4_hf_config()

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch(
                "megatron.bridge.models.deepseek.deepseek_v4_bridge.deepseek_v4_supports_fused_dsa_kernels",
                return_value=True,
            ),
        ):
            out = DeepSeekV4Bridge().hf_config_to_model_config(hf_config)

        assert out.dsa_kernel_backend == expected_dsa_backend
        assert out.enable_mhc_connections is True
        assert out.mhc_num_residual_streams == hf_config.hc_mult
        assert out.use_fused_mhc is expected_fused_mhc

    def test_model_config_disables_fusions_without_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            out = DeepSeekV4Bridge().hf_config_to_model_config(_deepseek_v4_hf_config())

        assert out.dsa_kernel_backend == "none"
        assert out.enable_mhc_connections is True
        assert out.use_fused_mhc is False

    def test_model_config_disables_dsa_fusion_when_optional_kernels_are_missing(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch(
                "megatron.bridge.models.deepseek.deepseek_v4_bridge.deepseek_v4_supports_fused_dsa_kernels",
                return_value=False,
            ),
        ):
            out = DeepSeekV4Bridge().hf_config_to_model_config(_deepseek_v4_hf_config())

        assert out.dsa_kernel_backend == "none"
        assert out.enable_mhc_connections is True
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


class TestDeepSeekV4HybridPattern:
    """DSv4 helper functions that translate the flat GPT-form recipe into a hybrid pattern."""

    def test_layer_pattern_maps_ratios_to_symbols(self):
        # 0 -> W (window), 4 -> C (CSA), 128 -> H (HCA); each logical layer gains a trailing E.
        assert _dsv4_hybrid_layer_pattern([0, 4, 128, 4], 4) == "WECEHECE"

    def test_layer_pattern_respects_num_hidden_layers(self):
        assert _dsv4_hybrid_layer_pattern([0, 4, 128, 4, 0], 3) == "WECEHE"

    def test_layer_pattern_rejects_unknown_ratio(self):
        with pytest.raises(ValueError, match="compression ratio"):
            _dsv4_hybrid_layer_pattern([7], 1)

    def test_csa_ratios_double_and_append_mtp(self):
        # Main "WECEHECE" -> [0,0,4,0,128,0,4,0]; one "/WE" MTP depth -> [0,0].
        assert _dsv4_hybrid_csa_compress_ratios("WECEHECE", "WE", 1) == [0, 0, 4, 0, 128, 0, 4, 0, 0, 0]

    def test_csa_ratios_no_mtp(self):
        assert _dsv4_hybrid_csa_compress_ratios("WECE", "", 0) == [0, 0, 4, 0]


class TestDeepSeekV4HybridProvider:
    """The provider must be both an MLA config carrier and a HybridModel builder."""

    def test_provider_is_both_mla_and_hybrid(self):
        assert issubclass(DeepSeekV4HybridModelProvider, MLAModelProvider)
        assert issubclass(DeepSeekV4HybridModelProvider, HybridModelProvider)

    def test_hybrid_provide_wins_in_mro(self):
        # HybridModelProvider must precede the GPT-model provider so provide()/finalize()
        # build a HybridModel and derive num_layers from hybrid_layer_pattern.
        assert DeepSeekV4HybridModelProvider.provide is HybridModelProvider.provide
        assert DeepSeekV4HybridModelProvider.finalize is HybridModelProvider.finalize


class TestDeepSeekV4BridgeHybridConfig:
    """The bridge must translate the DSv4 HF config into the hybrid layer pattern."""

    def test_model_config_builds_hybrid_pattern(self):
        out = DeepSeekV4Bridge().hf_config_to_model_config(_deepseek_v4_hf_config())

        # compress_ratios [0,4,128,4] over 4 layers -> WECEHECE.
        assert out.hybrid_layer_pattern == "WECEHECE"
        assert out.num_layers == 8
        assert out.mtp_hybrid_override_pattern == "WE"
        assert out.hybrid_stack_spec is transformer_engine_hybrid_stack_spec
        assert out.output_projection_groups == 8
        assert out.output_projection_lora_rank == 1024
        # MCore counts MoE positions, so this remains the logical HF layer count.
        assert out.moe_num_hash_layers == 3
        assert out.hash_moe_vocab_size == 129280
        # Doubled per-hybrid-layer ratios (main) + one MTP depth [0, 0].
        assert out.csa_compress_ratios == [0, 0, 4, 0, 128, 0, 4, 0, 0, 0]

    @pytest.mark.parametrize("mtp_depth", [0, 1])
    def test_checkpoint_config_reloads_native_stack(self, tmp_path, mtp_depth):
        from megatron.bridge.training.config import ConfigContainer
        from megatron.bridge.training.model_load_save import load_model_config

        hf_config = _deepseek_v4_hf_config()
        hf_config.num_nextn_predict_layers = mtp_depth
        model_config = DeepSeekV4Bridge().hf_config_to_model_config(hf_config)
        model_config.hidden_size = 128
        model_config.num_attention_heads = 16
        model_config.vocab_size = 129280

        config = ConfigContainer(
            model=model_config,
            train=None,
            optimizer=None,
            scheduler=None,
            dataset=None,
            logger=None,
            tokenizer=None,
            checkpoint=None,
        )
        config.to_yaml(str(tmp_path / "run_config.yaml"))
        restored, mlm_args = load_model_config(str(tmp_path))

        assert mlm_args is None
        assert isinstance(restored, DeepSeekV4ModelConfig)
        assert restored.hybrid_stack_spec is transformer_engine_hybrid_stack_spec
        assert (
            BridgeHybridModelBuilder(restored)._resolve_hybrid_stack_spec() is transformer_engine_hybrid_stack_spec()
        )
        assert restored.hybrid_layer_pattern == model_config.hybrid_layer_pattern
        assert (restored.mtp_num_layers or 0) == mtp_depth

    def test_megatron_to_hf_config_restores_renamed_mcore_fields(self):
        provider = SimpleNamespace(
            hybrid_layer_pattern="WECEHECE",
            mtp_num_layers=1,
            moe_num_hash_layers=3,
            activation_func_clamp_value=10.0,
            output_projection_groups=8,
            output_projection_lora_rank=1024,
            csa_window_size=128,
            mhc_num_residual_streams=4,
            mhc_sinkhorn_iterations=20,
            moe_shared_expert_intermediate_size=1024,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
        )

        with patch.object(MegatronModelBridge, "megatron_to_hf_config", return_value={}):
            hf_config = DeepSeekV4Bridge.megatron_to_hf_config(provider)

        assert hf_config["num_hash_layers"] == 3
        assert hf_config["o_groups"] == 8
        assert hf_config["o_lora_rank"] == 1024
        assert hf_config["hc_mult"] == 4
        assert hf_config["head_dim"] == 512
        assert hf_config["partial_rotary_factor"] == 0.125

    def test_megatron_to_hf_config_leaves_training_overrides_to_the_reference(self):
        """Import pins seq_length, moe_aux_loss_coeff and rotary_percent, so they cannot describe the HF model."""
        provider = SimpleNamespace(
            hybrid_layer_pattern="WEWECEHE",
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            output_projection_groups=8,
            output_projection_lora_rank=1024,
        )
        generic_config = {
            "max_position_embeddings": 4096,
            "router_aux_loss_coef": 0.0,
            "aux_loss_alpha": 0.0,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "head_dim": 64,
            "qk_rope_head_dim": 64,
        }

        with patch.object(MegatronModelBridge, "megatron_to_hf_config", return_value=generic_config):
            hf_config = DeepSeekV4Bridge.megatron_to_hf_config(provider)

        for hf_name in ("max_position_embeddings", "router_aux_loss_coef", "aux_loss_alpha", "rope_theta"):
            assert hf_name not in hf_config
        assert (hf_config["head_dim"], hf_config["qk_rope_head_dim"], hf_config["partial_rotary_factor"]) == (
            512,
            64,
            0.125,
        )


# DeepSeek-V4-Flash config.json (legacy key names) cut to four layers plus MTP, like the GPU truncation checks.
_DEEPSEEK_V4_FLASH_FOUR_LAYER_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "compress_ratios": [0, 0, 4, 128, 0],
    "compress_rope_theta": 160000,
    "eos_token_id": 1,
    "expert_dtype": "fp4",
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "initializer_range": 0.02,
    "max_position_embeddings": 1048576,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hash_layers": 3,
    "num_hidden_layers": 4,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "vocab_size": 129280,
}


class TestDeepSeekV4CheckpointConfigExport:
    """``convert.sh export`` rebuilds the HF config from the checkpoint through ``AutoBridge.from_auto_config``."""

    @pytest.mark.parametrize("use_model_config", [False, True], ids=["provider", "model_config"])
    def test_from_auto_config_reproduces_the_reference_config(self, tmp_path, use_model_config):
        pytest.importorskip("transformers.models.deepseek_v4")
        from transformers import AutoConfig

        from megatron.bridge import AutoBridge
        from megatron.bridge.training.config import ConfigContainer

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text(json.dumps(_DEEPSEEK_V4_FLASH_FOUR_LAYER_CONFIG))
        reference = AutoConfig.from_pretrained(hf_dir)
        bridge = AutoBridge.from_hf_config(reference)
        if use_model_config and bridge._model_bridge.MODEL_CONFIG_CLASS is None:
            pytest.skip("DeepSeek-V4 has no builder-backed model config yet.")
        model_config = (
            bridge.get_model_config() if use_model_config else bridge.to_megatron_provider(load_weights=False)
        )
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        ConfigContainer(
            model=model_config,
            train=None,
            optimizer=None,
            scheduler=None,
            dataset=None,
            logger=None,
            tokenizer=None,
            checkpoint=None,
        ).to_yaml(str(checkpoint_dir / "run_config.yaml"))

        rebuilt = AutoBridge.from_auto_config(str(checkpoint_dir), str(hf_dir)).hf_pretrained

        # The export writes this JSON, so it must match the reference byte for byte.
        assert rebuilt.to_json_string() == reference.to_json_string()
        # Native configs derive qk_rope_head_dim from head_dim and partial_rotary_factor; a re-import must
        # keep v_head_dim > qk_pos_emb_head_dim, which MCore asserts for the DSv4 hybrid attention.
        rebuilt_bridge = AutoBridge.from_hf_config(rebuilt)
        reimported = (
            rebuilt_bridge.get_model_config()
            if use_model_config
            else rebuilt_bridge.to_megatron_provider(load_weights=False)
        )
        assert (reimported.v_head_dim, reimported.qk_pos_emb_head_dim) == (512, 64)


class TestDeepSeekV4HybridMappingLayout:
    """Each logical HF layer must split across attention (2*i) and MoE (2*i+1) hybrid layers."""

    def test_mapping_splits_hf_layer_across_attention_and_moe(self, bridge_with_mtp):
        registry = bridge_with_mtp.mapping_registry()

        attn = registry.hf_to_megatron_lookup("layers.3.attn.wo_b.weight")
        moe = registry.hf_to_megatron_lookup("layers.3.ffn.gate.weight")
        attn_hc = registry.hf_to_megatron_lookup("layers.3.hc_attn_fn")
        moe_hc = registry.hf_to_megatron_lookup("layers.3.hc_ffn_fn")

        assert attn.megatron_param == "decoder.layers.6.inner_layer.self_attention.linear_proj.weight"
        assert moe.megatron_param == "decoder.layers.7.inner_layer.mlp.router.weight"
        assert attn_hc.megatron_param == "decoder.layers.6.hyper_connection.mapping_proj.weight"
        assert moe_hc.megatron_param == "decoder.layers.7.hyper_connection.mapping_proj.weight"

    def test_mapping_resolves_hybrid_final_norm(self, bridge_with_mtp):
        registry = bridge_with_mtp.mapping_registry()
        final_norm = registry.megatron_to_hf_lookup("decoder.final_norm.weight")
        assert final_norm.hf_param == "norm.weight"

    def test_mtp_inner_layers_nest_under_hybrid_stack(self, bridge_with_mtp):
        registry = bridge_with_mtp.mapping_registry()

        mtp_attn = registry.hf_to_megatron_lookup("mtp.0.attn.wo_b.weight")
        mtp_moe = registry.hf_to_megatron_lookup("mtp.0.ffn.gate.weight")

        assert mtp_attn.megatron_param == (
            "mtp.layers.0.mtp_model_layer.layers.0.inner_layer.self_attention.linear_proj.weight"
        )
        assert mtp_moe.megatron_param == "mtp.layers.0.mtp_model_layer.layers.1.inner_layer.mlp.router.weight"


def _native_deepseek_v4_config(**overrides) -> DeepseekV4Config:
    """DeepSeek-V4-Flash's architecture with eight logical layers, in Transformers' native schema."""
    values = dict(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=4096,
        moe_intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        vocab_size=129280,
        max_position_embeddings=1048576,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 128, 0],
        num_hash_layers=3,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        swiglu_limit=10.0,
        sliding_window=128,
        o_groups=8,
        o_lora_rank=1024,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    return DeepseekV4Config(**values)


def _same_value(left, right) -> bool:
    # finalize() creates a new init-method partial and pipeline layout object on each config.
    if isinstance(left, partial) and isinstance(right, partial):
        return (left.func, left.args, left.keywords) == (right.func, right.args, right.keywords)
    if isinstance(left, PipelineParallelLayerLayout) and isinstance(right, PipelineParallelLayerLayout):
        return left.input_data == right.input_data
    return left == right


def _shared_fields_differing(model_config, provider) -> dict:
    model_fields = {f.name for f in fields(model_config)} | {f.name for f in fields(model_config.transformer)}
    shared = {f.name for f in fields(provider)} & model_fields
    return {
        name: (getattr(provider, name), getattr(model_config, name))
        for name in sorted(shared)
        if not _same_value(getattr(provider, name), getattr(model_config, name))
    }


class TestDeepSeekV4BridgeModelConfig:
    def test_provider_bridge_is_inherited_compatibility_only(self):
        assert "provider_bridge" not in DeepSeekV4Bridge.__dict__

    def test_conversion_uses_builder_config(self):
        assert DeepSeekV4Bridge.MODEL_CONFIG_CLASS is DeepSeekV4ModelConfig
        assert DeepSeekV4Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
        assert DeepSeekV4ModelConfig.transformer_config_class is DeepSeekV4TransformerConfig

    def test_native_config_maps_rope_to_the_mla_transformer(self):
        model_config = DeepSeekV4Bridge().hf_config_to_model_config(_native_deepseek_v4_config())

        assert isinstance(model_config.transformer, DeepSeekV4TransformerConfig)
        assert model_config.position_embedding_type == "rope"
        assert model_config.transformer.rotary_base == 10000
        assert model_config.transformer.rotary_percent == 1.0
        assert model_config.csa_compress_rotary_base == 160000
        assert model_config.rotary_scaling_factor == 16
        assert model_config.hybrid_layer_pattern == "WEWECEHECEHECEHE"
        assert model_config.mtp_hybrid_override_pattern == "WE"

    @pytest.mark.parametrize(
        "overrides",
        [{}, {"num_nextn_predict_layers": 0}, {"tie_word_embeddings": True}],
        ids=["mtp", "no-mtp", "tied"],
    )
    @pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
    def test_model_config_matches_provider(self, overrides, pipeline_parallel_size):
        hf_config = _native_deepseek_v4_config(**overrides)
        bridge = DeepSeekV4Bridge()
        model_config = bridge.hf_config_to_model_config(hf_config)
        with pytest.warns(FutureWarning, match="deprecated for builder-backed models"):
            provider = bridge.provider_bridge(SimpleNamespace(config=hf_config))
        assert _shared_fields_differing(model_config, provider) == {}

        for config in (model_config, provider):
            config.pipeline_model_parallel_size = pipeline_parallel_size
            if pipeline_parallel_size > 1:
                set_deepseek_v4_pipeline_model_parallel_layout(config)
            config.finalize()

        assert _shared_fields_differing(model_config, provider) == {}
        assert DeepSeekV4Bridge.megatron_to_hf_config(model_config) == DeepSeekV4Bridge.megatron_to_hf_config(provider)

    def test_provider_no_longer_carries_unread_norm_topk_prob(self):
        hf_config = _native_deepseek_v4_config()
        with pytest.warns(FutureWarning):
            provider = DeepSeekV4Bridge().provider_bridge(SimpleNamespace(config=hf_config))

        assert "norm_topk_prob" not in vars(provider)

    def test_cp_partition_mode_is_declared_on_the_transformer_config(self):
        @dataclass
        class _Root:
            model: DeepSeekV4ModelConfig

        root = _Root(model=DeepSeekV4Bridge().hf_config_to_model_config(_native_deepseek_v4_config()))
        assert root.model.cp_partition_mode == "zigzag"

        process_config_with_overrides(root, cli_overrides=["model.cp_partition_mode=contiguous"])

        assert root.model.transformer.cp_partition_mode == "contiguous"
        with pytest.raises(AttributeError, match="declares a field"):
            root.model.mtp_eval_in_bf16 = True

    def test_round_trip_restores_the_pipeline_layout(self):
        model_config = DeepSeekV4Bridge().hf_config_to_model_config(_native_deepseek_v4_config())
        model_config.pipeline_model_parallel_size = 2
        set_deepseek_v4_pipeline_model_parallel_layout(model_config)
        model_config.finalize()

        restored = ModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, DeepSeekV4ModelConfig)
        assert isinstance(restored.transformer, DeepSeekV4TransformerConfig)
        assert restored.hybrid_layer_pattern == model_config.hybrid_layer_pattern
        assert restored.pipeline_model_parallel_layout == model_config.pipeline_model_parallel_layout.input_data
        restored.finalize()
        assert restored.hybrid_layer_pattern == model_config.hybrid_layer_pattern
