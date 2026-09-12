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

import logging
import sys
import types
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.fp8_export import FP8ExportLayout
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    MegatronModelBridge,
    WeightConversionTask,
    _HFNameSuffixMapping,
)
from megatron.bridge.models.conversion.param_mapping import split_qkv_weights
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


_QKV_GLOBAL = "decoder.layers.0.self_attention.linear_qkv.weight"
_MODEL_MB = "megatron.bridge.models.conversion.model_bridge"


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
    monkeypatch.setattr(
        bridge,
        "_detect_fp8_params",
        kwargs.get("detect_fp8", lambda *_a, **_k: {global_name: SimpleNamespace(block_shape=(None, None))}),
    )
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


class _IdentityMapping:
    def __init__(self, hf_param, megatron_param="dummy.megatron.weight"):
        self.hf_param = hf_param
        self.megatron_param = megatron_param

    def hf_to_megatron(self, hf_weights, _megatron_module):
        return hf_weights

    def megatron_to_hf(self, megatron_weights, _megatron_module):
        return {"model.weight": megatron_weights}

    def resolve(self, _captures):
        return _IdentityMapping(self.hf_param, self.megatron_param)


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

    def test_megatron_to_hf_scale_passes_row_block_size(self):
        class Base:
            def megatron_to_hf_scale(self, megatron_weights, megatron_module, *, row_block_size):
                assert row_block_size == 1
                return {"model.a": megatron_weights}

            def megatron_to_hf(self, megatron_weights, megatron_module):
                raise AssertionError("scale export should use the explicit row block size")

        w = _HFNameSuffixMapping(Base(), "_scale_inv", 1)
        t = torch.tensor([3.0])
        assert w.megatron_to_hf(t, None) == {"model.a_scale_inv": t}


class TestFp8ParamExport:
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
            ("fp8", {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": True}, False, 1),
            ("fp8", {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": False}, True, 0),
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
                        with pytest.raises(ValueError, match="only supports blockwise or MXFP8 parameter export"):
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
        caplog.set_level(logging.WARNING, logger="megatron.bridge.models.conversion.fp8_export")
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        MappingT = _make_qkv_mapping_type(gname)

        rowwise = torch.ones(scale_shape, dtype=torch.float32)
        metadata = {
            "fp8_dtype": SimpleNamespace(name="kFloat8E4M3"),
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

    def test_build_export_fp8_tasks_trims_mxfp8_scale_padding(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        MappingT = _make_qkv_mapping_type(gname)

        rowwise_scale_inv = torch.ones((128, 4), dtype=torch.uint8)
        metadata = {
            "fp8_dtype": SimpleNamespace(name="kFloat8E4M3"),
            "rowwise_data": torch.zeros((96, 64), dtype=torch.uint8),
            "rowwise_scale_inv": rowwise_scale_inv,
            "with_gemm_swizzled_scales": False,
        }
        fake_w = SimpleNamespace(get_metadata=lambda: metadata, shape=(96, 64))
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False, fp8_recipe="mxfp8"),
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

        assert tasks[0].param_weight.shape == (96, 64)
        assert tasks[0].param_weight.dtype == torch.float8_e4m3fn
        assert tasks[1].param_weight.shape == (96, 2)
        assert torch.all(tasks[1].param_weight == 1)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("shape", [(96, 64), (128, 128)], ids=["padded", "compact"])
    def test_real_te_mxfp8_export_dequantization_parity(self, monkeypatch, shape):
        pytest.importorskip("transformer_engine.pytorch")
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
        from transformer_engine_torch import DType

        supported, reason = FP8GlobalStateManager.is_mxfp8_available()
        if not supported:
            pytest.skip(reason)

        generator = torch.Generator(device="cuda").manual_seed(1234)
        source = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
        quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=False)
        quantizer.optimize_for_gemm = False
        quantized = quantizer(source)

        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        MappingT = _make_qkv_mapping_type(gname)
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False, fp8_recipe="mxfp8"),
            named_parameters=lambda: [(gname, quantized)],
        )
        _patch_export_task_context(
            monkeypatch,
            bridge,
            gname,
            registry_factory=lambda: MegatronMappingRegistry(MappingT()),
            detect_fp8=bridge._detect_fp8_params,
        )
        monkeypatch.setattr(f"{_MODEL_MB}.get_module_and_param_from_name", lambda *_a, **_k: (model, quantized))
        monkeypatch.setattr(f"{_MODEL_MB}.get_pg_size", lambda _g: 1)

        def gather(output, value, group=None):
            output[0] = value

        monkeypatch.setattr(f"{_MODEL_MB}.torch.distributed.all_gather_object", gather)
        tasks = bridge.build_export_fp8_tasks(SimpleNamespace(state=SimpleNamespace(source={})), [model])
        assert len(tasks) == 2
        exported = {}
        for task in tasks:
            exported.update(task.mapping.megatron_to_hf(task.param_weight, task.megatron_module))

        weight_name = "model.layers.0.self_attn.q_proj.weight"
        weight, scale = exported[weight_name], exported[f"{weight_name}_scale_inv"]
        assert weight.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.uint8
        assert weight.shape == shape
        assert scale.shape == (shape[0], shape[1] // 32)
        # E8M0 stores a biased exponent for each block of 32 weight values.
        dequantized = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)
        torch.testing.assert_close(dequantized, quantized.dequantize(dtype=torch.float32), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "layout_overrides, error_match",
        [
            pytest.param({"with_gemm_swizzled_scales": True}, "compact, non-swizzled scales", id="swizzled"),
            pytest.param({"scale_shape": (95, 2)}, "smaller than the compact scale shape", id="undersized"),
            pytest.param({"fp8_dtype": "kFloat8E5M2"}, "requires fp8_dtype=kFloat8E4M3", id="e5m2"),
            pytest.param({"data_dtype": torch.float32}, "requires uint8 data", id="data-dtype"),
            pytest.param({"scale_dtype": torch.float32}, "requires torch.uint8 scales", id="scale-dtype"),
        ],
    )
    def test_detect_fp8_params_propagates_remote_mxfp8_error(self, monkeypatch, layout_overrides, error_match):
        bridge = DummyBridge()
        model = SimpleNamespace(named_parameters=lambda: [])
        remote_layout = FP8ExportLayout(
            format_name="mxfp8",
            fp8_dtype="kFloat8E4M3",
            block_shape=(1, 32),
            data_dtype=torch.uint8,
            scale_dtype=torch.uint8,
            scale_shape=(128, 4),
            compact_scale_shape=(96, 2),
            with_gemm_swizzled_scales=False,
        )
        remote_layout = replace(remote_layout, **layout_overrides)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda *_a, **_k: [])
        monkeypatch.setattr(f"{_MODEL_MB}.get_pg_size", lambda _g: 2)

        def ag(output_list, obj, group=None):
            output_list[:] = [obj, {_QKV_GLOBAL: remote_layout}]

        monkeypatch.setattr(f"{_MODEL_MB}.torch.distributed.all_gather_object", ag)

        with pytest.raises(ValueError, match=error_match):
            bridge._detect_fp8_params([model], SimpleNamespace(), [], None, "_rowwise_scale_inv")

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
        holder.get_metadata = lambda: {
            "fp8_dtype": SimpleNamespace(name="kFloat8E4M3"),
            "rowwise_data": torch.zeros((32, 32), dtype=torch.uint8),
            "rowwise_scale_inv": torch.ones(1),
            "is_2D_scaled": False,
        }
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
            output_list[1] = {"decoder.layers.1.other.weight": next(iter(obj.values()))}

        monkeypatch.setattr(f"{_MODEL_MB}.torch.distributed.all_gather_object", ag)
        layouts = bridge._detect_fp8_params(
            [model], model.config, [gname, "decoder.layers.1.other.weight"], None, "_rowwise_scale_inv"
        )
        assert layouts[gname] and layouts["decoder.layers.1.other.weight"]

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

    @pytest.mark.parametrize(
        "metadata_overrides, error_match",
        [
            ({}, None),
            ({"fp8_dtype": SimpleNamespace(name="kFloat8E5M2")}, "requires fp8_dtype=kFloat8E4M3"),
            ({"fp8_dtype": None}, "requires fp8_dtype=kFloat8E4M3"),
            ({"rowwise_data": torch.zeros((96, 64))}, "requires uint8 data"),
            ({"rowwise_data": None}, "requires uint8 data"),
            ({"rowwise_scale_inv": torch.ones((128, 4))}, "requires torch.uint8 scales"),
            ({"rowwise_scale_inv": object()}, "requires torch.uint8 scales"),
        ],
    )
    def test_detect_fp8_params_from_mxfp8_metadata(self, monkeypatch, metadata_overrides, error_match):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        metadata = {
            "fp8_dtype": SimpleNamespace(name="kFloat8E4M3"),
            "rowwise_data": torch.zeros((96, 64), dtype=torch.uint8),
            "rowwise_scale_inv": torch.ones((128, 4), dtype=torch.uint8),
            "with_gemm_swizzled_scales": False,
        }
        metadata.update(metadata_overrides)
        holder = SimpleNamespace(get_metadata=lambda: metadata, shape=(96, 64), ndim=2)
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False, fp8_recipe="mxfp8"),
            named_parameters=lambda: [(gname, torch.nn.Parameter(torch.zeros(1)))],
        )
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_a, **_k: (SimpleNamespace(config=model.config), holder),
        )
        monkeypatch.setattr(f"{_MODEL_MB}._megatron_local_name_to_global", lambda *_a, **_k: gname)
        monkeypatch.setattr(f"{_MODEL_MB}.persistent_buffers", lambda *_a, **_k: [])
        monkeypatch.setattr(f"{_MODEL_MB}.get_pg_size", lambda _g: 1)

        gathered = []

        def _ag1(out, obj, group=None):
            gathered.append(True)
            out[0] = obj

        monkeypatch.setattr(f"{_MODEL_MB}.torch.distributed.all_gather_object", _ag1)

        if error_match is not None:
            with pytest.raises(ValueError, match=error_match):
                bridge._detect_fp8_params([model], model.config, [gname], None, "_rowwise_scale_inv")
            assert gathered == [True]
            return

        layouts = bridge._detect_fp8_params([model], model.config, [gname], None, "_rowwise_scale_inv")
        assert isinstance(layouts[gname], FP8ExportLayout)
        assert layouts[gname].format_name == "mxfp8"
        assert layouts[gname].fp8_dtype == "kFloat8E4M3"
        assert layouts[gname].block_shape == (1, 32)
        assert layouts[gname].data_dtype == torch.uint8
        assert layouts[gname].scale_dtype == torch.uint8
        assert layouts[gname].scale_shape == (128, 4)
        assert layouts[gname].compact_scale_shape == (96, 2)
        assert layouts[gname].with_gemm_swizzled_scales is False

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
            detect_fp8=lambda *_a, **_k: {gname: SimpleNamespace(block_shape=(1, 32))},
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

    def test_split_qkv_does_not_infer_compressed_layout(self):
        provider = SimpleNamespace(
            num_attention_heads=4,
            num_query_groups=2,
            hidden_size=128,
            kv_channels=None,
            attention_output_gate=False,
        )
        with pytest.raises(RuntimeError, match="shape"):
            split_qkv_weights(provider, torch.randn(256, 4))
