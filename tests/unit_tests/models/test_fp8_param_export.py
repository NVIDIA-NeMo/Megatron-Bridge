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

import builtins
import logging
import sys
import types
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
from megatron.bridge.models.conversion.param_mapping import split_qkv_weights
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


_QKV_GLOBAL = "decoder.layers.0.self_attention.linear_qkv.weight"
_MODEL_MB = "megatron.bridge.models.conversion.model_bridge"


def _make_qkv_mapping_type(global_name: str = _QKV_GLOBAL):
    class MegatronQkvMapping:
        hf_param = "hf.qkv.weight"
        megatron_param = global_name

        def resolve(self, _captures):
            return MegatronQkvMapping()

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
        monkeypatch.setattr(DummyBridge, "_broadcast_shared_embeddings", lambda self, *_a, **_k: None)
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

        with patch.object(AutoBridge, "_model_bridge", mock_mb):
            with patch(
                "megatron.bridge.models.conversion.auto_bridge.model_bridge.stream_weights_megatron_to_hf"
            ) as stream:
                stream.return_value = iter([("model.layers.0.self_attn.q_proj.weight", torch.ones(1))])
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
            assert stream.call_args.kwargs["conversion_tasks"] == fp8_tasks
        elif expect_raise:
            mock_mb.build_export_fp8_tasks.assert_not_called()
        else:
            assert stream.call_args.kwargs["conversion_tasks"] is None

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

        class Reg:
            @staticmethod
            def megatron_to_hf_lookup(_n):
                return MappingT()

        rowwise = torch.ones(scale_shape, dtype=torch.float32)
        fake_w = SimpleNamespace(
            _rowwise_data=torch.zeros((2, 256), dtype=torch.uint8),
            _rowwise_scale_inv=rowwise,
            _fp8_dtype=None,
            _quantizer=quantizer,
            _is_2D_scaled=is_2d,
            shape=(2, 256),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [(gname, torch.nn.Parameter(torch.zeros(1)))],
        )
        _patch_export_task_context(monkeypatch, bridge, gname, registry_factory=lambda: Reg())
        monkeypatch.setattr(
            f"{_MODEL_MB}.get_module_and_param_from_name",
            lambda *_a, **_k: (SimpleNamespace(config=model.config), fake_w),
        )
        tasks = bridge.build_export_fp8_tasks(
            SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace())), [model]
        )
        assert len(tasks) == 2 and tasks[1].global_param_name == f"{gname}_scale_inv"
        assert tasks[1].param_weight.shape == expect_shape
        assert torch.all(tasks[1].param_weight == 1.0)
        assert ("block_len or not is_2d_scaled" in caplog.text) is warn_trim
        if tasks[1].param_weight.shape == rowwise.shape:
            assert tasks[1].param_weight.data_ptr() == rowwise.data_ptr()

    def test_detect_fp8_params_blockwise(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL

        class TeTensor:
            pass

        monkeypatch.setitem(
            sys.modules,
            "transformer_engine.pytorch.tensor",
            types.ModuleType("transformer_engine.pytorch.tensor"),
        )
        sys.modules["transformer_engine.pytorch.tensor"].Float8BlockwiseQTensor = TeTensor

        holder = TeTensor()
        holder._rowwise_scale_inv = torch.ones(1)
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

    def test_detect_fp8_params_te_import_fails(self, monkeypatch):
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        real_imp = builtins.__import__

        def guard(name, glb=None, loc=None, fromlist=(), level=0):
            if name == "transformer_engine.pytorch.tensor":
                raise ImportError("no te")
            return real_imp(name, glb, loc, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guard)
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
        "mode",
        [
            pytest.param("remote_pp", id="remote_pp"),
            pytest.param("scale_lookup_none", id="scale_lookup_none"),
        ],
    )
    def test_build_export_fp8_tasks_placeholder(self, monkeypatch, caplog, mode):
        caplog.set_level(logging.WARNING, logger="megatron.bridge.models.conversion.model_bridge")
        bridge = DummyBridge()
        gname = _QKV_GLOBAL
        MappingT = _make_qkv_mapping_type(gname)

        if mode == "remote_pp":

            class Reg:
                @staticmethod
                def megatron_to_hf_lookup(_n):
                    return MappingT()

            _patch_export_task_context(
                monkeypatch, bridge, gname, registry_factory=lambda: Reg(), pp_rank=1, pp_size=2
            )
        else:
            n = {"c": 0}

            class Reg:
                @staticmethod
                def megatron_to_hf_lookup(_n):
                    n["c"] += 1
                    return MappingT() if n["c"] == 1 else None

            _patch_export_task_context(monkeypatch, bridge, gname, registry_factory=lambda: Reg())

        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [],
        )
        tasks = bridge.build_export_fp8_tasks(
            SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace())), [model]
        )
        assert len(tasks) == 2
        if mode == "remote_pp":
            assert tasks[0] and tasks[1]
            assert tasks[0].megatron_module is None and isinstance(tasks[0].mapping, MappingT)
            assert isinstance(tasks[1].mapping, _HFNameSuffixMapping)
        else:
            assert tasks[0] and tasks[0].global_param_name == gname
            assert tasks[1] is None
            assert "No mapping found for global_name" in caplog.text

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
