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

import math
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import split_qkv_weights
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


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


class TestFp8ParamExport:
    @pytest.mark.parametrize(
        "export_weight_dtype, expect_unquantized",
        [
            ("fp8", True),
            ("bf16", False),
        ],
    )
    def test_load_weights_hf_to_megatron_unquantized_capture_by_fp8_flag(
        self, monkeypatch, export_weight_dtype, expect_unquantized
    ):
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

        monkeypatch.setattr(DummyBridge, "build_conversion_tasks", lambda self, *_args, **_kwargs: [task])
        monkeypatch.setattr(DummyBridge, "_with_progress_tracking", lambda self, tasks, *_args, **_kwargs: tasks)
        monkeypatch.setattr(DummyBridge, "_broadcast_shared_embeddings", lambda self, *_args, **_kwargs: None)

        hf_pretrained = SimpleNamespace(state={"hf.w0": converted}, model_name_or_path="dummy")
        models = [SimpleNamespace()]
        returned_models = bridge.load_weights_hf_to_megatron(hf_pretrained, models)

        assert returned_models == models
        torch.testing.assert_close(target_param.detach(), converted)
        if expect_unquantized:
            assert bridge.unquantized_state_dict is not None
            assert "model" in bridge.unquantized_state_dict
            assert "decoder.layers.0.linear.weight" in bridge.unquantized_state_dict["model"]
        else:
            assert bridge.unquantized_state_dict is None

    @pytest.mark.parametrize(
        "export_weight_dtype, runtime_fp8_cfg, should_raise, expected_fp8_call_count",
        [
            (
                "fp8",
                {"fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": True},
                False,
                1,
            ),
            (
                "fp8",
                {"fp8": "e4m3", "fp8_recipe": "tensorwise", "fp8_param": True},
                True,
                0,
            ),
            (
                "fp8",
                {"fp8": None, "fp8_recipe": "blockwise", "fp8_param": True},
                True,
                0,
            ),
            (
                "bf16",
                {"fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": True},
                False,
                0,
            ),
        ],
    )
    def test_export_hf_weights_fp8_task_branching(
        self, export_weight_dtype, runtime_fp8_cfg, should_raise, expected_fp8_call_count
    ):
        """export_hf_weights FP8 branching with valid/invalid runtime configs."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        mock_megatron_model = [SimpleNamespace(config=SimpleNamespace(**runtime_fp8_cfg))]

        mock_model_bridge = Mock()
        fp8_tasks = [Mock(name="fp8_weight_task"), Mock(name="fp8_scale_inv_task")]
        mock_model_bridge.build_export_fp8_tasks.return_value = fp8_tasks

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            with patch(
                "megatron.bridge.models.conversion.auto_bridge.model_bridge.stream_weights_megatron_to_hf"
            ) as mock_stream:
                mock_stream.return_value = iter([("model.layers.0.self_attn.q_proj.weight", torch.ones(1))])

                with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
                    mock_arch_class = Mock()
                    mock_transformers.LlamaForCausalLM = mock_arch_class

                    bridge = AutoBridge(mock_hf_model)
                    bridge.export_weight_dtype = export_weight_dtype

                    with patch.object(AutoBridge, "_causal_lm_architecture", new_callable=PropertyMock) as mock_prop:
                        mock_prop.return_value = mock_arch_class
                        if should_raise:
                            with pytest.raises(ValueError, match="only supports blockwise FP8 parameter export"):
                                _ = list(bridge.export_hf_weights(mock_megatron_model, cpu=True))
                        else:
                            _ = list(bridge.export_hf_weights(mock_megatron_model, cpu=True))

                assert mock_model_bridge.build_export_fp8_tasks.call_count == expected_fp8_call_count
                if export_weight_dtype == "fp8" and not should_raise:
                    mock_model_bridge.build_export_fp8_tasks.assert_called_once_with(
                        mock_hf_model, mock_megatron_model
                    )
                    assert mock_stream.call_args.kwargs["conversion_tasks"] == fp8_tasks
                elif should_raise:
                    mock_model_bridge.build_export_fp8_tasks.assert_not_called()
                else:
                    assert mock_stream.call_args.kwargs["conversion_tasks"] is None

    def test_build_export_fp8_tasks_adds_scale_inv_task_when_fp8_detected(self, monkeypatch):
        """When FP8 flags are detected, build_export_fp8_tasks should insert scale_inv task."""
        bridge = DummyBridge()

        global_name = "decoder.layers.0.self_attention.linear_qkv.weight"
        local_name = global_name
        model = SimpleNamespace(
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            named_parameters=lambda: [(local_name, torch.nn.Parameter(torch.zeros(1)))],
        )

        class _FakeMapping:
            hf_param = "hf.qkv.weight"
            megatron_param = global_name

            def resolve(self, _captures):
                return _FakeMapping()

            def hf_to_megatron(self, hf_weights, _module):
                return hf_weights

            def megatron_to_hf(self, megatron_weights, _module):
                return {"model.layers.0.self_attn.q_proj.weight": megatron_weights}

        class _FakeRegistry:
            def megatron_to_hf_lookup(self, _name):
                return _FakeMapping()

        fake_local_weights = SimpleNamespace(
            _rowwise_data=torch.zeros((2, 256), dtype=torch.uint8),
            _rowwise_scale_inv=torch.ones((2, 8), dtype=torch.float32),
            _fp8_dtype=None,
            _quantizer=SimpleNamespace(block_len=128),
            _is_2D_scaled=True,
            shape=(2, 256),
        )

        monkeypatch.setattr(bridge, "mapping_registry", lambda: _FakeRegistry())
        monkeypatch.setattr(bridge, "_share_embeddings_and_output_weights", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(
            bridge, "_megatron_global_param_names_all_pp_ranks", lambda *_args, **_kwargs: [global_name]
        )
        monkeypatch.setattr(bridge, "_detect_fp8_params", lambda *_args, **_kwargs: {global_name: True})
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda models: models if isinstance(models, list) else [models],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
            lambda: SimpleNamespace(size=lambda: 1),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.persistent_buffers",
            lambda *_args, **_kwargs: [],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge._megatron_local_name_to_global",
            lambda *_args, **_kwargs: _args[2],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_args, **_kwargs: (SimpleNamespace(config=model.config), fake_local_weights),
        )

        hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=SimpleNamespace()))
        tasks = bridge.build_export_fp8_tasks(hf_pretrained, [model])

        assert len(tasks) == 2
        assert tasks[0].global_param_name == global_name
        assert tasks[1].global_param_name == f"{global_name}_scale_inv"
        assert tasks[1].param_weight is not None
        expected_k_tiles = math.ceil(fake_local_weights.shape[-1] / fake_local_weights._quantizer.block_len)
        assert tasks[1].param_weight.shape[1] == expected_k_tiles

    @pytest.mark.parametrize(
        "hidden_size,last_dim,expected_shapes,expected_error",
        [
            # divisor=4: 16 -> 4
            (16, 4, ((4, 4), (2, 4), (2, 4)), None),
            # divisor=128: 4096 -> 32
            (4096, 32, ((32, 32), (16, 32), (16, 32)), None),
            # invalid divisor inference: hidden_size % last_dim != 0
            (10, 4, None, "Cannot infer block divisor"),
            # invalid scaled head_size: head_size % divisor != 0
            (12, 3, None, "Cannot scale head_size"),
        ],
    )
    def test_split_qkv_weights_compressed_scale_cases(self, hidden_size, last_dim, expected_shapes, expected_error):
        """split_qkv_weights handles compressed scales and error branches."""
        qkv_total_dim = 8  # head_num + 2 * num_query_groups = 4 + 4
        provider = SimpleNamespace(
            num_attention_heads=4,
            num_query_groups=2,
            hidden_size=hidden_size,
            kv_channels=None,
            attention_output_gate=False,
        )
        if expected_error is None:
            head_size = hidden_size // provider.num_attention_heads
            divisor = hidden_size // last_dim
            scaled_head_size = head_size // divisor
            qkv = torch.randn(qkv_total_dim * scaled_head_size, last_dim)
        else:
            # Error branches are triggered before reshape, so minimal qkv rows are enough.
            qkv = torch.randn(qkv_total_dim, last_dim)

        if expected_error is not None:
            with pytest.raises(ValueError, match=expected_error):
                _ = split_qkv_weights(provider, qkv)
            return

        q, k, v = split_qkv_weights(provider, qkv)
        assert q.shape == expected_shapes[0]
        assert k.shape == expected_shapes[1]
        assert v.shape == expected_shapes[2]
