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

"""Unit tests for adapter export helpers in peft_bridge and auto_bridge."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.models.conversion.peft_bridge import (
    MegatronPeftBridge,
    build_adapter_config_dict,
    convert_adapter_weights_to_peft_state,
    infer_rank_pattern_from_adapter_weights,
    infer_target_modules_from_adapter_weights,
)


# ---------------------------------------------------------------------------
# infer_target_modules_from_adapter_weights
# ---------------------------------------------------------------------------


class TestInferTargetModules:
    def test_basic_lora_names(self):
        names = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
            "base_model.model.model.layers.1.mlp.gate_proj.lora_A.weight",
            "base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight",
        ]
        result = infer_target_modules_from_adapter_weights(names)
        assert result == ["gate_proj", "q_proj", "v_proj"]

    def test_empty_input(self):
        assert infer_target_modules_from_adapter_weights([]) == []

    def test_no_lora_suffixes(self):
        names = ["model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"]
        assert infer_target_modules_from_adapter_weights(names) == []

    def test_deduplication(self):
        names = [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "model.layers.1.self_attn.q_proj.lora_A.weight",
            "model.layers.1.self_attn.q_proj.lora_B.weight",
        ]
        result = infer_target_modules_from_adapter_weights(names)
        assert result == ["q_proj"]

    def test_sorted_output(self):
        names = [
            "model.layers.0.mlp.down_proj.lora_A.weight",
            "model.layers.0.self_attn.k_proj.lora_A.weight",
            "model.layers.0.mlp.gate_proj.lora_B.weight",
        ]
        result = infer_target_modules_from_adapter_weights(names)
        assert result == sorted(result)

    def test_mixed_lora_and_non_lora(self):
        names = [
            "model.layers.0.self_attn.o_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
        ]
        result = infer_target_modules_from_adapter_weights(names)
        assert result == ["o_proj"]


# ---------------------------------------------------------------------------
# build_adapter_config_dict
# ---------------------------------------------------------------------------


@dataclass
class FakeLoRA:
    dim: int = 16
    alpha: int = 32
    dropout: float = 0.1


class _ToyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_transposed = True
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 6, 4))
        self.down_proj = torch.nn.Parameter(torch.zeros(2, 4, 3))

    def forward(self, x):
        return x


class _ToySelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(64, 64, bias=False)
        self.v_proj = torch.nn.Linear(64, 64, bias=False)


class _ToyMLP(torch.nn.Module):
    def __init__(self, with_packed_experts: bool = False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(32, 32, bias=False)
        if with_packed_experts:
            self.experts = _ToyExperts()


class _ToyLayer(torch.nn.Module):
    def __init__(self, with_packed_experts: bool = False):
        super().__init__()
        self.self_attn = _ToySelfAttention()
        self.mlp = _ToyMLP(with_packed_experts=with_packed_experts)


class _ToyAdapterModel(torch.nn.Module):
    _keys_to_ignore_on_load_unexpected = (r"^mtp.*",)

    def __init__(self, *, model_name_or_path: str, with_packed_experts: bool = False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyLayer(with_packed_experts=with_packed_experts)])
        self.config = {"tie_word_embeddings": False}
        self.model_name_or_path = model_name_or_path

    def forward(self, x=None, **kwargs):
        return x

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


def _adapter_export(name: str, tensor: torch.Tensor) -> HFWeightTuple:
    """Build a raw adapter export record."""

    return HFWeightTuple(param_name=name, weight=tensor)


class TestBuildAdapterConfigDict:
    def test_basic_config(self):
        peft_config = FakeLoRA(dim=16, alpha=32, dropout=0.1)
        target_modules = ["q_proj", "v_proj"]
        config = build_adapter_config_dict(peft_config, target_modules, base_model_name_or_path="foo/bar")

        assert config["peft_type"] == "LORA"
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.1
        assert config["target_modules"] == ["q_proj", "v_proj"]
        assert config["base_model_name_or_path"] == "foo/bar"
        assert config["task_type"] == "CAUSAL_LM"
        assert config["use_dora"] is False
        assert config["inference_mode"] is True
        assert config["bias"] == "none"
        assert "target_parameters" not in config

    def test_target_parameters_config(self):
        peft_config = FakeLoRA(dim=8, alpha=16, dropout=0.0)
        config = build_adapter_config_dict(
            peft_config,
            target_modules=[],
            target_parameters=["model.layers.0.mlp.experts.down_proj"],
            base_model_name_or_path="foo/bar",
        )

        assert config["target_modules"] == []
        assert config["target_parameters"] == ["model.layers.0.mlp.experts.down_proj"]

    def test_target_parameters_zero_out_dropout(self):
        peft_config = FakeLoRA(dim=8, alpha=16, dropout=0.1)
        config = build_adapter_config_dict(
            peft_config,
            target_modules=[],
            target_parameters=["model.layers.0.self_attn.q_proj.weight"],
        )

        assert config["lora_dropout"] == 0.0

    def test_default_base_model_path(self):
        config = build_adapter_config_dict(FakeLoRA(), ["q_proj"])
        assert config["base_model_name_or_path"] == ""

    def test_none_base_model_path(self):
        config = build_adapter_config_dict(FakeLoRA(), ["q_proj"], base_model_name_or_path=None)
        assert config["base_model_name_or_path"] == ""

    def test_use_dora_detection(self):
        from megatron.bridge.peft.dora import DoRA

        dora = DoRA(dim=8, alpha=16)
        config = build_adapter_config_dict(dora, ["q_proj"])
        assert config["use_dora"] is True

    def test_lora_is_not_dora(self):
        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=8, alpha=16)
        config = build_adapter_config_dict(lora, ["q_proj"])
        assert config["use_dora"] is False

    def test_config_json_serializable(self):
        config = build_adapter_config_dict(FakeLoRA(), ["q_proj", "k_proj"])
        serialized = json.dumps(config)
        roundtripped = json.loads(serialized)
        assert roundtripped == config

    def test_empty_target_modules(self):
        config = build_adapter_config_dict(FakeLoRA(), [])
        assert config["target_modules"] == []
        assert "target_parameters" not in config


# ---------------------------------------------------------------------------
# _merge_single_adapter_weight — float32 precision
# ---------------------------------------------------------------------------


class TestMergeSingleAdapterWeight:
    """Tests for _merge_single_adapter_weight; patch parallel_state so merge runs as tp_size=1."""

    @pytest.fixture(autouse=True)
    def _patch_tensor_parallel(self):
        """Avoid 'tensor model parallel group is not initialized' in unit test (no dist init)."""
        with patch(
            "megatron.bridge.peft.lora.parallel_state.get_tensor_model_parallel_world_size",
            return_value=1,
        ):
            yield

    def test_merge_in_float32(self):
        """Verify that LoRA merge happens in float32 and result cast back."""
        bridge = MegatronPeftBridge()
        base = torch.zeros(4, 4, dtype=torch.bfloat16)
        lin_in = torch.eye(4, dtype=torch.bfloat16)
        lin_out = torch.eye(4, dtype=torch.bfloat16)

        merged = bridge._merge_single_adapter_weight(
            base, alpha=4, dim=4, linear_in_weight=lin_in, linear_out_weight=lin_out
        )

        assert merged.dtype == torch.bfloat16
        expected = (torch.eye(4, dtype=torch.float32)).to(torch.bfloat16)
        torch.testing.assert_close(merged, expected)

    def test_merge_preserves_float32_dtype(self):
        bridge = MegatronPeftBridge()
        base = torch.ones(2, 2, dtype=torch.float32)
        lin_in = torch.eye(2, dtype=torch.float32)
        lin_out = torch.eye(2, dtype=torch.float32) * 2

        merged = bridge._merge_single_adapter_weight(
            base, alpha=2, dim=2, linear_in_weight=lin_in, linear_out_weight=lin_out
        )

        assert merged.dtype == torch.float32
        scale = 2 / 2  # alpha / dim
        expected = base + scale * (lin_out @ lin_in)
        torch.testing.assert_close(merged, expected)

    def test_merge_device_handling(self):
        """Adapter weights on different device should be moved to base device."""
        bridge = MegatronPeftBridge()
        base = torch.zeros(2, 2)
        lin_in = torch.eye(2)
        lin_out = torch.eye(2)

        merged = bridge._merge_single_adapter_weight(
            base, alpha=1, dim=1, linear_in_weight=lin_in, linear_out_weight=lin_out
        )
        assert merged.device == base.device


class TestFusedFc1NameHelpers:
    def test_gate_and_up_detection_supports_standard_and_minimax_names(self):
        bridge = MegatronPeftBridge()

        assert bridge._is_fused_fc1_gate_proj("model.layers.0.mlp.gate_proj.weight")
        assert bridge._is_fused_fc1_gate_proj("model.layers.0.block_sparse_moe.experts.0.w1.weight")
        assert not bridge._is_fused_fc1_gate_proj("model.layers.0.mlp.up_proj.weight")

        assert bridge._is_fused_fc1_up_proj("model.layers.0.mlp.up_proj.weight")
        assert bridge._is_fused_fc1_up_proj("model.layers.0.block_sparse_moe.experts.0.w3.weight")
        assert not bridge._is_fused_fc1_up_proj("model.layers.0.mlp.gate_proj.weight")


# ---------------------------------------------------------------------------
# AutoBridge.save_hf_adapter
# ---------------------------------------------------------------------------


class TestSaveHfAdapter:
    def test_convert_packed_expert_adapter_to_target_parameters(self):
        down_a = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
                [[2.0, 1.0, 0.0], [3.0, 1.0, 1.0]],
            ]
        )
        down_b = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            ]
        )
        gate_up_a = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5]],
            ]
        )
        gate_up_b = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ]
        )

        adapter_state, module_keys, target_parameters = convert_adapter_weights_to_peft_state(
            [
                _adapter_export("model.layers.0.mlp.experts.down_proj.lora_A.weight", down_a),
                _adapter_export("model.layers.0.mlp.experts.down_proj.lora_B.weight", down_b),
                _adapter_export("model.layers.0.mlp.experts.gate_up_proj.lora_A.weight", gate_up_a),
                _adapter_export("model.layers.0.mlp.experts.gate_up_proj.lora_B.weight", gate_up_b),
            ]
        )

        assert module_keys == []
        assert target_parameters == [
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]

        expected_gate_up_a = torch.cat([chunk.transpose(0, 1) for chunk in gate_up_b], dim=0)
        expected_gate_up_b = torch.cat([chunk.transpose(0, 1) for chunk in gate_up_a], dim=1)
        expected_down_a = torch.cat([chunk.transpose(0, 1) for chunk in down_b], dim=0)
        expected_down_b = torch.cat([chunk.transpose(0, 1) for chunk in down_a], dim=1)

        torch.testing.assert_close(
            adapter_state["base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight"],
            expected_gate_up_a,
        )
        torch.testing.assert_close(
            adapter_state["base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"],
            expected_gate_up_b,
        )
        torch.testing.assert_close(
            adapter_state["base_model.model.model.layers.0.mlp.experts.lora_A.weight"],
            expected_down_a,
        )
        torch.testing.assert_close(
            adapter_state["base_model.model.model.layers.0.mlp.experts.lora_B.weight"],
            expected_down_b,
        )

    def test_convert_linear_module_adapter_stays_module_target(self):
        adapter_state, module_keys, target_parameters = convert_adapter_weights_to_peft_state(
            [
                _adapter_export("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(2, 4)),
                _adapter_export("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(4, 2)),
            ]
        )

        assert module_keys == [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        assert target_parameters == []
        assert set(adapter_state) == {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        }

    def test_infer_rank_pattern_uses_module_names_for_linear_targets(self):
        rank_pattern = infer_rank_pattern_from_adapter_weights(
            [
                _adapter_export("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(2, 4)),
                _adapter_export("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(4, 2)),
            ],
            default_rank=4,
        )

        assert rank_pattern == {"model.layers.0.self_attn.q_proj": 2}

    def test_save_creates_files(self, tmp_path):
        """save_hf_adapter should produce adapter_config.json and adapter_model.safetensors."""
        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=8, alpha=16)
        output_dir = tmp_path / "adapter_out"

        fake_weights = [
            _adapter_export("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(8, 64)),
            _adapter_export("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(64, 8)),
            _adapter_export("model.layers.0.self_attn.v_proj.lora_A.weight", torch.randn(8, 64)),
            _adapter_export("model.layers.0.self_attn.v_proj.lora_B.weight", torch.randn(64, 8)),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = _ToyAdapterModel(model_name_or_path="test/model")

        with patch("torch.distributed.is_initialized", return_value=False):
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge

            AutoBridge.save_hf_adapter(
                mock_bridge,
                model=[MagicMock()],
                path=output_dir,
                peft_config=lora,
                base_model_name_or_path="test/model",
            )

        assert (output_dir / "adapter_config.json").exists()
        assert (output_dir / "adapter_model.safetensors").exists()

        with open(output_dir / "adapter_config.json") as f:
            cfg = json.load(f)
        assert cfg["r"] == 8
        assert cfg["lora_alpha"] == 16
        assert cfg["target_modules"] == ["q_proj", "v_proj"]
        assert "target_parameters" not in cfg
        assert cfg["base_model_name_or_path"] == "test/model"

    def test_save_with_nonzero_dropout_keeps_linear_target_modules(self, tmp_path):
        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=8, alpha=16, dropout=0.1)
        output_dir = tmp_path / "adapter_out_dropout"

        fake_weights = [
            _adapter_export("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(8, 64)),
            _adapter_export("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(64, 8)),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = _ToyAdapterModel(model_name_or_path="test/model")

        with patch("torch.distributed.is_initialized", return_value=False):
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge

            AutoBridge.save_hf_adapter(
                mock_bridge,
                model=[MagicMock()],
                path=output_dir,
                peft_config=lora,
                base_model_name_or_path="test/model",
            )

        with open(output_dir / "adapter_config.json") as f:
            cfg = json.load(f)
        assert cfg["lora_dropout"] == 0.1
        assert cfg["target_modules"] == ["q_proj"]
        assert "target_parameters" not in cfg

    def test_save_raises_on_empty_adapter(self, tmp_path):
        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter([])

        with patch("torch.distributed.is_initialized", return_value=False):
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge
            from megatron.bridge.peft.lora import LoRA

            with pytest.raises(RuntimeError, match="No adapter weights"):
                AutoBridge.save_hf_adapter(
                    mock_bridge,
                    model=[MagicMock()],
                    path=tmp_path / "empty",
                    peft_config=LoRA(),
                )

    def test_save_infers_base_model_path(self, tmp_path):
        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=4, alpha=8)
        output_dir = tmp_path / "adapter_infer"

        fake_weights = [
            _adapter_export("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(4, 32)),
            _adapter_export("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(32, 4)),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = _ToyAdapterModel(model_name_or_path="inferred/model-id")

        with patch("torch.distributed.is_initialized", return_value=False):
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge

            AutoBridge.save_hf_adapter(
                mock_bridge,
                model=[MagicMock()],
                path=output_dir,
                peft_config=lora,
                base_model_name_or_path=None,
            )

        with open(output_dir / "adapter_config.json") as f:
            cfg = json.load(f)
        assert cfg["base_model_name_or_path"] == "inferred/model-id"
        assert cfg["target_modules"] == ["gate_proj"]
        assert "target_parameters" not in cfg

    def test_save_normalized_expert_adapter_writes_rank_pattern(self, tmp_path):
        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=8, alpha=16, normalize_moe_lora=True)
        output_dir = tmp_path / "normalized_expert_adapter"

        fake_weights = [
            _adapter_export(
                "model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
                torch.tensor(
                    [
                        [[1.0, 2.0, 3.0]],
                        [[4.0, 5.0, 6.0]],
                    ]
                ),
            ),
            _adapter_export(
                "model.layers.0.mlp.experts.gate_up_proj.lora_B.weight",
                torch.tensor(
                    [
                        [[1.0], [2.0], [3.0], [4.0]],
                        [[5.0], [6.0], [7.0], [8.0]],
                    ]
                ),
            ),
            _adapter_export(
                "model.layers.0.mlp.experts.down_proj.lora_A.weight",
                torch.tensor(
                    [
                        [[1.0, 0.0, 2.0]],
                        [[0.0, 1.0, 3.0]],
                    ]
                ),
            ),
            _adapter_export(
                "model.layers.0.mlp.experts.down_proj.lora_B.weight",
                torch.tensor(
                    [
                        [[1.0], [2.0], [3.0], [4.0]],
                        [[5.0], [6.0], [7.0], [8.0]],
                    ]
                ),
            ),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = _ToyAdapterModel(
            model_name_or_path="test/normalized-expert-model",
            with_packed_experts=True,
        )

        with patch("torch.distributed.is_initialized", return_value=False):
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge

            AutoBridge.save_hf_adapter(
                mock_bridge,
                model=[MagicMock()],
                path=output_dir,
                peft_config=lora,
                base_model_name_or_path="test/normalized-expert-model",
            )

        with open(output_dir / "adapter_config.json") as f:
            cfg = json.load(f)

        assert cfg["r"] == 8
        assert cfg["rank_pattern"] == {
            "model.layers.0.mlp.experts.down_proj": 1,
            "model.layers.0.mlp.experts.gate_up_proj": 1,
        }

    def test_save_packed_expert_adapter_uses_target_parameters(self, tmp_path):
        peft = pytest.importorskip("peft", reason="peft library not installed")
        from safetensors.torch import load_file

        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=2, alpha=4)
        output_dir = tmp_path / "packed_expert_adapter"

        down_a = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
                [[2.0, 1.0, 0.0], [3.0, 1.0, 1.0]],
            ]
        )
        down_b = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            ]
        )
        gate_up_a = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5]],
            ]
        )
        gate_up_b = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ]
        )

        fake_weights = [
            _adapter_export("model.layers.0.mlp.experts.down_proj.lora_A.weight", down_a),
            _adapter_export("model.layers.0.mlp.experts.down_proj.lora_B.weight", down_b),
            _adapter_export("model.layers.0.mlp.experts.gate_up_proj.lora_A.weight", gate_up_a),
            _adapter_export("model.layers.0.mlp.experts.gate_up_proj.lora_B.weight", gate_up_b),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = _ToyAdapterModel(
            model_name_or_path="test/packed-experts",
            with_packed_experts=True,
        )

        with patch("torch.distributed.is_initialized", return_value=False):
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge

            AutoBridge.save_hf_adapter(
                mock_bridge,
                model=[MagicMock()],
                path=output_dir,
                peft_config=lora,
                base_model_name_or_path="test/packed-experts",
            )

        with open(output_dir / "adapter_config.json") as f:
            cfg = json.load(f)
        assert cfg["target_modules"] == []
        assert cfg["target_parameters"] == [
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]

        state = load_file(str(output_dir / "adapter_model.safetensors"))
        assert set(state) == {
            "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
            "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
            "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
            "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight",
        }

        expected_gate_up_a = torch.cat([chunk.transpose(0, 1) for chunk in gate_up_b], dim=0)
        expected_gate_up_b = torch.cat([chunk.transpose(0, 1) for chunk in gate_up_a], dim=1)
        expected_down_a = torch.cat([chunk.transpose(0, 1) for chunk in down_b], dim=0)
        expected_down_b = torch.cat([chunk.transpose(0, 1) for chunk in down_a], dim=1)

        torch.testing.assert_close(
            state["base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight"],
            expected_gate_up_a,
        )
        torch.testing.assert_close(
            state["base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"],
            expected_gate_up_b,
        )
        torch.testing.assert_close(state["base_model.model.model.layers.0.mlp.experts.lora_A.weight"], expected_down_a)
        torch.testing.assert_close(state["base_model.model.model.layers.0.mlp.experts.lora_B.weight"], expected_down_b)

        loaded = peft.PeftModel.from_pretrained(
            _ToyAdapterModel(model_name_or_path="test/packed-experts", with_packed_experts=True),
            output_dir,
        )
        assert loaded is not None


# ---------------------------------------------------------------------------
# AutoBridge.export_adapter_ckpt  (orchestrator method)
# ---------------------------------------------------------------------------


class TestExportAdapterCkpt:
    """Tests for the export_adapter_ckpt orchestrator.

    Heavy infrastructure (dist_checkpointing, distributed context, model
    materialisation) is mocked out so that these tests exercise the
    config-parsing logic and the wiring between components.
    """

    @pytest.fixture()
    def bridge(self):
        """Return a minimal AutoBridge-like mock with the real export_adapter_ckpt bound."""
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge

        mock = MagicMock(spec=AutoBridge)
        mock.export_adapter_ckpt = AutoBridge.export_adapter_ckpt.__get__(mock, AutoBridge)
        mock.hf_pretrained = SimpleNamespace(model_name_or_path="test-org/test-model")

        fake_provider = MagicMock()
        fake_provider.provide_distributed_model.return_value = [MagicMock()]
        mock.to_megatron_provider.return_value = fake_provider
        return mock

    @pytest.fixture()
    def ckpt_dir(self, tmp_path):
        """Create a fake PEFT checkpoint directory with a run_config.yaml."""
        ckpt = tmp_path / "peft_ckpt"
        ckpt.mkdir()
        run_cfg = {
            "peft": {
                "_target_": "megatron.bridge.peft.lora.LoRA",
                "dim": 16,
                "alpha": 32,
                "dropout": 0.05,
                "normalize_moe_lora": True,
                "share_expert_adapters": False,
                "extra_ignored_key": "should_be_filtered",
            }
        }
        (ckpt / "run_config.yaml").write_text(yaml.dump(run_cfg))
        return ckpt

    @pytest.fixture(autouse=True)
    def _patch_heavy_deps(self):
        """Mock out dist_checkpointing, distributed context, and checkpoint helpers."""
        fake_sd = {"model": {}}
        with (
            patch(
                "megatron.core.dist_checkpointing.load",
                return_value=fake_sd,
            ) as self.mock_dist_load,
            patch(
                "megatron.bridge.training.checkpointing._generate_model_state_dict",
                return_value={"model": {}},
            ),
            patch(
                "megatron.bridge.training.checkpointing.apply_peft_adapter_filter_to_state_dict",
                side_effect=lambda sd, _lora: sd,
            ),
            patch(
                "megatron.bridge.training.model_load_save.temporary_distributed_context",
                return_value=nullcontext(),
            ),
        ):
            yield

    def test_basic_export_calls_save_hf_adapter(self, bridge, ckpt_dir, tmp_path):
        """Happy path: run_config.yaml is found, LoRA constructed, save_hf_adapter called."""
        output = tmp_path / "out"
        bridge.export_adapter_ckpt(str(ckpt_dir), output, show_progress=False)

        bridge.save_hf_adapter.assert_called_once()
        kw = bridge.save_hf_adapter.call_args
        assert Path(kw.kwargs.get("path", kw.args[1] if len(kw.args) > 1 else None)) == output
        assert kw.kwargs.get("base_model_name_or_path") == "test-org/test-model"
        assert kw.kwargs.get("show_progress") is False

    def test_lora_config_parsed_from_run_config(self, bridge, ckpt_dir, tmp_path):
        """LoRA dim/alpha/dropout read from run_config.yaml; extra keys filtered out."""
        bridge.export_adapter_ckpt(str(ckpt_dir), tmp_path / "out")

        peft_config = bridge.save_hf_adapter.call_args.kwargs.get(
            "peft_config",
            bridge.save_hf_adapter.call_args.args[2] if len(bridge.save_hf_adapter.call_args.args) > 2 else None,
        )
        assert peft_config.dim == 16
        assert peft_config.alpha == 32
        assert peft_config.dropout == 0.05
        assert peft_config.normalize_moe_lora is True
        assert peft_config.share_expert_adapters is False
        assert not hasattr(peft_config, "extra_ignored_key")

    def test_vlmlora_selected_when_target_matches(self, bridge, tmp_path):
        """VLMLoRA is instantiated when _target_ contains 'VLMLoRA'."""
        from megatron.bridge.peft.lora import VLMLoRA

        ckpt = tmp_path / "vlm_ckpt"
        ckpt.mkdir()
        run_cfg = {
            "peft": {
                "_target_": "megatron.bridge.peft.lora.VLMLoRA",
                "dim": 8,
                "alpha": 16,
            }
        }
        (ckpt / "run_config.yaml").write_text(yaml.dump(run_cfg))

        bridge.export_adapter_ckpt(str(ckpt), tmp_path / "out")

        peft_config = bridge.save_hf_adapter.call_args.kwargs.get(
            "peft_config",
            bridge.save_hf_adapter.call_args.args[2] if len(bridge.save_hf_adapter.call_args.args) > 2 else None,
        )
        assert isinstance(peft_config, VLMLoRA)

    def test_missing_checkpoint_raises(self, bridge, tmp_path):
        with pytest.raises(FileNotFoundError, match="PEFT checkpoint not found"):
            bridge.export_adapter_ckpt(str(tmp_path / "nonexistent"), tmp_path / "out")

    def test_missing_run_config_uses_defaults(self, bridge, tmp_path):
        """When run_config.yaml is absent, default LoRA settings are used."""
        from megatron.bridge.peft.lora import LoRA

        ckpt = tmp_path / "bare_ckpt"
        ckpt.mkdir()

        bridge.export_adapter_ckpt(str(ckpt), tmp_path / "out")

        peft_config = bridge.save_hf_adapter.call_args.kwargs.get(
            "peft_config",
            bridge.save_hf_adapter.call_args.args[2] if len(bridge.save_hf_adapter.call_args.args) > 2 else None,
        )
        default_lora = LoRA()
        assert isinstance(peft_config, LoRA)
        assert peft_config.dim == default_lora.dim
        assert peft_config.alpha == default_lora.alpha

    def test_run_config_in_parent_dir(self, bridge, tmp_path):
        """run_config.yaml in the parent directory is found (iter_* sub-dir pattern)."""
        parent = tmp_path / "run_dir"
        parent.mkdir()
        iter_dir = parent / "iter_0001000"
        iter_dir.mkdir()
        run_cfg = {"peft": {"_target_": "LoRA", "dim": 4, "alpha": 8}}
        (parent / "run_config.yaml").write_text(yaml.dump(run_cfg))

        bridge.export_adapter_ckpt(str(iter_dir), tmp_path / "out")

        peft_config = bridge.save_hf_adapter.call_args.kwargs.get(
            "peft_config",
            bridge.save_hf_adapter.call_args.args[2] if len(bridge.save_hf_adapter.call_args.args) > 2 else None,
        )
        assert peft_config.dim == 4
        assert peft_config.alpha == 8

    def test_corrupt_run_config_falls_back_to_defaults(self, bridge, tmp_path):
        """A run_config.yaml that fails to parse falls back to default LoRA."""
        from megatron.bridge.peft.lora import LoRA

        ckpt = tmp_path / "corrupt_ckpt"
        ckpt.mkdir()
        (ckpt / "run_config.yaml").write_text("not: valid: yaml: [[[")

        with patch(
            "megatron.bridge.training.utils.checkpoint_utils.read_run_config",
            side_effect=Exception("parse error"),
        ):
            bridge.export_adapter_ckpt(str(ckpt), tmp_path / "out")

        peft_config = bridge.save_hf_adapter.call_args.kwargs.get(
            "peft_config",
            bridge.save_hf_adapter.call_args.args[2] if len(bridge.save_hf_adapter.call_args.args) > 2 else None,
        )
        assert isinstance(peft_config, LoRA)
        assert peft_config.dim == LoRA().dim

    def test_provider_set_to_float32(self, bridge, ckpt_dir, tmp_path):
        """Provider dtypes must be forced to float32 for full-precision adapter export."""
        bridge.export_adapter_ckpt(str(ckpt_dir), tmp_path / "out")

        provider = bridge.to_megatron_provider.return_value
        assert provider.pipeline_dtype == torch.float32
        assert provider.params_dtype == torch.float32
        provider.finalize.assert_called_once()

    def test_dist_checkpointing_called_with_ckpt_path(self, bridge, ckpt_dir, tmp_path):
        bridge.export_adapter_ckpt(str(ckpt_dir), tmp_path / "out")

        self.mock_dist_load.assert_called_once()
        load_path = self.mock_dist_load.call_args.args[1]
        assert load_path == str(ckpt_dir.resolve())

    def test_base_model_name_from_name_or_path_fallback(self, bridge, ckpt_dir, tmp_path):
        """Falls back to hf_pretrained.name_or_path when model_name_or_path is empty."""
        bridge.hf_pretrained = SimpleNamespace(model_name_or_path="", name_or_path="fallback/model")

        bridge.export_adapter_ckpt(str(ckpt_dir), tmp_path / "out")

        base_name = bridge.save_hf_adapter.call_args.kwargs.get("base_model_name_or_path")
        assert base_name == "fallback/model"

    def test_multi_rank_export_runs_in_current_process(self, bridge, ckpt_dir, tmp_path):
        """Initialized multi-rank export should reuse the current process group."""
        output = tmp_path / "out"

        with (
            patch(
                "megatron.bridge.training.model_load_save.temporary_distributed_context",
                side_effect=AssertionError("temporary_distributed_context should not be used"),
            ),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.get_rank", return_value=3),
        ):
            bridge.export_adapter_ckpt(str(ckpt_dir), output)

        bridge.save_hf_adapter.assert_called_once()
