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
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.conversion.peft_bridge import (
    MegatronPeftBridge,
    build_adapter_config_dict,
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


# ---------------------------------------------------------------------------
# _merge_single_adapter_weight — float32 precision
# ---------------------------------------------------------------------------


class TestMergeSingleAdapterWeight:
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


# ---------------------------------------------------------------------------
# AutoBridge.save_hf_adapter
# ---------------------------------------------------------------------------


class TestSaveHfAdapter:
    def test_save_creates_files(self, tmp_path):
        """save_hf_adapter should produce adapter_config.json and adapter_model.safetensors."""
        from megatron.bridge.peft.lora import LoRA

        lora = LoRA(dim=8, alpha=16)
        output_dir = tmp_path / "adapter_out"

        fake_weights = [
            ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(8, 64)),
            ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(64, 8)),
            ("model.layers.0.self_attn.v_proj.lora_A.weight", torch.randn(8, 64)),
            ("model.layers.0.self_attn.v_proj.lora_B.weight", torch.randn(64, 8)),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = SimpleNamespace(model_name_or_path="test/model")

        with (
            patch("torch.distributed.is_available", return_value=False),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
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
        assert set(cfg["target_modules"]) == {"q_proj", "v_proj"}
        assert cfg["base_model_name_or_path"] == "test/model"

    def test_save_raises_on_empty_adapter(self, tmp_path):
        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter([])

        with (
            patch("torch.distributed.is_available", return_value=False),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
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
            ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(4, 32)),
            ("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(32, 4)),
        ]

        mock_bridge = MagicMock()
        mock_bridge.export_adapter_weights.return_value = iter(fake_weights)
        mock_bridge.hf_pretrained = SimpleNamespace(model_name_or_path="inferred/model-id")

        with (
            patch("torch.distributed.is_available", return_value=False),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
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
