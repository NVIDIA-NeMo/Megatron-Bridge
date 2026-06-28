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

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt.model_config import ACTIVATION_FUNC_METADATA_KEY, BridgeGPTModelConfig
from megatron.bridge.training.model_load_save import load_model_config


pytestmark = pytest.mark.unit


class _TestBridge(MegatronModelBridge):
    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry()


class _HookBridge(_TestBridge):
    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            {
                "normalization": "RMSNorm",
                "gated_linear_unit": True,
            }
        )
        return config_kwargs


class _UnknownFieldBridge(_TestBridge):
    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs["phantom_field"] = 1
        return config_kwargs


class _LegacyOnlyBridge(_TestBridge):
    def provider_bridge(self, hf_pretrained: Any) -> object:
        return object()


@dataclass
class _DerivedFieldTransformerConfig(TransformerConfig):
    derived_field: int = field(init=False, default=1)


def _hf_pretrained() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=16,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=32,
            rope_theta=10000,
            torch_dtype=torch.float32,
            hidden_act="silu",
            rope_scaling=None,
        )
    )


def test_partition_model_config_kwargs_routes_to_owning_dataclass() -> None:
    model_kwargs, transformer_kwargs = _TestBridge._partition_model_config_kwargs(
        {
            "seq_length": 32,
            "share_embeddings_and_output_weights": True,
            "num_layers": 2,
            "hidden_size": 16,
        },
        GPTModelConfig,
        TransformerConfig,
    )

    assert model_kwargs == {
        "seq_length": 32,
        "share_embeddings_and_output_weights": True,
    }
    assert transformer_kwargs == {
        "num_layers": 2,
        "hidden_size": 16,
    }


def test_model_config_bridge_uses_direct_mcore_transformer_config() -> None:
    model_config = _TestBridge().model_config_bridge(_hf_pretrained())

    assert isinstance(model_config, GPTModelConfig)
    assert type(model_config.transformer) is TransformerConfig
    assert model_config.seq_length == 32
    assert model_config.vocab_size == 128
    assert model_config.transformer.num_layers == 2
    assert model_config.transformer.hidden_size == 16
    assert model_config.transformer.ffn_hidden_size == 64


def test_model_config_bridge_applies_family_kwargs_before_construction() -> None:
    model_config = _HookBridge().model_config_bridge(_hf_pretrained())

    assert model_config.transformer.normalization == "RMSNorm"
    assert model_config.transformer.gated_linear_unit is True


def test_model_config_bridge_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="phantom_field"):
        _UnknownFieldBridge().model_config_bridge(_hf_pretrained())


def test_partition_model_config_kwargs_rejects_non_init_fields() -> None:
    with pytest.raises(ValueError, match="derived_field"):
        _TestBridge._partition_model_config_kwargs(
            {"derived_field": 2},
            GPTModelConfig,
            _DerivedFieldTransformerConfig,
        )


def test_model_config_bridge_rejects_legacy_only_bridge() -> None:
    with pytest.raises(NotImplementedError, match="without a builder-backed config path"):
        _LegacyOnlyBridge().model_config_bridge(_hf_pretrained())


def test_model_config_lookup_prefers_builder_config_and_supports_legacy_models(monkeypatch) -> None:
    bridge = _TestBridge()
    builder_config = SimpleNamespace(share_embeddings_and_output_weights=True)
    builder_model = SimpleNamespace(_bridge_model_config=builder_config, config=SimpleNamespace())
    wrapped_model = SimpleNamespace(_bridge_model_config=builder_config)
    legacy_model = SimpleNamespace(
        config=SimpleNamespace(share_embeddings_and_output_weights=False),
    )

    assert bridge._get_model_config_from_model(builder_model) is builder_config
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda model: legacy_model if model is wrapped_model else model,
    )
    assert bridge._get_model_config_from_model(wrapped_model) is builder_config
    assert bridge._get_model_config_from_model(legacy_model) is legacy_model.config
    assert bridge._share_embeddings_and_output_weights(builder_config) is True
    assert bridge._share_embeddings_and_output_weights(legacy_model.config) is False


def test_silu_activation_round_trips_through_model_config_dict() -> None:
    model_config = _HookBridge().model_config_bridge(_hf_pretrained())

    serialized = model_config.as_dict()
    restored = BridgeGPTModelConfig.from_dict(serialized)

    assert "activation_func" not in serialized["transformer"]
    assert serialized["extra_checkpoint_metadata"][ACTIVATION_FUNC_METADATA_KEY] == "silu"
    assert type(restored.transformer) is TransformerConfig
    assert restored.transformer.activation_func is F.silu


def test_load_model_config_restores_silu_before_nested_config_construction() -> None:
    serialized = _HookBridge().model_config_bridge(_hf_pretrained()).as_dict()

    with (
        patch(
            "megatron.bridge.training.checkpointing.get_checkpoint_run_config_filename",
            return_value="/checkpoint/run_config.yaml",
        ),
        patch("megatron.bridge.training.checkpointing.read_run_config", return_value={"model": serialized}),
        patch("megatron.bridge.training.model_load_save.file_exists", return_value=True),
    ):
        restored, megatron_args = load_model_config("/checkpoint")

    assert megatron_args is None
    assert isinstance(restored, BridgeGPTModelConfig)
    assert type(restored.transformer) is TransformerConfig
    assert restored.transformer.activation_func is F.silu


def test_model_config_modules_do_not_import_or_inherit_providers() -> None:
    models_root = Path(__file__).parents[3] / "src" / "megatron" / "bridge" / "models"
    violations: list[str] = []

    for path in models_root.rglob("model_config.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            else:
                imported_modules = []
            if any("provider" in module.lower() for module in imported_modules):
                violations.append(f"{path}: provider import")

            if isinstance(node, ast.ClassDef):
                base_names = [
                    base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
                    for base in node.bases
                ]
                if any("Provider" in base_name or base_name == "ModelProviderMixin" for base_name in base_names):
                    violations.append(f"{path}:{node.lineno}: provider inheritance")

    assert not violations, "\n".join(violations)
