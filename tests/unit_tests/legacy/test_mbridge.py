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

"""Contract tests for the bounded legacy MBridge facade."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import PretrainedConfig, Qwen3MoeConfig

from megatron.bridge.legacy import AutoBridge
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge


class _ProviderDouble:
    """Provider double with an explicit, non-extensible field contract."""

    tensor_model_parallel_size = 1
    params_dtype = torch.float32
    fp16 = False
    bf16 = False

    def __init__(self, models: list[object] | None = None) -> None:
        self.models = models or []
        self.apply_calls: list[tuple[torch.dtype | None, dict[str, object]]] = []
        self.finalize_calls = 0
        self.provide_calls: list[dict[str, object]] = []
        self.events: list[str] = []

    def apply_overrides_and_finalize(
        self,
        dtype: torch.dtype | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> _ProviderDouble:
        """Record and apply the already-validated overrides."""
        values = dict(overrides or {})
        self.events.append("apply_overrides")
        self.apply_calls.append((dtype, values))
        if dtype is not None:
            self.params_dtype = dtype
            self.fp16 = dtype == torch.float16
            self.bf16 = dtype == torch.bfloat16
        for name, value in values.items():
            setattr(self, name, value)
        self.finalize()
        return self

    def finalize(self) -> None:
        """Record finalization."""
        self.events.append("finalize")
        self.finalize_calls += 1

    def provide_distributed_model(self, **kwargs: object) -> list[object]:
        """Return the exact legacy model list."""
        self.events.append("provide_distributed_model")
        self.provide_calls.append(kwargs)
        return self.models


def _facade(
    *,
    models: list[object] | None = None,
) -> tuple[AutoBridge, Mock, _ProviderDouble]:
    current_bridge = Mock()
    provider = _ProviderDouble(models)
    return AutoBridge(current_bridge, provider), current_bridge, provider


def test_public_facade_surface_is_exactly_six_methods() -> None:
    """Keep the compatibility surface bounded to the owner-approved methods."""
    public_callables = {
        name
        for name, value in AutoBridge.__dict__.items()
        if not name.startswith("_") and (callable(value) or isinstance(value, classmethod))
    }

    assert public_callables == {
        "export_weights",
        "from_config",
        "get_model",
        "load_weights",
        "save_weights",
        "set_extra_args",
    }


def test_from_config_wraps_current_auto_bridge_and_applies_validated_overrides() -> None:
    """Build the facade through current bridge/provider code."""
    config = PretrainedConfig()
    current_bridge = Mock()
    provider = _ProviderDouble()
    current_bridge.to_megatron_provider.return_value = provider

    with patch(
        "megatron.bridge.legacy.mbridge.CurrentAutoBridge.from_hf_config",
        return_value=current_bridge,
    ) as from_hf_config:
        facade = AutoBridge.from_config(
            config,
            dtype=torch.bfloat16,
            tensor_model_parallel_size=2,
        )

    assert isinstance(facade, AutoBridge)
    assert facade._bridge is current_bridge
    assert facade._provider is provider
    from_hf_config.assert_called_once_with(config)
    current_bridge.to_megatron_provider.assert_called_once_with(load_weights=False)
    assert provider.apply_calls == [
        (torch.bfloat16, {"tensor_model_parallel_size": 2}),
    ]
    assert provider.params_dtype is torch.bfloat16
    assert provider.tensor_model_parallel_size == 2


def test_from_config_migrates_qwen3_moe_through_current_provider() -> None:
    """Select the production Qwen3 MoE bridge and preserve its key config."""
    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_local_experts=128,
        norm_topk_prob=True,
        router_aux_loss_coef=1e-3,
        mlp_only_layers=[],
        attention_bias=False,
        tie_word_embeddings=False,
    )

    facade = AutoBridge.from_config(config)

    assert isinstance(facade._bridge._model_bridge, Qwen3MoEBridge)
    assert facade._provider.num_layers == 48
    assert facade._provider.num_moe_experts == 128
    assert facade._provider.moe_router_topk == 8
    assert facade._provider.add_qkv_bias is False
    assert facade._provider.qk_layernorm is True


def test_from_config_rejects_invalid_dtype_without_rebuilding_provider() -> None:
    """Reject a non-torch dtype before provider finalization."""
    config = PretrainedConfig()
    current_bridge = Mock()
    provider = _ProviderDouble()
    current_bridge.to_megatron_provider.return_value = provider

    with (
        patch(
            "megatron.bridge.legacy.mbridge.CurrentAutoBridge.from_hf_config",
            return_value=current_bridge,
        ),
        pytest.raises(TypeError, match="dtype must be a torch.dtype"),
    ):
        AutoBridge.from_config(config, dtype="bfloat16")

    assert provider.apply_calls == []
    assert provider.finalize_calls == 0


def test_set_extra_args_rejects_all_overrides_before_any_mutation() -> None:
    """Prevent foreign attributes and partial updates on provider dataclasses."""
    facade, _, provider = _facade()

    with pytest.raises(AttributeError, match="'phantom_field'"):
        facade.set_extra_args(
            tensor_model_parallel_size=4,
            phantom_field=True,
        )

    assert provider.tensor_model_parallel_size == 1
    assert not hasattr(provider, "phantom_field")
    assert provider.apply_calls == []
    assert provider.finalize_calls == 0


def test_set_extra_args_applies_existing_fields_and_rebuilds_provider() -> None:
    """Apply valid provider fields through the current rebuild path."""
    facade, _, provider = _facade()

    assert facade.set_extra_args(tensor_model_parallel_size=4) is None

    assert provider.tensor_model_parallel_size == 4
    assert provider.apply_calls == [(None, {"tensor_model_parallel_size": 4})]
    assert provider.finalize_calls == 1
    assert provider.events == ["apply_overrides", "finalize"]


def test_from_config_set_extra_args_get_model_finalization_order() -> None:
    """Finalize each override phase before final model construction."""
    config = PretrainedConfig()
    models = [object()]
    current_bridge = Mock()
    provider = _ProviderDouble(models)
    current_bridge.to_megatron_provider.return_value = provider

    with patch(
        "megatron.bridge.legacy.mbridge.CurrentAutoBridge.from_hf_config",
        return_value=current_bridge,
    ):
        facade = AutoBridge.from_config(config, tensor_model_parallel_size=2)

    facade.set_extra_args(tensor_model_parallel_size=4)
    result = facade.get_model(wrap_with_ddp=False)

    assert result is models
    assert provider.tensor_model_parallel_size == 4
    assert provider.finalize_calls == 3
    assert provider.events == [
        "apply_overrides",
        "finalize",
        "apply_overrides",
        "finalize",
        "finalize",
        "provide_distributed_model",
    ]


def test_from_config_validates_dtype_backing_fields() -> None:
    """Require dtype aliases to correspond to real provider/config fields."""

    class _MissingDtypeFieldProvider:
        fp16 = False
        bf16 = False

        def __init__(self) -> None:
            self.apply_calls: list[object] = []

        def apply_overrides_and_finalize(self, **kwargs: object) -> None:
            self.apply_calls.append(kwargs)

    config = PretrainedConfig()
    current_bridge = Mock()
    provider = _MissingDtypeFieldProvider()
    current_bridge.to_megatron_provider.return_value = provider

    with (
        patch(
            "megatron.bridge.legacy.mbridge.CurrentAutoBridge.from_hf_config",
            return_value=current_bridge,
        ),
        pytest.raises(AttributeError, match="'params_dtype'"),
    ):
        AutoBridge.from_config(config, dtype=torch.bfloat16)

    assert provider.apply_calls == []


def test_get_model_preserves_list_shape_and_optionally_loads_weights() -> None:
    """Return the provider's list unchanged and load into that same list."""
    models = [object(), object()]
    facade, _, provider = _facade(models=models)
    weights_path = Path("/model")

    with patch.object(facade, "load_weights") as load_weights:
        result = facade.get_model(
            weights_path,
            wrap_with_ddp=False,
            use_cpu_initialization=True,
        )

    assert result is models
    assert provider.finalize_calls == 1
    assert provider.provide_calls == [
        {
            "wrap_with_ddp": False,
            "use_cpu_initialization": True,
        }
    ]
    load_weights.assert_called_once_with(models, weights_path)


@pytest.mark.parametrize("method_name", ("load_weights", "export_weights", "save_weights"))
def test_weight_apis_require_legacy_list_shape(method_name: str) -> None:
    """Reject a single model instead of silently changing legacy shape."""
    facade, _, _ = _facade()
    method = getattr(facade, method_name)

    with pytest.raises(TypeError, match="list of model chunks"):
        if method_name == "export_weights":
            method(object())
        else:
            method(object(), "/weights")


def test_load_weights_delegates_to_current_bridge() -> None:
    """Delegate supported loading directly to current conversion code."""
    models = [object()]
    facade, current_bridge, _ = _facade()

    assert facade.load_weights(models, "/weights") is None

    current_bridge.load_hf_weights.assert_called_once_with(models, hf_path="/weights")


def test_load_weights_rejects_unrepresentable_memory_efficient_mode() -> None:
    """Fail actionably instead of ignoring legacy load semantics."""
    models = [object()]
    facade, current_bridge, _ = _facade()

    with pytest.raises(NotImplementedError, match="memory_efficient=True.*not representable"):
        facade.load_weights(models, "/weights", memory_efficient=True)

    current_bridge.load_hf_weights.assert_not_called()


def test_export_weights_delegates_and_preserves_iterable() -> None:
    """Return the current bridge's export iterable without materializing it."""
    models = [object()]
    facade, current_bridge, _ = _facade()
    exported = iter((("weight", torch.ones(1)),))
    current_bridge.export_hf_weights.return_value = exported

    assert facade.export_weights(models) is exported

    current_bridge.export_hf_weights.assert_called_once_with(models)


def test_export_weights_rejects_unrepresentable_expert_layout() -> None:
    """Fail actionably instead of ignoring separate-expert output."""
    models = [object()]
    facade, current_bridge, _ = _facade()

    with pytest.raises(NotImplementedError, match="keep_stacked_experts=False.*not representable"):
        facade.export_weights(models, keep_stacked_experts=False)

    current_bridge.export_hf_weights.assert_not_called()


def test_save_weights_delegates_to_current_bridge() -> None:
    """Delegate supported saving directly to current save code."""
    models = [object()]
    facade, current_bridge, _ = _facade()

    assert facade.save_weights(models, "/output") is None

    current_bridge.save_hf_pretrained.assert_called_once_with(models, "/output")


def test_save_weights_rejects_unrepresentable_memory_efficient_mode() -> None:
    """Fail actionably instead of ignoring legacy save semantics."""
    models = [object()]
    facade, current_bridge, _ = _facade()

    with pytest.raises(NotImplementedError, match="memory_efficient=True.*not representable"):
        facade.save_weights(models, "/output", memory_efficient=True)

    current_bridge.save_hf_pretrained.assert_not_called()
