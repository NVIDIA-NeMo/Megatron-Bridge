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

"""Unit tests for megatron.bridge.models.transformer_config."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.transformer_config import (
    _HYBRIDEP_PADDING_FIELDS,
    HeterogeneousTransformerConfig,
    MLATransformerConfig,
    TransformerConfig,
    _enable_safe_hybridep_dispatch,
    _resolve_string_fields,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FINALIZE_PATCH = "megatron.bridge.models.transformer_config.MCoreTransformerConfig.__post_init__"
_MLA_FINALIZE_PATCH = "megatron.bridge.models.transformer_config.MCoreMLATransformerConfig.__post_init__"
_HETERO_FINALIZE_PATCH = "megatron.bridge.models.transformer_config.MCoreHeterogeneousTransformerConfig.__post_init__"


def _make_config(**kwargs) -> TransformerConfig:
    """Build a minimal TransformerConfig with MCore post_init skipped."""
    defaults = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_hybridep_config(config_type=TransformerConfig, **kwargs):
    """Build a HybridEP config using the padding field exposed by this MCore ref."""
    defaults = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
    )
    padding_field = next(field for field in _HYBRIDEP_PADDING_FIELDS if field in config_type.__dataclass_fields__)
    return config_type(**defaults, **{padding_field: False}, **kwargs), padding_field


class TestEnableSafeHybridepDispatch:
    """Tests for cross-branch Megatron Core HybridEP padding compatibility."""

    def test_enables_dev_padding_field(self):
        cfg = SimpleNamespace(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_hybridep_pad_variable_tokens=False,
            cuda_graph_impl="none",
        )

        _enable_safe_hybridep_dispatch(cfg)

        assert cfg.moe_hybridep_pad_variable_tokens is True

    def test_cuda_graph_config_does_not_require_padding_field(self):
        cfg = SimpleNamespace(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            cuda_graph_impl="full_iteration",
        )

        _enable_safe_hybridep_dispatch(cfg)


# ---------------------------------------------------------------------------
# _resolve_string_fields
# ---------------------------------------------------------------------------


class TestResolveStringFields:
    """Tests for the module-level _resolve_string_fields helper."""

    def test_activation_func_string_resolved_to_callable(self):
        cfg = _make_config(activation_func="silu")
        assert isinstance(cfg.activation_func, str)
        _resolve_string_fields(cfg)
        import torch.nn.functional as F

        assert cfg.activation_func is F.silu

    def test_activation_func_callable_left_unchanged(self):
        import torch.nn.functional as F

        cfg = _make_config(activation_func=F.gelu)
        _resolve_string_fields(cfg)
        assert cfg.activation_func is F.gelu

    def test_activation_func_none_left_unchanged(self):
        cfg = _make_config()
        cfg.activation_func = None
        _resolve_string_fields(cfg)
        assert cfg.activation_func is None

    def test_params_dtype_string_resolved_to_torch_dtype(self):
        cfg = _make_config()
        cfg.params_dtype = "bf16"
        _resolve_string_fields(cfg)
        assert cfg.params_dtype is torch.bfloat16

    def test_params_dtype_torch_dtype_left_unchanged(self):
        cfg = _make_config()
        cfg.params_dtype = torch.float32
        _resolve_string_fields(cfg)
        assert cfg.params_dtype is torch.float32

    def test_pipeline_dtype_string_resolved_to_torch_dtype(self):
        cfg = _make_config()
        cfg.pipeline_dtype = "fp16"
        _resolve_string_fields(cfg)
        assert cfg.pipeline_dtype is torch.float16

    def test_pipeline_dtype_none_left_unchanged(self):
        cfg = _make_config()
        cfg.pipeline_dtype = None
        _resolve_string_fields(cfg)
        assert cfg.pipeline_dtype is None

    def test_all_three_string_fields_resolved_together(self):
        cfg = _make_config(activation_func="gelu")
        cfg.params_dtype = "bf16"
        cfg.pipeline_dtype = "bf16"
        _resolve_string_fields(cfg)
        import torch.nn.functional as F

        assert cfg.activation_func is F.gelu
        assert cfg.params_dtype is torch.bfloat16
        assert cfg.pipeline_dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# TransformerConfig.finalize
# ---------------------------------------------------------------------------


class TestTransformerConfigFinalize:
    """Tests for TransformerConfig.finalize()."""

    def test_finalize_calls_mcore_post_init(self):
        cfg = _make_config()
        with patch(_FINALIZE_PATCH) as mock_post_init:
            cfg.finalize()
        mock_post_init.assert_called_once()

    def test_finalize_resolves_string_activation_func(self):
        cfg = _make_config(activation_func="silu")
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        import torch.nn.functional as F

        assert cfg.activation_func is F.silu

    def test_finalize_rejects_unregistered_activation_func(self):
        cfg = _make_config(activation_func="attacker_pkg.payload.fn")
        with pytest.raises(ValueError, match="attacker_pkg.payload.fn"):
            cfg.finalize()

    def test_finalize_resolves_string_params_dtype(self):
        cfg = _make_config()
        cfg.params_dtype = "bf16"
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.params_dtype is torch.bfloat16

    def test_finalize_resolves_string_pipeline_dtype(self):
        cfg = _make_config()
        cfg.pipeline_dtype = "fp16"
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.pipeline_dtype is torch.float16

    def test_sequence_parallel_disabled_when_tp1(self):
        cfg = _make_config(sequence_parallel=True, tensor_model_parallel_size=1)
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.sequence_parallel is False

    def test_sequence_parallel_preserved_when_tp_gt1(self):
        cfg = _make_config(sequence_parallel=True, tensor_model_parallel_size=2)
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.sequence_parallel is True

    def test_sequence_parallel_false_unchanged_with_tp1(self):
        cfg = _make_config(sequence_parallel=False, tensor_model_parallel_size=1)
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.sequence_parallel is False

    def test_pipeline_dtype_propagated_from_params_dtype_when_pp_gt1(self):
        cfg = _make_config()
        cfg.params_dtype = torch.bfloat16
        cfg.pipeline_dtype = None
        cfg.pipeline_model_parallel_size = 2
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.pipeline_dtype is torch.bfloat16

    def test_pipeline_dtype_not_overwritten_when_already_set(self):
        cfg = _make_config()
        cfg.params_dtype = torch.bfloat16
        cfg.pipeline_dtype = torch.float16
        cfg.pipeline_model_parallel_size = 2
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.pipeline_dtype is torch.float16

    def test_pipeline_dtype_not_set_when_pp1(self):
        cfg = _make_config()
        cfg.params_dtype = torch.bfloat16
        cfg.pipeline_dtype = None
        cfg.pipeline_model_parallel_size = 1
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.pipeline_dtype is None

    def test_moe_unset_expert_tensor_parallel_size_defaults_to_one(self):
        cfg = _make_config(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=None,
        )
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.expert_tensor_parallel_size == 1

    def test_moe_explicit_expert_tensor_parallel_size_is_preserved(self):
        cfg = _make_config(
            tensor_model_parallel_size=4,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=2,
        )
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.expert_tensor_parallel_size == 2

    def test_dense_unset_expert_tensor_parallel_size_is_not_changed_by_bridge(self):
        cfg = _make_config(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=None,
        )
        with patch(_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.expert_tensor_parallel_size is None

    def test_hybridep_finalization_enables_uneven_dispatch_padding(self):
        """HybridEP must safely handle different token counts on each EP rank."""
        cfg, padding_field = _make_hybridep_config()

        with patch(_FINALIZE_PATCH):
            cfg.finalize()

        assert getattr(cfg, padding_field) is True

    def test_non_hybridep_finalization_preserves_uneven_dispatch_padding(self):
        """Other flex backends must retain their configured padding behavior."""
        cfg, padding_field = _make_hybridep_config()
        cfg.moe_flex_dispatcher_backend = "deepep"

        with patch(_FINALIZE_PATCH):
            cfg.finalize()

        assert getattr(cfg, padding_field) is False

    @pytest.mark.parametrize(
        "cuda_graph_settings",
        [
            {"cuda_graph_impl": "full_iteration"},
            {"cuda_graph_impl": "transformer_engine"},
            {"cuda_graph_impl": "local"},
            {"enable_cuda_graph": True},
            {"external_cuda_graph": True},
            {"cuda_graph_modules": "full_iteration"},
            {"cuda_graph_scope": "full_iteration"},
        ],
    )
    def test_hybridep_cuda_graph_finalization_preserves_padding_setting(self, cuda_graph_settings):
        """CUDA-graph HybridEP configs must not gain a host scalar synchronization."""
        cfg, padding_field = _make_hybridep_config(**cuda_graph_settings)

        with patch(_FINALIZE_PATCH):
            cfg.finalize()

        assert getattr(cfg, padding_field) is False


class TestMLATransformerConfigFinalize:
    """Tests for MLATransformerConfig.finalize()."""

    def test_moe_unset_expert_tensor_parallel_size_defaults_to_one(self):
        cfg = MLATransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=None,
        )
        with patch(_MLA_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.expert_tensor_parallel_size == 1

    def test_hybridep_finalization_enables_uneven_dispatch_padding(self):
        cfg, padding_field = _make_hybridep_config(MLATransformerConfig)

        with patch(_MLA_FINALIZE_PATCH):
            cfg.finalize()

        assert getattr(cfg, padding_field) is True


# ---------------------------------------------------------------------------
# HeterogeneousTransformerConfig.finalize
# ---------------------------------------------------------------------------


class TestHeterogeneousTransformerConfigFinalize:
    """Tests for HeterogeneousTransformerConfig.finalize()."""

    def _make_hetero(self, **kwargs) -> HeterogeneousTransformerConfig:
        defaults = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
        defaults.update(kwargs)
        return HeterogeneousTransformerConfig(**defaults)

    def _make_valid_hetero(self, **kwargs) -> HeterogeneousTransformerConfig:
        block = {
            "attention": {"no_op": False, "replace_with_linear": False, "num_query_groups": 4},
            "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 256},
        }
        return self._make_hetero(
            heterogeneous_layers_config_encoded_json=json.dumps({"block_configs": [block, block]}),
            **kwargs,
        )

    def test_finalize_calls_mcore_hetero_post_init(self):
        cfg = self._make_hetero()
        with patch(_HETERO_FINALIZE_PATCH) as mock_post_init:
            cfg.finalize()
        mock_post_init.assert_called_once()

    def test_finalize_resolves_string_activation_func(self):
        cfg = self._make_hetero(activation_func="silu")
        with patch(_HETERO_FINALIZE_PATCH):
            cfg.finalize()
        import torch.nn.functional as F

        assert cfg.activation_func is F.silu

    def test_finalize_resolves_string_params_dtype(self):
        cfg = self._make_hetero()
        cfg.params_dtype = "bf16"
        with patch(_HETERO_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.params_dtype is torch.bfloat16

    def test_sequence_parallel_disabled_when_tp1(self):
        cfg = self._make_hetero(sequence_parallel=True, tensor_model_parallel_size=1)
        with patch(_HETERO_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.sequence_parallel is False

    def test_sequence_parallel_preserved_when_tp_gt1(self):
        cfg = self._make_hetero(sequence_parallel=True, tensor_model_parallel_size=2)
        with patch(_HETERO_FINALIZE_PATCH):
            cfg.finalize()
        assert cfg.sequence_parallel is True

    def test_hybridep_finalization_enables_uneven_dispatch_padding(self):
        cfg, padding_field = _make_hybridep_config(HeterogeneousTransformerConfig)

        with patch(_HETERO_FINALIZE_PATCH):
            cfg.finalize()

        assert getattr(cfg, padding_field) is True

    def test_pipeline_dtype_propagated_from_params_dtype_when_pp_gt1(self):
        cfg = self._make_valid_hetero(
            params_dtype=torch.bfloat16,
            pipeline_dtype=None,
            pipeline_model_parallel_size=2,
        )

        cfg.finalize()

        assert cfg.pipeline_dtype is torch.bfloat16

    def test_explicit_pipeline_dtype_is_preserved_when_pp_gt1(self):
        cfg = self._make_valid_hetero(
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.float16,
            pipeline_model_parallel_size=2,
        )

        cfg.finalize()

        assert cfg.pipeline_dtype is torch.float16
