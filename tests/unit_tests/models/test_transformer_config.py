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

from unittest.mock import patch

import torch

from megatron.bridge.models.transformer_config import (
    HeterogeneousTransformerConfig,
    TransformerConfig,
    _resolve_string_fields,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FINALIZE_PATCH = "megatron.bridge.models.transformer_config.MCoreTransformerConfig.__post_init__"
_HETERO_FINALIZE_PATCH = "megatron.bridge.models.transformer_config.MCoreHeterogeneousTransformerConfig.__post_init__"


def _make_config(**kwargs) -> TransformerConfig:
    """Build a minimal TransformerConfig with MCore post_init skipped."""
    defaults = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


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


# ---------------------------------------------------------------------------
# HeterogeneousTransformerConfig.finalize
# ---------------------------------------------------------------------------


class TestHeterogeneousTransformerConfigFinalize:
    """Tests for HeterogeneousTransformerConfig.finalize()."""

    def _make_hetero(self, **kwargs) -> HeterogeneousTransformerConfig:
        defaults = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
        defaults.update(kwargs)
        return HeterogeneousTransformerConfig(**defaults)

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
