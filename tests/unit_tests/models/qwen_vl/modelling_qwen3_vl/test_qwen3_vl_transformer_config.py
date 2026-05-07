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

"""Unit tests for Qwen3-VL vision transformer_config (get_vision_model_config)."""

from enum import Enum
from types import SimpleNamespace

import pytest

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


def _hf_config():
    return SimpleNamespace(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        num_position_embeddings=256,
        out_hidden_size=64,
        deepstack_visual_indexes=[1],
    )


def _megatron_base(**overrides):
    """Minimal megatron_config for get_vision_model_config (shared fields + overrides)."""
    base = dict(
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        tensor_model_parallel_size=1,
        enable_cuda_graph=False,
        cuda_graph_use_single_mempool=False,
        cuda_graph_retain_backward_graph=False,
        cuda_graph_warmup_steps=0,
        external_cuda_graph=False,
        cuda_graph_impl="none",
        cuda_graph_scope=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _cuda_graph_storage(config):
    """Return the concrete CUDA graph field used by this MCore version."""
    modules = getattr(config, "cuda_graph_modules", None)
    if modules is not None:
        return modules
    return getattr(config, "cuda_graph_scope", [])


class TestGetVisionModelConfigVisionCudaGraph:
    """Vision encoder CUDA graph propagation from megatron_config (provider)."""

    def test_vision_cuda_graph_defaults_when_impl_none(self):
        megatron = _megatron_base(
            vision_cuda_graph_impl="none",
            cuda_graph_impl="local_transformer_engine",
            cuda_graph_scope=["attn"],
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cfg.cuda_graph_impl == "none"
        assert cuda_graph_module_names(cfg) == []

    def test_vision_cuda_graph_defaults_when_attr_missing(self):
        megatron = _megatron_base(
            cuda_graph_impl="local_transformer_engine",
            cuda_graph_scope=["attn"],
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cfg.cuda_graph_impl == "none"
        assert cuda_graph_module_names(cfg) == []

    def test_vision_cuda_graph_ignores_language_legacy_full_iteration_when_disabled(self):
        megatron = _megatron_base(
            cuda_graph_impl="local",
            cuda_graph_scope=["full_iteration"],
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cfg.cuda_graph_impl == "none"
        assert cuda_graph_module_names(cfg) == []

    def test_vision_cuda_graph_impl_propagated_scope_empty_without_scope_attr(self):
        megatron = _megatron_base(vision_cuda_graph_impl="local_transformer_engine")
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cfg.cuda_graph_impl == "local_transformer_engine"
        assert cuda_graph_module_names(cfg) == []

    def test_vision_cuda_graph_scope_string_list_converted_to_enums(self):
        megatron = _megatron_base(
            vision_cuda_graph_impl="local_transformer_engine",
            vision_cuda_graph_scope=["attn", "mlp"],
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        cuda_graph_values = _cuda_graph_storage(cfg)

        assert cfg.cuda_graph_impl == "local_transformer_engine"
        assert cuda_graph_module_names(cfg) == ["attn", "mlp"]
        assert all(isinstance(value, Enum) for value in cuda_graph_values)

    def test_vision_cuda_graph_scope_list_propagated(self):
        scopes = ["attn"]
        megatron = _megatron_base(
            vision_cuda_graph_impl="local_transformer_engine",
            vision_cuda_graph_scope=scopes,
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cuda_graph_module_names(cfg) == scopes

    def test_vision_cuda_graph_scope_empty_list_clears_scope(self):
        megatron = _megatron_base(
            vision_cuda_graph_impl="local_transformer_engine",
            vision_cuda_graph_scope=[],
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cuda_graph_module_names(cfg) == []

    def test_max_vision_cuda_graph_seq_length_propagated(self):
        megatron = _megatron_base(
            vision_cuda_graph_impl="none",
            max_vision_cuda_graph_seq_length=4096,
        )
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cfg.max_vision_cuda_graph_seq_length == 4096

    def test_max_vision_cuda_graph_seq_length_unchanged_when_attr_missing(self):
        megatron = _megatron_base(vision_cuda_graph_impl="none")
        cfg = get_vision_model_config(_hf_config(), megatron)
        assert cfg.max_vision_cuda_graph_seq_length is None

    def test_invalid_scope_string_raises_keyerror(self):
        megatron = _megatron_base(
            vision_cuda_graph_impl="local_transformer_engine",
            vision_cuda_graph_scope=["not_a_valid_cuda_graph_scope_member"],
        )
        with pytest.raises(KeyError):
            get_vision_model_config(_hf_config(), megatron)
