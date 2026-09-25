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


@pytest.mark.parametrize(
    "granularity,method,num_layers",
    [(None, None, None), ("selective", None, None), ("full", "block", 2)],
)
@pytest.mark.parametrize(
    "vision_policy,expected",
    [
        ({"vision_recompute_granularity": "inherit"}, "inherit"),
        ({"vision_recompute_granularity": None}, (None, None, None)),
        (
            {
                "vision_recompute_granularity": "full",
                "vision_recompute_method": "uniform",
                "vision_recompute_num_layers": 1,
            },
            ("full", "uniform", 1),
        ),
        (
            {
                "vision_recompute_granularity": "full",
                "vision_recompute_method": "block",
                "vision_recompute_num_layers": 2,
            },
            ("full", "block", 2),
        ),
        (
            {"vision_recompute_granularity": "selective", "vision_recompute_modules": ["core_attn", "mlp"]},
            ("selective", None, None),
        ),
    ],
)
def test_vision_recompute_is_independent_of_decoder(granularity, method, num_layers, vision_policy, expected):
    megatron = _megatron_base(
        recompute_granularity=granularity,
        recompute_method=method,
        recompute_num_layers=num_layers,
        recompute_modules=["gdn_norm_out", "moe"],
        **vision_policy,
    )

    config = get_vision_model_config(_hf_config(), megatron)

    if expected == "inherit":
        expected = (granularity, method, num_layers)
        assert config.recompute_modules == ["core_attn"]
    else:
        assert config.recompute_modules == vision_policy.get("vision_recompute_modules", [])
    assert (config.recompute_granularity, config.recompute_method, config.recompute_num_layers) == expected
    assert (megatron.recompute_granularity, megatron.recompute_method, megatron.recompute_num_layers) == (
        granularity,
        method,
        num_layers,
    )
    assert megatron.recompute_modules == ["gdn_norm_out", "moe"]


def test_vision_recompute_missing_fields_preserve_inheritance():
    megatron = _megatron_base(recompute_granularity="full", recompute_method="block", recompute_num_layers=2)

    config = get_vision_model_config(_hf_config(), megatron)

    assert config.recompute_granularity == "full"
    assert config.recompute_method == "block"
    assert config.recompute_num_layers == 2


@pytest.mark.parametrize("implementation", ["transformer_engine", "local", "local_transformer_engine"])
@pytest.mark.parametrize("granularity", ["inherit", "full"])
def test_full_vision_recompute_rejects_per_layer_vision_graphs(implementation, granularity):
    megatron = _megatron_base(
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        vision_recompute_granularity=granularity,
        vision_recompute_method="uniform" if granularity == "full" else None,
        vision_recompute_num_layers=1 if granularity == "full" else None,
        vision_cuda_graph_impl=implementation,
        vision_cuda_graph_scope=["attn", "mlp"],
    )

    with pytest.raises(ValueError, match="incompatible with per-layer vision CUDA graphs"):
        get_vision_model_config(_hf_config(), megatron)


@pytest.mark.parametrize("vision_implementation", ["none", "full_iteration"])
def test_full_vision_recompute_allows_decoder_graphs(vision_implementation):
    megatron = _megatron_base(
        vision_recompute_granularity="full",
        vision_recompute_method="uniform",
        vision_recompute_num_layers=1,
        vision_cuda_graph_impl=vision_implementation,
        cuda_graph_impl="transformer_engine",
        cuda_graph_scope=["attn", "mlp"],
    )

    config = get_vision_model_config(_hf_config(), megatron)

    assert config.recompute_granularity == "full"
    assert config.cuda_graph_impl == vision_implementation
    assert megatron.cuda_graph_impl == "transformer_engine"


@pytest.mark.parametrize("num_layers", [None, 0, -1, 3, 1.5, True])
def test_full_vision_recompute_rejects_invalid_layer_counts(num_layers):
    with pytest.raises(ValueError, match="vision_recompute_num_layers"):
        get_vision_model_config(
            _hf_config(),
            _megatron_base(
                vision_recompute_granularity="full",
                vision_recompute_method="uniform",
                vision_recompute_num_layers=num_layers,
            ),
        )


@pytest.mark.parametrize(
    "policy,message",
    [
        ({"vision_recompute_granularity": "typo"}, "vision_recompute_granularity"),
        ({"vision_recompute_method": "uniform"}, "explicitly"),
        ({"vision_recompute_num_layers": 1}, "explicitly"),
        ({"vision_recompute_modules": ["core_attn"]}, "explicitly"),
        ({"vision_recompute_granularity": None, "vision_recompute_method": "uniform"}, "Disabled"),
        ({"vision_recompute_granularity": None, "vision_recompute_num_layers": 1}, "Disabled"),
        ({"vision_recompute_granularity": None, "vision_recompute_modules": []}, "Disabled"),
        ({"vision_recompute_granularity": "full", "vision_recompute_num_layers": 1}, "vision_recompute_method"),
        (
            {
                "vision_recompute_granularity": "full",
                "vision_recompute_method": "typo",
                "vision_recompute_num_layers": 1,
            },
            "vision_recompute_method",
        ),
        (
            {
                "vision_recompute_granularity": "full",
                "vision_recompute_method": "uniform",
                "vision_recompute_num_layers": 1,
                "vision_recompute_modules": [],
            },
            "only supported for selective",
        ),
        ({"vision_recompute_granularity": "selective"}, "non-empty list"),
        ({"vision_recompute_granularity": "selective", "vision_recompute_modules": []}, "non-empty list"),
        ({"vision_recompute_granularity": "selective", "vision_recompute_modules": ["moe"]}, "non-empty list"),
        (
            {"vision_recompute_granularity": "selective", "vision_recompute_modules": ["gdn_norm_out"]},
            "non-empty list",
        ),
        ({"vision_recompute_granularity": "selective", "vision_recompute_modules": ["layernorm"]}, "non-empty list"),
        ({"vision_recompute_granularity": "selective", "vision_recompute_method": "block"}, "cannot specify"),
        ({"vision_recompute_granularity": "selective", "vision_recompute_num_layers": 1}, "cannot specify"),
    ],
)
def test_invalid_vision_recompute_policies_fail_early(policy, message):
    with pytest.raises(ValueError, match=message):
        get_vision_model_config(_hf_config(), _megatron_base(**policy))


def test_selective_vision_modules_are_copied():
    modules = ["core_attn", "mlp"]
    megatron = _megatron_base(vision_recompute_granularity="selective", vision_recompute_modules=modules)
    config = get_vision_model_config(_hf_config(), megatron)
    config.recompute_modules.append("changed")
    assert modules == ["core_attn", "mlp"]


def test_disabled_vision_recompute_allows_graphs_with_full_decoder_recompute():
    config = get_vision_model_config(
        _hf_config(),
        _megatron_base(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            vision_recompute_granularity=None,
            vision_cuda_graph_impl="local_transformer_engine",
        ),
    )
    assert config.recompute_granularity is None
    assert config.recompute_modules == []
    assert config.cuda_graph_impl == "local_transformer_engine"
