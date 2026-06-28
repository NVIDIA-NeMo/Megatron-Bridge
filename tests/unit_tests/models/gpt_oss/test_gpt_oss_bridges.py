#!/usr/bin/env python3
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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
from megatron.bridge.models.gpt_oss.model_config import GPTOSSModelBuilder
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestGptOssBridge:
    """Unit tests for GPT-OSS bridge provider mapping."""

    @pytest.fixture
    def gpt_oss_cfg(self):
        return {
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_local_experts": 32,
            "torch_dtype": "bfloat16",
            "vocab_size": 201088,
            "hidden_act": "silu",
            "sliding_window": 4096,
        }

    @pytest.fixture
    def mock_pretrained(self, gpt_oss_cfg):
        # Use spec to prevent Mock from auto-creating undefined attributes
        cfg = Mock(spec=list(gpt_oss_cfg.keys()))
        for k, v in gpt_oss_cfg.items():
            setattr(cfg, k, v)

        m = Mock(spec=PreTrainedCausalLM)
        m.config = cfg
        m.generation_config = Mock()
        return m

    def test_registration(self):
        assert issubclass(GPTOSSBridge, MegatronModelBridge)

    def test_model_config_bridge_preserves_complete_yarn_config(self, gpt_oss_cfg):
        config = dict(gpt_oss_cfg)
        config["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 2048,
            "beta_fast": 24.0,
            "beta_slow": 2.0,
            "mscale": 1.25,
            "mscale_all_dim": 0.5,
            "truncate": True,
        }
        result = GPTOSSBridge().model_config_bridge(SimpleNamespace(config=SimpleNamespace(**config)))

        assert type(result.transformer) is TransformerConfig
        assert result.yarn_rotary_scaling_factor == 16.0
        assert result.yarn_original_max_position_embeddings == 2048
        assert result.yarn_beta_fast == 24.0
        assert result.yarn_beta_slow == 2.0
        assert result.yarn_mscale == 1.25
        assert result.yarn_mscale_all_dim == 0.5
        assert result.yarn_correction_range_round_to_int is True
        assert "yarn_rotary_scaling_factor" not in result.transformer.__dict__
        restored = type(result).from_dict(result.as_dict())
        assert restored.yarn_rotary_scaling_factor == 16.0

        def inspect_bound_yarn(_builder, _pg_collection, **_kwargs):
            assert result.transformer.yarn_rotary_scaling_factor == 16.0
            assert result.transformer.yarn_original_max_position_embeddings == 2048
            return Mock()

        with patch.object(GPTModelBuilder, "build_model", autospec=True, side_effect=inspect_bound_yarn):
            GPTOSSModelBuilder(result).build_model(Mock())
        assert "yarn_rotary_scaling_factor" not in result.transformer.__dict__

    def test_provider_bridge_maps_config(self, mock_pretrained):
        bridge = GPTOSSBridge()
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, GPTModelProvider)
        # Key fields mapped from HF config
        assert provider.num_layers == mock_pretrained.config.num_hidden_layers
        assert provider.num_moe_experts == mock_pretrained.config.num_local_experts
        # dtype mapping
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16
