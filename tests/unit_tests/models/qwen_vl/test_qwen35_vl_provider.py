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

import pytest

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    _TRANSFORMERS_HAS_QWEN3_5,
    _TRANSFORMERS_HAS_QWEN3_5_MOE,
    Qwen35VLModelProvider,
    Qwen35VLMoEModelProvider,
)


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have qwen3_5 support")


class TestQwen35VLModelProvider:
    """Tests for the dense Qwen3.5 VL model provider."""

    def test_initialization_defaults(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert provider.num_layers == 64
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 24

    def test_hybrid_architecture_defaults(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.attention_output_gate is True
        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 4

    def test_gdn_defaults(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert provider.linear_conv_kernel_dim == 4
        assert provider.linear_key_head_dim == 128
        assert provider.linear_value_head_dim == 128
        assert provider.linear_num_key_heads == 16
        assert provider.linear_num_value_heads == 48

    def test_vl_defaults(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert provider.position_embedding_type == "mrope"
        assert provider.mrope_section == [11, 11, 10]
        assert provider.image_token_id == 248056
        assert provider.video_token_id == 248057
        assert provider.vision_start_token_id == 248053
        assert provider.vision_end_token_id == 248054
        assert provider.bos_token_id == 248045
        assert provider.eos_token_id == 248044

    def test_common_llm_defaults(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.kv_channels == 256
        assert provider.num_query_groups == 4
        assert provider.hidden_dropout == 0.0
        assert provider.rotary_base == 10000000.0
        assert provider.rotary_percent == 0.25

    def test_freeze_options_defaults(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_freeze_options_custom(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
            freeze_language_model=True,
            freeze_vision_model=True,
        )
        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True

    def test_custom_mrope_section(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
            mrope_section=[8, 12, 12],
        )
        assert provider.mrope_section == [8, 12, 12]

    def test_vision_config_default_type(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert isinstance(provider.vision_config, Qwen3_5VisionConfig)

    def test_inherits_from_gpt_provider(self):
        assert issubclass(Qwen35VLModelProvider, GPTModelProvider)

    def test_provide_methods_exist(self):
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            num_attention_heads=24,
        )
        assert hasattr(provider, "provide") and callable(provider.provide)
        assert hasattr(provider, "provide_language_model") and callable(provider.provide_language_model)

    def test_tp_validation(self):
        with pytest.raises(ValueError, match="TP size"):
            Qwen35VLModelProvider(
                num_layers=64,
                hidden_size=5120,
                num_attention_heads=24,
                num_query_groups=2,
                tensor_model_parallel_size=4,
            )


@pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5_MOE, reason="transformers does not have qwen3_5_moe support")
class TestQwen35VLMoEModelProvider:
    """Tests for the MoE Qwen3.5 VL model provider."""

    def test_initialization_defaults(self):
        provider = Qwen35VLMoEModelProvider(
            num_layers=60,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.num_layers == 60
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32

    def test_moe_defaults(self):
        provider = Qwen35VLMoEModelProvider(
            num_layers=60,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.num_moe_experts == 512
        assert provider.moe_router_topk == 10
        assert provider.moe_shared_expert_gate is True
        assert provider.moe_grouped_gemm is True
        assert provider.moe_router_load_balancing_type == "global_aux_loss"
        assert provider.moe_router_pre_softmax is False
        assert provider.moe_token_dispatcher_type == "alltoall"

    def test_hybrid_architecture_defaults(self):
        provider = Qwen35VLMoEModelProvider(
            num_layers=60,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 4
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.attention_output_gate is True

    def test_gdn_defaults(self):
        provider = Qwen35VLMoEModelProvider(
            num_layers=60,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.linear_num_value_heads == 64
        assert provider.linear_num_key_heads == 16
        assert provider.linear_key_head_dim == 128
        assert provider.linear_value_head_dim == 128

    def test_vl_defaults(self):
        provider = Qwen35VLMoEModelProvider(
            num_layers=60,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert provider.position_embedding_type == "mrope"
        assert provider.mrope_section == [11, 11, 10]
        assert provider.bos_token_id == 248045
        assert provider.eos_token_id == 248046

    def test_inherits_from_gpt_provider(self):
        assert issubclass(Qwen35VLMoEModelProvider, GPTModelProvider)

    def test_vision_config_default_type(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig

        provider = Qwen35VLMoEModelProvider(
            num_layers=60,
            hidden_size=4096,
            num_attention_heads=32,
        )
        assert isinstance(provider.vision_config, Qwen3_5MoeVisionConfig)

    def test_tp_validation(self):
        with pytest.raises(ValueError, match="TP size"):
            Qwen35VLMoEModelProvider(
                num_layers=60,
                hidden_size=4096,
                num_attention_heads=32,
                num_query_groups=2,
                tensor_model_parallel_size=4,
            )
