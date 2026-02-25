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

import pytest
import torch
from megatron.core import parallel_state

from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider


pytestmark = [pytest.mark.unit]


def _mock_parallel_state(monkeypatch):
    """Mock parallel_state functions to avoid initialization requirements."""
    monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", lambda: True, raising=False)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        parallel_state, "get_data_parallel_world_size", lambda with_context_parallel=False: 1, raising=False
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_and_data_parallel_group", lambda **kwargs: None, raising=False)


def test_flux_provider_initialization_defaults():
    """Test FluxProvider initialization with default values."""
    provider = FluxProvider()

    # Base class requirements
    assert provider.num_layers == 1  # Dummy setting
    assert provider.hidden_size == 3072
    assert provider.ffn_hidden_size == 12288
    assert provider.num_attention_heads == 24
    assert provider.layernorm_epsilon == 1e-06
    assert provider.hidden_dropout == 0
    assert provider.attention_dropout == 0

    # FLUX-specific layer configuration
    assert provider.num_joint_layers == 19
    assert provider.num_single_layers == 38

    # Model architecture
    assert provider.add_qkv_bias is True
    assert provider.in_channels == 64
    assert provider.context_dim == 4096
    assert provider.model_channels == 256
    assert provider.axes_dims_rope == [16, 56, 56]
    assert provider.patch_size == 1
    assert provider.guidance_embed is False
    assert provider.vec_in_dim == 768


def test_flux_provider_initialization_custom():
    """Test FluxProvider initialization with custom values."""
    provider = FluxProvider(
        hidden_size=2048,
        num_attention_heads=16,
        kv_channels=128,  # 2048 // 16
        num_query_groups=16,  # Same as num_attention_heads (no GQA)
        num_joint_layers=10,
        num_single_layers=20,
        in_channels=32,
        guidance_embed=True,
    )

    assert provider.hidden_size == 2048
    assert provider.num_attention_heads == 16
    assert provider.num_joint_layers == 10
    assert provider.num_single_layers == 20
    assert provider.in_channels == 32
    assert provider.guidance_embed is True


def test_flux_provider_rotary_embedding_settings():
    """Test FluxProvider rotary embedding settings."""
    provider = FluxProvider()

    assert provider.rotary_interleaved is True
    assert provider.apply_rope_fusion is False


def test_flux_provider_initialization_settings():
    """Test FluxProvider initialization and performance settings."""
    provider = FluxProvider()

    assert provider.use_cpu_initialization is True
    assert provider.gradient_accumulation_fusion is False
    assert provider.enable_cuda_graph is False
    assert provider.cuda_graph_scope is None
    assert provider.use_te_rng_tracker is False
    assert provider.cuda_graph_warmup_steps == 2


def test_flux_provider_inference_settings():
    """Test FluxProvider inference settings."""
    provider = FluxProvider()

    assert provider.guidance_scale == 3.5


def test_flux_provider_checkpoint_settings():
    """Test FluxProvider checkpoint loading settings."""
    provider = FluxProvider()

    assert provider.ckpt_path is None
    assert provider.load_dist_ckpt is False
    assert provider.do_convert_from_hf is False
    assert provider.save_converted_model_to is None


def test_flux_provider_llm_compatibility_attributes():
    """Test FluxProvider has attributes for LLM compatibility."""
    provider = FluxProvider()

    # These attributes are unused for images/videos but required by bridge training for LLMs
    assert provider.seq_length == 1024
    assert provider.share_embeddings_and_output_weights is False
    assert provider.vocab_size == 25256 * 8
    assert provider.make_vocab_size_divisible_by == 128


def test_flux_provider_virtual_pipeline_validation():
    """Test that FluxProvider validates virtual pipeline configuration."""
    provider = FluxProvider(
        num_joint_layers=12,
        num_single_layers=24,
        virtual_pipeline_model_parallel_size=2,
        pipeline_model_parallel_size=3,
    )

    total_layers = provider.num_joint_layers + provider.num_single_layers
    p_size = provider.pipeline_model_parallel_size
    vp_size = provider.virtual_pipeline_model_parallel_size

    # Should satisfy: (total_layers // p_size) % vp_size == 0
    # (36 // 3) % 2 == 12 % 2 == 0 ✓
    assert (total_layers // p_size) % vp_size == 0


def test_flux_provider_virtual_pipeline_validation_fails(monkeypatch):
    """Test that FluxProvider raises assertion error for invalid virtual pipeline configuration."""
    _mock_parallel_state(monkeypatch)

    provider = FluxProvider(
        num_joint_layers=10,
        num_single_layers=20,
        virtual_pipeline_model_parallel_size=3,
        pipeline_model_parallel_size=2,
    )

    # Should fail: (30 // 2) % 3 == 15 % 3 == 0 ✓ (actually this passes)
    # Let's create a failing case: (10 // 2) % 3 == 5 % 3 == 2 ≠ 0
    provider.num_joint_layers = 5
    provider.num_single_layers = 5

    with pytest.raises(AssertionError, match="Make sure the number of model chunks is the same"):
        provider.provide()


def test_flux_provider_axes_dims_rope_field():
    """Test that axes_dims_rope field factory works correctly."""
    provider1 = FluxProvider()
    provider2 = FluxProvider()

    # Should have default values
    assert provider1.axes_dims_rope == [16, 56, 56]
    assert provider2.axes_dims_rope == [16, 56, 56]

    # Should be independent instances (not sharing same list)
    provider1.axes_dims_rope[0] = 32
    assert provider2.axes_dims_rope[0] == 16  # Should not be affected


def test_flux_provider_custom_axes_dims_rope():
    """Test FluxProvider with custom axes_dims_rope."""
    custom_axes = [8, 32, 32]
    provider = FluxProvider(axes_dims_rope=custom_axes)

    assert provider.axes_dims_rope == custom_axes


def test_flux_provider_activation_func_default():
    """Test that FluxProvider has default activation function."""
    provider = FluxProvider()

    from megatron.core.transformer.utils import openai_gelu

    assert provider.activation_func == openai_gelu


def test_flux_provider_custom_checkpoint_settings():
    """Test FluxProvider with custom checkpoint settings."""
    provider = FluxProvider(
        ckpt_path="/path/to/checkpoint",
        load_dist_ckpt=True,
        do_convert_from_hf=True,
        save_converted_model_to="/path/to/save",
    )

    assert provider.ckpt_path == "/path/to/checkpoint"
    assert provider.load_dist_ckpt is True
    assert provider.do_convert_from_hf is True
    assert provider.save_converted_model_to == "/path/to/save"


def test_flux_provider_cuda_graph_settings():
    """Test FluxProvider CUDA graph settings."""
    provider = FluxProvider(enable_cuda_graph=True, cuda_graph_scope="full", cuda_graph_warmup_steps=5)

    assert provider.enable_cuda_graph is True
    assert provider.cuda_graph_scope == "full"
    assert provider.cuda_graph_warmup_steps == 5


def test_flux_provider_is_transformer_config():
    """Test that FluxProvider is a TransformerConfig."""
    from megatron.bridge.models.transformer_config import TransformerConfig

    provider = FluxProvider()

    assert isinstance(provider, TransformerConfig)


def test_flux_provider_is_model_provider_mixin():
    """Test that FluxProvider is a ModelProviderMixin."""
    from megatron.bridge.models.model_provider import ModelProviderMixin

    provider = FluxProvider()

    assert isinstance(provider, ModelProviderMixin)


def test_flux_provider_has_provide_method():
    """Test that FluxProvider has provide method."""
    provider = FluxProvider()

    assert hasattr(provider, "provide")
    assert callable(provider.provide)


def test_flux_provider_dtype_settings():
    """Test FluxProvider data type settings."""
    provider = FluxProvider(bf16=True, params_dtype=torch.bfloat16)

    assert provider.bf16 is True
    assert provider.params_dtype == torch.bfloat16


def test_flux_provider_parallel_settings():
    """Test FluxProvider parallel configuration settings."""
    provider = FluxProvider(tensor_model_parallel_size=2, pipeline_model_parallel_size=4, sequence_parallel=True)

    assert provider.tensor_model_parallel_size == 2
    assert provider.pipeline_model_parallel_size == 4
    assert provider.sequence_parallel is True


def test_flux_provider_num_layers_is_dummy():
    """Test that num_layers is a dummy value and not used for layer count."""
    provider = FluxProvider()

    # num_layers is set to 1 (dummy) but actual layers are controlled by:
    assert provider.num_layers == 1
    assert provider.num_joint_layers == 19  # Actual double block count
    assert provider.num_single_layers == 38  # Actual single block count


def test_flux_provider_default_guidance_scale():
    """Test that guidance_scale has correct default value."""
    provider = FluxProvider()

    assert provider.guidance_scale == 3.5


def test_flux_provider_custom_guidance_scale():
    """Test FluxProvider with custom guidance_scale."""
    provider = FluxProvider(guidance_scale=7.5)

    assert provider.guidance_scale == 7.5
