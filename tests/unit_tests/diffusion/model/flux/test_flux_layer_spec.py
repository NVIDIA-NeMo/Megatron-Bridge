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
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.models.flux.flux_layer_spec import (
    AdaLNContinuous,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)


pytestmark = [pytest.mark.unit]


def test_adaln_continuous_initialization():
    """Test AdaLNContinuous module initialization."""
    hidden_size = 512
    conditioning_dim = 768

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )

    # Test with layer norm
    adaln_ln = AdaLNContinuous(config=config, conditioning_embedding_dim=conditioning_dim, norm_type="layer_norm")
    assert hasattr(adaln_ln, "adaLN_modulation")
    assert hasattr(adaln_ln, "norm")

    # Test with RMS norm
    adaln_rms = AdaLNContinuous(config=config, conditioning_embedding_dim=conditioning_dim, norm_type="rms_norm")
    assert hasattr(adaln_rms, "adaLN_modulation")
    assert hasattr(adaln_rms, "norm")


def test_adaln_continuous_invalid_norm_type():
    """Test AdaLNContinuous raises error for invalid norm type."""
    hidden_size = 512
    conditioning_dim = 768

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )

    with pytest.raises(ValueError, match="Unknown normalization type"):
        AdaLNContinuous(config=config, conditioning_embedding_dim=conditioning_dim, norm_type="invalid_norm")


def test_adaln_continuous_forward():
    """Test AdaLNContinuous forward pass."""
    hidden_size = 512
    conditioning_dim = 768
    seq_len = 8
    batch_size = 2

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )

    adaln_continuous = AdaLNContinuous(
        config=config, conditioning_embedding_dim=conditioning_dim, norm_type="layer_norm"
    )

    x = torch.randn(seq_len, batch_size, hidden_size)
    conditioning_emb = torch.randn(batch_size, conditioning_dim)

    output = adaln_continuous(x, conditioning_emb)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_get_flux_double_transformer_engine_spec():
    """Test get_flux_double_transformer_engine_spec returns valid ModuleSpec."""
    spec = get_flux_double_transformer_engine_spec()

    # Basic structure checks
    assert hasattr(spec, "module")
    assert hasattr(spec, "submodules")

    # Check module type
    from megatron.bridge.diffusion.models.flux.flux_layer_spec import MMDiTLayer

    assert spec.module == MMDiTLayer

    # Check submodules exist
    sub = spec.submodules
    assert hasattr(sub, "self_attention")
    assert hasattr(sub, "mlp")

    # Check self_attention submodules
    attn_spec = sub.self_attention
    assert hasattr(attn_spec, "module")
    assert hasattr(attn_spec, "submodules")
    assert hasattr(attn_spec, "params")

    # Verify attention mask type is no_mask
    from megatron.core.transformer.enums import AttnMaskType

    assert attn_spec.params.get("attn_mask_type") == AttnMaskType.no_mask

    # Check attention submodules
    attn_sub = attn_spec.submodules
    for attr in [
        "q_layernorm",
        "k_layernorm",
        "added_q_layernorm",
        "added_k_layernorm",
        "linear_qkv",
        "added_linear_qkv",
        "core_attention",
        "linear_proj",
    ]:
        assert hasattr(attn_sub, attr), f"Missing attention submodule: {attr}"

    # Check MLP submodules
    mlp_spec = sub.mlp
    assert hasattr(mlp_spec, "submodules")
    mlp_sub = mlp_spec.submodules
    assert hasattr(mlp_sub, "linear_fc1")
    assert hasattr(mlp_sub, "linear_fc2")


def test_get_flux_single_transformer_engine_spec():
    """Test get_flux_single_transformer_engine_spec returns valid ModuleSpec."""
    spec = get_flux_single_transformer_engine_spec()

    # Basic structure checks
    assert hasattr(spec, "module")
    assert hasattr(spec, "submodules")

    # Check module type
    from megatron.bridge.diffusion.models.flux.flux_layer_spec import FluxSingleTransformerBlock

    assert spec.module == FluxSingleTransformerBlock

    # Check submodules exist
    sub = spec.submodules
    assert hasattr(sub, "self_attention")
    assert hasattr(sub, "mlp")

    # Check self_attention submodules
    attn_spec = sub.self_attention
    assert hasattr(attn_spec, "module")
    assert hasattr(attn_spec, "submodules")
    assert hasattr(attn_spec, "params")

    # Verify attention mask type is no_mask
    from megatron.core.transformer.enums import AttnMaskType

    assert attn_spec.params.get("attn_mask_type") == AttnMaskType.no_mask

    # Check attention submodules
    attn_sub = attn_spec.submodules
    for attr in ["linear_qkv", "core_attention", "q_layernorm", "k_layernorm", "linear_proj"]:
        assert hasattr(attn_sub, attr), f"Missing attention submodule: {attr}"

    # Check MLP submodules
    mlp_spec = sub.mlp
    assert hasattr(mlp_spec, "submodules")
    mlp_sub = mlp_spec.submodules
    assert hasattr(mlp_sub, "linear_fc1")
    assert hasattr(mlp_sub, "linear_fc2")


def test_flux_double_and_single_specs_are_different():
    """Test that double and single transformer specs have different modules."""
    double_spec = get_flux_double_transformer_engine_spec()
    single_spec = get_flux_single_transformer_engine_spec()

    # Should have different module types
    assert double_spec.module != single_spec.module

    # Should have different attention modules
    assert double_spec.submodules.self_attention.module != single_spec.submodules.self_attention.module


def test_adaln_continuous_with_rms_norm_forward():
    """Test AdaLNContinuous forward pass with RMS norm."""
    hidden_size = 512
    conditioning_dim = 768
    seq_len = 8
    batch_size = 2

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )

    adaln_continuous = AdaLNContinuous(
        config=config, conditioning_embedding_dim=conditioning_dim, norm_type="rms_norm"
    )

    x = torch.randn(seq_len, batch_size, hidden_size)
    conditioning_emb = torch.randn(batch_size, conditioning_dim)

    output = adaln_continuous(x, conditioning_emb)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
