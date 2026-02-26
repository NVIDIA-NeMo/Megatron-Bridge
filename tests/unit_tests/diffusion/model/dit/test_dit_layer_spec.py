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
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.models.dit.dit_layer_spec import (
    AdaLN,
    DiTLayerWithAdaLN,
    RMSNorm,
    get_dit_adaln_block_with_transformer_engine_spec,
)


def test_rmsnorm_basic():
    """Test RMSNorm forward pass with basic properties."""
    hidden_size = 768
    batch_size = 2
    seq_len = 16

    # Create RMSNorm instance
    rms_norm = RMSNorm(hidden_size=hidden_size, eps=1e-6)

    # Create random input tensor
    x = torch.randn(seq_len, batch_size, hidden_size)

    # Forward pass
    output = rms_norm(x)

    # Check output shape matches input shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Check that weight parameter exists and has correct shape
    assert rms_norm.weight.shape == (hidden_size,), (
        f"Expected weight shape ({hidden_size},), got {rms_norm.weight.shape}"
    )

    # Check that output is not NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    # Check that RMS normalization approximately normalizes the variance along last dim
    # After RMS norm (before scaling by weight), variance should be close to 1
    normalized = rms_norm._norm(x.float()).type_as(x)
    rms_variance = normalized.pow(2).mean(-1)
    assert torch.allclose(rms_variance, torch.ones_like(rms_variance), atol=1e-5), (
        "RMS normalization should result in variance close to 1"
    )


def test_adaln_forward_chunking():
    """Test AdaLN forward pass returns correct number of chunks."""
    hidden_size = 768
    batch_size = 2
    n_adaln_chunks = 4

    # Create TransformerConfig object
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )

    # Create AdaLN instance with 9 chunks (for full DiT layer with cross attention)
    adaln = AdaLN(config=config, n_adaln_chunks=n_adaln_chunks, use_adaln_lora=True, adaln_lora_dim=256)

    # Create timestep embedding input
    timestep_emb = torch.randn(batch_size, hidden_size)

    # Forward pass should return n_adaln_chunks tensors
    chunks = adaln(timestep_emb)

    # Check that we get the correct number of chunks
    assert len(chunks) == n_adaln_chunks, f"Expected {n_adaln_chunks} chunks, got {len(chunks)}"

    # Check that each chunk has the correct shape
    for i, chunk in enumerate(chunks):
        assert chunk.shape == (batch_size, hidden_size), (
            f"Chunk {i} has shape {chunk.shape}, expected ({batch_size}, {hidden_size})"
        )


def test_adaln_modulation_methods():
    """Test AdaLN modulation and scaling methods."""
    hidden_size = 512
    seq_len = 8
    batch_size = 2

    # Create TransformerConfig object
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        layernorm_epsilon=1e-5,
        sequence_parallel=False,
    )

    # Create AdaLN instance
    adaln = AdaLN(config=config, n_adaln_chunks=6)

    # Create test tensors
    x = torch.randn(seq_len, batch_size, hidden_size)
    shift = torch.randn(batch_size, hidden_size)
    scale = torch.randn(batch_size, hidden_size)
    gate = torch.randn(batch_size, hidden_size)
    residual = torch.randn(seq_len, batch_size, hidden_size)

    # Test modulate method
    modulated = adaln.modulate(x, shift, scale)
    assert modulated.shape == x.shape, f"Modulated output shape {modulated.shape} != input shape {x.shape}"
    # Verify the modulation formula: x * (1 + scale) + shift
    expected_modulated = x * (1 + scale) + shift
    assert torch.allclose(modulated, expected_modulated, atol=1e-6), "Modulate formula incorrect"

    # Test scale_add method
    scaled_added = adaln.scale_add(residual, x, gate)
    assert scaled_added.shape == residual.shape, (
        f"scale_add output shape {scaled_added.shape} != residual shape {residual.shape}"
    )
    # Verify the formula: residual + gate * x
    expected_scaled_added = residual + gate * x
    assert torch.allclose(scaled_added, expected_scaled_added, atol=1e-6), "scale_add formula incorrect"

    # Test modulated_layernorm method
    modulated_ln = adaln.modulated_layernorm(x, shift, scale)
    assert modulated_ln.shape == x.shape, (
        f"modulated_layernorm output shape {modulated_ln.shape} != input shape {x.shape}"
    )
    assert not torch.isnan(modulated_ln).any(), "modulated_layernorm output contains NaN"

    # Test scaled_modulated_layernorm method
    hidden_states, shifted_output = adaln.scaled_modulated_layernorm(residual, x, gate, shift, scale)
    assert hidden_states.shape == residual.shape, (
        f"hidden_states shape {hidden_states.shape} != residual shape {residual.shape}"
    )
    assert shifted_output.shape == x.shape, f"shifted_output shape {shifted_output.shape} != x shape {x.shape}"
    assert not torch.isnan(hidden_states).any(), "hidden_states contains NaN"
    assert not torch.isnan(shifted_output).any(), "shifted_output contains NaN"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDiTLayerWithAdaLN:
    """Test class for DiTLayerWithAdaLN with shared setup."""

    def setup_method(self, method=None, monkeypatch=None):
        """Set up test fixtures before each test method."""
        # Stub parallel_state functions to avoid requiring initialization
        from megatron.core import parallel_state

        if monkeypatch:
            monkeypatch.setattr(
                parallel_state,
                "get_data_parallel_rank",
                lambda with_context_parallel=False, **kwargs: 0,
                raising=False,
            )
            monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda **kwargs: 1, raising=False)
            monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
            monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda **kwargs: None, raising=False)
            monkeypatch.setattr(
                parallel_state,
                "get_data_parallel_group",
                lambda with_context_parallel=False, **kwargs: None,
                raising=False,
            )
            monkeypatch.setattr(
                parallel_state, "get_tensor_model_parallel_group", lambda **kwargs: None, raising=False
            )

        # Common dimensions
        self.hidden_size = 512
        self.seq_len = 16
        self.batch_size = 2
        self.context_len = 32

        # Create TransformerConfig object
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            ffn_hidden_size=self.hidden_size * 4,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            bf16=False,
            fp16=False,
            params_dtype=torch.float32,
            apply_residual_connection_post_layernorm=False,
            add_bias_linear=False,
            gated_linear_unit=False,
            activation_func=torch.nn.functional.gelu,
            num_query_groups=None,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )

    def test_dit_layer_without_cross_attention(self, monkeypatch):
        """Test DiTLayerWithAdaLN forward pass without cross attention."""
        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Create submodules without cross attention using real attention modules
        submodules = get_dit_adaln_block_with_transformer_engine_spec().submodules
        submodules.cross_attention = IdentityOp
        dit_layer = (
            DiTLayerWithAdaLN(
                config=self.config,
                submodules=submodules,
                layer_number=1,
            )
            .to("cuda")
            .to(torch.bfloat16)
        )

        # Create input tensors
        hidden_states = torch.randn(self.seq_len, self.batch_size, self.hidden_size)
        hidden_states = hidden_states.reshape(1, -1, self.hidden_size).to("cuda")
        hidden_states = hidden_states.transpose(0, 1).to(torch.bfloat16)
        timestep_emb = torch.randn(1, self.hidden_size).to("cuda").to(torch.bfloat16)  # This acts as attention_mask

        cu_seqlens_q = torch.arange(
            0, (self.batch_size + 1) * self.seq_len, self.seq_len, dtype=torch.int32, device="cuda"
        )

        packed_seq_params = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_kv_padded=cu_seqlens_q,
                qkv_format="thd",
            ),
        }

        # Forward pass
        output, _ = dit_layer(
            hidden_states=hidden_states,
            attention_mask=timestep_emb,
            context=None,
            context_mask=None,
            packed_seq_params=packed_seq_params,
        )

        # Check output shape
        assert output.shape == hidden_states.shape, f"Output shape {output.shape} != input shape {hidden_states.shape}"

        # Check that output is valid (no NaN or Inf)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        # Check that adaLN has 6 chunks (for layer without cross attention)
        assert dit_layer.adaLN.n_adaln_chunks == 6, (
            f"Expected 6 adaLN chunks without cross attention, got {dit_layer.adaLN.n_adaln_chunks}"
        )

    def test_dit_layer_with_cross_attention(self, monkeypatch):
        """Test DiTLayerWithAdaLN forward pass with cross attention."""
        # Initialize with monkeypatch
        self.setup_method(None, monkeypatch)

        # Create submodules with cross attention using real attention modules
        submodules = get_dit_adaln_block_with_transformer_engine_spec().submodules

        dit_layer = (
            DiTLayerWithAdaLN(
                config=self.config,
                submodules=submodules,
                layer_number=1,
            )
            .to("cuda")
            .to(torch.bfloat16)
        )

        # Create input tensors
        hidden_states = torch.randn(1, self.seq_len * self.batch_size, self.hidden_size)
        hidden_states = hidden_states.reshape(1, -1, self.hidden_size).to("cuda")
        hidden_states = hidden_states.transpose(0, 1).to(torch.bfloat16)
        timestep_emb = torch.randn(1, self.hidden_size).to("cuda").to(torch.bfloat16)  # This acts as attention_mask
        context = torch.randn(1, self.context_len * self.batch_size, self.hidden_size)
        context = context.reshape(1, -1, self.hidden_size).to("cuda")
        context = context.transpose(0, 1).to(torch.bfloat16)
        context_mask = None

        cu_seqlens_q = torch.arange(
            0, (self.batch_size + 1) * self.seq_len, self.seq_len, dtype=torch.int32, device="cuda"
        )
        cu_seqlens_kv = torch.arange(
            0, (self.batch_size + 1) * self.context_len, self.context_len, dtype=torch.int32, device="cuda"
        )

        packed_seq_params = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_kv_padded=cu_seqlens_q,
                qkv_format="thd",
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_kv_padded=cu_seqlens_kv,
                qkv_format="thd",
            ),
        }

        # Forward pass
        output, _ = dit_layer(
            hidden_states=hidden_states,
            attention_mask=timestep_emb,
            context=context,
            context_mask=context_mask,
            packed_seq_params=packed_seq_params,
        )

        # Check output shape
        assert output.shape == hidden_states.shape, f"Output shape {output.shape} != input shape {hidden_states.shape}"

        # Check that output is valid (no NaN or Inf)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        # Check that adaLN has 9 chunks (for layer with cross attention)
        assert dit_layer.adaLN.n_adaln_chunks == 9, (
            f"Expected 9 adaLN chunks with cross attention, got {dit_layer.adaLN.n_adaln_chunks}"
        )
