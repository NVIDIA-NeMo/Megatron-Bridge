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

"""Unit tests for flop_utils module."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.training.utils.flop_utils import num_floating_point_operations


@dataclass
class MockModelConfig:
    """Mock model config for testing flop_utils helper functions."""

    num_layers: int = 24
    hidden_size: int = 4096
    seq_length: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int | None = 8
    kv_channels: int = 128
    vocab_size: int = 128256
    make_vocab_size_divisible_by: int = 128
    tensor_model_parallel_size: int = 1
    # Hybrid model settings
    is_hybrid_model: bool = False
    hybrid_override_pattern: str | None = None
    hybrid_attention_ratio: float = 0
    hybrid_mlp_ratio: float = 0
    # Mamba settings
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_groups: int = 8
    mamba_num_heads: int = 128
    # MoE settings
    num_moe_experts: int | None = None
    moe_layer_freq: int = 1
    moe_router_topk: int = 1
    moe_ffn_hidden_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_latent_size: int | None = None
    # MTP settings
    mtp_num_layers: int | None = None
    # Attention settings
    multi_latent_attention: bool = False
    group_query_attention: bool = True
    gated_linear_unit: bool = True
    activation_func: object = field(default=None)

    def __post_init__(self):
        import torch.nn.functional as F

        if self.activation_func is None:
            self.activation_func = F.silu


@dataclass
class MockConfigContainer:
    """Mock ConfigContainer for testing."""

    model: MockModelConfig


class TestMoELayerFlops:
    """Unit tests for moe_layer_flops helper function via hybrid_flops."""

    def test_moe_layer_flops_without_latent(self):
        """Test MoE layer FLOPs calculation without latent compression.

        Formula: routed_flops = 4 * B * S * H * moe_ffn_hidden * topk * scale_factor
                 shared_flops = 4 * B * S * H * shared_expert_size * scale_factor
                 total = (routed_flops + shared_flops) * 3 (fwd + bwd)
        """
        batch_size = 1
        seq_len = 1024
        hidden_size = 2048
        moe_ffn_hidden = 4096
        shared_expert_size = 2048
        topk = 2
        vocab_size = 32000
        swiglu = False  # scale_factor = 1.0

        model_cfg = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern="E",  # Single MoE layer
            num_layers=1,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=8192,
            num_attention_heads=16,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=shared_expert_size,
            moe_router_topk=topk,
            moe_latent_size=None,
            gated_linear_unit=swiglu,
        )
        cfg = MockConfigContainer(model=model_cfg)

        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # Calculate expected MoE layer FLOPs (scale_factor=1.0 for non-swiglu)
        expected_routed = 4 * batch_size * seq_len * hidden_size * moe_ffn_hidden * topk * 1.0
        expected_shared = 4 * batch_size * seq_len * hidden_size * shared_expert_size * 1.0
        expected_moe_layer = expected_routed + expected_shared

        # Logit computation: 2 * B * S * H * vocab_size
        expected_logit = 2 * batch_size * seq_len * hidden_size * vocab_size

        # Total: (moe_layer + logit) * 3 (for fwd + bwd)
        expected_total = (expected_moe_layer + expected_logit) * 3

        assert actual_flops == expected_total, f"Expected {expected_total:.2e} but got {actual_flops:.2e}"

    def test_moe_layer_flops_with_latent(self):
        """Test MoE layer FLOPs calculation with latent compression.

        With latent:
            routed_flops = 4 * B * S * latent * moe_ffn_hidden * topk * scale
                         + 4 * B * S * H * latent (up/down proj)
            shared_flops = 4 * B * S * H * shared_expert_size * scale
        """
        batch_size = 1
        seq_len = 1024
        hidden_size = 2048
        moe_ffn_hidden = 4096
        shared_expert_size = 0  # No shared expert for simpler calculation
        topk = 1
        latent_size = 512
        vocab_size = 32000
        swiglu = False

        model_cfg = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern="E",
            num_layers=1,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=8192,
            num_attention_heads=16,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=shared_expert_size,
            moe_router_topk=topk,
            moe_latent_size=latent_size,
            gated_linear_unit=swiglu,
        )
        cfg = MockConfigContainer(model=model_cfg)

        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # Expected with latent compression
        expected_routed_core = 4 * batch_size * seq_len * latent_size * moe_ffn_hidden * topk * 1.0
        expected_up_down_proj = 4 * batch_size * seq_len * hidden_size * latent_size
        expected_routed = expected_routed_core + expected_up_down_proj
        expected_shared = 4 * batch_size * seq_len * hidden_size * shared_expert_size * 1.0
        expected_moe_layer = expected_routed + expected_shared

        expected_logit = 2 * batch_size * seq_len * hidden_size * vocab_size
        expected_total = (expected_moe_layer + expected_logit) * 3

        assert actual_flops == expected_total, f"Expected {expected_total:.2e} but got {actual_flops:.2e}"

    def test_latent_vs_non_latent_flops_difference(self):
        """Verify latent MoE produces predictably different FLOPs than non-latent."""
        batch_size = 1
        seq_len = 1024
        hidden_size = 2048
        moe_ffn_hidden = 4096
        topk = 2
        latent_size = 512
        vocab_size = 32000

        base_config = dict(
            is_hybrid_model=True,
            hybrid_override_pattern="E",
            num_layers=1,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=8192,
            num_attention_heads=16,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=0,
            moe_router_topk=topk,
            gated_linear_unit=False,
        )

        # Without latent
        cfg_no_latent = MockConfigContainer(model=MockModelConfig(**base_config, moe_latent_size=None))
        flops_no_latent = num_floating_point_operations(cfg_no_latent, batch_size=batch_size)

        # With latent
        cfg_latent = MockConfigContainer(model=MockModelConfig(**base_config, moe_latent_size=latent_size))
        flops_latent = num_floating_point_operations(cfg_latent, batch_size=batch_size)

        # Calculate expected difference in MoE FLOPs only (logit term is same)
        # Non-latent routed: 4 * B * S * H * moe_ffn * topk
        non_latent_routed = 4 * batch_size * seq_len * hidden_size * moe_ffn_hidden * topk
        # Latent routed: 4 * B * S * latent * moe_ffn * topk + 4 * B * S * H * latent
        latent_routed = (
            4 * batch_size * seq_len * latent_size * moe_ffn_hidden * topk
            + 4 * batch_size * seq_len * hidden_size * latent_size
        )

        expected_diff = (non_latent_routed - latent_routed) * 3  # times 3 for fwd+bwd
        actual_diff = flops_no_latent - flops_latent

        assert actual_diff == expected_diff, f"Expected difference {expected_diff:.2e} but got {actual_diff:.2e}"


class TestHybridMoEFlops:
    """Tests for hybrid model FLOPs calculations with MoE layers."""

    def test_moe_only_pattern_exact_flops(self):
        """Test hybrid model with only MoE layers produces exact expected FLOPs."""
        batch_size = 1
        seq_len = 512
        hidden_size = 1024
        moe_ffn_hidden = 2048
        shared_expert_size = 1024
        topk = 1
        vocab_size = 16000
        num_moe_layers = 2

        model_cfg = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern="EE",
            num_layers=num_moe_layers,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=shared_expert_size,
            moe_router_topk=topk,
            moe_latent_size=None,
            gated_linear_unit=False,
        )
        cfg = MockConfigContainer(model=model_cfg)

        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # Expected calculation
        moe_routed = 4 * batch_size * seq_len * hidden_size * moe_ffn_hidden * topk
        moe_shared = 4 * batch_size * seq_len * hidden_size * shared_expert_size
        moe_per_layer = moe_routed + moe_shared
        total_moe = moe_per_layer * num_moe_layers

        logit = 2 * batch_size * seq_len * hidden_size * vocab_size

        expected_flops = (total_moe + logit) * 3

        assert actual_flops == expected_flops, f"Expected {expected_flops:.2e} but got {actual_flops:.2e}"


class TestHybridLayerCounting:
    """Tests to verify layer counting with different hybrid patterns."""

    @pytest.mark.parametrize(
        "pattern,expected_attn,expected_mamba,expected_mlp,expected_moe",
        [
            ("M-*E", 1, 1, 1, 1),
            ("MMMM", 0, 4, 0, 0),
            ("----", 0, 0, 4, 0),
            ("****", 4, 0, 0, 0),
            ("EEEE", 0, 0, 0, 4),
            ("M-*E-*M", 2, 2, 2, 1),
        ],
    )
    def test_layer_counting_patterns(self, pattern, expected_attn, expected_mamba, expected_mlp, expected_moe):
        """Test that patterns with different layer types produce different FLOPs."""
        batch_size = 1
        seq_len = 512
        hidden_size = 1024
        vocab_size = 16000

        model_cfg = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern=pattern,
            num_layers=len(pattern),
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=2048,
            moe_shared_expert_intermediate_size=1024,
            moe_router_topk=1,
            mamba_state_dim=64,
            mamba_head_dim=32,
            mamba_num_groups=4,
            mamba_num_heads=64,
            gated_linear_unit=False,
        )
        cfg = MockConfigContainer(model=model_cfg)

        flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # Verify the FLOPs reflect the layer composition
        # At minimum, patterns with more compute-heavy layers should have higher FLOPs
        assert flops > 0, f"FLOPs should be positive for pattern '{pattern}'"

        # More specific: verify the contribution from each layer type
        # by checking FLOPs scales with expected layer count
        if expected_moe > 0:
            # Verify MoE contribution is present
            moe_per_layer = (
                4 * batch_size * seq_len * hidden_size * 2048 * 1  # routed
                + 4 * batch_size * seq_len * hidden_size * 1024  # shared
            ) * 3
            min_expected = expected_moe * moe_per_layer
            assert flops >= min_expected, (
                f"FLOPs {flops:.2e} should include at least {min_expected:.2e} from {expected_moe} MoE layers"
            )

    def test_swiglu_scaling_factor(self):
        """Test that SwiGLU activation properly scales MoE FLOPs by 1.5x."""
        batch_size = 1
        seq_len = 512
        hidden_size = 1024
        moe_ffn_hidden = 2048
        vocab_size = 16000

        base_config = dict(
            is_hybrid_model=True,
            hybrid_override_pattern="E",
            num_layers=1,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=0,
            moe_router_topk=1,
            moe_latent_size=None,
        )

        # Without SwiGLU
        cfg_no_swiglu = MockConfigContainer(model=MockModelConfig(**base_config, gated_linear_unit=False))
        flops_no_swiglu = num_floating_point_operations(cfg_no_swiglu, batch_size=batch_size)

        # With SwiGLU
        cfg_swiglu = MockConfigContainer(model=MockModelConfig(**base_config, gated_linear_unit=True))
        flops_swiglu = num_floating_point_operations(cfg_swiglu, batch_size=batch_size)

        # Logit term (same for both)
        logit = 2 * batch_size * seq_len * hidden_size * vocab_size

        # MoE term without swiglu
        moe_no_swiglu = 4 * batch_size * seq_len * hidden_size * moe_ffn_hidden * 1 * 1.0
        # MoE term with swiglu (1.5x)
        moe_swiglu = 4 * batch_size * seq_len * hidden_size * moe_ffn_hidden * 1 * 1.5

        expected_no_swiglu = (moe_no_swiglu + logit) * 3
        expected_swiglu = (moe_swiglu + logit) * 3

        assert flops_no_swiglu == expected_no_swiglu, (
            f"Non-SwiGLU: expected {expected_no_swiglu:.2e} but got {flops_no_swiglu:.2e}"
        )
        assert flops_swiglu == expected_swiglu, f"SwiGLU: expected {expected_swiglu:.2e} but got {flops_swiglu:.2e}"


class TestHybridMtpPatternParsing:
    """Tests for hybrid/MTP pattern parsing in FLOPs accounting."""

    def test_inferred_mtp_depth_scales_hybrid_logit_flops(self):
        """When mtp_num_layers is inferred from parsed pattern, logits FLOPs should scale accordingly."""
        batch_size = 1
        seq_len = 256
        hidden_size = 1024
        vocab_size = 32000  # divisible by 128, so padded vocab is unchanged.

        base_cfg = dict(
            is_hybrid_model=True,
            hybrid_override_pattern="M*/MM/MM",
            num_layers=2,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=8,
            vocab_size=vocab_size,
            moe_ffn_hidden_size=2048,
            moe_shared_expert_intermediate_size=0,
            moe_router_topk=1,
            gated_linear_unit=False,
            mtp_num_layers=0,  # overridden below for inferred-vs-explicit comparison
        )

        cfg_explicit_zero = MockConfigContainer(model=MockModelConfig(**base_cfg))
        cfg_inferred = MockConfigContainer(model=MockModelConfig(**(base_cfg | {"mtp_num_layers": None})))

        parsed_pattern = SimpleNamespace(main_pattern="M*", mtp_pattern="MM", mtp_num_depths=2)
        mock_module = MagicMock()
        mock_module.parse_hybrid_pattern.return_value = parsed_pattern

        with patch("megatron.bridge.training.utils.flop_utils.importlib.import_module", return_value=mock_module):
            flops_explicit_zero = num_floating_point_operations(cfg_explicit_zero, batch_size=batch_size)
            flops_inferred = num_floating_point_operations(cfg_inferred, batch_size=batch_size)

        # Only the logits term should differ here:
        #   delta = 2 * B * S * H * vocab * inferred_mtp_num_layers, then *3 for fwd+bwd factor.
        expected_delta = 2 * batch_size * seq_len * hidden_size * vocab_size * 2 * 3
        actual_delta = flops_inferred - flops_explicit_zero
        assert actual_delta == expected_delta, f"Expected logits delta {expected_delta:.2e} but got {actual_delta:.2e}"
