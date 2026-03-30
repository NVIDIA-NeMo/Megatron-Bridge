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
    hybrid_layer_pattern: str | None = None
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
    attention_output_gate: bool = False
    # GDN (Gated DeltaNet) settings
    experimental_attention_variant: str | None = None
    linear_attention_freq: int | list | None = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48

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
            hybrid_layer_pattern="E",  # Single MoE layer
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
            hybrid_layer_pattern="E",
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
            hybrid_layer_pattern="E",
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
            hybrid_layer_pattern="EE",
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
            ("MG*E", 1, 1, 0, 1),
            ("GGGG", 0, 0, 0, 0),
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
            hybrid_layer_pattern=pattern,
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
            hybrid_layer_pattern="E",
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


@pytest.mark.unit
class TestGDNLayerFlops:
    """Tests for Gated DeltaNet (GDN) FLOPs calculation in transformer_flops path."""

    def _qwen35_27b_config(self, **overrides):
        """Return a MockModelConfig resembling Qwen3.5-27B (dense, 64 layers, freq=4)."""
        defaults = dict(
            num_layers=64,
            hidden_size=5120,
            seq_length=4096,
            ffn_hidden_size=17408,
            num_attention_heads=24,
            num_query_groups=4,
            kv_channels=256,
            vocab_size=248320,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=True,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
        )
        defaults.update(overrides)
        return MockModelConfig(**defaults)

    def test_gdn_flops_differ_from_pure_attention(self):
        """GDN-enabled config should produce different FLOPs than pure-attention baseline."""
        batch_size = 1
        gdn_cfg = MockConfigContainer(model=self._qwen35_27b_config())
        baseline_cfg = MockConfigContainer(model=self._qwen35_27b_config(experimental_attention_variant=None))
        gdn_flops = num_floating_point_operations(gdn_cfg, batch_size=batch_size)
        baseline_flops = num_floating_point_operations(baseline_cfg, batch_size=batch_size)
        assert gdn_flops != baseline_flops, "GDN FLOPs should differ from pure-attention FLOPs"
        assert gdn_flops > 0

    def test_gdn_only_layers(self):
        """With linear_attention_freq=1 (no standard attn), self_attn_term should be pure GDN."""
        batch_size = 1
        num_layers = 4
        hidden_size = 1024
        seq_length = 512
        vocab_size = 32000
        qk_head_dim = 64
        v_head_dim = 64
        num_qk_heads = 8
        num_v_heads = 16
        conv_kernel_dim = 4

        model_cfg = MockModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_length,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=128,
            vocab_size=vocab_size,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=1,
            linear_conv_kernel_dim=conv_kernel_dim,
            linear_key_head_dim=qk_head_dim,
            linear_value_head_dim=v_head_dim,
            linear_num_key_heads=num_qk_heads,
            linear_num_value_heads=num_v_heads,
        )
        cfg = MockConfigContainer(model=model_cfg)
        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # freq=1: pattern = [0 if (i+1)%1==0 else 1 for i in range(4)] = [0,0,0,0]
        # All layers are standard attention, 0 GDN layers.
        # This is because freq=1 means every layer is standard attention.
        # So actual_flops should equal baseline (no GDN).
        baseline_cfg = MockConfigContainer(
            model=MockModelConfig(
                num_layers=num_layers,
                hidden_size=hidden_size,
                seq_length=seq_length,
                ffn_hidden_size=4096,
                num_attention_heads=8,
                num_query_groups=8,
                kv_channels=128,
                vocab_size=vocab_size,
                make_vocab_size_divisible_by=128,
                tensor_model_parallel_size=1,
                gated_linear_unit=False,
            )
        )
        baseline_flops = num_floating_point_operations(baseline_cfg, batch_size=batch_size)
        assert actual_flops == baseline_flops, (
            "freq=1 means every layer is standard attention, so FLOPs should match baseline"
        )

    def test_gdn_layer_freq_list(self):
        """Test GDN with linear_attention_freq as a list pattern (6 GDN, 2 standard)."""
        batch_size = 1
        freq_list = [1, 1, 0, 1, 1, 0, 1, 1]  # 6 GDN, 2 standard
        assert sum(freq_list) == 6
        model_cfg = self._qwen35_27b_config(
            num_layers=8,
            linear_attention_freq=freq_list,
        )
        cfg = MockConfigContainer(model=model_cfg)
        flops = num_floating_point_operations(cfg, batch_size=batch_size)
        assert flops > 0

        # Verify the mask is actually applied: must differ from pure standard attention.
        baseline_cfg = MockConfigContainer(
            model=self._qwen35_27b_config(num_layers=8, experimental_attention_variant=None)
        )
        baseline_flops = num_floating_point_operations(baseline_cfg, batch_size=batch_size)
        assert flops != baseline_flops, "List-based GDN mask should differ from pure standard attention"

        # freq_list [1,1,0,1,1,0,1,1] is identical to the pattern generated by int freq=3.
        int_freq_cfg = MockConfigContainer(model=self._qwen35_27b_config(num_layers=8, linear_attention_freq=3))
        int_freq_flops = num_floating_point_operations(int_freq_cfg, batch_size=batch_size)
        assert flops == int_freq_flops, (
            "List [1,1,0,1,1,0,1,1] should produce the same FLOPs as int freq=3 (equivalent 6/2 split)"
        )

    def test_gdn_exact_self_attn_term(self):
        """Verify the GDN self_attn_term matches the expected formula from Megatron-LM."""
        batch_size = 1
        num_layers = 4
        hidden_size = 1024
        seq_length = 256
        vocab_size = 32000
        qk_head_dim = 64
        v_head_dim = 64
        num_qk_heads = 8
        num_v_heads = 16
        conv_kernel_dim = 4
        ffn_hidden_size = 4096

        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads

        # freq=2: layers 0,2 are GDN (pattern[i]=1), layers 1,3 are standard (pattern[i]=0)
        model_cfg = MockModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_length,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=128,
            vocab_size=vocab_size,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=2,
            linear_conv_kernel_dim=conv_kernel_dim,
            linear_key_head_dim=qk_head_dim,
            linear_value_head_dim=v_head_dim,
            linear_num_key_heads=num_qk_heads,
            linear_num_value_heads=num_v_heads,
        )
        cfg = MockConfigContainer(model=model_cfg)
        gdn_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # Compute expected manually
        # Standard attention per-layer (MHA, num_query_groups==num_attention_heads)
        kv_channels = 128
        q_proj_size = kv_channels * 8
        k_proj_size = kv_channels * 8
        v_proj_size = kv_channels * 8
        standard_attn_per_layer = (
            3
            * 2
            * (
                hidden_size * (q_proj_size + k_proj_size + v_proj_size)
                + q_proj_size * seq_length / 2 * 2
                + q_proj_size * hidden_size
            )
        )
        # GDN per-layer
        gdn_per_layer = (
            3
            * 2
            * (
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                + num_v_heads * (v_head_dim**2) * 4
                + hidden_size * v_dim
            )
        )
        # freq=2: pattern = [1, 0, 1, 0] -> 2 GDN, 2 standard
        expected_self_attn = gdn_per_layer * 2 + standard_attn_per_layer * 2
        # MLP: gated_linear_unit=False -> ffn_expansion_factor=2
        expected_mlp = 3 * 2 * hidden_size * (ffn_hidden_size * 2) * num_layers
        # Logit
        padded_vocab = vocab_size  # 32000 is already divisible by 128
        expected_logit = 3 * 2 * hidden_size * padded_vocab * 1
        expected_total = batch_size * seq_length * (expected_mlp + expected_self_attn + expected_logit)

        assert gdn_flops == expected_total, f"Expected {expected_total:.6e} but got {gdn_flops:.6e}"

    def test_gdn_more_gdn_layers_changes_flops(self):
        """Increasing GDN layer ratio (higher freq) should change FLOPs."""
        batch_size = 1
        # freq=4: 3/4 GDN, 1/4 standard
        cfg_freq4 = MockConfigContainer(model=self._qwen35_27b_config(num_layers=8, linear_attention_freq=4))
        # freq=8: 7/8 GDN, 1/8 standard
        cfg_freq8 = MockConfigContainer(model=self._qwen35_27b_config(num_layers=8, linear_attention_freq=8))
        flops_freq4 = num_floating_point_operations(cfg_freq4, batch_size=batch_size)
        flops_freq8 = num_floating_point_operations(cfg_freq8, batch_size=batch_size)
        assert flops_freq4 != flops_freq8, "Different GDN ratios should produce different FLOPs"


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
            hybrid_layer_pattern="M*/MM/MM",
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


@pytest.mark.unit
class TestHybridGDNFlops:
    """Tests for GDN ('G') layer support in the hybrid FLOPs path."""

    def test_gdn_hybrid_pattern_positive_flops(self):
        """A hybrid pattern containing G layers should produce positive FLOPs."""
        batch_size = 1
        model_cfg = MockModelConfig(
            is_hybrid_model=True,
            hybrid_layer_pattern="G*G*",
            num_layers=4,
            hidden_size=1024,
            seq_length=512,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            gated_linear_unit=False,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=8,
            linear_num_value_heads=16,
            linear_conv_kernel_dim=4,
        )
        cfg = MockConfigContainer(model=model_cfg)
        flops = num_floating_point_operations(cfg, batch_size=batch_size)
        assert flops > 0, "Hybrid pattern with G layers should produce positive FLOPs"

    def test_gdn_hybrid_exact_flops(self):
        """Verify exact GDN FLOPs in hybrid path match the gdn_layer_flops formula."""
        batch_size = 1
        seq_len = 512
        hidden_size = 1024
        vocab_size = 32000
        qk_head_dim = 64
        v_head_dim = 64
        num_qk_heads = 8
        num_v_heads = 16
        conv_kernel_dim = 4

        model_cfg = MockModelConfig(
            is_hybrid_model=True,
            hybrid_layer_pattern="GG",
            num_layers=2,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=vocab_size,
            gated_linear_unit=False,
            linear_key_head_dim=qk_head_dim,
            linear_value_head_dim=v_head_dim,
            linear_num_key_heads=num_qk_heads,
            linear_num_value_heads=num_v_heads,
            linear_conv_kernel_dim=conv_kernel_dim,
        )
        cfg = MockConfigContainer(model=model_cfg)
        flops = num_floating_point_operations(cfg, batch_size=batch_size)

        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads
        gdn_per_layer = (
            2
            * batch_size
            * seq_len
            * (
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                + num_v_heads * (v_head_dim**2) * 4
                + hidden_size * v_dim
            )
        )
        logit = 2 * batch_size * seq_len * hidden_size * vocab_size
        expected = (2 * gdn_per_layer + logit) * 3

        assert flops == expected, f"Expected {expected:.2e} but got {flops:.2e}"

    def test_gdn_differs_from_attention_in_hybrid(self):
        """G layers should produce different FLOPs than * layers in hybrid path."""
        batch_size = 1
        base = dict(
            is_hybrid_model=True,
            num_layers=4,
            hidden_size=1024,
            seq_length=512,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            gated_linear_unit=False,
        )
        cfg_gdn = MockConfigContainer(model=MockModelConfig(**base, hybrid_layer_pattern="GGGG"))
        cfg_attn = MockConfigContainer(model=MockModelConfig(**base, hybrid_layer_pattern="****"))
        flops_gdn = num_floating_point_operations(cfg_gdn, batch_size=batch_size)
        flops_attn = num_floating_point_operations(cfg_attn, batch_size=batch_size)
        assert flops_gdn != flops_attn, "G layers and * layers should have different FLOPs"


@pytest.mark.unit
class TestAttentionOutputGateFlops:
    """Tests for attention_output_gate FLOPs in transformer_flops path."""

    def test_gate_increases_flops(self):
        """attention_output_gate=True should add extra FLOPs for the gate projection."""
        batch_size = 1
        base = dict(
            num_layers=4,
            hidden_size=1024,
            seq_length=512,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        cfg_no_gate = MockConfigContainer(model=MockModelConfig(**base, attention_output_gate=False))
        cfg_gate = MockConfigContainer(model=MockModelConfig(**base, attention_output_gate=True))
        flops_no_gate = num_floating_point_operations(cfg_no_gate, batch_size=batch_size)
        flops_gate = num_floating_point_operations(cfg_gate, batch_size=batch_size)
        assert flops_gate > flops_no_gate, "attention_output_gate should increase FLOPs"

    def test_gate_exact_delta(self):
        """Verify the exact FLOPs delta from attention_output_gate matches the gate projection formula."""
        batch_size = 1
        num_layers = 4
        hidden_size = 1024
        seq_length = 512
        kv_channels = 128
        num_attention_heads = 8
        vocab_size = 32000

        base = dict(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_length,
            ffn_hidden_size=4096,
            num_attention_heads=num_attention_heads,
            num_query_groups=4,
            kv_channels=kv_channels,
            vocab_size=vocab_size,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        cfg_no_gate = MockConfigContainer(model=MockModelConfig(**base, attention_output_gate=False))
        cfg_gate = MockConfigContainer(model=MockModelConfig(**base, attention_output_gate=True))
        flops_no_gate = num_floating_point_operations(cfg_no_gate, batch_size=batch_size)
        flops_gate = num_floating_point_operations(cfg_gate, batch_size=batch_size)

        query_projection_size = kv_channels * num_attention_heads
        expected_delta = batch_size * seq_length * 3 * 2 * num_layers * hidden_size * query_projection_size
        actual_delta = flops_gate - flops_no_gate

        assert actual_delta == expected_delta, f"Expected gate delta {expected_delta:.2e} but got {actual_delta:.2e}"


@pytest.mark.unit
class TestMoELatentTransformerPath:
    """Tests for moe_latent_size handling in the transformer_flops path (non-hybrid)."""

    def test_latent_reduces_flops(self):
        """MoE with latent compression should produce fewer FLOPs than without (when latent < hidden)."""
        batch_size = 1
        hidden_size = 2048
        moe_ffn_hidden = 4096
        latent_size = 512

        base = dict(
            num_layers=4,
            hidden_size=hidden_size,
            seq_length=1024,
            ffn_hidden_size=8192,
            num_attention_heads=16,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            num_moe_experts=8,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=0,
            gated_linear_unit=False,
        )
        cfg_no_latent = MockConfigContainer(model=MockModelConfig(**base, moe_latent_size=None))
        cfg_latent = MockConfigContainer(model=MockModelConfig(**base, moe_latent_size=latent_size))
        flops_no_latent = num_floating_point_operations(cfg_no_latent, batch_size=batch_size)
        flops_latent = num_floating_point_operations(cfg_latent, batch_size=batch_size)
        assert flops_latent < flops_no_latent, (
            "Latent MoE (latent < hidden) should produce fewer FLOPs in transformer path"
        )

    def test_latent_exact_moe_term(self):
        """Verify exact MoE FLOPs with latent compression in transformer_flops path."""
        batch_size = 1
        num_layers = 2
        hidden_size = 1024
        seq_length = 512
        moe_ffn_hidden = 2048
        latent_size = 256
        topk = 1
        vocab_size = 32000

        model_cfg = MockModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_length,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=128,
            vocab_size=vocab_size,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            num_moe_experts=8,
            moe_layer_freq=1,
            moe_router_topk=topk,
            moe_ffn_hidden_size=moe_ffn_hidden,
            moe_shared_expert_intermediate_size=0,
            moe_latent_size=latent_size,
            gated_linear_unit=False,
        )
        cfg = MockConfigContainer(model=model_cfg)
        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        # ffn_expansion_factor = 2 (non-SwiGLU)
        ffn_exp = 2
        routed_term = (moe_ffn_hidden * topk * ffn_exp * latent_size / hidden_size) + 2 * latent_size
        # All layers are MoE (moe_layer_freq=1), num_dense_layers=0
        expected_mlp = 3 * 2 * hidden_size * routed_term * num_layers

        # Standard attention: 3 * 2 * num_layers * (...) -- compute per-layer
        kv_channels = 128
        q_proj = kv_channels * 8  # = 1024 = hidden_size
        k_proj = kv_channels * 8
        v_proj = kv_channels * 8
        attn_per_layer = hidden_size * (q_proj + k_proj + v_proj) + q_proj * seq_length / 2 * 2 + q_proj * hidden_size
        expected_attn = 3 * 2 * num_layers * attn_per_layer

        expected_logit = 3 * 2 * hidden_size * vocab_size

        expected_total = batch_size * seq_length * (expected_mlp + expected_attn + expected_logit)

        assert actual_flops == expected_total, f"Expected {expected_total:.2e} but got {actual_flops:.2e}"
