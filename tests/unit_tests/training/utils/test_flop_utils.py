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
import torch

from megatron.bridge.training.utils.flop_utils import (
    _lora_seq_stats_cache,
    _packed_data_exists,
    accumulate_flops_metadata,
    num_floating_point_operations,
    resolve_global_flops_seqlen_stats,
    vit_flops,
)


@dataclass
class MockVisionConfig:
    """Mock ViT vision config for testing vit_flops."""

    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096


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
    # MLA (Multi-Latent Attention) settings — DeepSeek-V2/V3 style
    q_lora_rank: int | None = None
    kv_lora_rank: int = 0
    qk_head_dim: int = 64
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 64
    o_lora_rank: int = 0
    o_groups: int = 1
    # Sliding window attention settings
    window_size: tuple | list | int | None = None
    window_attn_skip_freq: int | list | None = None
    # DeepSeek-V4 hybrid attention settings
    csa_compress_ratios: list[int] | None = None
    csa_window_size: int = 128
    dsa_indexer_n_heads: int | None = None
    dsa_indexer_head_dim: int | None = None
    dsa_indexer_topk: int | None = None
    # GDN (Gated DeltaNet) settings
    experimental_attention_variant: str | None = None
    linear_attention_freq: int | list | None = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    # Optional ViT vision config (for VLM FLOPS tests)
    vision_config: object | None = None

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

    def test_hybrid_attention_term_scales_with_seqlen_squared_sum(self):
        """Hybrid attention core FLOPs must use Σ L², not only average sequence length."""
        batch_size = 4
        seq_len = 1024
        hidden_size = 1024
        num_heads = 8
        kv_channels = 128
        cfg = MockConfigContainer(
            model=MockModelConfig(
                is_hybrid_model=True,
                hybrid_layer_pattern="*",
                num_layers=1,
                hidden_size=hidden_size,
                seq_length=seq_len,
                ffn_hidden_size=4096,
                num_attention_heads=num_heads,
                num_query_groups=num_heads,
                kv_channels=kv_channels,
                vocab_size=32000,
                gated_linear_unit=False,
            )
        )
        seqlen_sum = batch_size * seq_len
        sq_base = batch_size * seq_len**2
        sq_bumped = sq_base + 1_000_000

        flops_base = num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=sq_base,
        )
        flops_bumped = num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=sq_bumped,
        )

        query_projection_size = kv_channels * num_heads
        expected_delta = 3 * 2 * query_projection_size * (sq_bumped - sq_base)
        assert flops_bumped - flops_base == expected_delta


@pytest.mark.unit
class TestHybridTransformerParity:
    """Parity tests for equivalent logical transformer and physical hybrid patterns."""

    @staticmethod
    def _base_config(**overrides):
        config = dict(
            num_layers=4,
            hidden_size=512,
            seq_length=256,
            ffn_hidden_size=1024,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=64,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=True,
            mtp_num_layers=0,
        )
        config.update(overrides)
        return config

    def test_dense_physical_pattern_matches_transformer(self):
        """A physical attention+MLP stack should match the equivalent dense transformer."""
        batch_size = 2
        transformer_cfg = MockConfigContainer(model=MockModelConfig(**self._base_config()))
        hybrid_cfg = MockConfigContainer(
            model=MockModelConfig(
                **self._base_config(
                    is_hybrid_model=True,
                    num_layers=8,
                    hybrid_layer_pattern="*-" * 4,
                )
            )
        )

        transformer_flops = num_floating_point_operations(transformer_cfg, batch_size=batch_size)
        hybrid_flops = num_floating_point_operations(hybrid_cfg, batch_size=batch_size)

        assert hybrid_flops == pytest.approx(transformer_flops)

    def test_moe_physical_pattern_matches_transformer(self):
        """GPT-OSS-style attention+MoE physical layers should not double-count FLOPs."""
        batch_size = 2
        moe_config = self._base_config(
            num_moe_experts=8,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_ffn_hidden_size=1024,
            moe_shared_expert_intermediate_size=256,
        )
        transformer_cfg = MockConfigContainer(model=MockModelConfig(**moe_config))
        hybrid_cfg = MockConfigContainer(
            model=MockModelConfig(
                **(
                    moe_config
                    | {
                        "is_hybrid_model": True,
                        "num_layers": 8,
                        "hybrid_layer_pattern": "*E" * 4,
                    }
                )
            )
        )

        transformer_flops = num_floating_point_operations(transformer_cfg, batch_size=batch_size)
        hybrid_flops = num_floating_point_operations(hybrid_cfg, batch_size=batch_size)

        assert hybrid_flops == pytest.approx(transformer_flops)

    def test_hybrid_pattern_routes_to_hybrid_flop_path_without_flag(self):
        """A hybrid_layer_pattern should be enough to select hybrid FLOP accounting."""
        batch_size = 2
        moe_config = self._base_config(
            num_moe_experts=8,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_ffn_hidden_size=1024,
            moe_shared_expert_intermediate_size=256,
        )
        transformer_cfg = MockConfigContainer(model=MockModelConfig(**moe_config))
        hybrid_cfg = MockConfigContainer(
            model=MockModelConfig(
                **(
                    moe_config
                    | {
                        "num_layers": 8,
                        "hybrid_layer_pattern": "*E" * 4,
                    }
                )
            )
        )

        transformer_flops = num_floating_point_operations(transformer_cfg, batch_size=batch_size)
        hybrid_flops = num_floating_point_operations(hybrid_cfg, batch_size=batch_size)

        assert hybrid_flops == pytest.approx(transformer_flops)

    def test_mixed_dense_and_moe_physical_pattern_matches_transformer(self):
        """Hybrid FLOPs should match transformer FLOPs for mixed dense/MoE logical layers."""
        batch_size = 2
        mixed_config = self._base_config(
            num_moe_experts=8,
            moe_layer_freq=[0, 1, 0, 1],
            moe_router_topk=2,
            moe_ffn_hidden_size=1024,
            moe_shared_expert_intermediate_size=256,
        )
        transformer_cfg = MockConfigContainer(model=MockModelConfig(**mixed_config))
        hybrid_cfg = MockConfigContainer(
            model=MockModelConfig(
                **(
                    mixed_config
                    | {
                        "is_hybrid_model": True,
                        "num_layers": 8,
                        "hybrid_layer_pattern": "*-*E*-*E",
                    }
                )
            )
        )

        transformer_flops = num_floating_point_operations(transformer_cfg, batch_size=batch_size)
        hybrid_flops = num_floating_point_operations(hybrid_cfg, batch_size=batch_size)

        assert hybrid_flops == pytest.approx(transformer_flops)

    def test_sliding_window_physical_pattern_matches_transformer(self):
        """Hybrid SWA counting should use physical layer positions in window_attn_skip_freq."""
        batch_size = 2
        moe_config = self._base_config(
            seq_length=1024,
            num_moe_experts=8,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_ffn_hidden_size=1024,
            moe_shared_expert_intermediate_size=256,
            window_size=(127, 0),
        )
        transformer_cfg = MockConfigContainer(
            model=MockModelConfig(**(moe_config | {"window_attn_skip_freq": [1, 0, 1, 0]}))
        )
        hybrid_cfg = MockConfigContainer(
            model=MockModelConfig(
                **(
                    moe_config
                    | {
                        "is_hybrid_model": True,
                        "num_layers": 8,
                        "hybrid_layer_pattern": "*E" * 4,
                        "window_attn_skip_freq": [1, 0, 0, 0, 1, 0, 0, 0],
                    }
                )
            )
        )

        transformer_flops = num_floating_point_operations(transformer_cfg, batch_size=batch_size)
        hybrid_flops = num_floating_point_operations(hybrid_cfg, batch_size=batch_size)

        assert hybrid_flops == pytest.approx(transformer_flops)

    def test_dynamic_sequence_lengths_match_transformer(self):
        """Hybrid attention core FLOPs should use seqlen_squared_sum like transformer FLOPs."""
        batch_size = 2
        seqlen_sum = 128 + 640
        seqlen_squared_sum = 128**2 + 640**2
        transformer_cfg = MockConfigContainer(model=MockModelConfig(**self._base_config(num_layers=2)))
        hybrid_cfg = MockConfigContainer(
            model=MockModelConfig(
                **self._base_config(
                    is_hybrid_model=True,
                    num_layers=4,
                    hybrid_layer_pattern="*-" * 2,
                )
            )
        )

        transformer_flops = num_floating_point_operations(
            transformer_cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=seqlen_squared_sum,
        )
        hybrid_flops = num_floating_point_operations(
            hybrid_cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=seqlen_squared_sum,
        )

        assert hybrid_flops == pytest.approx(transformer_flops)


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

    @pytest.mark.parametrize(
        ("linear_attention_freq", "last_layer_is_gdn"),
        [
            pytest.param(4, False, id="int-last-standard"),
            pytest.param([1, 1, 1, 0], False, id="list-last-standard"),
            pytest.param(3, True, id="int-last-gdn"),
            pytest.param([1, 1, 0, 1], True, id="list-last-gdn"),
        ],
    )
    def test_gdn_mtp_reuses_last_decoder_attention_type(self, linear_attention_freq, last_layer_is_gdn):
        """Every MTP depth should reuse the final decoder layer's attention type."""
        batch_size = 1
        hidden_size = 128
        seq_length = 32
        ffn_hidden_size = 256
        num_attention_heads = 4
        num_query_groups = 2
        kv_channels = 32
        vocab_size = 1024
        mtp_num_layers = 2
        linear_key_head_dim = 16
        linear_value_head_dim = 16
        linear_num_key_heads = 4
        linear_num_value_heads = 8
        linear_conv_kernel_dim = 4
        model_overrides = {
            "num_layers": 4,
            "hidden_size": hidden_size,
            "seq_length": seq_length,
            "ffn_hidden_size": ffn_hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_query_groups": num_query_groups,
            "kv_channels": kv_channels,
            "vocab_size": vocab_size,
            "gated_linear_unit": False,
            "linear_attention_freq": linear_attention_freq,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_value_head_dim": linear_value_head_dim,
            "linear_num_key_heads": linear_num_key_heads,
            "linear_num_value_heads": linear_num_value_heads,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
        }
        base_cfg = MockConfigContainer(model=self._qwen35_27b_config(**model_overrides))
        mtp_cfg = MockConfigContainer(model=self._qwen35_27b_config(**model_overrides, mtp_num_layers=mtp_num_layers))

        query_projection_size = kv_channels * num_attention_heads
        key_value_projection_size = kv_channels * num_query_groups
        dense_mlp = 3 * 2 * hidden_size * (ffn_hidden_size * 2)
        standard_attention = (
            3
            * 2
            * (
                hidden_size * (query_projection_size + 2 * key_value_projection_size)
                + query_projection_size * hidden_size
                + query_projection_size * seq_length
            )
        )
        qk_dim = linear_key_head_dim * linear_num_key_heads
        v_dim = linear_value_head_dim * linear_num_value_heads
        gdn_attention = (
            3
            * 2
            * (
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * linear_num_value_heads)
                + linear_conv_kernel_dim * (2 * qk_dim + v_dim)
                + linear_num_value_heads * linear_value_head_dim**2 * 4
                + hidden_size * v_dim
            )
        )
        mtp_aux = 3 * 2 * (3 * hidden_size + 2 * hidden_size**2)
        extra_logits = 3 * 2 * hidden_size * vocab_size
        last_attention = gdn_attention if last_layer_is_gdn else standard_attention
        expected_delta = (
            batch_size * seq_length * mtp_num_layers * (dense_mlp + last_attention + mtp_aux + extra_logits)
        )

        actual_delta = num_floating_point_operations(mtp_cfg, batch_size=batch_size) - num_floating_point_operations(
            base_cfg, batch_size=batch_size
        )

        assert actual_delta == pytest.approx(expected_delta)

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


class TestDeepSeekV4HybridFlops:
    """Tests for DeepSeek-V4 hybrid attention FLOPs in the transformer path."""

    def test_dsv4_hybrid_exact_flops(self):
        """DSv4 hybrid FLOPs include sparse attention, compressor, and indexer terms."""
        batch_size = 1
        seq_len = 512
        hidden_size = 128
        num_layers = 3
        num_heads = 4
        v_head_dim = 32
        q_lora_rank = 16
        qk_head_dim = 24
        qk_pos_emb_head_dim = 8
        o_lora_rank = 16
        o_groups = 2
        window = 64
        idx_n_heads = 2
        idx_head_dim = 8
        idx_topk = 32
        ffn_hidden_size = 256
        vocab_size = 1024

        model_cfg = MockModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            q_lora_rank=q_lora_rank,
            qk_head_dim=qk_head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
            csa_compress_ratios=[0, 4, 128],
            csa_window_size=window,
            dsa_indexer_n_heads=idx_n_heads,
            dsa_indexer_head_dim=idx_head_dim,
            dsa_indexer_topk=idx_topk,
            gated_linear_unit=False,
        )
        cfg = MockConfigContainer(model=model_cfg)

        q_term = q_lora_rank * (hidden_size + num_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1)
        kv_term = hidden_size * v_head_dim + v_head_dim
        o_term = num_heads * v_head_dim * o_lora_rank + o_groups * o_lora_rank * hidden_size
        projection_term = 3 * 2 * num_layers * (q_term + kv_term + o_term)

        sparse_attn_r0 = num_heads * window * v_head_dim * 2
        sparse_attn_r128 = num_heads * (window + (seq_len // 128) / 2) * v_head_dim * 2
        effective_topk = min(idx_topk, seq_len // 4)
        avg_comp_4 = effective_topk * (1 - effective_topk * 4 / (2 * seq_len))
        sparse_attn_r4 = num_heads * (window + avg_comp_4) * v_head_dim * 2
        main_compressor_term = hidden_size * (2 * v_head_dim) * 2 + hidden_size * v_head_dim * 2
        indexer_term = (
            hidden_size * (2 * idx_head_dim) * 2
            + q_lora_rank * idx_n_heads * idx_head_dim
            + hidden_size * idx_n_heads
            + idx_n_heads * idx_head_dim * (seq_len // 4)
        )
        dsv4_extra_term = (
            3 * 2 * (sparse_attn_r0 + sparse_attn_r4 + sparse_attn_r128 + main_compressor_term + indexer_term)
        )
        self_attention_term = projection_term + dsv4_extra_term

        mlp_term = 3 * 2 * hidden_size * (ffn_hidden_size * 2 * num_layers)
        logit_term = 3 * 2 * hidden_size * vocab_size
        expected_flops = batch_size * seq_len * (mlp_term + self_attention_term + logit_term)

        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        assert actual_flops == expected_flops

    def test_dsv4_hybrid_validates_compress_ratio_length(self):
        """CSA compress-ratio count must match decoder plus MTP layer count."""
        model_cfg = MockModelConfig(
            num_layers=2,
            mtp_num_layers=1,
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            q_lora_rank=16,
            o_lora_rank=16,
            csa_compress_ratios=[0, 4],
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=32,
        )
        cfg = MockConfigContainer(model=model_cfg)

        with pytest.raises(ValueError, match=r"expected 3 \(num_layers=2, mtp_num_layers=1\)"):
            num_floating_point_operations(cfg, batch_size=1)

    def test_dsv4_hybrid_validates_supported_compress_ratios(self):
        """CSA compress ratios must be recognized before layer counts are used."""
        model_cfg = MockModelConfig(
            num_layers=3,
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            q_lora_rank=16,
            o_lora_rank=16,
            csa_compress_ratios=[0, 8, 128],
            dsa_indexer_n_heads=2,
            dsa_indexer_head_dim=8,
            dsa_indexer_topk=32,
        )
        cfg = MockConfigContainer(model=model_cfg)

        with pytest.raises(ValueError, match=r"unsupported values: \[8\]"):
            num_floating_point_operations(cfg, batch_size=1)

    def test_dsv4_hybrid_requires_q_lora_rank(self):
        """DSv4 hybrid FLOPs require q_lora_rank for projection accounting."""
        model_cfg = MockModelConfig(
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            q_lora_rank=None,
            csa_compress_ratios=[0] * 24,
        )
        cfg = MockConfigContainer(model=model_cfg)

        with pytest.raises(ValueError, match="q_lora_rank must be set"):
            num_floating_point_operations(cfg, batch_size=1)

    def test_dsv4_hybrid_requires_compress_ratios(self):
        """DSv4 hybrid FLOPs require per-layer CSA compress ratios."""
        model_cfg = MockModelConfig(
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            q_lora_rank=16,
            csa_compress_ratios=None,
        )
        cfg = MockConfigContainer(model=model_cfg)

        with pytest.raises(ValueError, match="csa_compress_ratios must be set"):
            num_floating_point_operations(cfg, batch_size=1)

    def test_dsv4_hybrid_without_ratio4_layers_skips_indexer_terms(self):
        """DSv4 hybrid FLOPs support CSA patterns without DSA-indexed ratio-4 layers."""
        batch_size = 1
        seq_len = 512
        hidden_size = 128
        num_layers = 2
        num_heads = 4
        v_head_dim = 32
        q_lora_rank = 16
        qk_head_dim = 24
        qk_pos_emb_head_dim = 8
        o_lora_rank = 16
        o_groups = 2
        window = 64
        ffn_hidden_size = 256
        vocab_size = 1024

        model_cfg = MockModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_len,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            q_lora_rank=q_lora_rank,
            qk_head_dim=qk_head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
            csa_compress_ratios=[0, 128],
            csa_window_size=window,
            gated_linear_unit=False,
        )
        cfg = MockConfigContainer(model=model_cfg)

        q_term = q_lora_rank * (hidden_size + num_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1)
        kv_term = hidden_size * v_head_dim + v_head_dim
        o_term = num_heads * v_head_dim * o_lora_rank + o_groups * o_lora_rank * hidden_size
        projection_term = 3 * 2 * num_layers * (q_term + kv_term + o_term)

        sparse_attn_r0 = num_heads * window * v_head_dim * 2
        sparse_attn_r128 = num_heads * (window + (seq_len // 128) / 2) * v_head_dim * 2
        main_compressor_term = hidden_size * v_head_dim * 2
        dsv4_extra_term = 3 * 2 * (sparse_attn_r0 + sparse_attn_r128 + main_compressor_term)
        self_attention_term = projection_term + dsv4_extra_term

        mlp_term = 3 * 2 * hidden_size * (ffn_hidden_size * 2 * num_layers)
        logit_term = 3 * 2 * hidden_size * vocab_size
        expected_flops = batch_size * seq_len * (mlp_term + self_attention_term + logit_term)

        actual_flops = num_floating_point_operations(cfg, batch_size=batch_size)

        assert actual_flops == expected_flops

    @pytest.mark.parametrize(
        ("missing_field", "error_message"),
        [
            ("dsa_indexer_n_heads", "dsa_indexer_n_heads must be set"),
            ("dsa_indexer_head_dim", "dsa_indexer_head_dim must be set"),
            ("dsa_indexer_topk", "dsa_indexer_topk must be set"),
        ],
    )
    def test_dsv4_hybrid_ratio4_requires_indexer_config(self, missing_field, error_message):
        """DSv4 hybrid ratio-4 layers require DSA indexer config for FLOPs accounting."""
        model_kwargs = {
            "num_layers": 3,
            "multi_latent_attention": True,
            "experimental_attention_variant": "dsv4_hybrid",
            "q_lora_rank": 16,
            "o_lora_rank": 16,
            "csa_compress_ratios": [0, 4, 128],
            "dsa_indexer_n_heads": 2,
            "dsa_indexer_head_dim": 8,
            "dsa_indexer_topk": 32,
        }
        model_kwargs[missing_field] = None
        model_cfg = MockModelConfig(**model_kwargs)
        cfg = MockConfigContainer(model=model_cfg)

        with pytest.raises(ValueError, match=error_message):
            num_floating_point_operations(cfg, batch_size=1)


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
        with patch("megatron.bridge.training.utils.flop_utils.parse_hybrid_pattern", return_value=parsed_pattern):
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


@pytest.mark.unit
class TestGluFfnExpansion:
    """Tests that gated_linear_unit uses ffn_expansion_factor=3 for any GLU activation."""

    def test_glu_non_silu_uses_expansion_factor_three(self):
        """GeGLU and SwiGLU both count three FFN projections; activation_func is not in the formula."""
        import torch.nn.functional as F

        batch_size = 1
        seq_length = 512
        hidden_size = 1024
        ffn_hidden_size = 4096
        num_layers = 4
        vocab_size = 32000

        base = dict(
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=seq_length,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=vocab_size,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

        cfg_no_glu = MockConfigContainer(model=MockModelConfig(**base, gated_linear_unit=False))
        cfg_glu_gelu = MockConfigContainer(
            model=MockModelConfig(**base, gated_linear_unit=True, activation_func=F.gelu)
        )
        cfg_glu_silu = MockConfigContainer(
            model=MockModelConfig(**base, gated_linear_unit=True, activation_func=F.silu)
        )

        flops_no_glu = num_floating_point_operations(cfg_no_glu, batch_size=batch_size)
        flops_glu_gelu = num_floating_point_operations(cfg_glu_gelu, batch_size=batch_size)
        flops_glu_silu = num_floating_point_operations(cfg_glu_silu, batch_size=batch_size)

        assert flops_glu_gelu == flops_glu_silu, "GLU FLOPs must not depend on activation_func"

        # Only the dense MLP term changes: expansion 3 vs 2 (attn + logit are identical).
        expected_mlp_delta = batch_size * seq_length * 3 * 2 * hidden_size * ffn_hidden_size * num_layers
        assert flops_glu_gelu - flops_no_glu == expected_mlp_delta
        assert flops_glu_silu - flops_no_glu == expected_mlp_delta


@pytest.mark.unit
class TestSlidingWindowAttentionFlops:
    """Tests for sliding window attention (SWA) FLOPs in transformer_flops path."""

    def test_swa_reduces_flops(self):
        """SWA layers should produce fewer FLOPs than full attention when window < seq_length."""
        batch_size = 1
        base = dict(
            num_layers=8,
            hidden_size=1024,
            seq_length=4096,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        cfg_full = MockConfigContainer(model=MockModelConfig(**base))
        cfg_swa = MockConfigContainer(model=MockModelConfig(**base, window_size=(511, 0), window_attn_skip_freq=2))
        flops_full = num_floating_point_operations(cfg_full, batch_size=batch_size)
        flops_swa = num_floating_point_operations(cfg_swa, batch_size=batch_size)
        assert flops_swa < flops_full, "SWA should reduce FLOPs when window < seq_length"

    def test_swa_no_effect_when_window_ge_seq(self):
        """SWA should have no effect when effective window >= seq_length."""
        batch_size = 1
        seq_length = 512
        base = dict(
            num_layers=4,
            hidden_size=1024,
            seq_length=seq_length,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        cfg_full = MockConfigContainer(model=MockModelConfig(**base))
        cfg_swa = MockConfigContainer(
            model=MockModelConfig(**base, window_size=(seq_length, 0), window_attn_skip_freq=2)
        )
        flops_full = num_floating_point_operations(cfg_full, batch_size=batch_size)
        flops_swa = num_floating_point_operations(cfg_swa, batch_size=batch_size)
        assert flops_swa == flops_full, "SWA with window >= seq should equal full attention FLOPs"

    def test_swa_exact_delta(self):
        """Verify the exact FLOPs reduction from SWA matches the core attention formula difference."""
        batch_size = 1
        num_layers = 4
        hidden_size = 1024
        seq_length = 4096
        kv_channels = 128
        num_attention_heads = 8
        window_left = 511
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
        cfg_full = MockConfigContainer(model=MockModelConfig(**base))
        cfg_swa = MockConfigContainer(
            model=MockModelConfig(**base, window_size=(window_left, 0), window_attn_skip_freq=2)
        )
        flops_full = num_floating_point_operations(cfg_full, batch_size=batch_size)
        flops_swa = num_floating_point_operations(cfg_swa, batch_size=batch_size)

        # skip_freq=2: layers [0,2] are SWA, layers [1,3] are full → 2 SWA layers
        num_swa_layers = 2
        query_projection_size = kv_channels * num_attention_heads
        effective_window = window_left + 0 + 1  # 512

        # Causal SWA: query at position t attends to min(t, W) keys.
        # Sum over t=1..T, then average (flop_utils divides by T; *T happens in seq_length).
        total_swa_keys = (
            effective_window * (effective_window + 1) / 2 + (seq_length - effective_window) * effective_window
        )
        avg_swa_keys = total_swa_keys / seq_length
        core_diff_per_layer = query_projection_size * (seq_length - 2 * avg_swa_keys)
        expected_delta = batch_size * seq_length * 3 * 2 * num_swa_layers * core_diff_per_layer
        actual_delta = flops_full - flops_swa

        assert actual_delta == expected_delta, f"Expected SWA delta {expected_delta:.2e} but got {actual_delta:.2e}"

    def test_swa_list_pattern(self):
        """Test SWA with a list pattern for window_attn_skip_freq."""
        batch_size = 1
        base = dict(
            num_layers=4,
            hidden_size=1024,
            seq_length=4096,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        # List [1,1,0,1] means 3 SWA layers, 1 full layer
        cfg_list = MockConfigContainer(
            model=MockModelConfig(**base, window_size=(511, 0), window_attn_skip_freq=[1, 1, 0, 1])
        )
        # Int freq=4 gives pattern [1,1,1,0] → 3 SWA, 1 full (same counts, different order)
        cfg_int = MockConfigContainer(model=MockModelConfig(**base, window_size=(511, 0), window_attn_skip_freq=4))
        flops_list = num_floating_point_operations(cfg_list, batch_size=batch_size)
        flops_int = num_floating_point_operations(cfg_int, batch_size=batch_size)
        assert flops_list == flops_int, "Same SWA/full split should produce same FLOPs regardless of order"

    def test_swa_all_layers_when_skip_freq_none(self):
        """When window_size is set but window_attn_skip_freq is None, all layers should be SWA."""
        batch_size = 1
        base = dict(
            num_layers=4,
            hidden_size=1024,
            seq_length=4096,
            ffn_hidden_size=4096,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=128,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        cfg_no_window = MockConfigContainer(model=MockModelConfig(**base))
        cfg_all_swa = MockConfigContainer(
            model=MockModelConfig(**base, window_size=(511, 0), window_attn_skip_freq=None)
        )
        flops_full = num_floating_point_operations(cfg_no_window, batch_size=batch_size)
        flops_all_swa = num_floating_point_operations(cfg_all_swa, batch_size=batch_size)
        assert flops_all_swa < flops_full, (
            "window_size set with skip_freq=None should make all layers SWA (fewer FLOPs)"
        )


class TestVitFlops:
    """Unit tests for the `vit_flops` helper (reviewer-requested config-based signature)."""

    @staticmethod
    def _base_cfg(**vision_overrides):
        vision = MockVisionConfig(**vision_overrides)
        return MockConfigContainer(model=MockModelConfig(vision_config=vision))

    def test_vit_flops_returns_zero_without_vision_config(self):
        """When the model has no attached vision_config, vit_flops must return 0.

        This lets callers unconditionally invoke the helper for LLM-only models
        without special-casing VLM detection at the call site.
        """
        cfg = MockConfigContainer(model=MockModelConfig(vision_config=None))
        assert vit_flops(cfg, batch_size=2, num_patches=256) == 0

    def test_vit_flops_returns_zero_for_non_positive_patches(self):
        """Non-positive patch counts should short-circuit to 0 regardless of config."""
        cfg = self._base_cfg()
        assert vit_flops(cfg, batch_size=1, num_patches=0) == 0
        assert vit_flops(cfg, batch_size=1, num_patches=-5) == 0

    def test_vit_flops_matches_closed_form(self):
        """Hand-computed closed form should match the helper output exactly.

        Per-token per-layer cost:
            8*h^2 (QKVO) + 4*h*num_patches (core attn) + 4*h*intermediate (MLP)
        Transformer total: per_token_per_layer * num_patches * depth
        Patch merger (spatial_merge=2 => merge_unit=4):
            num_merged * (2*(4h)^2 + 2*(4h)*out_h)
        Training multiplier: * batch_size * 3 (fwd + bwd)
        """
        depth, h, inter, spatial, out_h = 4, 256, 1024, 2, 2048
        num_patches = 16  # divisible by spatial_merge_size**2 = 4
        batch_size = 2
        cfg = self._base_cfg(
            depth=depth,
            hidden_size=h,
            intermediate_size=inter,
            spatial_merge_size=spatial,
            out_hidden_size=out_h,
        )

        per_token_per_layer = 8 * h**2 + 4 * h * num_patches + 4 * h * inter
        transformer_total = per_token_per_layer * num_patches * depth
        merge_unit = spatial**2
        merged_hidden = h * merge_unit
        num_merged = num_patches // merge_unit
        merger_total = num_merged * (2 * merged_hidden * merged_hidden + 2 * merged_hidden * out_h)
        expected = (transformer_total + merger_total) * batch_size * 3

        assert vit_flops(cfg, batch_size=batch_size, num_patches=num_patches) == expected

    def test_vit_flops_scales_linearly_with_batch_size(self):
        """Doubling batch_size should double the returned FLOPS (per-image attn is fixed)."""
        cfg = self._base_cfg()
        f1 = vit_flops(cfg, batch_size=1, num_patches=64)
        f2 = vit_flops(cfg, batch_size=2, num_patches=64)
        assert f2 == 2 * f1

    def test_vit_flops_quadratic_in_num_patches_attention_term(self):
        """Attention core term should grow faster than linear in per-image patch count.

        Doubling ``num_patches`` more than doubles the returned FLOPS because the
        core-attn contribution scales as O(num_patches^2) while other terms are linear.
        """
        cfg = self._base_cfg(depth=2, hidden_size=128, intermediate_size=256)
        f_low = vit_flops(cfg, batch_size=1, num_patches=32)
        f_high = vit_flops(cfg, batch_size=1, num_patches=64)
        assert f_high > 2 * f_low, "Attention quadratic term must make doubling patches > 2x FLOPS"

    def test_vit_flops_out_hidden_size_defaults_to_model_hidden_size(self):
        """When vision_config lacks out_hidden_size, fall back to cfg.model.hidden_size."""

        class _VisionNoOut:
            depth = 2
            hidden_size = 128
            intermediate_size = 256
            spatial_merge_size = 2
            # intentionally no out_hidden_size attribute

        cfg_fallback = MockConfigContainer(model=MockModelConfig(hidden_size=512, vision_config=_VisionNoOut()))
        cfg_explicit = MockConfigContainer(
            model=MockModelConfig(
                hidden_size=512,
                vision_config=MockVisionConfig(
                    depth=2,
                    hidden_size=128,
                    intermediate_size=256,
                    spatial_merge_size=2,
                    out_hidden_size=512,
                ),
            )
        )
        assert vit_flops(cfg_fallback, 1, 16) == vit_flops(cfg_explicit, 1, 16)


class TestDynamicSeqLenFlops:
    """Unit tests for dynamic-length FLOPS accounting .

    Covers the ``seqlen_sum``, ``seqlen_squared_sum`` and ``num_vision_patches``
    parameters added to ``num_floating_point_operations`` for accurate VLM
    reporting with variable-length padded batches.
    """

    @staticmethod
    def _llm_cfg():
        # Small GQA transformer so the numbers are cheap to compute.
        return MockConfigContainer(
            model=MockModelConfig(
                num_layers=2,
                hidden_size=128,
                seq_length=1024,
                ffn_hidden_size=256,
                num_attention_heads=8,
                num_query_groups=4,
                kv_channels=16,
                vocab_size=1024,
                make_vocab_size_divisible_by=128,
                gated_linear_unit=False,
            )
        )

    def test_seqlen_sum_fallback_matches_legacy(self):
        """Omitting the new parameters must reproduce the legacy constant-length result."""
        cfg = self._llm_cfg()
        legacy = num_floating_point_operations(cfg, batch_size=4)
        equivalent = num_floating_point_operations(
            cfg,
            batch_size=4,
            seqlen_sum=4 * cfg.model.seq_length,
            seqlen_squared_sum=4 * cfg.model.seq_length**2,
        )
        assert legacy == equivalent

    def test_shorter_seqlen_sum_reduces_flops(self):
        """Passing a smaller effective seq length than cfg must reduce the reported FLOPS."""
        cfg = self._llm_cfg()
        legacy = num_floating_point_operations(cfg, batch_size=2)
        short = num_floating_point_operations(
            cfg,
            batch_size=2,
            seqlen_sum=2 * 256,  # effective seq_length = 256 << 1024
            seqlen_squared_sum=2 * 256**2,
        )
        assert short < legacy

    def test_seqlen_squared_sum_is_wired_into_core_attention(self):
        """Higher seqlen_squared_sum (same mean) must increase FLOPS via core-attn.

        Two batches with identical ``seqlen_sum`` but different ``seqlen_squared_sum``
        (more variance) must not produce identical FLOPS — otherwise the squared-sum
        pipeline is dead code (the exact bug the reviewer flagged).
        """
        cfg = self._llm_cfg()
        batch_size = 2
        total_tokens = 2048  # equal mean seq_len = 1024 for both batches

        # Equal-length case: seq_lens = [1024, 1024] => sq_sum = 2*1024^2
        equal_sq = 2 * 1024**2
        # Imbalanced case: seq_lens = [256, 1792] => sq_sum = 256^2 + 1792^2 (larger)
        imbalanced_sq = 256**2 + 1792**2
        assert imbalanced_sq > equal_sq  # sanity for the test input

        flops_equal = num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=total_tokens,
            seqlen_squared_sum=equal_sq,
        )
        flops_imbalanced = num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=total_tokens,
            seqlen_squared_sum=imbalanced_sq,
        )
        assert flops_imbalanced > flops_equal, (
            "seqlen_squared_sum must feed core attention FLOPS; otherwise the accumulator pipeline is dead code."
        )

    def test_attention_term_scales_linearly_with_seqlen_squared_sum(self):
        """Attention core FLOPS must be *linear* in Σ L² .

        Strategy: keep every other input identical (same cfg, same batch_size,
        same ``seqlen_sum``) and vary only ``seqlen_squared_sum``. All linear
        terms (MLP, QKV/O projection, logits, MTP) depend on ``seqlen_sum``
        only, so they cancel out in the delta. The remaining increment must
        equal the closed-form quadratic coefficient times Δ(Σ L²).

        Core-attn contribution per layer (standard MHA/GQA):
            ``full_core = query_projection_size * core_attn_seq_factor``
        After outer ``seqlen_sum *`` and the training factor ``3 * 2``:
            ``ΔFLOPS = 3 * 2 * num_layers * Q * Δ(Σ L²)``
        where ``Q = kv_channels * num_attention_heads``. Anything weaker than
        a linear dependency (e.g. sqrt/log) would fail this assertion.
        """
        cfg = self._llm_cfg()
        batch_size = 4
        seqlen_sum = batch_size * cfg.model.seq_length
        sq_base = batch_size * cfg.model.seq_length**2
        sq_bumped = sq_base + 1_000_000  # arbitrary non-trivial increment

        flops_base = num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=sq_base,
        )
        flops_bumped = num_floating_point_operations(
            cfg,
            batch_size=batch_size,
            seqlen_sum=seqlen_sum,
            seqlen_squared_sum=sq_bumped,
        )

        query_projection_size = cfg.model.kv_channels * cfg.model.num_attention_heads
        expected_delta = 3 * 2 * cfg.model.num_layers * query_projection_size * (sq_bumped - sq_base)
        assert flops_bumped - flops_base == expected_delta, (
            f"attention core should be linear in Σ L²: expected Δ = {expected_delta}, got {flops_bumped - flops_base}"
        )

    def test_num_vision_patches_adds_vit_flops(self):
        """Supplying num_vision_patches must add a strictly positive ViT contribution."""
        cfg_llm = self._llm_cfg()
        cfg_vlm = MockConfigContainer(
            model=MockModelConfig(
                **{k: v for k, v in cfg_llm.model.__dict__.items() if k != "vision_config"},
                vision_config=MockVisionConfig(
                    depth=2,
                    hidden_size=128,
                    num_heads=8,
                    intermediate_size=256,
                    spatial_merge_size=2,
                    out_hidden_size=128,
                ),
            )
        )
        batch_size = 2
        llm_only = num_floating_point_operations(cfg_vlm, batch_size=batch_size)
        vlm_flops = num_floating_point_operations(
            cfg_vlm,
            batch_size=batch_size,
            num_vision_patches=128,
        )
        assert vlm_flops > llm_only
        # ViT-only path matches the delta (invoked with per-image patches)
        vit_only = vit_flops(cfg_vlm, batch_size=batch_size, num_patches=128 / batch_size)
        assert vlm_flops - llm_only == vit_only

    def test_num_vision_patches_zero_has_no_effect(self):
        """Zero vision patches should match the pure LLM path exactly."""
        cfg = self._llm_cfg()
        baseline = num_floating_point_operations(cfg, batch_size=2)
        assert num_floating_point_operations(cfg, batch_size=2, num_vision_patches=0) == baseline


@pytest.mark.unit
class TestMLAFlops:
    """Tests for Multi-Latent Attention (MLA) FLOPs in transformer_flops path.

    MLA is the attention variant used in DeepSeek-V2/V3. Q and KV projections
    are low-rank-factored to compress the KV cache. Per-layer FLOPs follow
    the closed form in flop_utils.py (lines 343-398):

        self_attn_term = 3 * 2 * num_layers * (
            q_term
          + kv_lora_rank * (hidden + n_heads * (qk_head_dim + v_head_dim) + 1)
          + hidden * qk_pos_emb_head_dim
          + n_heads * v_head_dim * hidden
          + seq_length * n_heads * (qk_head_dim + qk_pos_emb_head_dim) / 2
          + seq_length * n_heads * v_head_dim / 2
        )

    where ``q_term`` switches form when ``q_lora_rank`` is set.
    """

    @staticmethod
    def _mla_inner(
        hidden: int,
        n_heads: int,
        seq_length: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_head_dim: int,
        qk_pos_emb_head_dim: int,
        v_head_dim: int,
    ) -> float:
        """Mirror flop_utils.py MLA formula — kept here for regression coverage."""
        if q_lora_rank is None:
            q_term = hidden * n_heads * (qk_head_dim + qk_pos_emb_head_dim)
        else:
            q_term = q_lora_rank * (hidden + n_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1)
        return (
            q_term
            + kv_lora_rank * (hidden + n_heads * (qk_head_dim + v_head_dim) + 1)
            + hidden * qk_pos_emb_head_dim
            + n_heads * v_head_dim * hidden
            + seq_length * n_heads * (qk_head_dim + qk_pos_emb_head_dim) / 2
            + seq_length * n_heads * v_head_dim / 2
        )

    def _base_mla_kwargs(self, **overrides):
        """Small DeepSeek-V3-shaped MLA config (dense, no MoE/MTP) — clean math."""
        defaults = dict(
            num_layers=2,
            hidden_size=256,
            seq_length=128,
            ffn_hidden_size=512,
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=32,
            vocab_size=32000,  # already divisible by 128 → padded == vocab
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,  # ffn_expansion_factor = 2, simpler MLP math
            multi_latent_attention=True,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
        )
        defaults.update(overrides)
        return defaults

    def test_mla_with_q_lora_exact_formula(self):
        """MLA with q_lora_rank (DeepSeek-V3 style) matches the closed-form FLOPs exactly."""
        batch_size = 1
        kw = self._base_mla_kwargs()
        cfg = MockConfigContainer(model=MockModelConfig(**kw))
        actual = num_floating_point_operations(cfg, batch_size=batch_size)

        inner = self._mla_inner(
            hidden=kw["hidden_size"],
            n_heads=kw["num_attention_heads"],
            seq_length=kw["seq_length"],
            q_lora_rank=kw["q_lora_rank"],
            kv_lora_rank=kw["kv_lora_rank"],
            qk_head_dim=kw["qk_head_dim"],
            qk_pos_emb_head_dim=kw["qk_pos_emb_head_dim"],
            v_head_dim=kw["v_head_dim"],
        )
        expected_self_attn = 3 * 2 * kw["num_layers"] * inner
        # MLP: ffn_expansion_factor = 2 (non-SwiGLU), all layers dense.
        expected_mlp = 3 * 2 * kw["hidden_size"] * (kw["ffn_hidden_size"] * 2) * kw["num_layers"]
        # Logit term: padded_vocab == vocab when already divisible by 128.
        expected_logit = 3 * 2 * kw["hidden_size"] * kw["vocab_size"] * 1
        # No MTP in baseline config.
        expected_total = batch_size * kw["seq_length"] * (expected_mlp + expected_self_attn + expected_logit)

        assert actual == expected_total, f"Expected {expected_total:.6e} but got {actual:.6e}"

    def test_mla_without_q_lora_exact_formula(self):
        """MLA without q_lora_rank uses the direct projection q_term (hidden * n_heads * head_dims)."""
        batch_size = 1
        kw = self._base_mla_kwargs(q_lora_rank=None)
        cfg = MockConfigContainer(model=MockModelConfig(**kw))
        actual = num_floating_point_operations(cfg, batch_size=batch_size)

        inner = self._mla_inner(
            hidden=kw["hidden_size"],
            n_heads=kw["num_attention_heads"],
            seq_length=kw["seq_length"],
            q_lora_rank=None,
            kv_lora_rank=kw["kv_lora_rank"],
            qk_head_dim=kw["qk_head_dim"],
            qk_pos_emb_head_dim=kw["qk_pos_emb_head_dim"],
            v_head_dim=kw["v_head_dim"],
        )
        expected_self_attn = 3 * 2 * kw["num_layers"] * inner
        expected_mlp = 3 * 2 * kw["hidden_size"] * (kw["ffn_hidden_size"] * 2) * kw["num_layers"]
        expected_logit = 3 * 2 * kw["hidden_size"] * kw["vocab_size"] * 1
        expected_total = batch_size * kw["seq_length"] * (expected_mlp + expected_self_attn + expected_logit)

        assert actual == expected_total, f"Expected {expected_total:.6e} but got {actual:.6e}"

    def test_q_lora_reduces_q_projection_flops(self):
        """Adding q_lora_rank should reduce q-projection FLOPs when q_lora_rank < n_heads * (qk_h + qk_pos)."""
        batch_size = 1
        # With these dims, the un-compressed Q projection is hidden * n_heads * 48 = 256 * 8 * 48 = 98304.
        # The Q-LoRA path uses q_lora_rank * (hidden + n_heads * 48 + 1) = 64 * (256 + 384 + 1) = 41024.
        # So enabling Q-LoRA reduces self-attn FLOPs.
        kw_q_lora = self._base_mla_kwargs(q_lora_rank=64)
        kw_no_q_lora = self._base_mla_kwargs(q_lora_rank=None)
        flops_q_lora = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw_q_lora)), batch_size=batch_size
        )
        flops_no_q_lora = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw_no_q_lora)), batch_size=batch_size
        )
        assert flops_q_lora < flops_no_q_lora, (
            "Q-LoRA compression should reduce attention FLOPs when q_lora_rank * (h + ...) < h * n_heads * (qk + qk_pos)"
        )

    def test_mla_differs_from_standard_attention(self):
        """An MLA config and a same-shape MHA config should produce different FLOPs."""
        batch_size = 1
        kw_mla = self._base_mla_kwargs()
        kw_mha = self._base_mla_kwargs(multi_latent_attention=False)
        flops_mla = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw_mla)), batch_size=batch_size
        )
        flops_mha = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw_mha)), batch_size=batch_size
        )
        assert flops_mla != flops_mha, "MLA and standard attention paths should produce different FLOPs"
        assert flops_mla > 0 and flops_mha > 0

    def test_mla_batch_size_scales_linearly(self):
        """FLOPs must scale linearly with batch_size for MLA."""
        kw = self._base_mla_kwargs()
        cfg = MockConfigContainer(model=MockModelConfig(**kw))
        f_b1 = num_floating_point_operations(cfg, batch_size=1)
        f_b4 = num_floating_point_operations(cfg, batch_size=4)
        assert f_b4 == 4 * f_b1, f"Linear scaling violated: f(B=4)={f_b4:.6e} vs 4*f(B=1)={4 * f_b1:.6e}"

    def test_mla_seq_length_quadratic_growth(self):
        """Doubling seq_length should grow MLA FLOPs by more than 2x (core attn term is O(s^2))."""
        kw_short = self._base_mla_kwargs(seq_length=128)
        kw_long = self._base_mla_kwargs(seq_length=256)
        f_short = num_floating_point_operations(MockConfigContainer(model=MockModelConfig(**kw_short)), batch_size=1)
        f_long = num_floating_point_operations(MockConfigContainer(model=MockModelConfig(**kw_long)), batch_size=1)
        # The core-attention component scales as B*s^2; total grows super-linearly.
        assert f_long > 2 * f_short, (
            f"Expected superlinear seq scaling but got f(s=256)={f_long:.6e} vs 2*f(s=128)={2 * f_short:.6e}"
        )


@pytest.mark.unit
class TestMLAWithMoE:
    """Sanity tests for MLA combined with MoE (DeepSeek-V3 architecture shape)."""

    def test_mla_moe_combination_positive_and_distinct(self):
        """MLA + MoE config should produce positive FLOPs distinct from MLA-only and MHA+MoE."""
        batch_size = 1
        base = dict(
            num_layers=2,
            hidden_size=256,
            seq_length=128,
            ffn_hidden_size=512,
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=32,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            num_moe_experts=8,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_ffn_hidden_size=512,
            moe_shared_expert_intermediate_size=0,
        )
        flops_mla_moe = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**base, multi_latent_attention=True)),
            batch_size=batch_size,
        )
        flops_mha_moe = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**base, multi_latent_attention=False)),
            batch_size=batch_size,
        )
        assert flops_mla_moe > 0
        assert flops_mha_moe > 0
        assert flops_mla_moe != flops_mha_moe, "MLA+MoE and MHA+MoE should differ in self-attention term"


@pytest.mark.unit
class TestExplicitMtpInTransformerPath:
    """Tests for explicit cfg.model.mtp_num_layers in the transformer_flops (non-hybrid) path.

    DeepSeek-V3 uses MTP. The current functional tests cover only Llama / Qwen3-MoE
    (no MTP), and the unit tests cover the inferred-from-pattern path through
    `hybrid_flops`. The transformer_flops branch where mtp_num_layers is set
    explicitly was previously uncovered.
    """

    def _base_kwargs(self, **overrides):
        defaults = dict(
            num_layers=4,
            hidden_size=512,
            seq_length=256,
            ffn_hidden_size=1024,
            num_attention_heads=8,
            num_query_groups=8,
            kv_channels=64,
            vocab_size=32000,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            gated_linear_unit=False,
        )
        defaults.update(overrides)
        return defaults

    def test_explicit_mtp_increases_flops(self):
        """Explicit mtp_num_layers > 0 must add MTP norms/proj FLOPs and grow logits."""
        kw = self._base_kwargs()
        f_no_mtp = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw, mtp_num_layers=None)), batch_size=1
        )
        f_mtp_2 = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw, mtp_num_layers=2)), batch_size=1
        )
        assert f_mtp_2 > f_no_mtp, (
            f"Explicit mtp_num_layers should grow FLOPs: got mtp=2 → {f_mtp_2:.6e} vs none → {f_no_mtp:.6e}"
        )

    def test_explicit_mtp_exact_delta(self):
        """Verify the exact FLOPs delta from explicit mtp_num_layers in non-MoE transformer path.

        For non-MoE: each MTP layer is added as a dense layer, contributing one
        extra layer worth of MLP and self-attention. The MTP norms/proj term
        and the logit factor (mtp+1) are also added.
        """
        batch_size = 1
        mtp = 2
        kw = self._base_kwargs()
        f_no_mtp = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw, mtp_num_layers=None)), batch_size=batch_size
        )
        f_mtp = num_floating_point_operations(
            MockConfigContainer(model=MockModelConfig(**kw, mtp_num_layers=mtp)), batch_size=batch_size
        )

        hidden = kw["hidden_size"]
        seq = kw["seq_length"]
        ffn = kw["ffn_hidden_size"]
        n_heads = kw["num_attention_heads"]
        n_query_groups = kw["num_query_groups"]  # MHA → equal to n_heads
        kv_ch = kw["kv_channels"]
        vocab = kw["vocab_size"]  # already padded for divisor 128

        # Per-layer MLP contribution to the inner sum (ffn_expansion=2 for non-SwiGLU).
        mlp_per_layer = 3 * 2 * hidden * (ffn * 2)
        # Per-layer attention contribution (MHA: n_query_groups == n_heads).
        q_proj = kv_ch * n_heads
        k_proj = kv_ch * n_query_groups
        v_proj = kv_ch * n_query_groups
        attn_per_layer = 3 * 2 * (hidden * (q_proj + k_proj + v_proj) + q_proj * seq / 2 * 2 + q_proj * hidden)
        # MTP norms+proj fixed term (added once when mtp_num_layers > 0).
        mtp_norms = 3 * 2 * mtp * (3 * hidden + 2 * hidden * hidden)
        # Extra logit factor: (mtp+1) - 1 = mtp.
        extra_logit = 3 * 2 * hidden * vocab * mtp

        # Each MTP layer adds one dense layer of MLP + self-attention.
        expected_delta = batch_size * seq * (mtp * mlp_per_layer + mtp * attn_per_layer + mtp_norms + extra_logit)

        actual_delta = f_mtp - f_no_mtp
        assert actual_delta == expected_delta, f"Expected MTP delta {expected_delta:.6e} but got {actual_delta:.6e}"


@pytest.mark.unit
class TestProviderOverride:
    """Tests for the `_get_num_floating_point_operations` model-provider override path.

    Some bridges (e.g., diffusion or MoE families with custom accounting) implement
    `_get_num_floating_point_operations` on the model config to bypass the generic
    calculator. The early-return at the top of `num_floating_point_operations`
    must call that method exactly once and return its result without entering
    the calculator.
    """

    def test_provider_override_short_circuits(self):
        """When the model exposes _get_num_floating_point_operations, it short-circuits."""
        sentinel = 1234567
        captured: list[int] = []

        m = MockModelConfig()

        def custom(batch_size):
            captured.append(batch_size)
            return sentinel * batch_size

        # Attach as instance attribute — `hasattr(cfg.model, "...")` becomes True.
        m._get_num_floating_point_operations = custom

        cfg = MockConfigContainer(model=m)
        assert num_floating_point_operations(cfg, batch_size=1) == sentinel
        assert num_floating_point_operations(cfg, batch_size=4) == sentinel * 4
        # Override must have been invoked twice with the right batch_size args.
        assert captured == [1, 4], f"Override call log mismatch: {captured}"


class _State:
    """Minimal stand-in for GlobalState — just an attribute bag."""


class TestAccumulateFlopsMetadata:
    """Unit tests for ``accumulate_flops_metadata``."""

    def test_bshd_no_cu_seqlens_uses_pack_length_squared(self):
        # Without cu_seqlens, the accumulator falls back to BSHD math —
        # mbs * seq_len² — matching the pre-existing behavior on dense
        # pretraining / non-packed paths.
        state = _State()
        tokens = torch.zeros(2, 512)
        accumulate_flops_metadata(state, tokens)
        assert state._flops_seqlen_sum == 2 * 512
        assert state._flops_seqlen_sq_sum == 2 * 512**2
        assert not getattr(state, "_flops_requires_global_reduce", False)

    def test_bshd_fallback_uses_full_sequence_length_for_cp_sliced_tokens(self):
        # Dense GPT batches are sliced along sequence dimension before the
        # forward step under context parallelism. FLOPS should still be based on
        # the full model sequence length, not the CP-local token length.
        state = _State()
        tokens = torch.zeros(1, 2048)
        accumulate_flops_metadata(state, tokens, config_seq_len=4096)
        assert state._flops_seqlen_sum == 4096
        assert state._flops_seqlen_sq_sum == 4096**2
        assert not getattr(state, "_flops_requires_global_reduce", False)

    def test_mock_state_accumulators_start_at_zero(self):
        state = MagicMock()
        tokens = torch.zeros(1, 8)
        accumulate_flops_metadata(state, tokens)
        assert state._flops_seqlen_sum == 8
        assert state._flops_seqlen_sq_sum == 64

    def test_thd_cu_seqlens_uses_sum_of_squares(self):
        # cu_seqlens = [0, 256, 512, 4096] → sub-seq lengths [256, 256, 3584].
        # THD attention work = 256² + 256² + 3584² = 12,975,488; the BSHD
        # approximation (1 × 4096²) would be 16,777,216 — much larger.
        state = _State()
        tokens = torch.zeros(1, 4096)
        cu_seqlens = torch.tensor([0, 256, 512, 4096])
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_seqlens)
        assert state._flops_seqlen_sum == 1 * 4096
        assert state._flops_seqlen_sq_sum == 256**2 + 256**2 + 3584**2
        assert state._flops_requires_global_reduce

    def test_config_seq_len_does_not_override_thd_cu_seqlens(self):
        # config_seq_len is only a dense/non-packed fallback. THD still uses
        # the actual packed tensor length for linear terms and cu_seqlens for
        # attention work.
        state = _State()
        tokens = torch.zeros(1, 2048)
        cu_seqlens = torch.tensor([0, 512, 2048])
        accumulate_flops_metadata(state, tokens, config_seq_len=4096, cu_seqlens=cu_seqlens)
        assert state._flops_seqlen_sum == 2048
        assert state._flops_seqlen_sq_sum == 512**2 + 1536**2
        assert state._flops_requires_global_reduce

    def test_thd_padded_cu_seqlens_with_argmin(self):
        # Offline packed SFT pads cu_seqlens for CUDA graphs; the real
        # entries end at cu_seqlens_argmin. Pad entries past argmin must be
        # ignored (here they would otherwise contribute zero-length, but we
        # exercise the truncation explicitly).
        state = _State()
        tokens = torch.zeros(1, 8192)
        cu_seqlens = torch.tensor([0, 1024, 4096, 8192, 8192, 8192, 8192])
        argmin = torch.tensor(4)  # real entries [0, 1024, 4096, 8192]
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_seqlens, cu_seqlens_argmin=argmin)
        assert state._flops_seqlen_sq_sum == 1024**2 + 3072**2 + 4096**2

    def test_thd_unpadded_takes_precedence_over_padded(self):
        # When both cu_seqlens_unpadded and cu_seqlens are present, the
        # unpadded variant describes the actual sub-sequence boundaries used
        # by the attention kernel (cu_seqlens_q in PackedSeqParams) and must
        # be the source of Σᵢ sᵢ².
        state = _State()
        tokens = torch.zeros(1, 4096)
        cu_seqlens_padded = torch.tensor([0, 4096, 4096, 4096])  # 1 pad-aligned sub-seq
        cu_seqlens_unpadded = torch.tensor([0, 1000, 3500, 4096])  # 3 real sub-seqs
        accumulate_flops_metadata(
            state,
            tokens,
            cu_seqlens=cu_seqlens_padded,
            cu_seqlens_unpadded=cu_seqlens_unpadded,
        )
        assert state._flops_seqlen_sq_sum == 1000**2 + 2500**2 + 596**2

    def test_accumulates_additively_across_microbatches(self):
        # Each call adds to existing accumulators (microbatch loop semantics).
        state = _State()
        tokens = torch.zeros(1, 128)
        cu_a = torch.tensor([0, 32, 128])
        cu_b = torch.tensor([0, 64, 128])
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_a)
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_b)
        assert state._flops_seqlen_sum == 2 * 128
        assert state._flops_seqlen_sq_sum == (32**2 + 96**2) + (64**2 + 64**2)

    @pytest.mark.parametrize("vp_size", [1, 2, 10])
    def test_vpp_accumulates_each_logical_microbatch_once(self, vp_size):
        # MCore's interleaved schedule calls forward_step once for every
        # (logical microbatch, model chunk) pair. FLOPS metadata describes the
        # data, not a model chunk, so only VP stage 0 may contribute it.
        state = _State()
        num_microbatches = 4
        tokens = torch.zeros(1, 128)
        cu_seqlens = torch.tensor([0, 32, 128])

        for vp_stage in range(vp_size):
            for _ in range(num_microbatches):
                accumulate_flops_metadata(
                    state,
                    tokens,
                    vp_stage=vp_stage,
                    cu_seqlens=cu_seqlens,
                    num_vision_patches=8,
                )

        seqlen_sum, seqlen_sq_sum, vision = resolve_global_flops_seqlen_stats(
            state,
            data_parallel_size=1,
            vp_size=vp_size,
            dp_group=None,
        )
        assert seqlen_sum == num_microbatches * 128
        assert seqlen_sq_sum == num_microbatches * (32**2 + 96**2)
        assert vision == num_microbatches * 8

    def test_tokens_none_is_noop(self):
        state = _State()
        accumulate_flops_metadata(state, None)
        assert not hasattr(state, "_flops_seqlen_sum")
        assert not hasattr(state, "_flops_seqlen_sq_sum")

    def test_num_vision_patches_int(self):
        # Vision-patch count is precomputed by the (model-specific) caller and
        # passed as a model-agnostic scalar.
        state = _State()
        tokens = torch.zeros(1, 64)
        accumulate_flops_metadata(state, tokens, num_vision_patches=32 + 8)
        assert state._flops_vision_patches == 40
        assert not getattr(state, "_flops_requires_global_reduce", False)

    def test_num_vision_patches_tensor_accumulates_across_microbatches(self):
        # A scalar device tensor (no host sync) accumulates correctly.
        state = _State()
        tokens = torch.zeros(1, 64)
        accumulate_flops_metadata(state, tokens, num_vision_patches=torch.tensor(32))
        accumulate_flops_metadata(state, tokens, num_vision_patches=torch.tensor(8))
        assert int(state._flops_vision_patches) == 40

    def test_num_vision_patches_none_is_noop(self):
        state = _State()
        tokens = torch.zeros(1, 64)
        accumulate_flops_metadata(state, tokens)
        assert not hasattr(state, "_flops_vision_patches")

    def test_empty_cu_seqlens_falls_back_to_bshd(self):
        # Degenerate cu_seqlens (only one element after argmin truncation)
        # yields no sub-seqs, so the helper must fall back to BSHD rather
        # than report 0 attention work.
        state = _State()
        tokens = torch.zeros(1, 256)
        cu_seqlens = torch.tensor([0])
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_seqlens)
        assert state._flops_seqlen_sq_sum == 1 * 256**2
        assert not getattr(state, "_flops_requires_global_reduce", False)

    def test_zero_length_subseq_contributes_zero(self):
        # A repeated cu_seqlens value yields a zero-length sub-seq. The Σᵢ sᵢ²
        # computation no longer filters with a boolean mask (which would force a
        # per-microbatch device sync); zero-length entries must still contribute 0,
        # giving the same result as if they were dropped.
        state = _State()
        tokens = torch.zeros(1, 500)
        # sub-seq lengths from [0, 100, 100, 500] -> [100, 0, 400]
        cu_seqlens = torch.tensor([0, 100, 100, 500])
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_seqlens)
        assert state._flops_seqlen_sq_sum == 100**2 + 0 + 400**2

    def test_thd_substantially_smaller_than_bshd_for_short_samples(self):
        # Regression check on the headline claim: a pack containing many
        # short samples has dramatically less attention work than the BSHD
        # approximation would suggest.
        state = _State()
        tokens = torch.zeros(1, 8192)
        # 32 sub-seqs of length 256 → pack length 8192.
        cu_seqlens = torch.tensor([i * 256 for i in range(33)])
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_seqlens)
        thd_sq = state._flops_seqlen_sq_sum
        bshd_sq = 1 * 8192**2
        # 32 * 256² = 2,097,152 vs 8192² = 67,108,864 → 32× smaller.
        assert thd_sq == 32 * 256**2
        assert bshd_sq // thd_sq == 32

    def test_thd_inline_no_host_sync_on_cpu_returns_int(self):
        # On CPU, _scalar_sum_for_accumulator returns a plain int; the inline THD path
        # must produce the same Σᵢ sᵢ² as the per-pack analytic value with no buffering.
        state = _State()
        tokens = torch.zeros(1, 4096)
        cu_seqlens = torch.tensor([0, 256, 512, 4096])
        accumulate_flops_metadata(state, tokens, cu_seqlens=cu_seqlens)
        assert state._flops_seqlen_sq_sum == 256**2 + 256**2 + 3584**2
        # No deferred cu records: the sum is computed inline, not buffered.
        assert getattr(state, "_flops_cu_records", None) is None


@pytest.mark.unit
class TestResolveGlobalFlopsSeqlenStats:
    """Unit tests for ``resolve_global_flops_seqlen_stats`` (non-distributed paths).

    ``torch.distributed`` is not initialized in unit tests, so the helper takes the
    extrapolation fallback (``local * data_parallel_size``). The exact all-reduce
    path is covered by functional/distributed tests.
    """

    def test_extrapolates_local_by_dp_size(self):
        state = _State()
        state._flops_seqlen_sum = 1000
        state._flops_seqlen_sq_sum = 250_000
        state._flops_vision_patches = 64
        seqlen_sum, seqlen_sq_sum, vision = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=4, dp_group=None
        )
        assert seqlen_sum == 1000 * 4
        assert seqlen_sq_sum == 250_000 * 4
        assert vision == 64 * 4

    def test_dp_size_one_returns_local(self):
        state = _State()
        state._flops_seqlen_sum = 512
        state._flops_seqlen_sq_sum = 4096
        seqlen_sum, seqlen_sq_sum, vision = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=1, dp_group=None
        )
        assert (seqlen_sum, seqlen_sq_sum, vision) == (512, 4096, 0)

    def test_vpp_size_does_not_rescale_before_extrapolation(self):
        # VPP accumulators already represent the executed training step; dividing
        # by vp_size undercounts reported FLOPS for virtual-pipeline jobs.
        state = _State()
        state._flops_seqlen_sum = 1000
        state._flops_seqlen_sq_sum = 250_000
        state._flops_vision_patches = 64
        seqlen_sum, seqlen_sq_sum, vision = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=2, vp_size=4, dp_group=None
        )
        assert seqlen_sum == 1000 * 2
        assert seqlen_sq_sum == 250_000 * 2
        assert vision == 64 * 2

    def test_no_accumulation_returns_none(self):
        # Step functions that don't set accumulators → caller falls back to fixed-length.
        state = _State()
        seqlen_sum, seqlen_sq_sum, vision = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=8, dp_group=None
        )
        assert seqlen_sum is None
        assert seqlen_sq_sum is None
        assert vision == 0

    def test_coerces_scalar_tensor_accumulators(self):
        # forward_step may leave accumulators as scalar tensors (deferred host sync).
        state = _State()
        state._flops_seqlen_sum = torch.tensor(800)
        state._flops_seqlen_sq_sum = torch.tensor(160_000)
        seqlen_sum, seqlen_sq_sum, _ = resolve_global_flops_seqlen_stats(state, data_parallel_size=2, dp_group=None)
        assert seqlen_sum == 1600
        assert seqlen_sq_sum == 320_000

    def test_dp_group_ignored_when_distributed_not_initialized(self):
        # A non-None dp_group must not trigger a collective when torch.distributed
        # is not initialized; the helper falls back to extrapolation.
        assert not torch.distributed.is_initialized()
        state = _State()
        state._flops_seqlen_sum = 10
        state._flops_seqlen_sq_sum = 100
        seqlen_sum, seqlen_sq_sum, _ = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=4, dp_group=object()
        )
        assert seqlen_sum == 40
        assert seqlen_sq_sum == 400

    def test_dense_stats_extrapolate_without_all_reduce_when_distributed_initialized(self, monkeypatch):
        # Dense BSHD stats are identical across DP ranks, so extrapolation is exact.
        # The helper should not create a tiny NCCL collective unless THD metadata
        # explicitly requested exact global reduction.
        state = _State()
        state._flops_seqlen_sum = 10
        state._flops_seqlen_sq_sum = 100
        all_reduce = MagicMock()
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

        seqlen_sum, seqlen_sq_sum, _ = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=4, dp_group=object()
        )

        all_reduce.assert_not_called()
        assert seqlen_sum == 40
        assert seqlen_sq_sum == 400

    def test_thd_stats_use_all_reduce_when_distributed_initialized(self, monkeypatch):
        # THD packed samples can differ across DP ranks, so a state marked by
        # accumulate_flops_metadata must take the exact all-reduce path.
        state = _State()
        state._flops_seqlen_sum = 10
        state._flops_seqlen_sq_sum = 100
        state._flops_requires_global_reduce = True

        def fake_all_reduce(stats, op=None, group=None):
            stats.mul_(4)

        all_reduce = MagicMock(side_effect=fake_all_reduce)
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        seqlen_sum, seqlen_sq_sum, vision = resolve_global_flops_seqlen_stats(
            state, data_parallel_size=4, dp_group=object()
        )

        all_reduce.assert_called_once()
        assert (seqlen_sum, seqlen_sq_sum, vision) == (40, 400, 0)


@pytest.mark.unit
class TestPackedDataExists:
    """Unit tests for ``_packed_data_exists`` (parquet spec + npy fallback gate).

    The helper decides whether the LoRA-aware FLOP branch fires. It must accept
    the same specs the packed-parquet loader does -- a single file, a glob, or a
    directory -- via ``resolve_packed_parquet_paths`` instead of a bare
    ``Path.exists()`` (which silently disabled LoRA accounting for glob/dir specs),
    while still recognising the deprecated single ``.npy`` file.
    """

    def test_none_or_empty_is_false(self):
        assert _packed_data_exists(None) is False
        assert _packed_data_exists("") is False

    def test_parquet_single_file(self, tmp_path):
        f = tmp_path / "training_4096.idx.parquet"
        f.touch()
        assert _packed_data_exists(str(f)) is True

    def test_parquet_glob_spec(self, tmp_path):
        (tmp_path / "shard_000.idx.parquet").touch()
        (tmp_path / "shard_001.idx.parquet").touch()
        assert _packed_data_exists(str(tmp_path / "shard_*.idx.parquet")) is True

    def test_parquet_directory_spec(self, tmp_path):
        (tmp_path / "shard_000.idx.parquet").touch()
        assert _packed_data_exists(str(tmp_path)) is True

    def test_missing_parquet_file_is_false(self, tmp_path):
        # Nonexistent parquet spec: the resolver raises, and the helper must
        # swallow it and report "not present" rather than propagate.
        assert _packed_data_exists(str(tmp_path / "missing.idx.parquet")) is False

    def test_legacy_npy_file(self, tmp_path):
        f = tmp_path / "training_4096.npy"
        f.touch()
        assert _packed_data_exists(str(f)) is True

    def test_missing_npy_file_is_false(self, tmp_path):
        assert _packed_data_exists(str(tmp_path / "training_4096.npy")) is False


@pytest.mark.unit
class TestLoraSquadPackedFlopBranch:
    """The LoRA + SQuAD + Llama-3-70B packed-dataset FLOP branch.

    Exercises the packed-data-path discovery (prefer ``*.idx.parquet``, fall back
    to the deprecated ``*.npy``) and confirms that when a packed dataset is found
    the LoRA-aware (~4ND) branch fires -- i.e. ``calculate_avg_seqlen`` is consulted
    -- instead of falling through to the full-model (6ND) accounting.
    ``calculate_avg_seqlen`` itself is patched (its parquet/npy loading is covered
    directly in tests/unit_tests/data/packing/test_algorithms.py); here we only
    validate the branch selection and packed-path resolution.
    """

    @staticmethod
    def _cfg(dataset_root, seq_len=512, packed_train_data_path=None):
        model = MockModelConfig(
            num_layers=2,
            hidden_size=128,
            seq_length=seq_len,
            ffn_hidden_size=256,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=16,
            vocab_size=1024,
            make_vocab_size_divisible_by=128,
            gated_linear_unit=False,
        )
        model.hf_model_id = "meta-llama/Meta-Llama-3-70B"
        cfg = MockConfigContainer(model=model)
        cfg.peft = LoRA()
        cfg.dataset = SimpleNamespace(
            dataset_name="squad",
            enable_offline_packing=True,
            offline_packing_specs=SimpleNamespace(
                packed_train_data_path=packed_train_data_path,
                packed_sequence_size=seq_len,
            ),
            dataset_root=str(dataset_root),
        )
        cfg.train = SimpleNamespace(global_batch_size=1)
        return cfg

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _lora_seq_stats_cache.clear()
        yield
        _lora_seq_stats_cache.clear()

    def test_prefers_parquet_and_takes_lora_branch(self, tmp_path):
        # dataset_root discovery: a packed parquet shard exists, so the branch fires.
        packed_dir = tmp_path / "packed" / "run0"
        packed_dir.mkdir(parents=True)
        (packed_dir / "training_512.idx.parquet").touch()
        cfg = self._cfg(tmp_path, seq_len=512)

        with patch(
            "megatron.bridge.training.utils.flop_utils.calculate_avg_seqlen",
            return_value=(2.0, 10.0, 5.0, 34.0),
        ) as mock_calc:
            flops = num_floating_point_operations(cfg, batch_size=1)

        mock_calc.assert_called_once()
        # The resolved path must be the parquet shard, not the npy.
        assert mock_calc.call_args.args[0].endswith("training_512.idx.parquet")
        assert flops > 0

    def test_falls_back_to_npy_when_no_parquet(self, tmp_path):
        # No parquet present -> discovery must fall back to the deprecated .npy.
        packed_dir = tmp_path / "packed" / "run0"
        packed_dir.mkdir(parents=True)
        (packed_dir / "training_512.npy").touch()
        cfg = self._cfg(tmp_path, seq_len=512)

        with patch(
            "megatron.bridge.training.utils.flop_utils.calculate_avg_seqlen",
            return_value=(2.0, 10.0, 5.0, 34.0),
        ) as mock_calc:
            flops = num_floating_point_operations(cfg, batch_size=1)

        mock_calc.assert_called_once()
        assert mock_calc.call_args.args[0].endswith("training_512.npy")
        assert flops > 0

    def test_lora_branch_differs_from_full_model_accounting(self, tmp_path):
        # The LoRA-aware (~4ND) result must diverge from the 6ND full-model path,
        # which is the whole point of the fix (parquet was being missed, silently
        # falling through to 6ND).
        packed_dir = tmp_path / "packed" / "run0"
        packed_dir.mkdir(parents=True)
        (packed_dir / "training_512.idx.parquet").touch()
        cfg = self._cfg(tmp_path, seq_len=512)

        with patch(
            "megatron.bridge.training.utils.flop_utils.calculate_avg_seqlen",
            return_value=(2.0, 10.0, 5.0, 34.0),
        ):
            lora_flops = num_floating_point_operations(cfg, batch_size=1)

        # Same model config but without the LoRA peft -> full-model accounting.
        full_cfg = MockConfigContainer(model=cfg.model)
        full_flops = num_floating_point_operations(full_cfg, batch_size=1)
        assert lora_flops != full_flops

    def test_result_is_cached_across_calls(self, tmp_path):
        # calculate_avg_seqlen is expensive; repeated calls with the same packed
        # path/gbs/seq_len must hit the module cache and not recompute.
        packed_dir = tmp_path / "packed" / "run0"
        packed_dir.mkdir(parents=True)
        (packed_dir / "training_512.idx.parquet").touch()
        cfg = self._cfg(tmp_path, seq_len=512)

        with patch(
            "megatron.bridge.training.utils.flop_utils.calculate_avg_seqlen",
            return_value=(2.0, 10.0, 5.0, 34.0),
        ) as mock_calc:
            first = num_floating_point_operations(cfg, batch_size=1)
            second = num_floating_point_operations(cfg, batch_size=1)

        mock_calc.assert_called_once()
        assert first == second
