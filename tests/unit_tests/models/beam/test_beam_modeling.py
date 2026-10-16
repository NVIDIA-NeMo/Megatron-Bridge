# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 Reflection AI. All rights reserved.
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

"""Beam modules that Megatron-Core lacks: SSMax, gate bias, sandwich norms, depth-scaled residuals, fp32 router."""

import math
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module

from megatron.bridge.models.beam import modeling_beam
from megatron.bridge.models.beam.beam_config import BeamTransformerConfig
from megatron.bridge.models.beam.modeling_beam import (
    AttentionOutputGateBias,
    BeamMLPLayer,
    BeamMoETransformerLayer,
    BeamSelfAttention,
    BeamTopKRouter,
    BeamTransformerLayer,
    BeamTransformerLayerSubmodules,
    apply_output_gate,
    apply_ssmax_to_query,
    beam_hybrid_stack_spec,
    beam_router_gating_linear,
    depth_scaled_residual_coefs,
    scaled_residual_add,
    ssmax_absolute_positions,
    ssmax_packed_positions,
)


def _config(**kw):
    config = BeamTransformerConfig(**kw)
    config.finalize()
    return config


def _distributed_context():
    """Single-rank process group and model-parallel state; returns a teardown callable."""
    from megatron.core import parallel_state

    created = False
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=0, world_size=1)
        created = True
    parallel_state.initialize_model_parallel(1, 1)

    def teardown():
        parallel_state.destroy_model_parallel()
        if created:
            torch.distributed.destroy_process_group()

    return teardown


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

ATTN = dict(num_layers=1, hidden_size=16, num_attention_heads=4)


class TestConfigValidation:
    def test_fields_default_off(self):
        config = _config(**ATTN)
        for name in modeling_beam.BEAM_FEATURE_FIELDS:
            assert getattr(config, name) is False, name

    def test_fields_declared_on_the_transformer_config(self):
        assert set(modeling_beam.BEAM_FEATURE_FIELDS) <= set(BeamTransformerConfig.__dataclass_fields__)

    def test_ssmax_requires_nope_layers(self):
        with pytest.raises(ValueError, match="ssmax requires no_rope_freq"):
            _config(**ATTN, ssmax=True)

    @pytest.mark.parametrize("option", ["flash_decode", "multi_latent_attention"])
    def test_ssmax_rejects_paths_that_bypass_core_attention(self, option):
        # Megatron-Core itself asserts flash_decode off on NoPE layers; the Beam check covers the rest.
        with pytest.raises((ValueError, AssertionError), match=option):
            _config(**ATTN, no_rope_freq=[1], ssmax=True, **{option: True})

    def test_gate_bias_requires_gate(self):
        with pytest.raises(ValueError, match="attention_output_gate_bias requires"):
            _config(**ATTN, attention_output_gate_bias=True)

    @pytest.mark.parametrize(
        "feature,option,value",
        [
            ("depth_scaled_residual", "inference_fuse_tp_communication", True),
            ("post_mlp_layernorm", "enable_mhc_connections", True),
            ("depth_scaled_residual", "mtp_num_layers", 1),
        ],
    )
    def test_rejects_incompatible_residual_paths(self, feature, option, value):
        # Megatron-Core asserts some of these combinations itself (e.g. fused TP inference).
        with pytest.raises((ValueError, AssertionError), match=option):
            _config(num_layers=4, hidden_size=16, num_attention_heads=4, **{feature: True, option: value})

    def test_accepts_sandwich_norms_with_depth_scaling(self):
        _config(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=4,
            post_self_attn_layernorm=True,
            post_mlp_layernorm=True,
            depth_scaled_residual=True,
        )

    @pytest.mark.parametrize("router_dtype", [None, "fp64"])
    def test_fp32_router_params_require_fp32_router_dtype(self, router_dtype):
        with pytest.raises(ValueError, match="moe_router_fp32_params requires"):
            _config(**ROUTER, moe_router_fp32_params=True, moe_router_dtype=router_dtype)


# ---------------------------------------------------------------------------
# SSMax
# ---------------------------------------------------------------------------


class TestSSMaxQueryScaling:
    @pytest.mark.parametrize("position", [0, 64, 2047, 8191, 131071])
    def test_reference_formula_and_decode_position(self, position):
        q = torch.ones(1, 2, 3, 4)
        expected = torch.full_like(q, (1 + (position + 1) / 2048) ** 0.1)
        torch.testing.assert_close(apply_ssmax_to_query(q, position_offset=position), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_scale_in_fp32_before_rounding(self, dtype):
        q = torch.tensor([-3.0, -1.0, 0.1, 1.5, 7.0], dtype=dtype).view(5, 1, 1, 1)
        positions = torch.arange(64, 69, dtype=torch.float32)
        scale = torch.exp(0.1 * torch.log1p((positions + 1) / 2048)).view(5, 1, 1, 1)
        expected = (q.float() * scale).to(dtype)
        torch.testing.assert_close(apply_ssmax_to_query(q, position_offset=64), expected, rtol=0, atol=0)

    def test_bf16_rounding_boundary(self):
        # Rounding the scale to bf16 first would leave 1.5 unchanged at position 64.
        q = torch.full((1, 1, 1, 1), 1.5, dtype=torch.bfloat16)
        assert apply_ssmax_to_query(q, position_offset=64).item() == 1.5078125


class TestSSMaxContextParallelPositions:
    def test_cp1_is_plain_arange(self):
        torch.testing.assert_close(ssmax_absolute_positions(8), torch.arange(8).float())

    @pytest.mark.parametrize("rank,positions", [(0, [0, 1, 2, 3, 12, 13, 14, 15]), (1, [4, 5, 6, 7, 8, 9, 10, 11])])
    def test_cp2_zigzag_layout_and_scaling(self, rank, positions):
        positions = torch.tensor(positions, dtype=torch.float32)
        torch.testing.assert_close(ssmax_absolute_positions(8, cp_rank=rank, cp_size=2), positions)
        q = torch.ones(8, 1, 1, 4)
        expected = (1 + (positions + 1) / 2048).pow(0.1)
        torch.testing.assert_close(apply_ssmax_to_query(q, cp_rank=rank, cp_size=2)[:, 0, 0, 0], expected)

    @pytest.mark.parametrize("cp_size", [2, 4, 8])
    def test_ranks_partition_the_sequence_exactly(self, cp_size):
        seen = torch.cat([ssmax_absolute_positions(16, cp_rank=r, cp_size=cp_size) for r in range(cp_size)])
        torch.testing.assert_close(torch.sort(seen).values, torch.arange(16 * cp_size).float())

    def test_odd_local_length_rejected(self):
        with pytest.raises(ValueError, match="must be even"):
            ssmax_absolute_positions(7, cp_rank=0, cp_size=2)


class TestSSMaxPackedPositions:
    def _packed(self, cu, **kw):
        return ssmax_packed_positions(torch.tensor(cu, dtype=torch.int32), **kw)

    def test_restart_at_every_sequence(self):
        assert self._packed([0, 3, 8]).tolist() == [0, 1, 2, 0, 1, 2, 3, 4]

    def test_offset_is_added(self):
        assert self._packed([0, 2, 5], position_offset=1).tolist() == [1, 2, 1, 2, 3]

    def test_single_sequence_matches_unpacked(self):
        for cp_size in (1, 2, 4):
            for cp_rank in range(cp_size):
                packed = self._packed([0, 16], cp_rank=cp_rank, cp_size=cp_size)
                torch.testing.assert_close(packed, ssmax_absolute_positions(16 // cp_size, 0, cp_rank, cp_size))

    def test_cp_zigzag_per_sequence(self):
        assert self._packed([0, 8, 16], cp_rank=0, cp_size=2).tolist() == [0, 1, 6, 7, 0, 1, 6, 7]
        assert self._packed([0, 8, 16], cp_rank=1, cp_size=2).tolist() == [2, 3, 4, 5, 2, 3, 4, 5]

    def test_cp_concatenates_per_sequence_mapping(self):
        cu = [0, 8, 12, 24]
        for cp_rank in range(2):
            expected = torch.cat([ssmax_absolute_positions(n // 2, 0, cp_rank, 2) for n in (8, 4, 12)])
            torch.testing.assert_close(self._packed(cu, cp_rank=cp_rank, cp_size=2), expected)

    def test_tail_padding_is_clamped_onto_last_sequence(self):
        assert self._packed([0, 3, 5], total_tokens=8).tolist() == [0, 1, 2, 0, 1, 2, 3, 4]

    def test_apply_on_thd_query_matches_per_sequence_scaling(self):
        torch.manual_seed(0)
        q = torch.randn(9, 2, 4)
        cu = torch.tensor([0, 4, 9], dtype=torch.int32)
        packed = apply_ssmax_to_query(q, cu_seqlens=cu)
        separate = torch.cat([apply_ssmax_to_query(q[:4]), apply_ssmax_to_query(q[4:])])
        torch.testing.assert_close(packed, separate, rtol=0, atol=0)
        assert not torch.allclose(packed, apply_ssmax_to_query(q))


class _PackedParams(SimpleNamespace):
    def __init__(self, cu, qkv_format="thd"):
        super().__init__(qkv_format=qkv_format, cu_seqlens_q=cu, cu_seqlens_q_padded=None)


class TestSSMaxCoreAttentionHook:
    """The mixin scales the query it hands to the wrapped core attention, on NoPE layers only."""

    class _Recorder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, query, key, value, attention_mask, *args, **kwargs):
            return query

    class _Attention(modeling_beam._BeamSSMaxCoreAttentionMixin, _Recorder):
        pass

    def _core(self, nope, ssmax=True, cp_group=None):
        config = SimpleNamespace(ssmax=ssmax, no_rope_freq=[int(nope), 0])
        pg = SimpleNamespace(cp=cp_group) if cp_group is not None else None
        return self._Attention(config=config, layer_number=1, pg_collection=pg)

    @pytest.mark.parametrize("ssmax,nope", [(False, True), (True, False)])
    def test_other_layers_are_untouched(self, ssmax, nope):
        q = torch.randn(6, 1, 2, 4)
        assert torch.equal(self._core(nope, ssmax)(q, q, q, None), q)

    def test_nope_layer_scales_the_query(self):
        q = torch.randn(6, 1, 2, 4)
        out = self._core(True, cp_group=SimpleNamespace(size=lambda: 1))(q, q, q, None, attn_mask_type=None)
        torch.testing.assert_close(out, apply_ssmax_to_query(q), rtol=0, atol=0)

    def test_packed_positions_restart_per_sequence(self):
        q = torch.randn(9, 2, 4)
        cu = torch.tensor([0, 4, 9], dtype=torch.int32)
        core = self._core(True, cp_group=SimpleNamespace(size=lambda: 1))
        out = core(q, q, q, None, attn_mask_type=None, packed_seq_params=_PackedParams(cu))
        torch.testing.assert_close(out, apply_ssmax_to_query(q, cu_seqlens=cu), rtol=0, atol=0)
        with pytest.raises(NotImplementedError, match="thd"):
            core(q, q, q, None, packed_seq_params=_PackedParams(cu, qkv_format="sbhd"))

    def test_decode_offset_from_cached_key_length(self):
        q, k = torch.randn(1, 1, 2, 4), torch.randn(7, 1, 2, 4)
        out = self._core(True, cp_group=SimpleNamespace(size=lambda: 1))(q, k, k, None)
        torch.testing.assert_close(out, apply_ssmax_to_query(q, position_offset=6), rtol=0, atol=0)

    def test_context_parallel_decode_rejected(self):
        q, k = torch.randn(2, 1, 2, 4), torch.randn(8, 1, 2, 4)
        core = self._core(True, cp_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0))
        with pytest.raises(NotImplementedError, match="context parallelism"):
            core(q, k, k, None)

    def test_context_parallel_uses_zigzag_positions(self):
        q = torch.randn(8, 1, 2, 4)
        core = self._core(True, cp_group=SimpleNamespace(size=lambda: 2, rank=lambda: 1))
        torch.testing.assert_close(core(q, q, q, None), apply_ssmax_to_query(q, cp_rank=1, cp_size=2), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Attention output gate bias
# ---------------------------------------------------------------------------

GATE = dict(num_layers=2, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True)


class TestApplyOutputGate:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("bias_values", [None, [0.0] * 4, [0.5, -1.0, 1.5, -0.25]])
    def test_matches_hf_formula(self, dtype, packed, bias_values):
        torch.manual_seed(0)
        s, b, heads, hd = 5, 1 if packed else 2, 4, 8
        x = torch.randn(s, b, heads * hd).to(dtype)
        gate = torch.randn(s, b, heads, hd).to(dtype)
        bias = torch.tensor(bias_values) if bias_values is not None else None
        pre_sigmoid = gate.float()
        if bias is not None:
            pre_sigmoid = pre_sigmoid + bias[None, None, :, None]
        expected = (x.float() * torch.sigmoid(pre_sigmoid).reshape_as(x)).to(dtype)
        # Packed attention produces a THD gate, but keeps a singleton batch dimension in x.
        out = apply_output_gate(x, gate.squeeze(1) if packed else gate, bias)
        assert out.dtype == dtype
        torch.testing.assert_close(out, expected)

    def test_rejects_mismatched_heads(self):
        with pytest.raises(ValueError, match="not divisible"):
            apply_output_gate(torch.ones(1, 1, 9), torch.zeros(1, 1, 9), torch.zeros(2))

    def test_self_attention_gate_uses_its_bias(self):
        attn = SimpleNamespace(output_gate_bias=SimpleNamespace(bias=torch.tensor([1.0, -1.0])))
        x, gate = torch.ones(2, 1, 4), torch.zeros(2, 1, 2, 2)
        out = BeamSelfAttention._apply_output_gate(attn, x, gate)
        torch.testing.assert_close(out, apply_output_gate(x, gate, attn.output_gate_bias.bias))
        attn.output_gate_bias = None
        torch.testing.assert_close(BeamSelfAttention._apply_output_gate(attn, x, gate), x * 0.5)


class TestAttentionOutputGateBias:
    def _cfg(self, **kw):
        return _config(**GATE, attention_output_gate=True, attention_output_gate_bias=True, **kw)

    def test_init_value_dtype_and_tp_attributes(self):
        mod = AttentionOutputGateBias(self._cfg(params_dtype=torch.bfloat16), 2)
        assert mod.bias.shape == (2,)
        assert mod.bias.dtype == torch.bfloat16
        assert torch.all(mod.bias == 0)
        assert mod.bias.tensor_model_parallel is True
        assert mod.bias.partition_dim == 0
        assert list(mod.state_dict().keys()) == ["bias"]
        assert mod.bias.device.type == "cpu"

    def test_gradient_reaches_bias(self):
        mod = AttentionOutputGateBias(self._cfg(), 4)
        x = torch.ones(3, 2, 16)
        gate = torch.zeros(3, 2, 4, 4)
        apply_output_gate(x, gate, mod.bias).sum().backward()
        # Each head contributes 3 * 2 * 4 terms with sigmoid'(0) = 1/4.
        torch.testing.assert_close(mod.bias.grad, torch.full((4,), 6.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires distributed CUDA setup")
    def test_sharded_state_dict_tp_axis(self):
        """TP=1: one shard and a distinct replica id for every DP rank."""
        from megatron.core import parallel_state

        teardown = _distributed_context()
        try:
            mod = AttentionOutputGateBias(self._cfg(), 4)
            ((key, tensor),) = mod.sharded_state_dict(
                prefix="decoder.layers.0.self_attention.output_gate_bias."
            ).items()
            assert key == "decoder.layers.0.self_attention.output_gate_bias.bias"
            assert tuple(tensor.global_shape) == (4,)
            assert tensor.axis_fragmentations[0] == 1
            dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            assert tuple(tensor.replica_id) == (0, 0, dp_rank)
        finally:
            teardown()


# ---------------------------------------------------------------------------
# Sandwich norms and depth-scaled residuals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer_number,expected",
    [
        (1, (1 / math.sqrt(2), 1 / math.sqrt(2))),  # Beam layer 0 attention
        (2, (1 / math.sqrt(3), math.sqrt(2 / 3))),  # Beam layer 0 MLP
        (103, (1 / math.sqrt(104), math.sqrt(103 / 104))),  # Beam layer 51 attention
        (104, (1 / math.sqrt(105), math.sqrt(104 / 105))),  # Beam layer 51 MLP
    ],
)
def test_depth_scaled_residual_coefs(layer_number, expected):
    assert depth_scaled_residual_coefs(layer_number) == pytest.approx(expected)


def test_scaled_residual_add_formula_and_dtype():
    torch.manual_seed(0)
    out = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    res = torch.randn(5, 2, 8, dtype=torch.float32)  # fp32_residual_connection
    got = scaled_residual_add(out, res, (0.6, 0.8))
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, out.float() * 0.6 + res * 0.8)


class _Branch(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
        self.bias = torch.nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, x, **kwargs):
        return torch.nn.functional.linear(x, self.weight), self.bias


class _PostNorm(torch.nn.RMSNorm):
    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps)


def _layer_config(post_norm, depth_scaling, hidden_dropout=0.0):
    return _config(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=2,
        transformer_impl="local",
        normalization="RMSNorm",
        hidden_dropout=hidden_dropout,
        attention_dropout=0.0,
        post_self_attn_layernorm=post_norm,
        post_mlp_layernorm=post_norm,
        depth_scaled_residual=depth_scaling,
    )


def _attention_layer(config, layer_number, post_norm=_PostNorm):
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

    return BeamTransformerLayer(
        config,
        BeamTransformerLayerSubmodules(
            self_attention=_Branch,
            self_attn_bda=get_bias_dropout_add,
            post_self_attn_layernorm=post_norm,
            post_mlp_layernorm=post_norm,
        ),
        layer_number=layer_number,
        pg_collection=SimpleNamespace(tp=None, pp=None),
        add_layer_offset=False,
    )


def _mlp_layer(config, layer_number, post_norm=_PostNorm):
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

    return BeamMLPLayer(
        config,
        BeamTransformerLayerSubmodules(
            mlp=_Branch,
            mlp_bda=get_bias_dropout_add,
            post_self_attn_layernorm=post_norm,
            post_mlp_layernorm=post_norm,
        ),
        layer_number=layer_number,
        pg_collection=SimpleNamespace(tp=None, pp=None),
        add_layer_offset=False,
    )


@pytest.mark.parametrize("beam_layer", [0, 3])
@pytest.mark.parametrize("post_norm,depth_scaling", [(True, True), (True, False), (False, True), (False, False)])
def test_split_layers_forward_backward(beam_layer, post_norm, depth_scaling):
    """Beam layer ``i`` = attention layer ``2i + 1`` then MLP layer ``2i + 2``; adds at ``2i + 2`` and ``2i + 3``."""
    torch.manual_seed(42)
    config = _layer_config(post_norm, depth_scaling)
    attn = _attention_layer(config, 2 * beam_layer + 1)
    mlp = _mlp_layer(config, 2 * beam_layer + 2)
    if post_norm:
        assert isinstance(attn.post_self_attn_layernorm, _PostNorm)
        assert isinstance(mlp.post_mlp_layernorm, _PostNorm)
    else:
        assert attn.post_self_attn_layernorm is None and mlp.post_mlp_layernorm is None
    # Each layer only builds the norm of the sublayer it owns.
    assert attn.post_mlp_layernorm is None and mlp.post_self_attn_layernorm is None
    x = torch.randn(5, 2, 8, requires_grad=True)
    hidden, _ = attn(x)
    actual, _ = mlp(hidden)
    expected = x
    for index, (branch, norm) in enumerate(
        [(attn.self_attention, attn.post_self_attn_layernorm), (mlp.mlp, mlp.post_mlp_layernorm)],
        start=2 * beam_layer + 2,
    ):
        out = torch.nn.functional.linear(expected, branch.weight, branch.bias)
        if post_norm:
            out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + config.layernorm_epsilon)
            out = out * norm.weight
        expected = (
            out / math.sqrt(index) + expected * math.sqrt((index - 1) / index) if depth_scaling else out + expected
        )
    torch.testing.assert_close(actual, expected)
    params = (x, *attn.parameters(), *mlp.parameters())
    actual_grads = torch.autograd.grad(actual.square().sum(), params)
    expected_grads = torch.autograd.grad(expected.square().sum(), params)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)
        assert torch.isfinite(actual_grad).all() and actual_grad.count_nonzero() > 0


def test_absent_sublayer_keeps_its_identity_bda():
    from megatron.core.transformer.identity_op import IdentityFuncOp

    config = _layer_config(True, True)
    assert isinstance(_attention_layer(config, 1).mlp_bda, IdentityFuncOp)
    assert isinstance(_mlp_layer(config, 2).self_attn_bda, IdentityFuncOp)


def test_depth_scaling_rejects_a_layer_with_both_sublayers():
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

    submodules = BeamTransformerLayerSubmodules(
        self_attention=_Branch,
        self_attn_bda=get_bias_dropout_add,
        mlp=_Branch,
        mlp_bda=get_bias_dropout_add,
        post_self_attn_layernorm=_PostNorm,
        post_mlp_layernorm=_PostNorm,
    )
    with pytest.raises(NotImplementedError, match="split hybrid layers"):
        BeamTransformerLayer(
            _layer_config(True, True), submodules, layer_number=1, pg_collection=SimpleNamespace(tp=None, pp=None)
        )


def test_dropout_follows_the_post_norm():
    """bias -> norm -> dropout -> add: normalizing after dropout would undo its 1/(1-p) rescale."""
    torch.manual_seed(0)
    layer = _attention_layer(_layer_config(True, False, hidden_dropout=0.5), 1).train()
    bda = layer.self_attn_bda(True, False)
    out, residual = torch.randn(64, 1, 8), torch.zeros(64, 1, 8)
    torch.manual_seed(1)
    got = bda((out, None), residual, 0.5)
    torch.manual_seed(1)
    expected = torch.nn.functional.dropout(layer.post_self_attn_layernorm(out), p=0.5)
    torch.testing.assert_close(got, expected)


def test_enabled_post_norm_needs_a_norm_module():
    from megatron.core.transformer.identity_op import IdentityOp

    with pytest.raises(ValueError, match="normalization module"):
        _attention_layer(_layer_config(True, False), 1, post_norm=IdentityOp)


def test_state_dict_names_match_the_bridge_mappings():
    config = _layer_config(True, True)
    assert "post_self_attn_layernorm.weight" in _attention_layer(config, 1).state_dict()
    assert "post_mlp_layernorm.weight" in _mlp_layer(config, 2).state_dict()


# ---------------------------------------------------------------------------
# fp32 router
# ---------------------------------------------------------------------------

ROUTER = dict(
    num_layers=2,
    hidden_size=12,
    num_attention_heads=4,
    num_moe_experts=4,
    moe_router_topk=2,
    moe_router_load_balancing_type="aux_loss",
    moe_aux_loss_coeff=0,
    use_cpu_initialization=True,
    bf16=True,
    params_dtype=torch.bfloat16,
    add_bias_linear=False,  # tests that want a router bias opt in
)


def _router_config(**kw):
    return _config(**{**ROUTER, **kw})


def _cpu_router(config):
    return BeamTopKRouter(config, pg_collection=ProcessGroupCollection())


class TestRouterParameters:
    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_parameter_dtype_and_marks(self, flag, with_bias):
        cfg = _router_config(moe_router_fp32_params=flag, moe_router_dtype="fp32", add_bias_linear=with_bias)
        router = _cpu_router(cfg)
        expected = torch.float32 if flag else torch.bfloat16
        assert (router.bias is not None) == with_bias
        for param in router.parameters():
            assert param.dtype == expected
            assert getattr(param, "keep_in_fp32", False) == flag
            assert param.sequence_parallel is False

    def test_float16_wrapper_keeps_router_fp32(self):
        cfg = _router_config(moe_router_fp32_params=True, moe_router_dtype="fp32", add_bias_linear=True)
        wrapped = Float16Module(cfg, _cpu_router(cfg))
        assert wrapped.module.weight.dtype == wrapped.module.bias.dtype == torch.float32

    def test_fp32_router_installs_the_optimizer_patch(self):
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

        _cpu_router(_router_config(moe_router_fp32_params=True, moe_router_dtype="fp32"))
        assert getattr(DistributedOptimizer.__init__, "_beam_group_index_fix", False)


class TestGatingTorchPath:
    """Force the torch GEMM path even when TE is installed."""

    @pytest.fixture(autouse=True)
    def _torch_path(self, monkeypatch):
        from megatron.core.transformer.moe import moe_utils

        monkeypatch.setattr(moe_utils, "te_general_gemm", None)

    @pytest.mark.parametrize("flag", [False, True])
    def test_loaded_weights_logits_and_gradients(self, flag):
        router = _cpu_router(_router_config(moe_router_fp32_params=flag, moe_router_dtype="fp32"))
        raw = torch.randn(4, 12, generator=torch.Generator().manual_seed(0))
        assert not torch.equal(raw.bfloat16().float(), raw)
        with torch.no_grad():
            router.weight.copy_(raw)
        assert torch.equal(router.weight.float(), raw) == flag
        expected_weight = raw if flag else raw.bfloat16().float()
        x = torch.randn(5, 12).bfloat16().requires_grad_(True)
        logits = beam_router_gating_linear(x, router.weight, router.bias, torch.float32)
        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits, x.float() @ expected_weight.t(), rtol=0, atol=0)
        logits.sum().backward()
        assert router.weight.grad.dtype == router.weight.dtype
        assert x.grad.dtype == torch.bfloat16
        torch.testing.assert_close(router.weight.grad, (torch.ones(5, 4).t() @ x.float()).to(router.weight.dtype))

    def test_router_gating_uses_the_bridge_function(self):
        router = _cpu_router(_router_config(moe_router_fp32_params=True, moe_router_dtype="fp32"))
        x = torch.randn(3, 12).bfloat16()
        with mock.patch.object(modeling_beam, "beam_router_gating_linear", wraps=beam_router_gating_linear) as spy:
            with mock.patch("torch.cuda.current_device", return_value="cpu"):
                logits = router.gating(x)
        spy.assert_called_once()
        assert logits.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "router_dtype,weight_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.float64, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_router_gating_linear_matches_reference(router_dtype, weight_dtype):
    """TE GEMM path (bf16 input) against a float64 reference, including fp32 weights."""
    torch.manual_seed(0)
    input_shape, experts = ((3, 2, 12), 4) if weight_dtype == torch.float32 else ((4096, 7168), 256)
    inp = torch.randn(input_shape, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    weight = torch.randn((experts, input_shape[-1]), dtype=weight_dtype, device="cuda").requires_grad_(True)
    grad = torch.randn((*input_shape[:-1], experts), dtype=router_dtype, device="cuda")
    out = beam_router_gating_linear(inp, weight, None, router_dtype)
    assert out.dtype == router_dtype
    out.backward(grad)
    assert inp.grad.dtype == inp.dtype and weight.grad.dtype == weight.dtype

    ref_inp, ref_weight = inp.detach().double().requires_grad_(True), weight.detach().double().requires_grad_(True)
    ref = ref_inp @ ref_weight.t()
    ref.backward(grad.double())
    tols = dict(rtol=2.0e-2, atol=1.0e-4 if weight_dtype == torch.float32 else 1.0e-3)
    torch.testing.assert_close(out.double(), ref, **tols)
    torch.testing.assert_close(inp.grad.double(), ref_inp.grad, rtol=2.0e-2, atol=1.0e-1)
    torch.testing.assert_close(weight.grad.double(), ref_weight.grad, rtol=2.0e-2, atol=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_router_gating_backward_accepts_expanded_grad_output(weight_dtype):
    """A stride-0 gradient from sum().backward() must match the contiguous case."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda").bfloat16()
    w0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda").to(weight_dtype)
    expected_w = torch.tensor([[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]], device="cuda")
    expected_x = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], device="cuda")
    for kind in ("contiguous", "expanded"):
        w = w0.clone().requires_grad_(True)
        xi = x.clone().requires_grad_(True)
        out = beam_router_gating_linear(xi, w, None, torch.float32)
        if kind == "contiguous":
            out.backward(torch.ones_like(out))
        else:
            out.sum().backward()
        torch.testing.assert_close(w.grad.float(), expected_w, rtol=0, atol=0, msg=kind)
        torch.testing.assert_close(xi.grad.float(), expected_x, rtol=0, atol=0, msg=kind)


class TestDistributedOptimizerGroupIndexMap:
    def test_fix_reorders_only_mixed_groups(self):
        a, gate, b, c = (torch.nn.Parameter(torch.zeros(1)) for _ in range(4))
        opt = SimpleNamespace(
            model_fp32_groups=[[gate], []],
            model_float16_groups=[[a, b], [c]],
            model_param_group_index_map={a: (0, 0), gate: (0, 1), b: (0, 2), c: (1, 5)},
        )
        modeling_beam._fix_group_index_map(opt)
        assert opt.model_param_group_index_map == {gate: (0, 0), a: (0, 1), b: (0, 2), c: (1, 5)}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_mixed_dtype_param_group_lookup(self):
        """Mixed bf16/fp32 groups must resolve each parameter to its own optimizer shard."""
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from megatron.core.transformer.module import mark_keep_in_fp32
        from megatron.core.transformer.transformer_config import TransformerConfig

        modeling_beam.patch_distributed_optimizer_group_index_map()
        teardown = _distributed_context()
        try:

            class Mixed(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.a = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device="cuda")
                    self.gate = torch.nn.Linear(64, 8, bias=False, device="cuda")
                    mark_keep_in_fp32(self.gate.weight)
                    self.b = torch.nn.Linear(64, 32, bias=False, dtype=torch.bfloat16, device="cuda")

            model = Mixed()
            fills = {"a.weight": 1.0, "gate.weight": 2.0, "b.weight": 3.0}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.fill_(fills[name])
            ddp = DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1, bf16=True),
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                model,
            )
            optim = get_megatron_optimizer(
                OptimizerConfig(optimizer="adam", lr=1e-3, bf16=True, use_distributed_optimizer=True), [ddp]
            )
            dist_optim = optim.chained_optimizers[0] if hasattr(optim, "chained_optimizers") else optim
            assert isinstance(dist_optim, DistributedOptimizer)
            names = {id(param): name for name, param in model.named_parameters()}
            for model_param in dist_optim.model_param_gbuf_map:
                name = names[id(model_param)]
                main = dist_optim._get_main_param_and_optimizer_states(model_param)["param"]
                local = dist_optim._get_model_param_range_map(model_param)["gbuf_local"]
                assert main.numel() == local.end - local.start, (name, main.numel(), local)
                assert main.dtype == torch.float32, (name, main.dtype)
                assert torch.all(main == fills[name]), (name, main.unique().tolist())
        finally:
            teardown()


# ---------------------------------------------------------------------------
# Layer spec
# ---------------------------------------------------------------------------


def _spec_config(**kw):
    defaults = dict(
        num_layers=8,  # 4 Beam layers: attention + MLP each
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        num_moe_experts=8,
        moe_grouped_gemm=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        attention_output_gate=True,
        attention_output_gate_bias=True,
        moe_router_dtype="fp32",
        moe_router_fp32_params=True,
        post_self_attn_layernorm=True,
        post_mlp_layernorm=True,
        depth_scaled_residual=True,
        no_rope_freq=[0, 0, 1, 0, 0, 0, 1, 0],
        ssmax=True,
    )
    defaults.update(kw)
    return _config(**defaults)


def test_hybrid_stack_spec_uses_the_beam_modules():
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.moe_layer import MoELayer

    spec = beam_hybrid_stack_spec(_spec_config())
    assert spec.module is HybridStack
    attention, mlp, moe = (spec.submodules.attention_layer, spec.submodules.mlp_layer, spec.submodules.moe_layer)
    for layer_spec, module in ((attention, BeamTransformerLayer), (mlp, BeamMLPLayer), (moe, BeamMoETransformerLayer)):
        assert layer_spec.module is module
        assert isinstance(layer_spec.submodules, BeamTransformerLayerSubmodules)

    sub = attention.submodules
    assert sub.self_attention.module is BeamSelfAttention
    assert sub.self_attention.submodules.core_attention is modeling_beam.BeamTEDotProductAttention
    assert sub.self_attention.submodules.q_layernorm is not IdentityOp
    assert sub.post_self_attn_layernorm is modeling_beam.TENorm
    assert sub.mlp is IdentityOp

    sub = mlp.submodules
    mlp_cls = getattr(sub.mlp, "module", getattr(sub.mlp, "func", None))
    assert sub.self_attention is IdentityOp and getattr(mlp_cls, "__self__", mlp_cls) is MLP
    assert sub.post_mlp_layernorm is modeling_beam.TENorm

    sub = moe.submodules
    assert sub.self_attention is IdentityOp and sub.pre_mlp_layernorm is not IdentityOp
    assert sub.mlp.func is MoELayer and sub.mlp.keywords["submodules"].router is BeamTopKRouter
    assert sub.post_mlp_layernorm is modeling_beam.TENorm


def test_hybrid_stack_spec_without_experts_has_no_moe_layer():
    from megatron.core.transformer.identity_op import IdentityOp

    spec = beam_hybrid_stack_spec(_spec_config(num_moe_experts=None, moe_router_fp32_params=False))
    assert spec.submodules.moe_layer is IdentityOp


def test_hybrid_stack_spec_validates_the_config():
    config = _spec_config()
    config.attention_output_gate = False
    with pytest.raises(ValueError, match="attention_output_gate_bias requires"):
        beam_hybrid_stack_spec(config)


def test_hybrid_stack_spec_requires_transformer_engine():
    with pytest.raises(NotImplementedError, match="Transformer Engine"):
        beam_hybrid_stack_spec(_spec_config(transformer_impl="local"))


# ---------------------------------------------------------------------------
# SSMax inside a real SelfAttention (GPU)
# ---------------------------------------------------------------------------

SEQ, BATCH, HEADS, HDIM = 8, 2, 4, 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestSSMaxInSelfAttention:
    """The query that reaches TE attention is exactly the unscaled query times the SSMax factor."""

    def setup_method(self, method):
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        self._teardown = _distributed_context()
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(0)

    def teardown_method(self, method):
        self._teardown()

    def _attention(self, nope, **kw):
        from megatron.core.transformer.enums import AttnMaskType

        defaults = dict(
            num_layers=1,
            hidden_size=HEADS * HDIM,
            num_attention_heads=HEADS,
            kv_channels=HDIM,
            use_cpu_initialization=True,
            bf16=False,
            params_dtype=torch.float32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            no_rope_freq=[int(nope)],
            ssmax=nope,
        )
        defaults.update(kw)
        config = _config(**defaults)
        spec = beam_hybrid_stack_spec(config).submodules.attention_layer.submodules.self_attention
        attn = BeamSelfAttention(
            config=config, submodules=spec.submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
        )
        return attn.cuda().eval()

    def _queries(self, attn, hidden, **kwargs):
        """Queries handed to TE with the SSMax hook off and on."""
        te_forward = modeling_beam.TEDotProductAttention.forward
        queries, outputs = [], []

        def spy(module, query, *args, **kw):
            queries.append(query.detach().clone())
            return te_forward(module, query, *args, **kw)

        with torch.no_grad(), mock.patch.object(modeling_beam.TEDotProductAttention, "forward", spy):
            for flag in (False, True):
                attn.core_attention._beam_ssmax = flag
                outputs.append(attn(hidden, attention_mask=None, **kwargs)[0])
        return queries, outputs

    def test_only_nope_layers_are_scaled(self):
        attn = self._attention(nope=False)
        assert attn.core_attention._beam_ssmax is False
        assert isinstance(attn.core_attention, modeling_beam.BeamTEDotProductAttention)

    def test_query_scaling_on_a_nope_layer(self):
        attn = self._attention(nope=True)
        hidden = torch.randn(SEQ, BATCH, HEADS * HDIM, device="cuda")
        (q_base, q_on), (out_base, out_on) = self._queries(attn, hidden)
        positions = torch.arange(SEQ, device="cuda", dtype=torch.float32)
        scale = torch.exp(0.1 * torch.log1p((positions + 1) / 2048)).view(SEQ, 1, 1, 1)
        expected = q_base * scale
        assert not torch.equal(expected, q_base) and not torch.equal(out_base, out_on)
        torch.testing.assert_close(q_on, expected, rtol=0, atol=0)

    def test_packed_thd_query_uses_per_sequence_positions(self):
        """thd batch of two sequences: the scale restarts at the second sequence."""
        from megatron.core.packed_seq_params import PackedSeqParams

        cu = torch.tensor([0, 128, 320], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=192, max_seqlen_kv=192
        )
        attn = self._attention(nope=True, bf16=True, params_dtype=torch.bfloat16)  # fused thd attention needs bf16
        hidden = torch.randn(320, 1, HEADS * HDIM, device="cuda", dtype=torch.bfloat16)
        (q_base, q_on), _ = self._queries(attn, hidden, packed_seq_params=packed)
        assert q_base.dim() == 3 and q_base.shape[0] == 320
        positions = torch.cat([torch.arange(128), torch.arange(192)]).cuda().float()
        scale = torch.exp(0.1 * torch.log1p((positions + 1) / 2048)).view(320, 1, 1)
        expected = (q_base.float() * scale).to(q_base.dtype)
        assert not torch.equal(expected, q_base)
        torch.testing.assert_close(q_on, expected, rtol=0, atol=0)

    def test_gate_bias_is_applied_per_head(self):
        attn = self._attention(nope=False, attention_output_gate=True, attention_output_gate_bias=True)
        assert isinstance(attn.output_gate_bias, AttentionOutputGateBias)
        hidden = torch.randn(SEQ, BATCH, HEADS * HDIM, device="cuda")
        with torch.no_grad():
            before = attn(hidden, attention_mask=None)[0]
            attn.output_gate_bias.bias.copy_(torch.linspace(-2.0, 2.0, HEADS))
            after = attn(hidden, attention_mask=None)[0]
        assert not torch.equal(before, after)
