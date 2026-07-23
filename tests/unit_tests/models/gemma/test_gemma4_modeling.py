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

"""CPU-only unit tests for Gemma 4 modeling helpers."""

import types
import weakref
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb

from megatron.bridge.models.gemma.modeling_gemma4 import (
    Gemma4DenseMLP,
    Gemma4DenseRotaryEmbedding,
    Gemma4DenseSelfAttention,
    Gemma4DenseTransformerLayer,
    Gemma4MoEExperts,
    Gemma4MoELayer,
    Gemma4MoERouter,
    Gemma4OutputLayer,
    Gemma4RMSNorm,
    Gemma4RotaryEmbedding,
    Gemma4SelfAttention,
    Gemma4TEDotProductAttention,
    Gemma4TopKRouter,
    Gemma4TransformerLayer,
    _attach_ple_modules,
    _compute_per_layer_inputs,
    _gemma4_block_spec,
    _gemma4_checkpointed_forward,
    _gemma4_layer_input,
    _install_ple_forward,
    _install_tied_kv,
    _is_gemma4_sliding_layer,
    _logit_softcapping,
    _patch_ple_block_threading,
    get_gemma4_layer_spec,
    wire_gemma4_kv_sharing,
)


def _config(**kwargs):
    defaults = {
        "hidden_size": 4,
        "num_experts": 3,
        "top_k_experts": 2,
        "layernorm_epsilon": 1e-6,
        "moe_intermediate_size": 3,
        "window_size": (511, 0),
        "window_attn_skip_freq": ["sliding_attention", "full_attention"],
        "kv_channels": 8,
        "global_kv_channels": 8,
        "rotary_interleaved": False,
        "sliding_window_rope_base": 10_000.0,
        "full_attention_rope_base": 1_000_000.0,
        "full_attention_rope_partial_factor": 0.5,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestGemma4RMSNorm:
    def test_matches_hf_style_rms_norm(self):
        norm = Gemma4RMSNorm(_config(), hidden_size=2, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2.0, 3.0]))
        hidden_states = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

        out = norm(hidden_states)

        expected = hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        expected = expected * torch.tensor([2.0, 3.0])
        torch.testing.assert_close(out, expected)

    def test_without_scale_has_no_weight(self):
        norm = Gemma4RMSNorm(_config(), hidden_size=2, with_scale=False)

        assert not hasattr(norm, "weight")

    def test_weight_uses_model_parameter_dtype(self):
        norm = Gemma4RMSNorm(
            _config(params_dtype=torch.bfloat16),
            hidden_size=2,
        )

        assert norm.weight.dtype is torch.bfloat16

    def test_weight_is_marked_for_sequence_parallel_reduction(self):
        norm = Gemma4RMSNorm(
            _config(params_dtype=torch.bfloat16, sequence_parallel=True),
            hidden_size=2,
        )

        assert norm.weight.sequence_parallel is True


class TestGemma4MoE:
    def test_router_returns_normalized_topk_weights(self):
        router = Gemma4MoERouter(_config(hidden_size=4, num_experts=3, top_k_experts=2))
        with torch.no_grad():
            router.proj.weight.zero_()
            router.per_expert_scale.fill_(1.0)
        hidden_states = torch.ones(5, 4)

        router_probs, top_k_weights, top_k_index = router(hidden_states)

        assert router_probs.shape == (5, 3)
        assert top_k_weights.shape == (5, 2)
        assert top_k_index.shape == (5, 2)
        torch.testing.assert_close(top_k_weights.sum(dim=-1), torch.ones(5))

    def test_experts_return_hidden_shape(self):
        experts = Gemma4MoEExperts(_config(hidden_size=4, num_experts=2, moe_intermediate_size=3))
        hidden_states = torch.ones(2, 4)
        top_k_index = torch.tensor([[0], [1]])
        top_k_weights = torch.ones(2, 1)

        out = experts(hidden_states, top_k_index, top_k_weights)

        assert out.shape == hidden_states.shape


class TestGemma4LayerSpec:
    @pytest.mark.parametrize(
        ("skip_freq", "layer_number", "expected"),
        [
            (["sliding_attention", "full_attention"], 1, True),
            (["sliding_attention", "full_attention"], 2, False),
            ([1, 0], 1, True),
            ([1, 0], 2, False),
        ],
    )
    def test_is_gemma4_sliding_layer_from_list(self, skip_freq, layer_number, expected):
        cfg = _config(window_attn_skip_freq=skip_freq)

        assert _is_gemma4_sliding_layer(cfg, layer_number) is expected

    def test_is_gemma4_sliding_layer_returns_false_without_window(self):
        cfg = _config(window_size=None)

        assert _is_gemma4_sliding_layer(cfg, 1) is False

    def test_is_gemma4_sliding_layer_uses_window_attention_helper_for_non_list(self):
        cfg = _config(window_attn_skip_freq=2)

        assert _is_gemma4_sliding_layer(cfg, 1) is True
        assert _is_gemma4_sliding_layer(cfg, 2) is False

    def test_get_gemma4_layer_spec_uses_dense_components(self):
        layer_spec = get_gemma4_layer_spec()

        assert layer_spec.module is Gemma4DenseTransformerLayer
        assert layer_spec.submodules.self_attention.module is Gemma4DenseSelfAttention
        assert layer_spec.submodules.post_self_attn_layernorm is Gemma4RMSNorm
        assert layer_spec.submodules.post_mlp_layernorm is Gemma4RMSNorm

    def test_double_wide_mlp_only_applies_to_shared_kv_layers(self, monkeypatch):
        mlp_builders = []

        def fake_layer_init(self, config, submodules, layer_number=1, **kwargs):
            del kwargs
            torch.nn.Module.__init__(self)
            self.config = config
            self.layer_number = layer_number
            mlp_builders.append(submodules.mlp)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TransformerLayer.__init__",
            fake_layer_init,
        )
        config = _config(
            hidden_size=4,
            ffn_hidden_size=6,
            num_layers=4,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            per_layer_embed_dim=0,
            enable_moe_block=False,
        )

        Gemma4DenseTransformerLayer(config, get_gemma4_layer_spec().submodules, layer_number=2)
        Gemma4DenseTransformerLayer(config, get_gemma4_layer_spec().submodules, layer_number=3)

        assert mlp_builders[0].func.__self__ is Gemma4DenseMLP
        assert mlp_builders[0].keywords["ffn_hidden_size"] == 6
        assert mlp_builders[1].func.__self__ is Gemma4DenseMLP
        assert mlp_builders[1].keywords["ffn_hidden_size"] == 12

    def test_dense_mlp_sets_width_on_layer_config_and_constructor(self, monkeypatch):
        captured = {}

        def fake_mlp_init(self, config, submodules, ffn_hidden_size=None, **kwargs):
            del submodules, kwargs
            torch.nn.Module.__init__(self)
            captured["config"] = config
            captured["ffn_hidden_size"] = ffn_hidden_size

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.MLP.__init__", fake_mlp_init)
        config = _config(ffn_hidden_size=6)

        Gemma4DenseMLP(config, submodules=SimpleNamespace(), ffn_hidden_size=12)

        assert captured["config"] is not config
        assert config.ffn_hidden_size == 6
        assert captured["config"].ffn_hidden_size == 12
        assert captured["ffn_hidden_size"] == 12


class TestGemma4DenseSelfAttention:
    def test_init_marks_shared_layer_and_source_index(self, monkeypatch):
        init_configs = []

        def fake_init(self, config, submodules, layer_number, *args, **kwargs):
            del submodules, args, kwargs
            self.config = config
            self.layer_number = layer_number
            init_configs.append(config)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.__init__",
            fake_init,
        )
        cfg = _config(
            softmax_scale=None,
            num_layers=6,
            num_kv_shared_layers=2,
            window_attn_skip_freq=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            attention_k_eq_v=True,
        )

        attn = Gemma4DenseSelfAttention(cfg, submodules=object(), layer_number=5)

        assert init_configs[0].softmax_scale == 1.0
        assert init_configs[0].qk_layernorm is True
        assert attn.is_gemma4_sliding_layer is True
        assert attn.is_kv_shared_layer is True
        assert attn.kv_shared_layer_index == 2
        assert attn.store_full_length_kv is False
        assert attn.attention_k_eq_v is False

    def test_init_sets_global_attention_config_and_store_full_length_kv(self, monkeypatch):
        def fake_init(self, config, submodules, layer_number, *args, **kwargs):
            del submodules, args, kwargs
            self.config = config
            self.layer_number = layer_number

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.__init__",
            fake_init,
        )
        cfg = _config(
            softmax_scale=0.5,
            num_layers=6,
            num_kv_shared_layers=2,
            window_attn_skip_freq=2,
            global_kv_channels=16,
            num_global_query_groups=3,
            attention_k_eq_v=True,
        )

        attn = Gemma4DenseSelfAttention(cfg, submodules=object(), layer_number=4)

        assert attn.config.softmax_scale == 0.5
        assert attn.config.kv_channels == 16
        assert attn.config.num_query_groups == 3
        assert attn.is_gemma4_sliding_layer is False
        assert attn.attention_k_eq_v is True
        assert attn.is_kv_shared_layer is False
        assert attn.store_full_length_kv is True

    def _make_attention_for_methods(self):
        attn = object.__new__(Gemma4DenseSelfAttention)
        attn.is_kv_shared_layer = False
        attn.attention_k_eq_v = False
        attn.store_full_length_kv = False
        attn._stored_kv = None
        attn._kv_source_ref = None
        attn.is_gemma4_sliding_layer = True
        attn.layer_number = 2
        attn.config = SimpleNamespace(
            num_layers=4,
            num_query_groups=2,
            test_mode=False,
        )
        attn.original_config = _config(
            num_layers=4,
            window_attn_skip_freq=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        )
        attn.hidden_size_per_attention_head = 2
        attn.world_size = 1
        attn.num_attention_heads_per_partition = 2
        attn.pg_collection = SimpleNamespace(tp=None)
        attn.q_layernorm = None
        attn.k_layernorm = None
        return attn

    def test_sharded_state_dict_uses_sliding_or_global_prefix(self, monkeypatch):
        calls = []

        def fake_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            calls.append((prefix, sharded_offsets, metadata))
            return {"plain": object()}

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.sharded_state_dict",
            fake_sharded_state_dict,
        )
        attn = self._make_attention_for_methods()

        result = Gemma4DenseSelfAttention.sharded_state_dict(attn, prefix="decoder.layers.0.self_attention.")

        assert result.keys() == {"plain"}
        assert calls[0][0] == "decoder.layers.0.self_attention_sliding."

        calls.clear()
        attn.is_gemma4_sliding_layer = False
        Gemma4DenseSelfAttention.sharded_state_dict(attn, prefix="attention")
        assert calls[0][0] == "attention_global"

    def test_sharded_state_dict_remaps_dense_layer_axis_metadata(self, monkeypatch):
        from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor

        tensor = ShardedTensor(
            key="weight",
            data=torch.zeros(2),
            dtype=torch.float32,
            local_shape=(2,),
            global_shape=(4, 2),
            global_offset=(2, 0),
            axis_fragmentations=(4, 1),
            prepend_axis_num=1,
        )
        obj = ShardedObject(key="obj", data={"x": 1}, global_shape=(4,), global_offset=(2,))
        untouched = ShardedObject(key="plain", data={"x": 1}, global_shape=(3,), global_offset=(0,))

        def fake_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            del self, prefix, sharded_offsets, metadata
            return {"tensor": tensor, "nested": {"object": obj, "untouched": untouched}}

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.sharded_state_dict",
            fake_sharded_state_dict,
        )
        attn = self._make_attention_for_methods()
        attn.layer_number = 3
        attn.is_gemma4_sliding_layer = True

        out = Gemma4DenseSelfAttention.sharded_state_dict(attn)

        assert out["tensor"].global_shape == (2, 2)
        assert out["tensor"].global_offset == (1, 0)
        assert out["tensor"].axis_fragmentations == (2, 1)
        assert out["nested"]["object"].global_shape == (2,)
        assert out["nested"]["object"].global_offset == (1,)
        assert out["nested"]["untouched"] is untouched

    def test_get_k_eq_v_query_key_value_tensors_splits_and_reshapes(self, monkeypatch):
        mixed = torch.arange(2 * 1 * 1 * 8, dtype=torch.float32).view(2, 1, 1, 8)

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states, output_gate, split_qkv
            return mixed, [4, 2, 2]

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention_for_methods()

        query, key, raw_key = Gemma4DenseSelfAttention._get_k_eq_v_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
        )

        assert query.shape == (2, 1, 2, 2)
        torch.testing.assert_close(key, mixed[..., 4:6])
        torch.testing.assert_close(raw_key, mixed[..., 4:6])

    def test_get_k_eq_v_query_key_value_tensors_slices_tp_and_applies_norms(self, monkeypatch):
        class AddModule(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return x + self.value

        mixed = torch.arange(2 * 1 * 1 * 12, dtype=torch.float32).view(2, 1, 1, 12)
        realtime_calls = []

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states, output_gate, split_qkv
            return mixed, [8, 2, 2]

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.get_pg_rank", lambda tp: 1)
        attn = self._make_attention_for_methods()
        attn.config.num_query_groups = 1
        attn.config.test_mode = True
        attn.world_size = 2
        attn.num_attention_heads_per_partition = 4
        object.__setattr__(attn, "q_layernorm", AddModule(10.0))
        object.__setattr__(attn, "k_layernorm", AddModule(20.0))
        attn.run_realtime_tests = lambda: realtime_calls.append(True)

        query, key, raw_key = Gemma4DenseSelfAttention._get_k_eq_v_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
        )

        assert query.shape == (2, 1, 2, 2)
        torch.testing.assert_close(query, mixed[..., :8].reshape(2, 1, 4, 2)[:, :, 2:4, :] + 10.0)
        torch.testing.assert_close(key, mixed[..., 8:10] + 20.0)
        torch.testing.assert_close(raw_key, mixed[..., 8:10])
        assert realtime_calls == [True]

    def test_shared_layer_reuses_source_kv_when_available(self, monkeypatch):
        query = torch.ones(2, 1, 1, 2)
        fallback_key = torch.full_like(query, 2.0)
        fallback_value = torch.full_like(query, 3.0)

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states, output_gate, split_qkv
            return query, fallback_key, fallback_value

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention_for_methods()
        attn.is_kv_shared_layer = True
        source_key = torch.full_like(query, 4.0)
        source_value = torch.full_like(query, 5.0)

        class Source:
            pass

        source = Source()
        source._stored_kv = (source_key, source_value)
        attn._kv_source_ref = weakref.ref(source)

        out_query, out_key, out_value = Gemma4DenseSelfAttention.get_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
        )

        assert out_query is query
        torch.testing.assert_close(out_key, source_key)
        torch.testing.assert_close(out_value, source_value)

    def test_shared_layer_normalizes_fallback_kv_when_source_missing(self, monkeypatch):
        query = torch.ones(2, 1, 1, 2)
        fallback_key = torch.full_like(query, 2.0)
        fallback_value = torch.full_like(query, 3.0)

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states, output_gate, split_qkv
            return query, fallback_key, fallback_value

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention_for_methods()
        attn.is_kv_shared_layer = True
        attn._kv_source_ref = None

        out_query, out_key, out_value = Gemma4DenseSelfAttention.get_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
        )

        assert out_query is query
        assert out_key is fallback_key
        torch.testing.assert_close(out_value, torch.ones_like(fallback_value))

    def test_shared_layer_delegates_unsupported_qkv_modes(self, monkeypatch):
        result = (torch.ones(1), [1])

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states
            assert output_gate is False
            assert split_qkv is False
            return result

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention_for_methods()
        attn.is_kv_shared_layer = True

        out = Gemma4DenseSelfAttention.get_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(1),
            split_qkv=False,
        )

        assert out is result

    def test_get_query_key_value_tensors_ties_value_and_stores_kv(self, monkeypatch):
        query = torch.ones(2, 1, 1, 2)
        key = torch.full_like(query, 2.0)
        raw_value = torch.full_like(query, 7.0)

        def fake_k_eq_v(self, hidden_states, key_value_states=None):
            del self, hidden_states, key_value_states
            return query, key, raw_value

        attn = self._make_attention_for_methods()
        attn.attention_k_eq_v = True
        attn.store_full_length_kv = True
        attn._get_k_eq_v_query_key_value_tensors = types.MethodType(fake_k_eq_v, attn)

        out_query, out_key, out_value = Gemma4DenseSelfAttention.get_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
        )

        assert out_query is query
        assert out_key is key
        torch.testing.assert_close(out_value, torch.ones_like(raw_value))
        stored_key, stored_value = attn._stored_kv
        assert stored_key is key
        torch.testing.assert_close(stored_value, torch.ones_like(raw_value))

    def test_get_query_key_value_tensors_output_gate_ties_value(self, monkeypatch):
        query = torch.ones(2, 1, 1, 2)
        key = torch.full_like(query, 2.0)
        value = torch.full_like(query, 9.0)
        gate = torch.full_like(query, 4.0)

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states, output_gate, split_qkv
            return query, key, value, gate

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention_for_methods()
        attn.attention_k_eq_v = True

        out_query, out_key, out_value, out_gate = Gemma4DenseSelfAttention.get_query_key_value_tensors(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
            output_gate=True,
        )

        assert out_query is query
        assert out_key is key
        assert out_gate is gate
        torch.testing.assert_close(out_value, torch.ones_like(key))

    def test_forward_selects_attention_mask_from_dict(self, monkeypatch):
        calls = []

        def fake_forward(self, hidden_states, attention_mask, *args, **kwargs):
            del self, args, kwargs
            calls.append(attention_mask)
            return hidden_states, None

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.forward",
            fake_forward,
        )
        attn = self._make_attention_for_methods()
        hidden_states = torch.zeros(2, 1, 4)
        sliding_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        full_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

        out, bias = Gemma4DenseSelfAttention.forward(
            attn,
            hidden_states,
            {"sliding_attention": sliding_mask, "full_attention": full_mask},
        )

        assert out is hidden_states
        assert bias is None
        assert calls[0] is sliding_mask


class TestGemma4SelfAttention:
    def _make_attention(self, *, layer_number):
        attn = object.__new__(Gemma4SelfAttention)
        object.__setattr__(attn, "layer_number", layer_number)
        object.__setattr__(
            attn,
            "config",
            SimpleNamespace(
                interleaved_attn_pattern=(1, 1),
                num_layers=4,
            ),
        )
        return attn

    def test_sharded_state_dict_remaps_global_layer_offsets(self, monkeypatch):
        from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor

        prefix = "layers.3.self_attention."
        tensor_key = f"{prefix}linear_qkv.weight"
        object_key = f"{prefix}linear_qkv._extra_state"
        tensor = ShardedTensor(
            key=tensor_key,
            data=torch.zeros(2),
            dtype=torch.float32,
            local_shape=(2,),
            global_shape=(4, 2),
            global_offset=(3, 0),
            axis_fragmentations=(4, 1),
            prepend_axis_num=1,
        )
        untouched = ShardedTensor(
            key="untouched",
            data=torch.zeros(1, 2),
            dtype=torch.float32,
            local_shape=(1, 2),
            global_shape=(4, 2),
            global_offset=(3, 0),
            axis_fragmentations=(4, 1),
            prepend_axis_num=0,
        )
        obj = ShardedObject(key=object_key, data={"x": 1}, global_shape=(4, 2), global_offset=(3, 0))
        calls = []

        def fake_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            del self
            calls.append((prefix, sharded_offsets, metadata))
            return {tensor_key: tensor, object_key: obj, "nested": {"untouched": untouched}, "plain": object()}

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.sharded_state_dict",
            fake_sharded_state_dict,
        )
        attn = self._make_attention(layer_number=4)

        out = Gemma4SelfAttention.sharded_state_dict(attn, prefix=prefix)

        assert tensor_key in out
        assert object_key in out
        assert calls[0][0] == prefix
        assert out[tensor_key].key == "layers.3.self_attention_global.linear_qkv.weight"
        assert out[tensor_key].global_shape == (2, 2)
        assert out[tensor_key].global_offset == (1, 0)
        assert out[tensor_key].axis_fragmentations == (2, 1)
        assert out[object_key].key == "layers.3.self_attention_global.linear_qkv._extra_state"
        assert out[object_key].global_shape == (2, 2)
        assert out[object_key].global_offset == (1, 0)
        assert out["nested"]["untouched"].key == "untouched"
        assert out["nested"]["untouched"].global_shape == (4, 2)
        assert out["nested"]["untouched"].global_offset == (3, 0)

    def test_sharded_state_dict_remaps_sliding_layer_offsets_without_dot_prefix(self, monkeypatch):
        from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor

        prefix = "self_attention"
        tensor_key = f"{prefix}.weight"
        object_key = f"{prefix}.obj"
        tensor = ShardedTensor(
            key=tensor_key,
            data=torch.zeros(2),
            dtype=torch.float32,
            local_shape=(2,),
            global_shape=(4, 2),
            global_offset=(2, 0),
            axis_fragmentations=None,
            prepend_axis_num=1,
        )
        obj = ShardedObject(key=object_key, data={"x": 1}, global_shape=(4,), global_offset=(2,))
        calls = []

        def fake_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            del self
            calls.append((prefix, sharded_offsets, metadata))
            return {tensor_key: tensor, object_key: obj}

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.sharded_state_dict",
            fake_sharded_state_dict,
        )
        attn = self._make_attention(layer_number=3)

        out = Gemma4SelfAttention.sharded_state_dict(attn, prefix=prefix)

        assert calls[0][0] == prefix
        assert tensor_key in out
        assert object_key in out
        assert out[tensor_key].key == "self_attention_sliding.weight"
        assert out[tensor_key].global_shape == (2, 2)
        assert out[tensor_key].global_offset == (1, 0)
        assert out[tensor_key].axis_fragmentations is None
        assert out[object_key].key == "self_attention_sliding.obj"
        assert out[object_key].global_shape == (2,)
        assert out[object_key].global_offset == (1,)

    def test_get_query_key_value_tensors_returns_short_super_result(self, monkeypatch):
        expected = (torch.ones(1), torch.zeros(1))

        def fake_get_qkv(self, hidden_states, key_value_states=None, **kwargs):
            del self, hidden_states, key_value_states, kwargs
            return expected

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention(layer_number=1)
        attn._v_norm_eps = 1e-6

        out = Gemma4SelfAttention.get_query_key_value_tensors(attn, torch.zeros(1))

        assert out is expected

    def test_get_query_key_value_tensors_normalizes_tied_value_from_raw_key(self, monkeypatch):
        query = torch.tensor([[[[1.0, 2.0]]]])
        raw_key = torch.tensor([[[[3.0, 4.0]]]])
        unused_value = torch.tensor([[[[5.0, 6.0]]]])
        mixed_qkv = torch.cat((query, raw_key, unused_value), dim=-1)

        def fake_get_qkv(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            del self, hidden_states, key_value_states
            assert output_gate is False
            assert split_qkv is False
            return mixed_qkv, [2, 2, 2]

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.get_query_key_value_tensors",
            fake_get_qkv,
        )
        attn = self._make_attention(layer_number=2)
        attn.config.num_query_groups = 1
        attn.config.test_mode = False
        attn.hidden_size_per_attention_head = 2
        attn.world_size = 1

        class ScaleKey(torch.nn.Module):
            def forward(self, tensor):
                return tensor * torch.tensor([2.0, 1.0])

        object.__setattr__(attn, "q_layernorm", torch.nn.Identity())
        object.__setattr__(attn, "k_layernorm", ScaleKey())
        attn._tied_kv = True
        attn._v_norm_eps = 1e-6

        out_query, out_key, out_value = Gemma4SelfAttention.get_query_key_value_tensors(
            attn,
            torch.zeros(1),
        )

        torch.testing.assert_close(out_query, query)
        torch.testing.assert_close(out_key, raw_key * torch.tensor([2.0, 1.0]))
        expected_value = raw_key / torch.sqrt(raw_key.pow(2).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(out_value, expected_value)
        assert not torch.allclose(
            out_value,
            out_key / torch.sqrt(out_key.pow(2).mean(-1, keepdim=True) + 1e-6),
        )

    def test_forward_selects_local_mask_and_rotary_embedding(self, monkeypatch):
        calls = {}

        def fake_forward(self, **kwargs):
            del self
            calls.update(kwargs)
            return "out", "bias"

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.forward", fake_forward)
        attn = self._make_attention(layer_number=1)
        hidden_states = torch.zeros(2, 1, 4)
        sliding_mask = object()
        full_mask = object()
        local_rope = object()
        global_rope = object()

        out = Gemma4SelfAttention.forward(
            attn,
            hidden_states=hidden_states,
            attention_mask={"sliding_attention": sliding_mask, "full_attention": full_mask},
            rotary_pos_emb=(local_rope, global_rope),
        )

        assert out == ("out", "bias")
        assert calls["hidden_states"] is hidden_states
        assert calls["attention_mask"] is sliding_mask
        assert calls["rotary_pos_emb"] is local_rope

    def test_forward_selects_global_mask_and_rotary_embedding(self, monkeypatch):
        calls = {}

        def fake_forward(self, **kwargs):
            del self
            calls.update(kwargs)
            return "out", "bias"

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.SelfAttention.forward", fake_forward)
        attn = self._make_attention(layer_number=2)
        global_mask = object()
        global_rope = object()

        Gemma4SelfAttention.forward(
            attn,
            hidden_states=torch.zeros(2, 1, 4),
            attention_mask={"sliding_attention": object(), "full_attention": global_mask},
            rotary_pos_emb=(object(), global_rope),
        )

        assert calls["attention_mask"] is global_mask
        assert calls["rotary_pos_emb"] is global_rope


class TestGemma4TEDotProductAttention:
    def test_init_sets_local_window_size(self, monkeypatch):
        calls = []

        def fake_init(self, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TEDotProductAttention.__init__",
            fake_init,
        )
        cfg = SimpleNamespace(interleaved_attn_pattern=(1, 1), window_size=512)

        Gemma4TEDotProductAttention(
            config=cfg,
            layer_number=1,
            attn_mask_type=object(),
            attention_type="self",
            attention_dropout=0.0,
        )

        assert calls[0]["config"].window_size == (511, 0)

    def test_init_clears_global_window_size(self, monkeypatch):
        calls = []

        def fake_init(self, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TEDotProductAttention.__init__",
            fake_init,
        )
        cfg = SimpleNamespace(interleaved_attn_pattern=(1, 1), window_size=512)

        Gemma4TEDotProductAttention(
            config=cfg,
            layer_number=2,
            attn_mask_type=object(),
            attention_type="self",
        )

        assert calls[0]["config"].window_size is None


class TestGemma4RotaryEmbeddings:
    def test_dense_rotary_uses_full_attention_partial_factor(self):
        rotary = Gemma4DenseRotaryEmbedding(_config(), use_cpu_initialization=True)

        assert rotary.rope_full.inv_freq.numel() == 4
        torch.testing.assert_close(rotary.rope_full.inv_freq[-2:], torch.zeros(2))

    def test_moe_rotary_builds_local_and_global_embeddings(self):
        rotary = Gemma4RotaryEmbedding(
            kv_channels=8,
            rotary_percent=1.0,
            rotary_base=1_000_000,
            rotary_base_local=10_000,
            global_kv_channels=16,
            global_rotary_percent=0.25,
            use_cpu_initialization=True,
        )

        assert rotary.inv_freq.numel() == 8
        expected_rotated = 1.0 / (1_000_000 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 16))
        torch.testing.assert_close(rotary.inv_freq[:2], expected_rotated)
        torch.testing.assert_close(rotary.inv_freq[2:], torch.zeros(6))
        assert rotary.rope_local.inv_freq.numel() == 4

    def test_moe_global_rotary_matches_hf_proportional_coordinate_layout(self):
        head_dim = 16
        partial_rotary_factor = 0.25
        rotary_base = 1_000_000
        position = 3
        rotary = Gemma4RotaryEmbedding(
            kv_channels=8,
            rotary_percent=1.0,
            rotary_base=rotary_base,
            rotary_base_local=10_000,
            global_kv_channels=head_dim,
            global_rotary_percent=partial_rotary_factor,
            use_cpu_initialization=True,
        )

        hidden_states = torch.arange(1, head_dim + 1, dtype=torch.float32).view(1, 1, 1, head_dim)
        freqs = rotary.get_freqs_non_repeated(1, offset=position)
        freqs = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
        config = SimpleNamespace(apply_rope_fusion=False, rotary_interleaved=False)
        actual = apply_rotary_pos_emb(hidden_states, freqs, config, cp_group=object())

        expected = hidden_states.clone()
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        angles = position / (rotary_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim))
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        left = hidden_states[..., :rope_angles]
        right = hidden_states[..., head_dim // 2 : head_dim // 2 + rope_angles]
        expected[..., :rope_angles] = left * cos - right * sin
        expected[..., head_dim // 2 : head_dim // 2 + rope_angles] = right * cos + left * sin

        torch.testing.assert_close(actual, expected)

    def test_dense_rotary_forwards_to_sliding_and_full_rope(self):
        class FakeRope:
            def __init__(self, name):
                self.name = name
                self.calls = []

            def __call__(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
                self.calls.append((max_seq_len, offset, packed_seq, cp_group))
                return f"{self.name}-emb"

            def get_rotary_seq_len(self, *args, **kwargs):
                self.calls.append(("seq", args, kwargs))
                return 123

            def get_cos_sin(self, max_seq_len, offset=0):
                self.calls.append(("cos", max_seq_len, offset))
                return f"{self.name}-cos-sin"

        rotary = object.__new__(Gemma4DenseRotaryEmbedding)
        object.__setattr__(rotary, "rope_sliding", FakeRope("sliding"))
        object.__setattr__(rotary, "rope_full", FakeRope("full"))

        out = Gemma4DenseRotaryEmbedding.forward(rotary, 8, offset=2, packed_seq=True, cp_group="pg")
        seq_len = Gemma4DenseRotaryEmbedding.get_rotary_seq_len(rotary, "hidden", sequence_len_offset=1)
        cos_sin = Gemma4DenseRotaryEmbedding.get_cos_sin(rotary, 4, offset=1)

        assert out == ("sliding-emb", "full-emb")
        assert rotary.rope_sliding.calls[0] == (8, 2, True, "pg")
        assert rotary.rope_full.calls[0] == (8, 2, True, "pg")
        assert seq_len == 123
        assert rotary.rope_sliding.calls[1] == ("seq", ("hidden",), {"sequence_len_offset": 1})
        assert cos_sin == ("sliding-cos-sin", "full-cos-sin")

    def test_moe_rotary_forward_uses_cached_path_without_cp_group(self, monkeypatch):
        class FakeLocalRope:
            def __init__(self):
                self.calls = []

            def forward(self, max_seq_len, offset, packed_seq, cp_group):
                self.calls.append((max_seq_len, offset, packed_seq, cp_group))
                return "local"

        global_calls = []

        def fake_base_forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
            del self
            global_calls.append((max_seq_len, offset, packed_seq, cp_group))
            return "global"

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.RotaryEmbedding.forward",
            fake_base_forward,
        )
        rotary = object.__new__(Gemma4RotaryEmbedding)
        object.__setattr__(rotary, "rope_local", FakeLocalRope())

        first = Gemma4RotaryEmbedding.forward(rotary, 8, offset=2, packed_seq=True)
        second = Gemma4RotaryEmbedding.forward(rotary, 8, offset=2, packed_seq=True)

        assert first == ("local", "global")
        assert second == first
        assert global_calls == [(8, 2, True, None)]
        assert rotary.rope_local.calls == [(8, 2, True, None)]

    def test_moe_rotary_forward_bypasses_cache_with_cp_group(self, monkeypatch):
        class FakeLocalRope:
            def __init__(self):
                self.calls = []

            def forward(self, max_seq_len, offset, packed_seq, cp_group):
                self.calls.append((max_seq_len, offset, packed_seq, cp_group))
                return "local"

        global_calls = []

        def fake_base_forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
            del self
            global_calls.append((max_seq_len, offset, packed_seq, cp_group))
            return "global"

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.RotaryEmbedding.forward",
            fake_base_forward,
        )
        rotary = object.__new__(Gemma4RotaryEmbedding)
        object.__setattr__(rotary, "rope_local", FakeLocalRope())
        cp_group = object()

        out = Gemma4RotaryEmbedding.forward(rotary, 8, offset=1, packed_seq=False, cp_group=cp_group)

        assert out == ("local", "global")
        assert global_calls == [(8, 1, False, cp_group)]
        assert rotary.rope_local.calls == [(8, 1, False, cp_group)]


class TestGemma4DenseTransformerLayerForward:
    def _make_layer(self, *, layer_number=1, fp32_residual_connection=True):
        layer = object.__new__(Gemma4DenseTransformerLayer)
        object.__setattr__(
            layer,
            "config",
            SimpleNamespace(
                fp32_residual_connection=fp32_residual_connection,
                bias_dropout_fusion=False,
                window_size=(511, 0),
                window_attn_skip_freq=["sliding_attention", "full_attention"],
            ),
        )
        object.__setattr__(layer, "layer_number", layer_number)
        object.__setattr__(layer, "training", False)
        object.__setattr__(layer, "hidden_dropout", 0.0)
        object.__setattr__(layer, "bias_dropout_add_exec_handler", lambda: nullcontext())
        return layer

    def test_forward_attention_uses_sliding_rotary_tuple_paths_and_fp32_residual(self):
        layer = self._make_layer(layer_number=1, fp32_residual_connection=True)
        hidden_states = torch.ones(2, 1, 4, dtype=torch.bfloat16)
        residual = torch.full_like(hidden_states, 2.0)
        attn_bias = torch.full_like(hidden_states, 0.5)
        calls = {}

        object.__setattr__(layer, "input_layernorm", lambda x: (x + 1, residual))

        def self_attention(hidden, **kwargs):
            calls["attention_hidden"] = hidden
            calls["rotary_pos_emb"] = kwargs["rotary_pos_emb"]
            calls["attention_mask"] = kwargs["attention_mask"]
            return torch.full_like(hidden, 3.0), attn_bias

        object.__setattr__(layer, "self_attention", self_attention)
        object.__setattr__(layer, "post_self_attn_layernorm", lambda x: x.float() + 4.0)

        def self_attn_bda(training, bias_dropout_fusion):
            assert training is False
            assert bias_dropout_fusion is False

            def apply(attention_output_with_bias, residual_arg, hidden_dropout):
                assert hidden_dropout == 0.0
                calls["residual_dtype"] = residual_arg.dtype
                attn_out, bias = attention_output_with_bias
                return attn_out + bias.float() + residual_arg

            return apply

        object.__setattr__(layer, "self_attn_bda", self_attn_bda)
        rotary_sliding = object()
        rotary_full = object()

        out, context = Gemma4DenseTransformerLayer._forward_attention(
            layer,
            hidden_states,
            attention_mask="mask",
            rotary_pos_emb=(rotary_sliding, rotary_full),
        )

        assert context is None
        assert calls["attention_mask"] == "mask"
        assert calls["rotary_pos_emb"] is rotary_sliding
        assert calls["residual_dtype"] == torch.float32
        torch.testing.assert_close(out, torch.full((2, 1, 4), 9.5))

    def test_forward_attention_uses_full_rotary_and_tensor_paths(self):
        layer = self._make_layer(layer_number=2, fp32_residual_connection=False)
        hidden_states = torch.ones(2, 1, 4)
        calls = {}

        object.__setattr__(layer, "input_layernorm", lambda x: x + 1.0)

        def self_attention(hidden, **kwargs):
            calls["rotary_pos_emb"] = kwargs["rotary_pos_emb"]
            return hidden + 2.0

        object.__setattr__(layer, "self_attention", self_attention)
        object.__setattr__(layer, "post_self_attn_layernorm", lambda x: x + 3.0)
        object.__setattr__(
            layer, "self_attn_bda", lambda training, fusion: lambda out, residual, dropout: out + residual
        )
        rotary_sliding = object()
        rotary_full = object()

        out, _ = Gemma4DenseTransformerLayer._forward_attention(
            layer,
            hidden_states,
            rotary_pos_emb=(rotary_sliding, rotary_full),
        )

        assert calls["rotary_pos_emb"] is rotary_full
        torch.testing.assert_close(out, torch.full_like(hidden_states, 8.0))

    def test_forward_mlp_combines_dense_and_moe_tuple_output(self):
        layer = self._make_layer(fp32_residual_connection=True)
        hidden_states = torch.ones(2, 1, 4, dtype=torch.bfloat16)
        residual = torch.full_like(hidden_states, 3.0)
        mlp_bias = torch.full_like(hidden_states, 0.25)

        object.__setattr__(layer, "_forward_pre_mlp_layernorm", lambda x: (x + 1.0, residual))
        object.__setattr__(layer, "mlp", lambda hidden, padding_mask=None: (torch.full_like(hidden, 5.0), mlp_bias))
        object.__setattr__(layer, "post_feedforward_layernorm_1", lambda x: x.float() + 10.0)
        object.__setattr__(layer, "pre_feedforward_layernorm_2", lambda x: x + 1.0)

        def moe_router(hidden_flat):
            assert hidden_flat.shape == (2, 4)
            return None, torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long)

        object.__setattr__(layer, "moe_router", moe_router)
        object.__setattr__(
            layer, "moe_experts", lambda hidden, top_k_index, top_k_weights: torch.full_like(hidden, 7.0)
        )
        object.__setattr__(layer, "post_feedforward_layernorm_2", lambda x: x + 20.0)
        object.__setattr__(layer, "post_mlp_layernorm", lambda x: x + 100.0)

        def mlp_bda(training, bias_dropout_fusion):
            assert training is False
            assert bias_dropout_fusion is False

            def apply(mlp_output_with_bias, residual_arg, hidden_dropout):
                assert hidden_dropout == 0.0
                mlp_out, bias = mlp_output_with_bias
                return mlp_out + bias.float() + residual_arg

            return apply

        object.__setattr__(layer, "mlp_bda", mlp_bda)

        out = Gemma4DenseTransformerLayer._forward_mlp(layer, hidden_states, padding_mask="mask")

        torch.testing.assert_close(out, torch.full((2, 1, 4), 145.25))

    def test_forward_mlp_without_moe_uses_tensor_paths(self):
        layer = self._make_layer(fp32_residual_connection=False)
        hidden_states = torch.ones(2, 1, 4)

        object.__setattr__(layer, "_forward_pre_mlp_layernorm", lambda x: x + 1.0)
        object.__setattr__(layer, "mlp", lambda hidden, padding_mask=None: hidden + 2.0)
        object.__setattr__(layer, "moe_router", None)
        object.__setattr__(layer, "post_mlp_layernorm", lambda x: x + 3.0)
        object.__setattr__(layer, "mlp_bda", lambda training, fusion: lambda out, residual, dropout: out + residual)

        out = Gemma4DenseTransformerLayer._forward_mlp(layer, hidden_states)

        torch.testing.assert_close(out, torch.full_like(hidden_states, 8.0))


class TestGemma4SharedKVWiring:
    def test_wire_gemma4_kv_sharing_links_shared_layers_to_sources(self):
        source = object.__new__(Gemma4DenseSelfAttention)
        source.layer_number = 1
        source.is_kv_shared_layer = False
        source.kv_shared_layer_index = None
        source._kv_source_ref = None

        shared = object.__new__(Gemma4DenseSelfAttention)
        shared.layer_number = 3
        shared.is_kv_shared_layer = True
        shared.kv_shared_layer_index = 0
        shared._kv_source_ref = None

        missing = object.__new__(Gemma4DenseSelfAttention)
        missing.layer_number = 4
        missing.is_kv_shared_layer = True
        missing.kv_shared_layer_index = 99
        missing._kv_source_ref = None

        model = SimpleNamespace(modules=lambda: [object(), source, shared, missing])

        wire_gemma4_kv_sharing(model)

        assert shared._kv_source_ref() is source
        assert missing._kv_source_ref is None


class TestGemma4PLEHelpers:
    def test_attach_ple_modules_returns_without_valid_dimensions(self):
        model = SimpleNamespace()
        config = SimpleNamespace(init_method=object())
        provider = SimpleNamespace(num_layers=2, per_layer_embed_dim=0, per_layer_embed_vocab_size=128)

        _attach_ple_modules(model, config, provider)

        assert not hasattr(model, "per_layer_embedding")

    def test_attach_ple_modules_installs_embedding_projection_and_norm(self, monkeypatch):
        calls = []

        class FakeVocabParallelEmbedding:
            def __init__(self, vocab_size, hidden_size, config, init_method):
                calls.append(("embedding", vocab_size, hidden_size, config, init_method))

        class FakeColumnParallelLinear:
            def __init__(self, input_size, output_size, config, init_method, bias, gather_output):
                calls.append(("projection", input_size, output_size, config, init_method, bias, gather_output))

        monkeypatch.setattr(
            "megatron.core.tensor_parallel.VocabParallelEmbedding",
            FakeVocabParallelEmbedding,
        )
        monkeypatch.setattr(
            "megatron.core.tensor_parallel.ColumnParallelLinear",
            FakeColumnParallelLinear,
        )
        model = SimpleNamespace()
        config = _config(init_method="init", layernorm_epsilon=1e-6)
        provider = SimpleNamespace(
            num_layers=3,
            per_layer_embed_dim=2,
            per_layer_embed_vocab_size=128,
            hidden_size=4,
            layernorm_epsilon=1e-5,
        )

        _attach_ple_modules(model, config, provider)

        assert isinstance(model.per_layer_embedding, FakeVocabParallelEmbedding)
        assert isinstance(model.per_layer_model_proj, FakeColumnParallelLinear)
        assert isinstance(model.per_layer_proj_norm, Gemma4RMSNorm)
        assert calls == [
            ("embedding", 128, 6, config, "init"),
            ("projection", 4, 6, config, "init", False, True),
        ]

    def test_compute_per_layer_inputs_combines_token_and_model_projections(self):
        class FakeEmbedding(torch.nn.Module):
            def forward(self, input_ids):
                batch, seq = input_ids.shape
                return torch.ones(batch, seq, 6)

        class FakeProjection(torch.nn.Module):
            def forward(self, hidden_states):
                batch, seq, _ = hidden_states.shape
                return torch.full((batch, seq, 6), 4.0), None

        model = SimpleNamespace(
            config=SimpleNamespace(
                per_layer_embed_dim=3,
                num_layers=2,
                hidden_size=4,
                sequence_parallel=False,
            ),
            per_layer_embedding=FakeEmbedding(),
            per_layer_model_proj=FakeProjection(),
            per_layer_proj_norm=torch.nn.Identity(),
        )
        input_ids = torch.ones(2, 3, dtype=torch.long)
        decoder_input = torch.zeros(3, 2, 4)

        out = _compute_per_layer_inputs(model, input_ids, decoder_input)

        assert out.shape == (2, 3, 2, 3)
        expected_value = (4.0 * (4**-0.5) + (3**0.5)) * (2.0**-0.5)
        torch.testing.assert_close(out, torch.full_like(out, expected_value))

    def test_compute_per_layer_inputs_returns_none_without_modules(self):
        model = SimpleNamespace(per_layer_embedding=None)

        assert _compute_per_layer_inputs(model, torch.ones(1, 2, dtype=torch.long), torch.ones(2, 1, 4)) is None

    def test_gemma4_layer_input_selects_global_layer(self):
        per_layer_inputs = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
        layer = SimpleNamespace(layer_number=2)

        out = _gemma4_layer_input(per_layer_inputs, layer)

        torch.testing.assert_close(out, per_layer_inputs[:, :, 1, :].transpose(0, 1))

    def test_install_ple_forward_injects_per_layer_inputs(self):
        class FakeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()

            def forward(self, *args, **kwargs):
                del args, kwargs
                return None

        class FakeEmbedding(torch.nn.Module):
            def forward(self, input_ids, position_ids=None):
                del position_ids
                seq = input_ids.shape[1]
                batch = input_ids.shape[0]
                return torch.ones(seq, batch, 4)

        class FakePLEmbedding(torch.nn.Module):
            def forward(self, input_ids):
                batch, seq = input_ids.shape
                return torch.ones(batch, seq, 6)

        class FakeProjection(torch.nn.Module):
            def forward(self, hidden_states):
                batch, seq, _ = hidden_states.shape
                return torch.full((batch, seq, 6), 2.0), None

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = FakeDecoder()
                self.embedding = FakeEmbedding()
                self.per_layer_embedding = FakePLEmbedding()
                self.per_layer_model_proj = FakeProjection()
                self.per_layer_proj_norm = torch.nn.Identity()
                self.pre_process = True
                self.config = SimpleNamespace(
                    per_layer_embed_dim=3,
                    num_layers=2,
                    hidden_size=4,
                    sequence_parallel=False,
                    scale_embeddings_by_hidden_size=True,
                )
                self.forward_calls = []

            def forward(
                self,
                input_ids,
                position_ids,
                attention_mask,
                decoder_input=None,
                labels=None,
                inference_context=None,
                packed_seq_params=None,
                extra_block_kwargs=None,
                runtime_gather_output=None,
                **kwargs,
            ):
                del labels, inference_context, packed_seq_params, runtime_gather_output, kwargs
                self.forward_calls.append(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "attention_mask": attention_mask,
                        "decoder_input": decoder_input,
                        "extra_block_kwargs": extra_block_kwargs,
                    }
                )
                return "ok"

        model = FakeModel()
        input_ids = torch.ones(2, 3, dtype=torch.long)
        attention_mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)

        _install_ple_forward(model)
        result = model(input_ids=input_ids, position_ids=None, attention_mask=attention_mask)

        assert result == "ok"
        assert model.decoder._gemma4_ple_threading_patched is True
        call = model.forward_calls[-1]
        assert call["decoder_input"].shape == (3, 2, 4)
        assert call["extra_block_kwargs"]["per_layer_inputs"].shape == (2, 3, 2, 3)

    def test_install_ple_forward_preserves_existing_extra_block_kwargs(self):
        class FakeDecoder(torch.nn.Module):
            layers = None

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = FakeDecoder()
                self.per_layer_embedding = None
                self.pre_process = False
                self.config = SimpleNamespace(sequence_parallel=False)
                self.forward_calls = []

            def forward(
                self,
                input_ids,
                position_ids,
                attention_mask,
                decoder_input=None,
                labels=None,
                inference_context=None,
                packed_seq_params=None,
                extra_block_kwargs=None,
                runtime_gather_output=None,
                **kwargs,
            ):
                del labels, inference_context, packed_seq_params, runtime_gather_output, kwargs
                self.forward_calls.append(extra_block_kwargs)
                return decoder_input

        model = FakeModel()
        decoder_input = torch.zeros(3, 1, 4)
        extra_kwargs = {"existing": object()}

        _install_ple_forward(model)
        result = model(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            position_ids=None,
            attention_mask=None,
            decoder_input=decoder_input,
            extra_block_kwargs=extra_kwargs,
        )

        assert result is decoder_input
        assert model.forward_calls[-1] is extra_kwargs
        assert model.decoder._gemma4_ple_threading_patched is True

    def test_patch_ple_block_threading_injects_layer_inputs_and_restores_state(self):
        class FakeLayer(torch.nn.Module):
            def __init__(self, layer_number):
                super().__init__()
                self.layer_number = layer_number
                self.calls = []

            def forward(self, hidden_states=None, **kwargs):
                self.calls.append((hidden_states, kwargs))
                return hidden_states + kwargs["per_layer_input"].sum()

        class FakeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer(1), FakeLayer(2)])

            def forward(self, hidden_states, **kwargs):
                del kwargs
                for layer in self.layers:
                    hidden_states = layer(hidden_states=hidden_states)
                return hidden_states

        decoder = FakeDecoder()
        per_layer_inputs = torch.arange(1 * 2 * 2 * 3, dtype=torch.float32).view(1, 2, 2, 3)

        _patch_ple_block_threading(decoder)
        out = decoder(torch.tensor(1.0), per_layer_inputs=per_layer_inputs)

        first_expected = _gemma4_layer_input(per_layer_inputs, decoder.layers[0])
        second_expected = _gemma4_layer_input(per_layer_inputs, decoder.layers[1])
        torch.testing.assert_close(decoder.layers[0].calls[0][1]["per_layer_input"], first_expected)
        torch.testing.assert_close(decoder.layers[1].calls[0][1]["per_layer_input"], second_expected)
        torch.testing.assert_close(out, torch.tensor(1.0) + first_expected.sum() + second_expected.sum())
        assert not hasattr(decoder, "_gemma4_current_per_layer_inputs")

    def test_patch_ple_block_threading_wraps_module_checkpointed_forward(self, monkeypatch):
        from megatron.core.transformer import transformer_block as transformer_block_module

        calls = []

        class FakeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()

            def forward(self, hidden_states, **kwargs):
                del kwargs
                return transformer_block_module.checkpointed_forward(self, hidden_states, "mask")

        def fake_orig_checkpointed_forward(block, *args, **kwargs):
            calls.append(("orig", block, args, kwargs))
            return "orig"

        def fake_gemma4_checkpointed_forward(block, *args, per_layer_inputs=None, **kwargs):
            calls.append(("gemma4", block, args, per_layer_inputs, kwargs))
            return "gemma4"

        monkeypatch.setattr(
            transformer_block_module,
            "checkpointed_forward",
            fake_orig_checkpointed_forward,
            raising=False,
        )
        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4._gemma4_checkpointed_forward",
            fake_gemma4_checkpointed_forward,
        )
        decoder = FakeDecoder()
        per_layer_inputs = torch.ones(1, 2, 1, 3)

        _patch_ple_block_threading(decoder)
        out = decoder(torch.tensor(1.0), per_layer_inputs=per_layer_inputs)

        assert out == "gemma4"
        assert calls[0][0] == "gemma4"
        assert calls[0][3] is per_layer_inputs
        assert transformer_block_module.checkpointed_forward is fake_orig_checkpointed_forward

    def test_patch_ple_block_threading_wraps_instance_checkpointed_forward(self, monkeypatch):
        calls = []

        class FakeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()

            def _checkpointed_forward(self, hidden_states, attention_mask, input_ids=None):
                calls.append(("orig", self, (hidden_states, attention_mask), {"input_ids": input_ids}))
                return "orig"

            def forward(self, hidden_states, **kwargs):
                del kwargs
                return self._checkpointed_forward(hidden_states, "mask", input_ids="tokens")

        def fake_gemma4_checkpointed_forward(block, *args, per_layer_inputs=None, **kwargs):
            calls.append(("gemma4", block, args, per_layer_inputs, kwargs))
            return "gemma4"

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4._gemma4_checkpointed_forward",
            fake_gemma4_checkpointed_forward,
        )
        decoder = FakeDecoder()
        per_layer_inputs = torch.ones(1, 2, 1, 3)

        _patch_ple_block_threading(decoder)
        out = decoder(torch.tensor(1.0), per_layer_inputs=per_layer_inputs)

        assert out == "gemma4"
        assert calls[0][0] == "gemma4"
        assert calls[0][3] is per_layer_inputs
        assert calls[0][4]["input_ids"] == "tokens"
        assert "_checkpointed_forward" not in decoder.__dict__

    def test_gemma4_checkpointed_forward_uniform_threads_ple_inputs(self, monkeypatch):
        from megatron.core import tensor_parallel

        checkpoint_calls = []

        class FakeTransformerLayer:
            def __init__(self, layer_number):
                self.layer_number = layer_number
                self.calls = []

            def _forward_attention(self, input_ids=None):
                del input_ids

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return (
                    kwargs["hidden_states"] + kwargs["per_layer_input"].sum() + float(self.layer_number),
                    f"context-{self.layer_number}",
                )

        class FakePlainLayer:
            layer_number = 2

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                assert "per_layer_input" not in kwargs
                assert "context" not in kwargs
                return kwargs["hidden_states"] + 100.0

        def fake_checkpoint(function, distribute_saved_activations, *args):
            checkpoint_calls.append(distribute_saved_activations)
            return function(*args)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TransformerLayer",
            FakeTransformerLayer,
        )
        monkeypatch.setattr(tensor_parallel, "checkpoint", fake_checkpoint)
        block = SimpleNamespace(
            layers=[FakeTransformerLayer(1), FakePlainLayer(), FakeTransformerLayer(3)],
            config=SimpleNamespace(
                recompute_method="uniform",
                recompute_num_layers=2,
                fp8=False,
                fp4=False,
                distribute_saved_activations=False,
            ),
            num_layers_per_pipeline_rank=3,
            pg_collection=SimpleNamespace(tp=None),
        )
        per_layer_inputs = torch.tensor([[[[10.0], [20.0], [30.0]]]])
        input_ids = torch.tensor([1])

        hidden_states, intermediates = _gemma4_checkpointed_forward(
            block,
            torch.tensor(0.0),
            input_ids=input_ids,
            attention_mask="mask",
            context="context",
            context_mask="context_mask",
            rotary_pos_emb="rope",
            attention_bias="bias",
            packed_seq_params="packed",
            use_inner_quantization_context=True,
            padding_mask="padding",
            extract_layer_indices={1},
            per_layer_inputs=per_layer_inputs,
        )

        torch.testing.assert_close(hidden_states, torch.tensor(144.0))
        torch.testing.assert_close(intermediates[0], torch.tensor(111.0))
        assert checkpoint_calls == [False, False]
        torch.testing.assert_close(
            block.layers[0].calls[0]["per_layer_input"], per_layer_inputs[:, :, 0, :].transpose(0, 1)
        )
        assert block.layers[0].calls[0]["input_ids"] is input_ids
        assert block.layers[1].calls[0]["attention_mask"] == "mask"
        assert "input_ids" not in block.layers[1].calls[0]
        torch.testing.assert_close(
            block.layers[2].calls[0]["per_layer_input"], per_layer_inputs[:, :, 2, :].transpose(0, 1)
        )
        assert block.layers[2].calls[0]["input_ids"] is input_ids

    def test_gemma4_checkpointed_forward_skips_input_ids_without_mcore_support(self, monkeypatch):
        from megatron.core import tensor_parallel

        class FakeTransformerLayer:
            def __init__(self, layer_number):
                self.layer_number = layer_number
                self.calls = []

            def _forward_attention(self):
                pass

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["hidden_states"] + float(self.layer_number), None

        def fake_checkpoint(function, distribute_saved_activations, *args):
            del distribute_saved_activations
            return function(*args)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TransformerLayer",
            FakeTransformerLayer,
        )
        monkeypatch.setattr(tensor_parallel, "checkpoint", fake_checkpoint)
        block = SimpleNamespace(
            layers=[FakeTransformerLayer(1)],
            config=SimpleNamespace(
                recompute_method="uniform",
                recompute_num_layers=1,
                fp8=False,
                fp4=False,
                distribute_saved_activations=False,
            ),
            num_layers_per_pipeline_rank=1,
            pg_collection=SimpleNamespace(tp=None),
        )
        input_ids = torch.tensor([1])

        hidden_states = _gemma4_checkpointed_forward(
            block,
            torch.tensor(0.0),
            input_ids=input_ids,
            attention_mask="mask",
            context="context",
            context_mask="context_mask",
            rotary_pos_emb="rope",
            attention_bias="bias",
            packed_seq_params="packed",
            use_inner_quantization_context=True,
            padding_mask="padding",
        )

        torch.testing.assert_close(hidden_states, torch.tensor(1.0))
        assert "input_ids" not in block.layers[0].calls[0]

    def test_gemma4_checkpointed_forward_block_recompute_extracts_start_layers(self, monkeypatch):
        from megatron.core import tensor_parallel

        checkpoint_calls = []

        class FakeTransformerLayer:
            def __init__(self, layer_number):
                self.layer_number = layer_number

            def __call__(self, **kwargs):
                return kwargs["hidden_states"] + float(self.layer_number), None

        def fake_checkpoint(function, distribute_saved_activations, *args):
            checkpoint_calls.append(distribute_saved_activations)
            return function(*args)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TransformerLayer",
            FakeTransformerLayer,
        )
        monkeypatch.setattr(tensor_parallel, "checkpoint", fake_checkpoint)
        block = SimpleNamespace(
            layers=[FakeTransformerLayer(1), FakeTransformerLayer(2)],
            config=SimpleNamespace(
                recompute_method="block",
                recompute_num_layers=1,
                fp8=False,
                fp4=False,
                distribute_saved_activations=True,
            ),
            num_layers_per_pipeline_rank=2,
            pg_collection=SimpleNamespace(tp=None),
        )

        hidden_states, intermediates = _gemma4_checkpointed_forward(
            block,
            torch.tensor(0.0),
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            attention_bias=None,
            packed_seq_params=None,
            use_inner_quantization_context=False,
            extract_layer_indices={5},
            layer_offset=5,
            per_layer_inputs=torch.zeros(1, 1, 2, 1),
        )

        torch.testing.assert_close(hidden_states, torch.tensor(3.0))
        torch.testing.assert_close(intermediates[0], torch.tensor(1.0))
        assert checkpoint_calls == [True]

    def test_gemma4_checkpointed_forward_rejects_invalid_recompute_method(self):
        block = SimpleNamespace(
            layers=[],
            config=SimpleNamespace(recompute_method="invalid", fp8=False, fp4=False),
            num_layers_per_pipeline_rank=0,
            pg_collection=SimpleNamespace(tp=None),
        )

        with pytest.raises(ValueError, match="Invalid activation recompute method"):
            _gemma4_checkpointed_forward(
                block,
                torch.tensor(0.0),
                attention_mask=None,
                context=None,
                context_mask=None,
                rotary_pos_emb=None,
                attention_bias=None,
                packed_seq_params=None,
                use_inner_quantization_context=False,
            )


class TestGemma4MoEHelpers:
    def test_topk_router_scale_state_is_trainable(self, monkeypatch):
        def fake_router_init(self, config, **kwargs):
            del kwargs
            torch.nn.Module.__init__(self)
            self.config = config

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TopKRouter.__init__",
            fake_router_init,
        )
        router = Gemma4TopKRouter(
            _config(
                num_moe_experts=3,
                params_dtype=torch.bfloat16,
                sequence_parallel=True,
            )
        )

        assert isinstance(router.scale, torch.nn.Parameter)
        assert isinstance(router.per_expert_scale, torch.nn.Parameter)
        assert router.scale.dtype is torch.bfloat16
        assert router.per_expert_scale.dtype is torch.bfloat16
        assert router.scale.sequence_parallel is True
        assert router.per_expert_scale.sequence_parallel is True

    def test_transformer_moe_norm_state_is_trainable(self, monkeypatch):
        def fake_layer_init(self, config, submodules, layer_number=1, **kwargs):
            del submodules, layer_number, kwargs
            torch.nn.Module.__init__(self)
            self.config = config

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TransformerLayer.__init__",
            fake_layer_init,
        )
        layer = Gemma4TransformerLayer(
            _config(params_dtype=torch.bfloat16, sequence_parallel=True),
            submodules=SimpleNamespace(),
        )

        assert isinstance(layer.pre_shared_expert_layernorm.weight, torch.nn.Parameter)
        assert isinstance(layer.post_ffn_layernorm.weight, torch.nn.Parameter)
        assert layer.pre_shared_expert_layernorm.weight.sequence_parallel is True
        assert layer.post_ffn_layernorm.weight.sequence_parallel is True

    def test_gemma4_block_spec_patches_attention_layer_and_moe_modules(self, monkeypatch):
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.moe.moe_layer import MoELayer

        calls = []
        attn_submodules = SimpleNamespace(core_attention="old_core", linear_proj="old_proj")
        mlp_submodules = SimpleNamespace(router="old_router")
        layer_spec = SimpleNamespace(
            module=object,
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(module=SelfAttention, submodules=attn_submodules),
                mlp=SimpleNamespace(module=MoELayer, submodules=mlp_submodules),
            ),
        )
        block_spec = SimpleNamespace(layer_specs=[layer_spec])

        def fake_get_gpt_decoder_block_spec(config, use_transformer_engine=True, **kwargs):
            calls.append((config, use_transformer_engine, kwargs))
            return block_spec

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.get_gpt_decoder_block_spec",
            fake_get_gpt_decoder_block_spec,
        )

        out = _gemma4_block_spec("config", use_transformer_engine=True, extra="value")

        assert out is block_spec
        assert calls == [("config", True, {"extra": "value"})]
        assert layer_spec.module is Gemma4TransformerLayer
        assert layer_spec.submodules.self_attention.module is Gemma4SelfAttention
        assert attn_submodules.core_attention is Gemma4TEDotProductAttention
        assert attn_submodules.linear_proj != "old_proj"
        assert layer_spec.submodules.mlp.module is Gemma4MoELayer
        assert mlp_submodules.router is Gemma4TopKRouter

    def test_gemma4_block_spec_patches_partial_moe_builder(self, monkeypatch):
        from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules

        mlp_submodules = MoESubmodules(experts=object(), shared_experts=object())
        layer_spec = SimpleNamespace(
            module=object,
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(module=object, submodules=None),
                mlp=partial(MoELayer, submodules=mlp_submodules),
            ),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.get_gpt_decoder_block_spec",
            lambda *args, **kwargs: SimpleNamespace(layer_specs=[layer_spec]),
        )

        _gemma4_block_spec("config", use_transformer_engine=True)

        assert layer_spec.submodules.mlp.func is Gemma4MoELayer
        assert layer_spec.submodules.mlp.keywords["submodules"].router is Gemma4TopKRouter

    def test_gemma4_block_spec_skips_te_projection_patch_when_disabled(self, monkeypatch):
        from megatron.core.transformer.attention import SelfAttention

        attn_submodules = SimpleNamespace(core_attention="old_core", linear_proj="old_proj")
        layer_spec = SimpleNamespace(
            module=object,
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(module=SelfAttention, submodules=attn_submodules),
                mlp=SimpleNamespace(module=object, submodules=None),
            ),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.get_gpt_decoder_block_spec",
            lambda *args, **kwargs: SimpleNamespace(layer_specs=[layer_spec]),
        )

        _gemma4_block_spec("config", use_transformer_engine=False)

        assert layer_spec.module is Gemma4TransformerLayer
        assert attn_submodules.core_attention is Gemma4TEDotProductAttention
        assert attn_submodules.linear_proj == "old_proj"

    def test_transformer_layer_post_mlp_adds_bias_and_layer_scalar(self):
        layer = object.__new__(Gemma4TransformerLayer)
        layer.layer_scalar = torch.tensor([0.5])
        layer.post_ffn_layernorm = lambda x: (x + 2.0, None)
        residual = torch.ones(2, 3)
        mlp_out = torch.full_like(residual, 4.0)
        mlp_bias = torch.full_like(residual, 1.0)

        out = Gemma4TransformerLayer._forward_post_mlp(layer, (mlp_out, mlp_bias), residual)

        torch.testing.assert_close(out, torch.full_like(residual, 4.0))
        assert out.requires_grad is False

    def test_topk_router_routing_normalizes_and_scales_probs(self, monkeypatch):
        routing_probs = torch.tensor([[0.2, 0.3, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
        routing_map = torch.tensor([[True, True, False], [True, True, False]])

        def fake_routing(self, logits, padding_mask=None, input_ids=None):
            del self, logits, padding_mask, input_ids
            return routing_probs, routing_map

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.TopKRouter.routing", fake_routing)
        router = object.__new__(Gemma4TopKRouter)
        router.per_expert_scale = torch.tensor([1.0, 2.0, 3.0])

        out_probs, out_map = Gemma4TopKRouter.routing(router, torch.zeros(2, 3))

        assert out_map is routing_map
        torch.testing.assert_close(out_probs[0], torch.tensor([0.4, 1.2, 0.0]))
        torch.testing.assert_close(out_probs[1], torch.tensor([0.5, 1.0, 0.0]))

    def test_topk_router_gating_applies_hf_norm_and_scale(self, monkeypatch):
        captured = []

        def fake_gating(self, hidden_states):
            del self
            captured.append(hidden_states)
            return hidden_states

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.TopKRouter.gating", fake_gating)
        router = object.__new__(Gemma4TopKRouter)
        torch.nn.Module.__init__(router)
        router.config = SimpleNamespace(layernorm_epsilon=1e-6)
        router.scale = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
        router.scalar_root_size = 2**-0.5
        hidden_states = torch.tensor([[3.0, 4.0]])

        output = Gemma4TopKRouter.gating(router, hidden_states)

        normed = hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        expected = normed * router.scale * router.scalar_root_size
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(captured[0], expected)

    def test_transformer_layer_uses_separate_moe_inputs(self):
        calls = []

        class FakeMoE:
            def forward_with_separate_inputs(self, expert, shared, router, padding_mask=None):
                calls.append((expert, shared, router, padding_mask))
                return torch.zeros_like(expert), None

        layer = SimpleNamespace(
            config=SimpleNamespace(fp32_residual_connection=False, layernorm_epsilon=1e-6),
            pre_mlp_layernorm=SimpleNamespace(weight=torch.tensor([2.0, 2.0])),
            pre_shared_expert_layernorm=SimpleNamespace(weight=torch.tensor([3.0, 3.0])),
            mlp=FakeMoE(),
            _forward_post_mlp=lambda output, residual: (output, residual),
        )
        hidden_states = torch.tensor([[[3.0, 4.0]]])

        output, residual = Gemma4TransformerLayer._forward_mlp(
            layer,
            hidden_states,
            padding_mask=torch.tensor([[True]]),
            input_ids=torch.tensor([[7]]),
        )

        base = hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        expert, shared, router, padding_mask = calls[0]
        torch.testing.assert_close(expert, base * 2.0)
        torch.testing.assert_close(shared, base * 3.0)
        torch.testing.assert_close(router, hidden_states)
        torch.testing.assert_close(output[0], torch.zeros_like(hidden_states))
        torch.testing.assert_close(residual, hidden_states)
        torch.testing.assert_close(padding_mask, torch.tensor([[True]]))

    def test_transformer_layer_preserves_packed_moe_batch_semantics(self):
        calls = []

        class FakeMoE:
            def forward_with_separate_inputs(self, expert, shared, router, padding_mask=None):
                calls.append((expert, shared, router, padding_mask))
                return expert, None

        layer = SimpleNamespace(
            config=SimpleNamespace(fp32_residual_connection=False, layernorm_epsilon=1e-6),
            is_moe_layer=True,
            pre_mlp_layernorm=SimpleNamespace(weight=torch.tensor([2.0, 2.0])),
            pre_shared_expert_layernorm=SimpleNamespace(weight=torch.tensor([3.0, 3.0])),
            mlp=FakeMoE(),
            _forward_post_mlp=lambda output, residual: (output[0], residual),
        )
        hidden_states = torch.arange(1, 9, dtype=torch.float32).view(4, 1, 2)
        padding_mask = torch.tensor([[False, True], [True, False]])

        output, residual = Gemma4TransformerLayer._forward_mlp(
            layer,
            hidden_states,
            padding_mask=padding_mask,
            packed_seq_params=SimpleNamespace(tokens_per_sample=2),
        )

        expected_router = hidden_states.view(2, 2, -1).transpose(0, 1).contiguous()
        normalized = expected_router * torch.pow(expected_router.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
        expert, shared, router, routed_padding_mask = calls[0]
        torch.testing.assert_close(expert, normalized * 2.0)
        torch.testing.assert_close(shared, normalized * 3.0)
        torch.testing.assert_close(router, expected_router)
        torch.testing.assert_close(routed_padding_mask, padding_mask)
        torch.testing.assert_close(output, (normalized * 2.0).transpose(0, 1).reshape(4, 1, 2))
        torch.testing.assert_close(residual, hidden_states)

    def test_topk_router_routing_keeps_probs_when_map_missing(self, monkeypatch):
        routing_probs = torch.ones(2, 3)

        def fake_routing(self, logits, padding_mask=None, input_ids=None):
            del self, logits, padding_mask, input_ids
            return routing_probs, None

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.TopKRouter.routing", fake_routing)
        router = object.__new__(Gemma4TopKRouter)
        router.per_expert_scale = torch.ones(3)

        out_probs, out_map = Gemma4TopKRouter.routing(router, torch.zeros(2, 3))

        assert out_probs is routing_probs
        assert out_map is None

    def test_moe_layer_postprocess_handles_latent_and_shared_expert(self):
        class Dispatcher:
            def combine_postprocess(self, output):
                return output + 1.0

        layer = object.__new__(Gemma4MoELayer)
        layer.token_dispatcher = Dispatcher()
        layer.config = SimpleNamespace(moe_latent_size=True)
        layer.fc2_latent_proj = lambda x: (x + 2.0, None)
        layer.post_moe_layernorm = lambda x: (x + 3.0, None)
        layer.post_shared_expert_layernorm = lambda x: (x + 4.0, None)
        output = torch.ones(2, 3)
        shared = torch.full_like(output, 10.0)

        out = Gemma4MoELayer.postprocess(layer, output, shared)

        torch.testing.assert_close(out, torch.full_like(output, 21.0))

    @pytest.mark.parametrize(
        ("fp8", "fp4", "expected_checkpoint"),
        [
            pytest.param(False, False, "tensor_parallel", id="bf16"),
            pytest.param(True, False, "te", id="fp8"),
            pytest.param(False, True, "te", id="fp4"),
        ],
    )
    def test_moe_layer_recompute_uses_expected_checkpoint_for_training(
        self, monkeypatch, fp8, fp4, expected_checkpoint
    ):
        calls = []

        def fake_te_checkpoint(function, distribute_saved_activations, get_rng_tracker, tp_group, *args):
            calls.append(("te", distribute_saved_activations, get_rng_tracker, tp_group))
            return function(*args)

        def fake_tensor_parallel_checkpoint(function, distribute_saved_activations, *args):
            calls.append(("tensor_parallel", distribute_saved_activations))
            return function(*args)

        monkeypatch.setattr("megatron.bridge.models.gemma.modeling_gemma4.te_checkpoint", fake_te_checkpoint)
        monkeypatch.setattr(tensor_parallel, "checkpoint", fake_tensor_parallel_checkpoint)

        layer = object.__new__(Gemma4MoELayer)
        torch.nn.Module.__init__(layer)
        layer.shared_expert_overlap = False
        layer.moe_layer_recompute = True
        layer.train()
        layer.config = SimpleNamespace(fp8=fp8, fp4=fp4)
        layer.tp_group = "tp-group"
        layer.shared_experts_compute = lambda tensor: tensor + 1.0
        layer.route = lambda tensor, padding_mask: (tensor, padding_mask)
        layer.preprocess = lambda tensor, probs, routing_map: (tensor + probs, routing_map)
        layer.dispatch = lambda tensor, probs: (tensor, probs)
        layer.routed_experts_compute = lambda tensor, probs: (tensor + 1.0, None)
        layer.combine = lambda tensor: tensor
        layer.postprocess = lambda output, shared: output + shared
        expert_input = torch.tensor(1.0)
        shared_input = torch.tensor(2.0)
        router_input = torch.tensor(3.0)

        output, bias = Gemma4MoELayer.forward_with_separate_inputs(
            layer,
            expert_input,
            shared_input,
            router_input,
        )

        assert bias is None
        torch.testing.assert_close(output, torch.tensor(8.0))
        if expected_checkpoint == "te":
            assert calls == [("te", False, tensor_parallel.random.get_cuda_rng_tracker, "tp-group")]
        else:
            assert calls == [("tensor_parallel", False)]

    def test_install_tied_kv_marks_only_global_attention_layers(self):
        local_attn = SimpleNamespace()
        global_attn = SimpleNamespace()
        model = SimpleNamespace(
            decoder=SimpleNamespace(
                layers=[
                    SimpleNamespace(layer_number=1, self_attention=local_attn),
                    SimpleNamespace(layer_number=2, self_attention=global_attn),
                    SimpleNamespace(layer_number=4),
                ]
            )
        )
        provider = SimpleNamespace(
            attention_k_eq_v=True,
            num_global_key_value_heads=1,
            interleaved_attn_pattern=(1, 1),
        )

        _install_tied_kv(model, provider)

        assert not hasattr(local_attn, "_tied_kv")
        assert global_attn._tied_kv is True

    def test_install_tied_kv_returns_when_disabled_or_missing_decoder(self):
        provider = SimpleNamespace(
            attention_k_eq_v=False,
            num_global_key_value_heads=1,
            interleaved_attn_pattern=(1, 1),
        )
        model = SimpleNamespace(decoder=SimpleNamespace(layers=[]))

        _install_tied_kv(model, provider)
        _install_tied_kv(
            SimpleNamespace(),
            SimpleNamespace(attention_k_eq_v=True, num_global_key_value_heads=1, interleaved_attn_pattern=(1, 1)),
        )
        _install_tied_kv(model, SimpleNamespace(attention_k_eq_v=True, num_global_key_value_heads=0))

        assert model.decoder.layers == []


class TestGemma4OutputHelpers:
    def test_logit_softcapping_applies_tanh_scale(self):
        logits = torch.tensor([-4.0, 0.0, 4.0])

        out = _logit_softcapping(logits, 2.0)

        torch.testing.assert_close(out, 2.0 * torch.tanh(logits / 2.0))

    def test_logit_softcapping_returns_input_without_scale(self):
        logits = torch.tensor([1.0])

        assert _logit_softcapping(logits, None) is logits

    def test_output_layer_applies_final_softcap(self):
        class BaseOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(final_logit_softcapping=2.0)

            def forward(self, x):
                return x, None

        class OutputLayer(Gemma4OutputLayer, BaseOutput):
            pass

        layer = OutputLayer()
        logits = torch.tensor([[-4.0, 0.0, 4.0]])

        out, bias = layer(logits)

        torch.testing.assert_close(out, 2.0 * torch.tanh(logits / 2.0))
        assert bias is None
