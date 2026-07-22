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

"""Unit tests for Gemma 4 text-only providers."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from megatron.bridge.models.gemma.gemma4_provider import (
    Gemma4DenseProvider,
    Gemma4ModelProvider,
    _install_gemma4_dense_load_state_aliases,
)
from megatron.bridge.models.gemma.modeling_gemma4 import (
    Gemma4OutputLayer,
    _gemma4_checkpointed_forward,
    _install_tied_kv,
    _patch_ple_block_threading,
)
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


class _IdentityOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states, None


class _ModelWithOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_layer = _IdentityOutputLayer(config)


def _build_dense_model(provider, model):
    provider._gemma4_dense_finalized = True
    with (
        patch("megatron.core.models.gpt.GPTModel", return_value=model),
        patch("megatron.bridge.models.gemma.gemma4_provider.Gemma4DenseRotaryEmbedding"),
        patch("megatron.bridge.models.gemma.gemma4_provider._attach_ple_modules"),
        patch("megatron.bridge.models.gemma.gemma4_provider.wire_gemma4_kv_sharing"),
        patch("megatron.bridge.models.gemma.gemma4_provider._install_ple_forward"),
        patch("megatron.bridge.models.gemma.gemma4_provider._install_gemma4_dense_load_state_aliases"),
    ):
        return provider.build()


class TestGemma4DenseProviderDefaults:
    """Config-level checks for Gemma 4 dense text providers."""

    @pytest.fixture
    def provider(self):
        return Gemma4DenseProvider()

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("num_layers", 42),
            ("hidden_size", 2560),
            ("ffn_hidden_size", 10240),
            ("num_attention_heads", 8),
            ("num_query_groups", 2),
            ("kv_channels", 256),
            ("global_kv_channels", 512),
            ("num_global_query_groups", 2),
            ("attention_k_eq_v", False),
            ("seq_length", 131_072),
            ("vocab_size", 262_143),
            ("make_vocab_size_divisible_by", 128),
            ("normalization", "RMSNorm"),
            ("layernorm_epsilon", 1e-6),
            ("window_size", (511, 0)),
            ("window_attn_skip_freq", 6),
            ("sliding_window_rope_base", 10_000.0),
            ("full_attention_rope_base", 1_000_000.0),
            ("full_attention_rope_partial_factor", 0.25),
            ("num_kv_shared_layers", 18),
            ("use_double_wide_mlp", False),
            ("per_layer_embed_vocab_size", 262_144),
            ("per_layer_embed_dim", 256),
            ("final_logit_softcapping", 30.0),
            ("num_moe_experts", None),
            ("moe_router_topk", None),
            ("moe_ffn_hidden_size", None),
        ],
    )
    def test_dense_e4b_defaults(self, provider, field, expected):
        assert getattr(provider, field) == expected

    def test_inherits_gpt_provider(self):
        assert issubclass(Gemma4DenseProvider, GPTModelProvider)

    def test_dtype_defaults(self, provider):
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.autocast_dtype == torch.bfloat16

    def test_finalize_sets_dense_flag(self, provider):
        assert not getattr(provider, "_gemma4_dense_finalized", False)
        provider.finalize()
        assert provider._gemma4_dense_finalized is True

    def test_finalize_marks_double_wide_layers_as_heterogeneous_for_checkpointing(self):
        provider = Gemma4DenseProvider(use_double_wide_mlp=True)

        provider.finalize()

        assert provider.hetereogenous_dist_checkpoint is True

    def test_provide_rejects_pipeline_parallel(self, provider):
        provider.pipeline_model_parallel_size = 2
        with pytest.raises(NotImplementedError, match="PP=1"):
            provider.provide()

    def test_provide_rejects_virtual_pipeline_stage(self, provider):
        with pytest.raises(NotImplementedError, match="PP=1"):
            provider.provide(vp_stage=0)

    def test_build_applies_final_logit_softcapping_once(self, provider):
        """Dense output must match HF's single tanh softcap in the saturation regime."""
        built = _build_dense_model(provider, _ModelWithOutputLayer(provider))

        raw_logits = torch.tensor([[-120.0, -30.0, 0.0, 30.0, 120.0]])
        logits, bias = built.output_layer(raw_logits)
        expected_once = 30.0 * torch.tanh(raw_logits / 30.0)
        expected_twice = 30.0 * torch.tanh(expected_once / 30.0)

        assert isinstance(built.output_layer, Gemma4OutputLayer)
        torch.testing.assert_close(logits, expected_once)
        assert not torch.allclose(logits, expected_twice)
        assert bias is None

    def test_build_does_not_double_softcap_existing_output_layer(self, provider):
        model = _ModelWithOutputLayer(provider)
        extend_instance(model.output_layer, Gemma4OutputLayer)
        built = _build_dense_model(provider, model)

        raw_logits = torch.tensor([[-120.0, 120.0]])
        logits, _ = built.output_layer(raw_logits)
        torch.testing.assert_close(logits, 30.0 * torch.tanh(raw_logits / 30.0))

    def test_build_leaves_output_uncapped_when_disabled(self, provider):
        provider.final_logit_softcapping = None
        built = _build_dense_model(provider, _ModelWithOutputLayer(provider))

        raw_logits = torch.tensor([[-120.0, 120.0]])
        logits, _ = built.output_layer(raw_logits)
        assert not isinstance(built.output_layer, Gemma4OutputLayer)
        torch.testing.assert_close(logits, raw_logits)


class TestGemma4DenseLoadStateAliases:
    """The Dense checkpoint uses sliding/global aliases; module load expects self_attention."""

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = nn.Module()
            self.self_attention.linear_proj = nn.Linear(2, 2, bias=False)
            self.self_attention.linear_qkv = nn.Linear(2, 2, bias=False)
            self.self_attention.q_layernorm = nn.LayerNorm(2)
            self.self_attention.k_layernorm = nn.LayerNorm(2)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([TestGemma4DenseLoadStateAliases._Layer()])

    @pytest.mark.parametrize("alias", ["self_attention_sliding", "self_attention_global"])
    def test_load_state_aliases_attention_keys(self, alias):
        model = self._Model()
        _install_gemma4_dense_load_state_aliases(model)

        state_dict = {
            f"decoder.layers.0.{alias}.linear_proj.weight": torch.full((2, 2), 1.0),
            f"decoder.layers.0.{alias}.linear_qkv.weight": torch.full((2, 2), 2.0),
            f"decoder.layers.0.{alias}.q_layernorm.weight": torch.full((2,), 3.0),
            f"decoder.layers.0.{alias}.q_layernorm.bias": torch.full((2,), 4.0),
            f"decoder.layers.0.{alias}.k_layernorm.weight": torch.full((2,), 5.0),
            f"decoder.layers.0.{alias}.k_layernorm.bias": torch.full((2,), 6.0),
        }

        load_result = model.load_state_dict(state_dict, strict=False)

        assert not load_result.unexpected_keys
        assert torch.allclose(model.decoder.layers[0].self_attention.linear_proj.weight, torch.full((2, 2), 1.0))
        assert torch.allclose(model.decoder.layers[0].self_attention.linear_qkv.weight, torch.full((2, 2), 2.0))
        assert torch.allclose(model.decoder.layers[0].self_attention.q_layernorm.weight, torch.full((2,), 3.0))
        assert torch.allclose(model.decoder.layers[0].self_attention.q_layernorm.bias, torch.full((2,), 4.0))
        assert torch.allclose(model.decoder.layers[0].self_attention.k_layernorm.weight, torch.full((2,), 5.0))
        assert torch.allclose(model.decoder.layers[0].self_attention.k_layernorm.bias, torch.full((2,), 6.0))

    def test_install_is_idempotent(self):
        model = self._Model()
        _install_gemma4_dense_load_state_aliases(model)
        _install_gemma4_dense_load_state_aliases(model)
        assert model._gemma4_dense_load_state_aliases_installed is True


class TestGemma4DenseDistributedCheckpoint:
    """Exercise a real heterogeneous E2B model checkpoint on CPU."""

    def test_double_wide_mlp_checkpoint_roundtrip(self, tmp_path):
        from megatron.core import parallel_state
        from megatron.core.dist_checkpointing import load, save

        rendezvous = tmp_path / "dist_init"
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
        parallel_state.initialize_model_parallel()

        def make_provider():
            return Gemma4DenseProvider(
                num_layers=4,
                hidden_size=8,
                ffn_hidden_size=16,
                num_attention_heads=2,
                num_query_groups=1,
                kv_channels=4,
                global_kv_channels=4,
                num_global_query_groups=1,
                seq_length=8,
                vocab_size=32,
                make_vocab_size_divisible_by=1,
                window_size=(3, 0),
                window_attn_skip_freq=[True, False, True, False],
                num_kv_shared_layers=2,
                use_double_wide_mlp=True,
                per_layer_embed_vocab_size=32,
                per_layer_embed_dim=0,
                bf16=False,
                params_dtype=torch.float32,
                autocast_dtype=torch.float32,
                use_cpu_initialization=True,
            )

        try:
            with (
                patch(
                    "megatron.bridge.models.gemma.gemma4_provider.Gemma4DenseRotaryEmbedding",
                    return_value=None,
                ),
                patch("torch.cuda.current_device", return_value="cpu"),
                patch("torch.cuda.synchronize"),
            ):
                source = make_provider().provide()
                source_state = {
                    name: tensor.detach().clone()
                    for name, tensor in source.state_dict().items()
                    if torch.is_tensor(tensor)
                }
                save(source.sharded_state_dict(), checkpoint_dir, async_sharded_save=False)

                destination = make_provider().provide()
                loaded_state = load(destination.sharded_state_dict(), checkpoint_dir)
                destination.load_state_dict(loaded_state, strict=False)

            fc1_shapes = [tuple(layer.mlp.linear_fc1.weight.shape) for layer in source.decoder.layers]
            assert fc1_shapes == [(32, 8), (32, 8), (64, 8), (64, 8)]
            for name, expected in source_state.items():
                torch.testing.assert_close(destination.state_dict()[name], expected)
        finally:
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()


class TestGemma4PLEBlockThreading:
    """Bridge-side compatibility patch for clean MCore TransformerBlock instances."""

    class _Layer(nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.per_layer_inputs_seen = []

        def forward(self, hidden_states, attention_mask=None, context=None, **kwargs):
            self.per_layer_inputs_seen.append(kwargs.get("per_layer_input"))
            return hidden_states, context

    class _Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    TestGemma4PLEBlockThreading._Layer(1),
                    TestGemma4PLEBlockThreading._Layer(2),
                ]
            )

        def _get_layer(self, index):
            return self.layers[index]

        def forward(self, hidden_states, attention_mask=None):
            context = None
            for layer in self.layers:
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                )
            return hidden_states

    def test_patches_decoder_instance_without_changing_class_signature(self):
        decoder = self._Decoder()
        class_forward = type(decoder).forward
        _patch_ple_block_threading(decoder)
        _patch_ple_block_threading(decoder)

        assert type(decoder).forward is class_forward
        assert decoder._gemma4_ple_threading_patched is True

    def test_threads_per_layer_inputs_to_each_layer(self):
        decoder = self._Decoder()
        _patch_ple_block_threading(decoder)

        hidden_states = torch.zeros(3, 2, 5)
        per_layer_inputs = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)

        decoder(hidden_states=hidden_states, attention_mask=None, per_layer_inputs=per_layer_inputs)

        assert torch.equal(
            decoder.layers[0].per_layer_inputs_seen[-1],
            per_layer_inputs[:, :, 0, :].transpose(0, 1),
        )
        assert torch.equal(
            decoder.layers[1].per_layer_inputs_seen[-1],
            per_layer_inputs[:, :, 1, :].transpose(0, 1),
        )
        assert not hasattr(decoder, "_gemma4_current_per_layer_inputs")

    def test_recompute_checkpoint_args_carry_per_layer_inputs(self, monkeypatch):
        class _RecomputeLayer(nn.Module):
            def __init__(self, layer_number):
                super().__init__()
                self.layer_number = layer_number
                self.per_layer_inputs_seen = []

            def forward(self, hidden_states, attention_mask=None, context=None, **kwargs):
                self.per_layer_inputs_seen.append(kwargs.get("per_layer_input"))
                return hidden_states, context

        class _RecomputeDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    fp8=False,
                    fp4=False,
                    recompute_method="uniform",
                    recompute_num_layers=1,
                    distribute_saved_activations=False,
                )
                self.layers = nn.ModuleList([_RecomputeLayer(1), _RecomputeLayer(2)])
                self.num_layers_per_pipeline_rank = len(self.layers)

        checkpoint_args = []

        def _fake_checkpoint(function, distribute_saved_activations, *args):
            del distribute_saved_activations
            checkpoint_args.append(args)
            return function(*args)

        monkeypatch.setattr(
            "megatron.bridge.models.gemma.modeling_gemma4.TransformerLayer",
            _RecomputeLayer,
        )
        monkeypatch.setattr(
            "megatron.core.tensor_parallel.checkpoint",
            _fake_checkpoint,
        )

        decoder = _RecomputeDecoder()
        hidden_states = torch.zeros(3, 2, 5)
        per_layer_inputs = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)

        _gemma4_checkpointed_forward(
            decoder,
            hidden_states=hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            attention_bias=None,
            packed_seq_params=None,
            use_inner_quantization_context=False,
            per_layer_inputs=per_layer_inputs,
        )

        assert checkpoint_args
        assert all(args[-1] is per_layer_inputs for args in checkpoint_args)
        assert torch.equal(
            decoder.layers[0].per_layer_inputs_seen[-1],
            per_layer_inputs[:, :, 0, :].transpose(0, 1),
        )
        assert torch.equal(
            decoder.layers[1].per_layer_inputs_seen[-1],
            per_layer_inputs[:, :, 1, :].transpose(0, 1),
        )


class TestGemma4ModelProviderDefaults:
    """Config-level checks for the MoE text provider."""

    @pytest.fixture
    def provider(self):
        return Gemma4ModelProvider()

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("seq_length", 262_144),
            ("position_embedding_type", "rope"),
            ("rotary_base", (10_000, 1_000_000)),
            ("normalization", "RMSNorm"),
            ("layernorm_zero_centered_gamma", False),
            ("layernorm_epsilon", 1e-6),
            ("kv_channels", 256),
            ("num_query_groups", 8),
            ("window_size", 1024),
            ("interleaved_attn_pattern", (5, 1)),
            ("global_head_dim", 512),
            ("num_global_key_value_heads", 2),
            ("global_rotary_percent", 0.25),
            ("num_moe_experts", 128),
            ("moe_router_topk", 8),
            ("moe_ffn_hidden_size", 704),
            ("moe_shared_expert_intermediate_size", 2112),
            ("final_logit_softcapping", 30.0),
        ],
    )
    def test_moe_defaults(self, provider, field, expected):
        assert getattr(provider, field) == expected

    def test_dtype_defaults(self, provider):
        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.autocast_dtype == torch.bfloat16

    @pytest.mark.parametrize(
        "provider",
        [
            Gemma4ModelProvider(transformer_impl="inference_optimized"),
            Gemma4ModelProvider(cuda_graph_impl="transformer_engine"),
            Gemma4ModelProvider(moe_shared_expert_overlap=True),
            Gemma4ModelProvider(mlp_chunks_for_prefill=2),
            Gemma4ModelProvider(mlp_chunks_for_training=2),
            Gemma4ModelProvider(inference_fuse_tp_communication=True),
            Gemma4ModelProvider(recompute_granularity="selective", recompute_modules=["layernorm"]),
            Gemma4ModelProvider(
                fine_grained_activation_offloading=True,
                offload_modules=["mlp_norm"],
            ),
        ],
    )
    def test_finalize_rejects_unsupported_custom_moe_orchestration(self, provider):
        with pytest.raises(ValueError, match="Gemma 4 MoE"):
            provider.finalize()

    def test_provide_restores_dual_rotary_base(self, provider):
        mock_model = Mock()
        del mock_model.embedding
        del mock_model.output_layer

        with (
            patch.object(GPTModelProvider, "provide", return_value=mock_model) as mock_super_provide,
            patch("megatron.bridge.models.gemma.gemma4_provider.Gemma4RotaryEmbedding") as mock_rotary,
            patch("megatron.bridge.models.gemma.gemma4_provider._install_tied_kv") as mock_tied_kv,
        ):
            result = provider.provide(pre_process=True, post_process=True)

        assert result is mock_model
        assert provider.rotary_base == (10_000, 1_000_000)
        mock_super_provide.assert_called_once_with(pre_process=True, post_process=True, vp_stage=None)
        mock_rotary.assert_called_once()
        mock_tied_kv.assert_called_once_with(mock_model, provider)

    def test_provide_uses_padded_model_vocab_for_custom_embedding(self):
        provider = Gemma4ModelProvider(vocab_size=262145)
        mock_model = Mock(vocab_size=262272)
        mock_model.embedding = Mock()
        mock_model.setup_embeddings_and_output_layer = Mock()

        with (
            patch.object(GPTModelProvider, "provide", return_value=mock_model),
            patch("megatron.bridge.models.gemma.gemma4_provider.Gemma3LanguageModelEmbedding") as mock_embedding,
            patch("megatron.bridge.models.gemma.gemma4_provider.Gemma4RotaryEmbedding"),
            patch("megatron.bridge.models.gemma.gemma4_provider._install_tied_kv"),
        ):
            provider.provide(pre_process=True, post_process=True)

        assert provider.vocab_size == 262145
        mock_embedding.assert_called_once_with(
            config=provider,
            vocab_size=262272,
            max_sequence_length=provider.seq_length,
            position_embedding_type=provider.position_embedding_type,
            scatter_to_sequence_parallel=provider.scatter_embedding_sequence_parallel,
        )

    def test_provide_restores_dual_rotary_base_on_error(self, provider):
        with patch.object(GPTModelProvider, "provide", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                provider.provide(pre_process=True, post_process=True)

        assert provider.rotary_base == (10_000, 1_000_000)

    def test_provide_does_not_double_softcap_existing_output_layer(self, provider):
        model = _ModelWithOutputLayer(provider)
        extend_instance(model.output_layer, Gemma4OutputLayer)
        model.setup_embeddings_and_output_layer = Mock()

        with (
            patch.object(GPTModelProvider, "provide", return_value=model),
            patch("megatron.bridge.models.gemma.gemma4_provider.Gemma4RotaryEmbedding"),
            patch("megatron.bridge.models.gemma.gemma4_provider._install_tied_kv"),
        ):
            built = provider.provide(pre_process=True, post_process=True)

        raw_logits = torch.tensor([[-120.0, 120.0]])
        logits, _ = built.output_layer(raw_logits)
        torch.testing.assert_close(logits, 30.0 * torch.tanh(raw_logits / 30.0))


class TestInstallTiedKV:
    def test_skips_when_attention_k_eq_v_false(self):
        provider = Gemma4ModelProvider(
            num_layers=6,
            hidden_size=64,
            num_attention_heads=4,
            attention_k_eq_v=False,
        )
        provider.num_moe_experts = None

        class FakeLayer:
            layer_number = 1

        class FakeModel:
            class decoder:
                layers = [FakeLayer()]

        _install_tied_kv(FakeModel(), provider)
        assert not getattr(FakeLayer, "_tied_kv", False)

    def test_marks_global_layers_only(self):
        provider = Gemma4ModelProvider(
            num_layers=6,
            hidden_size=64,
            num_attention_heads=4,
            num_global_key_value_heads=2,
            global_head_dim=16,
            interleaved_attn_pattern=(5, 1),
            num_moe_experts=4,
            attention_k_eq_v=True,
        )

        class FakeLinear(nn.Module):
            def forward(self, x):
                return x, None

        class FakeAttn:
            def __init__(self):
                self.linear_qkv = FakeLinear()

        class FakeLayer:
            def __init__(self, number):
                self.layer_number = number
                self.self_attention = FakeAttn()

        class FakeDecoder:
            def __init__(self):
                self.layers = [FakeLayer(i) for i in range(1, 7)]

        class FakeModel:
            def __init__(self):
                self.decoder = FakeDecoder()

        model = FakeModel()
        _install_tied_kv(model, provider)

        for layer in model.decoder.layers:
            is_global = layer.layer_number == 6
            has_flag = getattr(layer.self_attention, "_tied_kv", False)
            assert has_flag == is_global, f"Layer {layer.layer_number}: expected _tied_kv={is_global}, got {has_flag}"
