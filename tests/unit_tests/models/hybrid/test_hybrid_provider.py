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

from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.hybrid import hybrid_provider
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider


class TestHybridModelProvider:
    def test_hybrid_provider_initialization(self):
        provider = HybridModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=1,
        )

        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 1
        assert provider.fp16_lm_cross_entropy is False
        assert provider.parallel_output is True
        assert provider.share_embeddings_and_output_weights is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.mamba_num_groups == 8
        assert provider.mamba_chunk_size == 128
        assert provider.hybrid_layer_pattern is None
        assert provider.hybrid_stack_spec is None
        assert provider.seq_length == 8192
        assert provider.position_embedding_type == "none"
        assert provider.vocab_size is None

    def test_modelopt_spec_remaps_te_layernorm_keys(self):
        mock_spec = Mock(spec=ModuleSpec)
        with patch(
            "megatron.bridge.models.hybrid.hybrid_provider.get_hybrid_stack_modelopt_spec",
            return_value=mock_spec,
        ) as mock_fn:
            result = hybrid_provider.modelopt_hybrid_stack_spec()

        mock_fn.assert_called_once_with(local_core_attention=False, remap_te_layernorm=True)
        assert result is mock_spec

    def test_modelopt_spec_preserves_grouped_moe_topology(self):
        provider = HybridModelProvider(
            hidden_size=128,
            num_attention_heads=1,
            num_moe_experts=4,
            moe_grouped_gemm=True,
        )
        mock_spec = Mock(spec=ModuleSpec)
        with patch(
            "megatron.bridge.models.hybrid.hybrid_provider.get_hybrid_stack_modelopt_spec",
            return_value=mock_spec,
        ) as mock_fn:
            result = hybrid_provider.modelopt_hybrid_stack_spec(provider)

        mock_fn.assert_called_once_with(use_default_te_spec=True, moe_grouped_gemm=True)
        assert result is mock_spec

    def test_rejects_mamba_stack_spec_argument(self):
        module_spec = ModuleSpec(module=object)

        with pytest.raises(TypeError, match="mamba_stack_spec"):
            HybridModelProvider(
                num_layers=2,
                hidden_size=128,
                num_attention_heads=1,
                mamba_stack_spec=module_spec,
            )

    def test_provide_method_basic(self):
        provider = HybridModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch("megatron.bridge.models.hybrid.hybrid_provider.calculate_padded_vocab_size", return_value=1024):
            with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance

                result = provider.provide(pre_process=True, post_process=True)

                assert result == mock_instance
                mock_model.assert_called_once()
                assert mock_model.call_args.kwargs["hybrid_stack_spec"] is hybrid_provider.default_hybrid_stack_spec

    def test_provide_method_with_vocab_padding(self):
        provider = HybridModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=True,
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch(
            "megatron.bridge.models.hybrid.hybrid_provider.calculate_padded_vocab_size", return_value=50176
        ) as mock_calc_vocab:
            with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel") as mock_model:
                provider.provide(pre_process=True, post_process=True)

                mock_calc_vocab.assert_called_once_with(50000, 128, 8)
                assert mock_model.call_args.kwargs["vocab_size"] == 50176

    def test_nondefault_mamba_chunk_size_is_applied_without_mutating_default_spec(self):
        provider = HybridModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            mamba_chunk_size=256,
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel") as mock_model:
            provider.provide(pre_process=True, post_process=True)

        configured_spec = mock_model.call_args.kwargs["hybrid_stack_spec"]
        configured_mixer = configured_spec.submodules.mamba_layer.submodules.mixer
        default_mixer = hybrid_provider.default_hybrid_stack_spec.submodules.mamba_layer.submodules.mixer
        assert configured_mixer.params["chunk_size"] == 256
        assert "chunk_size" not in default_mixer.params

    @patch("megatron.bridge.models.hybrid.hybrid_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.hybrid.hybrid_provider.is_pp_last_stage", return_value=True)
    def test_provide_method_respects_explicit_pipeline_stages(self, *_):
        provider = HybridModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel") as mock_model:
            provider.provide(pre_process=False, post_process=True)

        assert mock_model.call_args.kwargs["pre_process"] is False
        assert mock_model.call_args.kwargs["post_process"] is True

    def test_hybrid_stack_spec_callable(self):
        def custom_stack_spec():
            spec = Mock()
            spec.info = "custom spec"
            return spec

        provider = HybridModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            hybrid_stack_spec=custom_stack_spec,
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel") as mock_model:
            provider.provide(pre_process=True, post_process=True)

        spec_call_kwarg = mock_model.call_args.kwargs["hybrid_stack_spec"]
        assert isinstance(spec_call_kwarg, Mock)
        assert spec_call_kwarg.info == "custom spec"

    def test_finalize_uses_compatible_hybrid_layer_count(self):
        provider = HybridModelProvider(
            hidden_size=768,
            num_attention_heads=8,
            hybrid_layer_pattern="M-M-|M-M*-/MM/MM",
        )

        with patch.object(hybrid_provider.TransformerConfig, "finalize", autospec=True) as mock_finalize:
            provider.finalize()

        assert provider.num_layers == 9
        mock_finalize.assert_called_once_with(provider)

    def test_finalize_mtp_num_layers_none_with_repeated_layer(self):
        sep = hybrid_provider.Symbols.MTP_SEPARATOR
        provider = HybridModelProvider(
            hidden_size=128,
            num_attention_heads=1,
            hybrid_layer_pattern="M-M-M-M-",
            mtp_hybrid_override_pattern="M*",
            mtp_num_layers=None,
            mtp_use_repeated_layer=True,
        )

        with patch.object(hybrid_provider.TransformerConfig, "finalize", autospec=True):
            provider.finalize()

        assert provider.hybrid_layer_pattern == "M-M-M-M-" + sep + "M*"
        assert provider.mtp_num_layers is not None
