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
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models import BertModelProvider
from megatron.bridge.models.bert import bert_provider


class TestBertModelProvider:
    """Test cases for BertModelProvider class."""

    def test_defaults(self):
        """Test that BertModelProvider has the expected masked-LM defaults."""
        provider = BertModelProvider(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            vocab_size=128,
        )

        # No NSP/pooler head, matching MegatronBertForMaskedLM (add_pooling_layer=False).
        assert provider.add_binary_head is False
        assert provider.num_tokentypes == 2
        assert provider.share_embeddings_and_output_weights is True
        assert provider.position_embedding_type == "learned_absolute"
        assert provider.max_position_embeddings == 512
        assert provider.should_pad_vocab is False

    def test_provide_requires_vocab_size(self):
        """Test that provide() fails fast when vocab_size was never configured."""
        provider = BertModelProvider(num_layers=2, hidden_size=32, num_attention_heads=4)

        with pytest.raises(AssertionError, match="vocab_size"):
            provider.provide()

    def test_provide_constructs_mcore_bert_with_provider_fields(self):
        """Test that provide() forwards BERT-specific configuration to Megatron Core."""
        layer_spec = ModuleSpec(module=torch.nn.Identity)
        provider = BertModelProvider(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            vocab_size=128,
            max_position_embeddings=64,
            num_tokentypes=3,
            share_embeddings_and_output_weights=False,
            transformer_layer_spec=layer_spec,
        )

        with patch.object(bert_provider, "MCoreBertModel") as model_cls:
            model = provider.provide(pre_process=True, post_process=False, vp_stage=1)

        assert model is model_cls.return_value
        model_cls.assert_called_once_with(
            config=provider,
            num_tokentypes=3,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=64,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="learned_absolute",
            rotary_percent=1.0,
            seq_len_interpolation_factor=None,
            add_binary_head=False,
            return_embeddings=False,
            pre_process=True,
            post_process=False,
            vp_stage=1,
            pg_collection=None,
        )

    def test_provide_resolves_callable_layer_spec_and_pads_vocab(self):
        """Test callable layer specs and optional vocabulary padding."""
        layer_spec = ModuleSpec(module=torch.nn.Identity)
        layer_spec_factory = Mock(return_value=layer_spec)
        provider = BertModelProvider(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            vocab_size=130,
            should_pad_vocab=True,
            transformer_layer_spec=layer_spec_factory,
        )

        with (
            patch.object(bert_provider, "calculate_padded_vocab_size", return_value=144) as calculate_vocab,
            patch.object(bert_provider, "MCoreBertModel") as model_cls,
        ):
            provider.provide(pre_process=True, post_process=True)

        calculate_vocab.assert_called_once_with(130, 128, 1)
        layer_spec_factory.assert_called_once_with()
        assert model_cls.call_args.kwargs["transformer_layer_spec"] is layer_spec
        assert model_cls.call_args.kwargs["vocab_size"] == 144
