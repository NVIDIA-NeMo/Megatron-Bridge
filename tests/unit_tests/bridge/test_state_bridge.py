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

from megatron.hub.bridge.state_bridge import MegatronStateBridge
from megatron.hub.bridge.weight_bridge import DirectWeightBridge, QKVWeightBridge


class TestMegatronStateBridge:
    @pytest.fixture
    def sample_mappings(self):
        """Provides a sample list of weight bridge mappings."""
        return [
            DirectWeightBridge(
                megatron="embedding.word_embeddings.weight",
                to="model.embed_tokens.weight",
            ),
            QKVWeightBridge(
                megatron="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            DirectWeightBridge(
                megatron="decoder.layers.*.mlp.linear_fc1.weight",
                to="model.layers.*.mlp.gate_proj.weight",
            ),
            DirectWeightBridge(
                megatron="output_layer.weight",
                to="lm_head.weight",
            ),
        ]

    @pytest.fixture
    def state_bridge(self, sample_mappings):
        """Initializes MegatronStateBridge with sample mappings."""
        return MegatronStateBridge(*sample_mappings)

    def test_init_and_len(self, state_bridge, sample_mappings):
        """Test initialization and length of the state bridge."""
        assert len(state_bridge) == len(sample_mappings)
        assert state_bridge.get_all_mappings() == sample_mappings

    def test_query_megatron_exact_match(self, state_bridge):
        """Test querying with an exact megatron parameter name."""
        mapping = state_bridge.query_megatron("embedding.word_embeddings.weight")
        assert mapping is not None
        assert mapping.megatron == "embedding.word_embeddings.weight"
        assert mapping.to == "model.embed_tokens.weight"
        assert isinstance(mapping, DirectWeightBridge)

    def test_query_megatron_wildcard_match(self, state_bridge):
        """Test querying with a wildcard in the megatron parameter name."""
        mapping = state_bridge.query_megatron("decoder.layers.10.mlp.linear_fc1.weight")
        assert mapping is not None
        assert mapping.megatron == "decoder.layers.10.mlp.linear_fc1.weight"
        assert mapping.to == "model.layers.10.mlp.gate_proj.weight"
        assert isinstance(mapping, DirectWeightBridge)

    def test_query_megatron_qkv_wildcard_match(self, state_bridge):
        """Test querying a QKV bridge with a wildcard."""
        mapping = state_bridge.query_megatron("decoder.layers.5.self_attention.linear_qkv.weight")
        assert mapping is not None
        assert isinstance(mapping, QKVWeightBridge)
        assert mapping.megatron == "decoder.layers.5.self_attention.linear_qkv.weight"
        assert mapping.to["q"] == "model.layers.5.self_attn.q_proj.weight"
        assert mapping.to["k"] == "model.layers.5.self_attn.k_proj.weight"
        assert mapping.to["v"] == "model.layers.5.self_attn.v_proj.weight"

    def test_query_megatron_no_match(self, state_bridge):
        """Test querying a non-existent parameter name."""
        mapping = state_bridge.query_megatron("non.existent.weight")
        assert mapping is None

    def test_query_to_exact_match(self, state_bridge):
        """Test reverse querying with an exact destination name."""
        mapping = state_bridge.query_to("lm_head.weight")
        assert mapping is not None
        assert mapping.megatron == "output_layer.weight"
        assert mapping.to == "lm_head.weight"

    def test_query_to_wildcard_match(self, state_bridge):
        """Test reverse querying with a wildcard in the destination name."""
        mapping = state_bridge.query_to("model.layers.3.mlp.gate_proj.weight")
        assert mapping is not None
        assert mapping.megatron == "decoder.layers.3.mlp.linear_fc1.weight"
        assert mapping.to == "model.layers.3.mlp.gate_proj.weight"

    def test_query_to_dict_destination_wildcard(self, state_bridge):
        """Test reverse querying for a QKV bridge with wildcards."""
        mapping_q = state_bridge.query_to("model.layers.12.self_attn.q_proj.weight")
        assert mapping_q is not None
        assert isinstance(mapping_q, QKVWeightBridge)
        assert mapping_q.megatron == "decoder.layers.12.self_attention.linear_qkv.weight"
        assert mapping_q.to["q"] == "model.layers.12.self_attn.q_proj.weight"

        mapping_k = state_bridge.query_to("model.layers.0.self_attn.k_proj.weight")
        assert mapping_k is not None
        assert mapping_k.megatron == "decoder.layers.0.self_attention.linear_qkv.weight"

    def test_query_to_no_match(self, state_bridge):
        """Test reverse querying a non-existent destination name."""
        mapping = state_bridge.query_to("non.existent.weight")
        assert mapping is None

    def test_get_all_mappings(self, state_bridge, sample_mappings):
        """Test retrieving all mappings."""
        all_mappings = state_bridge.get_all_mappings()
        assert all_mappings == sample_mappings
        # Ensure it's a copy
        all_mappings.append("new_item")
        assert len(state_bridge.get_all_mappings()) == len(sample_mappings)

    def test_get_mappings_by_pattern(self, state_bridge):
        """Test retrieving mappings by a regex pattern."""
        mlp_mappings = state_bridge.get_mappings_by_pattern("decoder.layers.*.mlp.*")
        assert len(mlp_mappings) == 1
        assert mlp_mappings[0].megatron == "decoder.layers.*.mlp.linear_fc1.weight"

        qkv_mappings = state_bridge.get_mappings_by_pattern("decoder.layers.*.self_attention.linear_qkv.weight")
        assert len(qkv_mappings) == 1
        assert isinstance(qkv_mappings[0], QKVWeightBridge)

        all_decoder = state_bridge.get_mappings_by_pattern("decoder.*")
        assert len(all_decoder) == 2

        no_match = state_bridge.get_mappings_by_pattern("encoder.*")
        assert len(no_match) == 0

    def test_describe(self, state_bridge):
        """Test the human-readable description of the bridge."""
        description = state_bridge.describe()
        assert isinstance(description, str)
        assert "MegatronStateBridge with 4 mappings" in description
        assert "embedding.word_embeddings.weight" in description
        assert "→ model.embed_tokens.weight" in description
        assert "decoder.layers.*.self_attention.linear_qkv.weight" in description
        assert "q: model.layers.*.self_attn.q_proj.weight" in description
        assert "bridge: QKVWeightBridge" in description
        assert "bridge: DirectWeightBridge" in description

    def test_iterator_and_repr(self, state_bridge, sample_mappings):
        """Test the iterator and string representation of the bridge."""
        assert repr(state_bridge) == "MegatronStateBridge(4 mappings)"

        count = 0
        for mapping in state_bridge:
            assert mapping in sample_mappings
            count += 1
        assert count == len(sample_mappings)


class TestMegatronStateBridgeEdgeCases:
    """Test edge cases and additional functionality."""

    def test_empty_state_bridge(self):
        """Test creating an empty state bridge."""
        bridge = MegatronStateBridge()
        assert len(bridge) == 0
        assert bridge.query_megatron("any.weight") is None
        assert bridge.query_to("any.weight") is None
        assert bridge.get_all_mappings() == []
        assert bridge.get_mappings_by_pattern("*") == []
        assert repr(bridge) == "MegatronStateBridge(0 mappings)"

        # Test iterator on empty bridge
        count = 0
        for _ in bridge:
            count += 1
        assert count == 0

    def test_multiple_wildcards(self):
        """Test patterns with multiple wildcards."""
        mapping = DirectWeightBridge(
            megatron="decoder.layers.*.blocks.*.weight", to="model.layers.*.sublayers.*.weight"
        )
        bridge = MegatronStateBridge(mapping)

        # Query with multiple indices
        result = bridge.query_megatron("decoder.layers.3.blocks.2.weight")
        assert result is not None
        assert result.megatron == "decoder.layers.3.blocks.2.weight"
        assert result.to == "model.layers.3.sublayers.2.weight"

        # Reverse query
        result = bridge.query_to("model.layers.5.sublayers.1.weight")
        assert result is not None
        assert result.megatron == "decoder.layers.5.blocks.1.weight"
        assert result.to == "model.layers.5.sublayers.1.weight"

    def test_non_numeric_wildcard_no_match(self):
        """Test that wildcards only match digits."""
        mapping = DirectWeightBridge(megatron="decoder.layers.*.weight", to="model.layers.*.weight")
        bridge = MegatronStateBridge(mapping)

        # Should not match non-numeric values
        assert bridge.query_megatron("decoder.layers.abc.weight") is None
        assert bridge.query_megatron("decoder.layers.12a.weight") is None
        assert bridge.query_megatron("decoder.layers.1.2.weight") is None

        # Should match numeric values
        assert bridge.query_megatron("decoder.layers.123.weight") is not None

    def test_duplicate_patterns(self):
        """Test behavior with duplicate patterns (first match wins)."""
        mapping1 = DirectWeightBridge(megatron="decoder.layers.*.weight", to="model.layers.*.weight_v1")
        mapping2 = DirectWeightBridge(megatron="decoder.layers.*.weight", to="model.layers.*.weight_v2")
        bridge = MegatronStateBridge(mapping1, mapping2)

        # First mapping should win
        result = bridge.query_megatron("decoder.layers.0.weight")
        assert result is not None
        assert result.to == "model.layers.0.weight_v1"

        # get_mappings_by_pattern should return both
        matches = bridge.get_mappings_by_pattern("decoder.layers.*.weight")
        assert len(matches) == 2

    def test_complex_qkv_patterns(self):
        """Test complex QKV patterns with multiple levels of nesting."""
        mapping = QKVWeightBridge(
            megatron="model.*.transformer.*.attention.qkv",
            q="transformer.blocks.*.layers.*.q",
            k="transformer.blocks.*.layers.*.k",
            v="transformer.blocks.*.layers.*.v",
        )
        bridge = MegatronStateBridge(mapping)

        # Test forward query
        result = bridge.query_megatron("model.0.transformer.5.attention.qkv")
        assert result is not None
        assert result.megatron == "model.0.transformer.5.attention.qkv"
        assert result.to["q"] == "transformer.blocks.0.layers.5.q"
        assert result.to["k"] == "transformer.blocks.0.layers.5.k"
        assert result.to["v"] == "transformer.blocks.0.layers.5.v"

        # Test reverse query for each component
        result_q = bridge.query_to("transformer.blocks.2.layers.3.q")
        assert result_q is not None
        assert result_q.megatron == "model.2.transformer.3.attention.qkv"

    def test_special_characters_in_names(self):
        """Test handling of special regex characters in parameter names."""
        # Names with special regex characters
        mapping = DirectWeightBridge(megatron="decoder.layers.*.weight[0]", to="model.layers.*.weight(0)")
        bridge = MegatronStateBridge(mapping)

        # Should properly escape special characters
        result = bridge.query_megatron("decoder.layers.5.weight[0]")
        assert result is not None
        assert result.to == "model.layers.5.weight(0)"

        # Should not match without proper brackets
        assert bridge.query_megatron("decoder.layers.5.weight0") is None

    def test_pattern_matching_edge_cases(self):
        """Test various edge cases in pattern matching."""
        mappings = [
            DirectWeightBridge(megatron="*.weight", to="*.w"),
            DirectWeightBridge(megatron="prefix.*.suffix", to="p.*.s"),
            DirectWeightBridge(megatron="*", to="transformed.*"),
        ]
        bridge = MegatronStateBridge(*mappings)

        # Test single component wildcard
        result = bridge.query_megatron("5.weight")
        assert result is not None
        assert result.to == "5.w"

        # Test wildcard in middle
        result = bridge.query_megatron("prefix.100.suffix")
        assert result is not None
        assert result.to == "p.100.s"

        # Test wildcard only
        result = bridge.query_megatron("42")
        assert result is not None
        assert result.to == "transformed.42"

    def test_get_mappings_by_pattern_complex(self):
        """Test get_mappings_by_pattern with various patterns."""
        mappings = [
            DirectWeightBridge("embedding.weight", "embed.weight"),
            DirectWeightBridge("decoder.layers.*.weight", "layers.*.w"),
            DirectWeightBridge("decoder.layers.*.bias", "layers.*.b"),
            DirectWeightBridge("encoder.layers.*.weight", "enc.*.w"),
            QKVWeightBridge("decoder.*.qkv", q="dec.*.q", k="dec.*.k", v="dec.*.v"),
        ]
        bridge = MegatronStateBridge(*mappings)

        # Test exact match pattern
        exact = bridge.get_mappings_by_pattern("embedding.weight")
        assert len(exact) == 1
        assert exact[0].megatron == "embedding.weight"

        # Test wildcard pattern
        decoder_all = bridge.get_mappings_by_pattern("decoder.*")
        assert len(decoder_all) == 3  # 2 DirectWeightBridge + 1 QKVWeightBridge

        # Test more specific wildcard
        decoder_weights = bridge.get_mappings_by_pattern("decoder.layers.*.weight")
        assert len(decoder_weights) == 1

        # Test pattern matching everything
        all_mappings = bridge.get_mappings_by_pattern("*")
        assert len(all_mappings) == len(mappings)

        # Test no matches
        no_match = bridge.get_mappings_by_pattern("nonexistent.*")
        assert len(no_match) == 0

    def test_describe_formatting(self):
        """Test the describe method formatting with various bridge types."""
        from megatron.hub.bridge.weight_bridge import GatedMLPWeightBridge, TPAwareWeightBridge

        mappings = [
            DirectWeightBridge("a.weight", "b.weight"),
            QKVWeightBridge("c.qkv", q="d.q", k="d.k", v="d.v"),
            GatedMLPWeightBridge("e.mlp", gate="f.gate", up="f.up"),
            TPAwareWeightBridge("g.*.weight", "h.*.weight"),
        ]
        bridge = MegatronStateBridge(*mappings)

        description = bridge.describe()

        # Check header
        assert "MegatronStateBridge with 4 mappings:" in description

        # Check each mapping is described
        assert "1. a.weight" in description
        assert "→ b.weight" in description
        assert "bridge: DirectWeightBridge" in description

        assert "2. c.qkv" in description
        assert "q: d.q" in description
        assert "k: d.k" in description
        assert "v: d.v" in description
        assert "bridge: QKVWeightBridge" in description

        assert "3. e.mlp" in description
        assert "gate: f.gate" in description
        assert "up: f.up" in description
        assert "bridge: GatedMLPWeightBridge" in description

        assert "4. g.*.weight" in description
        assert "→ h.*.weight" in description
        assert "bridge: TPAwareWeightBridge" in description

    def test_initialization_with_list(self):
        """Test that MegatronStateBridge can be initialized from a list using *."""
        mappings_list = [DirectWeightBridge("a.weight", "b.weight"), DirectWeightBridge("c.weight", "d.weight")]

        # Initialize using * to unpack list
        bridge = MegatronStateBridge(*mappings_list)
        assert len(bridge) == 2
        assert bridge.get_all_mappings() == mappings_list

    def test_immutability_of_returned_mappings(self):
        """Test that modifications to returned mappings don't affect the bridge."""
        mapping1 = DirectWeightBridge("a.weight", "b.weight")
        mapping2 = DirectWeightBridge("c.weight", "d.weight")
        bridge = MegatronStateBridge(mapping1, mapping2)

        # Get all mappings and modify the returned list
        all_mappings = bridge.get_all_mappings()
        original_len = len(all_mappings)
        all_mappings.append(DirectWeightBridge("e.weight", "f.weight"))

        # Bridge should remain unchanged
        assert len(bridge) == original_len
        assert len(bridge.get_all_mappings()) == original_len
