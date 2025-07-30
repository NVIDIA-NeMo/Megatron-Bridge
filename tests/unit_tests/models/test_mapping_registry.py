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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import DirectMapping, QKVMapping


class TestMegatronMappingRegistry:
    @pytest.fixture
    def sample_mappings(self):
        """Provides a sample list of param mapping mappings."""
        return [
            DirectMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            DirectMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                hf_param="model.layers.*.mlp.gate_proj.weight",
            ),
            DirectMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
        ]

    @pytest.fixture
    def mapping_registry(self, sample_mappings):
        """Initializes MegatronMappingRegistry with sample mappings."""
        return MegatronMappingRegistry(*sample_mappings)

    def test_init_and_len(self, mapping_registry, sample_mappings):
        """Test initialization and length of the mapping registry."""
        assert len(mapping_registry) == len(sample_mappings)
        assert mapping_registry.get_all_mappings() == sample_mappings

    def test_megatron_to_hf_lookup_exact_match(self, mapping_registry):
        """Test querying with an exact megatron parameter name."""
        mapping = mapping_registry.megatron_to_hf_lookup("embedding.word_embeddings.weight")
        assert mapping is not None
        assert mapping.megatron_param == "embedding.word_embeddings.weight"
        assert mapping.hf_param == "model.embed_tokens.weight"
        assert isinstance(mapping, DirectMapping)

    def test_megatron_to_hf_lookup_wildcard_match(self, mapping_registry):
        """Test querying with a wildcard in the megatron parameter name."""
        mapping = mapping_registry.megatron_to_hf_lookup("decoder.layers.10.mlp.linear_fc1.weight")
        assert mapping is not None
        assert mapping.megatron_param == "decoder.layers.10.mlp.linear_fc1.weight"
        assert mapping.hf_param == "model.layers.10.mlp.gate_proj.weight"
        assert isinstance(mapping, DirectMapping)

    def test_megatron_to_hf_lookup_qkv_wildcard_match(self, mapping_registry):
        """Test querying a QKV bridge with a wildcard."""
        mapping = mapping_registry.megatron_to_hf_lookup("decoder.layers.5.self_attention.linear_qkv.weight")
        assert mapping is not None
        assert isinstance(mapping, QKVMapping)
        assert mapping.megatron_param == "decoder.layers.5.self_attention.linear_qkv.weight"
        assert mapping.hf_param["q"] == "model.layers.5.self_attn.q_proj.weight"
        assert mapping.hf_param["k"] == "model.layers.5.self_attn.k_proj.weight"
        assert mapping.hf_param["v"] == "model.layers.5.self_attn.v_proj.weight"

    def test_megatron_to_hf_lookup_no_match(self, mapping_registry):
        """Test querying a non-existent parameter name."""
        mapping = mapping_registry.megatron_to_hf_lookup("non.existent.weight")
        assert mapping is None

    def test_hf_to_megatron_lookup_exact_match(self, mapping_registry):
        """Test reverse querying with an exact destination name."""
        mapping = mapping_registry.hf_to_megatron_lookup("lm_head.weight")
        assert mapping is not None
        assert mapping.megatron_param == "output_layer.weight"
        assert mapping.hf_param == "lm_head.weight"

    def test_hf_to_megatron_lookup_wildcard_match(self, mapping_registry):
        """Test reverse querying with a wildcard in the destination name."""
        mapping = mapping_registry.hf_to_megatron_lookup("model.layers.3.mlp.gate_proj.weight")
        assert mapping is not None
        assert mapping.megatron_param == "decoder.layers.3.mlp.linear_fc1.weight"
        assert mapping.hf_param == "model.layers.3.mlp.gate_proj.weight"

    def test_hf_to_megatron_lookup_dict_destination_wildcard(self, mapping_registry):
        """Test reverse querying for a QKV bridge with wildcards."""
        mapping_q = mapping_registry.hf_to_megatron_lookup("model.layers.12.self_attn.q_proj.weight")
        assert mapping_q is not None
        assert isinstance(mapping_q, QKVMapping)
        assert mapping_q.megatron_param == "decoder.layers.12.self_attention.linear_qkv.weight"
        assert mapping_q.hf_param["q"] == "model.layers.12.self_attn.q_proj.weight"

        mapping_k = mapping_registry.hf_to_megatron_lookup("model.layers.0.self_attn.k_proj.weight")
        assert mapping_k is not None
        assert mapping_k.megatron_param == "decoder.layers.0.self_attention.linear_qkv.weight"

    def test_hf_to_megatron_lookup_no_match(self, mapping_registry):
        """Test reverse querying a non-existent destination name."""
        mapping = mapping_registry.hf_to_megatron_lookup("non.existent.weight")
        assert mapping is None

    def test_get_all_mappings(self, mapping_registry, sample_mappings):
        """Test retrieving all mappings."""
        all_mappings = mapping_registry.get_all_mappings()
        assert all_mappings == sample_mappings
        # Ensure it's a copy
        all_mappings.append("new_item")
        assert len(mapping_registry.get_all_mappings()) == len(sample_mappings)

    def test_get_mappings_by_pattern(self, mapping_registry):
        """Test retrieving mappings by a regex pattern."""
        mlp_mappings = mapping_registry.get_mappings_by_pattern("decoder.layers.*.mlp.*")
        assert len(mlp_mappings) == 1
        assert mlp_mappings[0].megatron_param == "decoder.layers.*.mlp.linear_fc1.weight"

        qkv_mappings = mapping_registry.get_mappings_by_pattern("decoder.layers.*.self_attention.linear_qkv.weight")
        assert len(qkv_mappings) == 1
        assert isinstance(qkv_mappings[0], QKVMapping)

        all_decoder = mapping_registry.get_mappings_by_pattern("decoder.*")
        assert len(all_decoder) == 2

        no_match = mapping_registry.get_mappings_by_pattern("encoder.*")
        assert len(no_match) == 0

    def test_describe(self, mapping_registry):
        """Test the human-readable description of the bridge."""
        description = mapping_registry.describe()
        assert isinstance(description, str)
        assert "MegatronMappingRegistry with 4 mappings" in description
        assert "embedding.word_embeddings.weight" in description
        assert "→ model.embed_tokens.weight" in description
        assert "decoder.layers.*.self_attention.linear_qkv.weight" in description
        assert "q: model.layers.*.self_attn.q_proj.weight" in description
        assert "bridge: QKVMapping" in description
        assert "bridge: DirectMapping" in description

    def test_iterator_and_repr(self, mapping_registry, sample_mappings):
        """Test the iterator and string representation of the bridge."""
        assert repr(mapping_registry) == "MegatronMappingRegistry(4 mappings)"

        count = 0
        for mapping in mapping_registry:
            assert mapping in sample_mappings
            count += 1
        assert count == len(sample_mappings)


class TestMegatronMappingRegistryEdgeCases:
    """Test edge cases and additional functionality."""

    def test_empty_mapping_registry(self):
        """Test creating an empty mapping registry."""
        bridge = MegatronMappingRegistry()
        assert len(bridge) == 0
        assert bridge.megatron_to_hf_lookup("any.weight") is None
        assert bridge.hf_to_megatron_lookup("any.weight") is None
        assert bridge.get_all_mappings() == []
        assert bridge.get_mappings_by_pattern("*") == []
        assert repr(bridge) == "MegatronMappingRegistry(0 mappings)"

        # Test iterator on empty bridge
        count = 0
        for _ in bridge:
            count += 1
        assert count == 0

    def test_multiple_wildcards(self):
        """Test patterns with multiple wildcards."""
        mapping = DirectMapping(
            megatron_param="decoder.layers.*.blocks.*.weight", hf_param="model.layers.*.sublayers.*.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Query with multiple indices
        result = bridge.megatron_to_hf_lookup("decoder.layers.3.blocks.2.weight")
        assert result is not None
        assert result.megatron_param == "decoder.layers.3.blocks.2.weight"
        assert result.hf_param == "model.layers.3.sublayers.2.weight"

        # Reverse query
        result = bridge.hf_to_megatron_lookup("model.layers.5.sublayers.1.weight")
        assert result is not None
        assert result.megatron_param == "decoder.layers.5.blocks.1.weight"
        assert result.hf_param == "model.layers.5.sublayers.1.weight"

    def test_non_numeric_wildcard_no_match(self):
        """Test that wildcards only match digits."""
        mapping = DirectMapping(megatron_param="decoder.layers.*.weight", hf_param="model.layers.*.weight")
        bridge = MegatronMappingRegistry(mapping)

        # Should not match non-numeric values
        assert bridge.megatron_to_hf_lookup("decoder.layers.abc.weight") is None
        assert bridge.megatron_to_hf_lookup("decoder.layers.12a.weight") is None
        assert bridge.megatron_to_hf_lookup("decoder.layers.1.2.weight") is None

        # Should match numeric values
        assert bridge.megatron_to_hf_lookup("decoder.layers.123.weight") is not None

    def test_duplicate_patterns(self):
        """Test behavior with duplicate patterns (first match wins)."""
        mapping1 = DirectMapping(megatron_param="decoder.layers.*.weight", hf_param="model.layers.*.weight_v1")
        mapping2 = DirectMapping(megatron_param="decoder.layers.*.weight", hf_param="model.layers.*.weight_v2")
        bridge = MegatronMappingRegistry(mapping1, mapping2)

        # First mapping should win
        result = bridge.megatron_to_hf_lookup("decoder.layers.0.weight")
        assert result is not None
        assert result.hf_param == "model.layers.0.weight_v1"

        # get_mappings_by_pattern should return both
        matches = bridge.get_mappings_by_pattern("decoder.layers.*.weight")
        assert len(matches) == 2

    def test_complex_qkv_patterns(self):
        """Test complex QKV patterns with multiple levels of nesting."""
        mapping = QKVMapping(
            megatron_param="model.*.transformer.*.attention.qkv",
            q="transformer.blocks.*.layers.*.q",
            k="transformer.blocks.*.layers.*.k",
            v="transformer.blocks.*.layers.*.v",
        )
        bridge = MegatronMappingRegistry(mapping)

        # Test forward query
        result = bridge.megatron_to_hf_lookup("model.0.transformer.5.attention.qkv")
        assert result is not None
        assert result.megatron_param == "model.0.transformer.5.attention.qkv"
        assert result.hf_param["q"] == "transformer.blocks.0.layers.5.q"
        assert result.hf_param["k"] == "transformer.blocks.0.layers.5.k"
        assert result.hf_param["v"] == "transformer.blocks.0.layers.5.v"

        # Test reverse query for each component
        result_q = bridge.hf_to_megatron_lookup("transformer.blocks.2.layers.3.q")
        assert result_q is not None
        assert result_q.megatron_param == "model.2.transformer.3.attention.qkv"

    def test_special_characters_in_names(self):
        """Test handling of special regex characters in parameter names."""
        # Names with special regex characters
        mapping = DirectMapping(megatron_param="decoder.layers.*.weight[0]", hf_param="model.layers.*.weight(0)")
        bridge = MegatronMappingRegistry(mapping)

        # Should properly escape special characters
        result = bridge.megatron_to_hf_lookup("decoder.layers.5.weight[0]")
        assert result is not None
        assert result.hf_param == "model.layers.5.weight(0)"

        # Should not match without proper brackets
        assert bridge.megatron_to_hf_lookup("decoder.layers.5.weight0") is None

    def test_pattern_matching_edge_cases(self):
        """Test various edge cases in pattern matching."""
        mappings = [
            DirectMapping(megatron_param="*.weight", hf_param="*.w"),
            DirectMapping(megatron_param="prefix.*.suffix", hf_param="p.*.s"),
            DirectMapping(megatron_param="*", hf_param="transformed.*"),
        ]
        bridge = MegatronMappingRegistry(*mappings)

        # Test single component wildcard
        result = bridge.megatron_to_hf_lookup("5.weight")
        assert result is not None
        assert result.hf_param == "5.w"

        # Test wildcard in middle
        result = bridge.megatron_to_hf_lookup("prefix.100.suffix")
        assert result is not None
        assert result.hf_param == "p.100.s"

        # Test wildcard only
        result = bridge.megatron_to_hf_lookup("42")
        assert result is not None
        assert result.hf_param == "transformed.42"

    def test_get_mappings_by_pattern_complex(self):
        """Test get_mappings_by_pattern with various patterns."""
        mappings = [
            DirectMapping("embedding.weight", "embed.weight"),
            DirectMapping("decoder.layers.*.weight", "layers.*.w"),
            DirectMapping("decoder.layers.*.bias", "layers.*.b"),
            DirectMapping("encoder.layers.*.weight", "enc.*.w"),
            QKVMapping("decoder.*.qkv", q="dec.*.q", k="dec.*.k", v="dec.*.v"),
        ]
        bridge = MegatronMappingRegistry(*mappings)

        # Test exact match pattern
        exact = bridge.get_mappings_by_pattern("embedding.weight")
        assert len(exact) == 1
        assert exact[0].megatron_param == "embedding.weight"

        # Test wildcard pattern
        decoder_all = bridge.get_mappings_by_pattern("decoder.*")
        assert len(decoder_all) == 3  # 2 DirectMapping + 1 QKVMapping

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
        from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, TPAwareMapping

        mappings = [
            DirectMapping("a.weight", "b.weight"),
            QKVMapping("c.qkv", q="d.q", k="d.k", v="d.v"),
            GatedMLPMapping("e.mlp", gate="f.gate", up="f.up"),
            TPAwareMapping(megatron_param="g.*.weight", hf_param="h.*.weight"),
        ]
        bridge = MegatronMappingRegistry(*mappings)

        description = bridge.describe()

        # Check header
        assert "MegatronMappingRegistry with 4 mappings:" in description

        # Check each mapping is described
        assert "1. a.weight" in description
        assert "→ b.weight" in description
        assert "bridge: DirectMapping" in description

        assert "2. c.qkv" in description
        assert "q: d.q" in description
        assert "k: d.k" in description
        assert "v: d.v" in description
        assert "bridge: QKVMapping" in description

        assert "3. e.mlp" in description
        assert "gate: f.gate" in description
        assert "up: f.up" in description
        assert "bridge: GatedMLPMapping" in description

        assert "4. g.*.weight" in description
        assert "→ h.*.weight" in description
        assert "bridge: TPAwareMapping" in description

    def test_initialization_with_list(self):
        """Test that MegatronMappingRegistry can be initialized from a list using *."""
        mappings_list = [DirectMapping("a.weight", "b.weight"), DirectMapping("c.weight", "d.weight")]

        # Initialize using * to unpack list
        bridge = MegatronMappingRegistry(*mappings_list)
        assert len(bridge) == 2
        assert bridge.get_all_mappings() == mappings_list

    def test_immutability_of_returned_mappings(self):
        """Test that modifications to returned mappings don't affect the bridge."""
        mapping1 = DirectMapping("a.weight", "b.weight")
        mapping2 = DirectMapping("c.weight", "d.weight")
        bridge = MegatronMappingRegistry(mapping1, mapping2)

        # Get all mappings and modify the returned list
        all_mappings = bridge.get_all_mappings()
        original_len = len(all_mappings)
        all_mappings.append(DirectMapping("e.weight", "f.weight"))

        # Bridge should remain unchanged
        assert len(bridge) == original_len
        assert len(bridge.get_all_mappings()) == original_len


class TestMegatronMappingRegistryDoubleWildcard:
    """Test the new "**" wildcard functionality."""

    def test_double_wildcard_basic_matching(self):
        """Test basic "**" wildcard matching that can cross dots."""
        mapping = DirectMapping(
            megatron_param="model.**.weight",
            hf_param="hf.**.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Should match patterns that cross multiple dot levels
        result = bridge.megatron_to_hf_lookup("model.layers.0.attention.weight")
        assert result is not None
        assert result.megatron_param == "model.layers.0.attention.weight"
        assert result.hf_param == "hf.layers.0.attention.weight"

        # Should match simple single level
        result = bridge.megatron_to_hf_lookup("model.simple.weight")
        assert result is not None
        assert result.hf_param == "hf.simple.weight"

        # Should match complex nested structures
        result = bridge.megatron_to_hf_lookup("model.transformer.layers.12.attention.self.weight")
        assert result is not None
        assert result.hf_param == "hf.transformer.layers.12.attention.self.weight"

    def test_double_wildcard_vs_single_wildcard(self):
        """Test difference between "**" and "*" wildcards."""
        mappings = [
            DirectMapping(
                megatron_param="model.*.simple.weight",  # Single wildcard - no dots
                hf_param="hf_single.*.weight"
            ),
            DirectMapping(
                megatron_param="model.**.complex.weight",  # Double wildcard - can have dots
                hf_param="hf_double.**.weight"
            )
        ]
        bridge = MegatronMappingRegistry(*mappings)

        # Single wildcard should match only single components (no dots)
        result = bridge.megatron_to_hf_lookup("model.layer0.simple.weight")
        assert result is not None
        assert result.hf_param == "hf_single.layer0.weight"

        # Single wildcard should NOT match components with dots
        result = bridge.megatron_to_hf_lookup("model.layers.0.simple.weight")
        assert result is None  # Should not match because "layers.0" contains a dot

        # Double wildcard should match complex nested structures
        result = bridge.megatron_to_hf_lookup("model.layers.5.attention.complex.weight")
        assert result is not None
        assert result.hf_param == "hf_double.layers.5.attention.weight"

        # Double wildcard should also match simple structures
        result = bridge.megatron_to_hf_lookup("model.simple.complex.weight")
        assert result is not None
        assert result.hf_param == "hf_double.simple.weight"

    def test_mixed_wildcards_in_pattern(self):
        """Test patterns with both "*" and "**" wildcards."""
        mapping = DirectMapping(
            megatron_param="model.layers.*.attention.**.weight",
            hf_param="hf.blocks.*.attn.**.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Should match with single component for * and complex path for **
        result = bridge.megatron_to_hf_lookup("model.layers.12.attention.self.qkv.weight")
        assert result is not None
        assert result.megatron_param == "model.layers.12.attention.self.qkv.weight"
        assert result.hf_param == "hf.blocks.12.attn.self.qkv.weight"

        # Should match with just the required structure
        result = bridge.megatron_to_hf_lookup("model.layers.0.attention.proj.weight")
        assert result is not None
        assert result.hf_param == "hf.blocks.0.attn.proj.weight"

        # Should NOT match if * part contains dots
        result = bridge.megatron_to_hf_lookup("model.layers.sub.layer.attention.proj.weight")
        assert result is None

    def test_double_wildcard_reverse_lookup(self):
        """Test reverse lookup with "**" wildcards."""
        mapping = DirectMapping(
            megatron_param="decoder.**.linear.weight",
            hf_param="model.**.proj.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Test reverse lookup from HF to Megatron
        result = bridge.hf_to_megatron_lookup("model.layers.5.attention.self.proj.weight")
        assert result is not None
        assert result.megatron_param == "decoder.layers.5.attention.self.linear.weight"
        assert result.hf_param == "model.layers.5.attention.self.proj.weight"

        # Test with simpler structure
        result = bridge.hf_to_megatron_lookup("model.embed.proj.weight")
        assert result is not None
        assert result.megatron_param == "decoder.embed.linear.weight"

    def test_double_wildcard_qkv_mapping(self):
        """Test "**" wildcards with QKV mappings."""
        mapping = QKVMapping(
            megatron_param="decoder.**.attention.qkv.weight",
            q="model.**.attn.q_proj.weight",
            k="model.**.attn.k_proj.weight", 
            v="model.**.attn.v_proj.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Test forward lookup
        result = bridge.megatron_to_hf_lookup("decoder.layers.8.blocks.2.attention.qkv.weight")
        assert result is not None
        assert isinstance(result, QKVMapping)
        assert result.megatron_param == "decoder.layers.8.blocks.2.attention.qkv.weight"
        assert result.hf_param["q"] == "model.layers.8.blocks.2.attn.q_proj.weight"
        assert result.hf_param["k"] == "model.layers.8.blocks.2.attn.k_proj.weight"
        assert result.hf_param["v"] == "model.layers.8.blocks.2.attn.v_proj.weight"

        # Test reverse lookup for each component
        result_q = bridge.hf_to_megatron_lookup("model.transformer.layer.5.attn.q_proj.weight")
        assert result_q is not None
        assert result_q.megatron_param == "decoder.transformer.layer.5.attention.qkv.weight"

        result_k = bridge.hf_to_megatron_lookup("model.deep.nested.layer.attn.k_proj.weight")
        assert result_k is not None
        assert result_k.megatron_param == "decoder.deep.nested.layer.attention.qkv.weight"

    def test_multiple_double_wildcards(self):
        """Test patterns with multiple "**" wildcards."""
        mapping = DirectMapping(
            megatron_param="model.**.layers.**.weight",
            hf_param="hf.**.blocks.**.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Should match complex nested structures with multiple variable parts
        result = bridge.megatron_to_hf_lookup("model.encoder.deep.layers.attention.self.weight")
        assert result is not None
        assert result.megatron_param == "model.encoder.deep.layers.attention.self.weight"
        assert result.hf_param == "hf.encoder.deep.blocks.attention.self.weight"

        # Should match simpler structures too
        result = bridge.megatron_to_hf_lookup("model.simple.layers.proj.weight")
        assert result is not None
        assert result.hf_param == "hf.simple.blocks.proj.weight"

    def test_double_wildcard_edge_cases(self):
        """Test edge cases with "**" wildcards."""
        mapping = DirectMapping(
            megatron_param="**.weight",  # Double wildcard at start
            hf_param="transformed.**.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Should match any path ending in .weight
        result = bridge.megatron_to_hf_lookup("model.layers.0.attention.weight")
        assert result is not None
        assert result.hf_param == "transformed.model.layers.0.attention.weight"

        # Should match simple cases too
        result = bridge.megatron_to_hf_lookup("embed.weight")
        assert result is not None
        assert result.hf_param == "transformed.embed.weight"

        # Test with ending double wildcard
        mapping2 = DirectMapping(
            megatron_param="model.**",
            hf_param="hf.**"
        )
        bridge2 = MegatronMappingRegistry(mapping2)

        result = bridge2.megatron_to_hf_lookup("model.anything.can.match.here")
        assert result is not None
        assert result.hf_param == "hf.anything.can.match.here"

    def test_double_wildcard_no_match_cases(self):
        """Test cases where "**" wildcards should not match."""
        mapping = DirectMapping(
            megatron_param="model.**.attention.weight",
            hf_param="hf.**.attn.weight"
        )
        bridge = MegatronMappingRegistry(mapping)

        # Should not match if required parts are missing
        result = bridge.megatron_to_hf_lookup("model.layers.0.weight")  # Missing "attention"
        assert result is None

        result = bridge.megatron_to_hf_lookup("model.attention.bias")  # Wrong ending
        assert result is None

        result = bridge.megatron_to_hf_lookup("decoder.layers.attention.weight")  # Wrong start
        assert result is None

    def test_double_wildcard_get_mappings_by_pattern(self):
        """Test get_mappings_by_pattern with "**" wildcards."""
        mappings = [
            DirectMapping("model.**.weight", "hf.**.weight"),
            DirectMapping("model.**.bias", "hf.**.bias"),
            DirectMapping("encoder.**.weight", "enc.**.weight"),
            DirectMapping("simple.weight", "simple.weight"),
        ]
        bridge = MegatronMappingRegistry(*mappings)

        # Test pattern matching with double wildcards
        model_all = bridge.get_mappings_by_pattern("model.**")
        assert len(model_all) == 2  # Both model.**.weight and model.**.bias

        weight_all = bridge.get_mappings_by_pattern("**.weight")
        assert len(weight_all) == 3  # model, encoder, and simple weight mappings

        # Test exact pattern matching
        exact = bridge.get_mappings_by_pattern("model.**.weight")
        assert len(exact) == 1
        assert exact[0].megatron_param == "model.**.weight"
