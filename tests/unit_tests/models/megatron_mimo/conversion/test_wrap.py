# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.megatron_mimo.conversion import (
    MIMOComponent,
    build_route_local_registry,
    make_route_local_bridge,
)


def _language_route() -> MIMOComponent:
    return MIMOComponent(
        name="language",
        source_prefix="language_model.",
        target_module_path="language_model",
    )


def _vision_route() -> MIMOComponent:
    return MIMOComponent(
        name="vision",
        source_prefix="vision_model.",
        target_module_path="modality_submodules.images.encoders.qwen_visual",
    )


def _vlm_source_registry() -> MegatronMappingRegistry:
    """A two-component registry: language_model.* and vision_model.* prefixes.

    Covers the four mapping subclasses MIMO conversion is expected to handle
    for the Qwen3.5-VL-shape source bridge: AutoMapping, QKVMapping,
    GatedMLPMapping, and ReplicatedMapping.
    """
    return MegatronMappingRegistry(
        AutoMapping(
            megatron_param="language_model.embedding.word_embeddings.weight",
            hf_param="model.embed_tokens.weight",
        ),
        AutoMapping(
            megatron_param="language_model.output_layer.weight",
            hf_param="lm_head.weight",
        ),
        AutoMapping(
            megatron_param="language_model.decoder.final_layernorm.weight",
            hf_param="model.norm.weight",
        ),
        QKVMapping(
            megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
            q="model.layers.*.self_attn.q_proj.weight",
            k="model.layers.*.self_attn.k_proj.weight",
            v="model.layers.*.self_attn.v_proj.weight",
        ),
        AutoMapping(
            megatron_param="language_model.decoder.layers.*.self_attention.linear_proj.weight",
            hf_param="model.layers.*.self_attn.o_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
            gate="model.layers.*.mlp.gate_proj.weight",
            up="model.layers.*.mlp.up_proj.weight",
        ),
        AutoMapping(
            megatron_param="language_model.decoder.layers.*.mlp.linear_fc2.weight",
            hf_param="model.layers.*.mlp.down_proj.weight",
        ),
        AutoMapping(
            megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            hf_param="model.layers.*.input_layernorm.weight",
        ),
        ReplicatedMapping(
            megatron_param="vision_model.**",
            hf_param="visual.**",
        ),
    )


class TestBuildRouteLocalRegistry:
    def test_filters_to_route_prefix(self):
        source = _vlm_source_registry()
        language_local = build_route_local_registry(source, _language_route())
        vision_local = build_route_local_registry(source, _vision_route())

        # Every language-local mapping has no language_model. prefix
        for mapping in language_local.mappings:
            assert not mapping.megatron_param.startswith("language_model.")

        # Every vision-local mapping has no vision_model. prefix
        for mapping in vision_local.mappings:
            assert not mapping.megatron_param.startswith("vision_model.")

        # Sum of route-local mapping counts >= source count after expansion
        # (registry init may add layernorm aliases on the route-local side).
        assert len(language_local.mappings) + len(vision_local.mappings) >= len(source.mappings)

    def test_route_local_names_are_prefix_stripped(self):
        source = _vlm_source_registry()
        local = build_route_local_registry(source, _language_route())
        names = {m.megatron_param for m in local.mappings}
        assert "embedding.word_embeddings.weight" in names
        assert "output_layer.weight" in names
        assert "decoder.layers.*.self_attention.linear_qkv.weight" in names
        assert "decoder.layers.*.mlp.linear_fc1.weight" in names

    def test_hf_param_preserved_unchanged(self):
        """Prefix strip only affects megatron_param; hf_param is untouched."""
        source = _vlm_source_registry()
        local = build_route_local_registry(source, _language_route())
        names = {m.megatron_param: m for m in local.mappings}

        embedding = names["embedding.word_embeddings.weight"]
        assert embedding.hf_param == "model.embed_tokens.weight"

        qkv = names["decoder.layers.*.self_attention.linear_qkv.weight"]
        # QKVMapping stores hf_param as dict
        assert qkv.hf_param == {
            "q": "model.layers.*.self_attn.q_proj.weight",
            "k": "model.layers.*.self_attn.k_proj.weight",
            "v": "model.layers.*.self_attn.v_proj.weight",
        }

    def test_subclass_preserved(self):
        source = _vlm_source_registry()
        local = build_route_local_registry(source, _language_route())
        by_name = {m.megatron_param: m for m in local.mappings}

        assert isinstance(by_name["embedding.word_embeddings.weight"], AutoMapping)
        assert isinstance(by_name["decoder.layers.*.self_attention.linear_qkv.weight"], QKVMapping)
        assert isinstance(by_name["decoder.layers.*.mlp.linear_fc1.weight"], GatedMLPMapping)

    def test_replicated_double_star_mapping(self):
        """vision_model.** with hf_param visual.** should strip cleanly."""
        source = _vlm_source_registry()
        local = build_route_local_registry(source, _vision_route())
        names = [m.megatron_param for m in local.mappings]
        assert "**" in names
        replicated = [m for m in local.mappings if m.megatron_param == "**"][0]
        assert isinstance(replicated, ReplicatedMapping)
        assert replicated.hf_param == "visual.**"

    def test_lookup_works_on_route_local_registry(self):
        """The route-local registry must be usable for normal lookups."""
        source = _vlm_source_registry()
        local = build_route_local_registry(source, _language_route())

        mapping = local.megatron_to_hf_lookup("decoder.layers.5.self_attention.linear_qkv.weight")
        assert mapping is not None
        assert mapping.hf_param["q"] == "model.layers.5.self_attn.q_proj.weight"

    def test_layernorm_alias_present_on_route_local(self):
        """The route-local registry should still have the layernorm-alias expansion.

        Source registry contains the fused TE name
        'language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight'.
        After strip+rebuild, the alias 'decoder.layers.*.input_layernorm.weight'
        should be present too (added by MegatronMappingRegistry.__init__).
        """
        source = _vlm_source_registry()
        local = build_route_local_registry(source, _language_route())
        names = {m.megatron_param for m in local.mappings}
        assert "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight" in names
        assert "decoder.layers.*.input_layernorm.weight" in names

    def test_idempotent_alias_expansion(self):
        """Re-running build on an already-expanded registry does not duplicate aliases."""
        source = _vlm_source_registry()
        local_once = build_route_local_registry(source, _language_route())
        # Wrap local_once and re-feed: layernorm aliases must not duplicate.
        passthrough = MIMOComponent(
            name="language",
            source_prefix="decoder.",  # not stripping the alias name; just filtering
            target_module_path="language_model",
        )
        passthrough_registry = build_route_local_registry(local_once, passthrough)
        names = [m.megatron_param for m in passthrough_registry.mappings]
        # No duplicates
        assert len(names) == len(set(names))


class _FakeSourceBridge:
    """Stand-in for MegatronModelBridge for instance-clone semantics testing.

    The wrap helper uses copy.copy + types.MethodType — both work on any
    Python object, so we can validate the override semantics without spinning
    up a real bridge subclass.
    """

    def __init__(self, registry: MegatronMappingRegistry):
        self._registry = registry
        self.some_state = "original"

    def mapping_registry(self) -> MegatronMappingRegistry:
        return self._registry

    def some_method(self) -> str:
        return f"hello from {self.some_state}"


class TestMakeRouteLocalBridge:
    def test_mapping_registry_overridden(self):
        registry = _vlm_source_registry()
        source = _FakeSourceBridge(registry)

        wrapped = make_route_local_bridge(source, _language_route())

        wrapped_registry = wrapped.mapping_registry()
        assert wrapped_registry is not registry
        names = {m.megatron_param for m in wrapped_registry.mappings}
        assert "embedding.word_embeddings.weight" in names

    def test_source_bridge_unchanged(self):
        registry = _vlm_source_registry()
        source = _FakeSourceBridge(registry)
        original_registry = source.mapping_registry()

        _ = make_route_local_bridge(source, _language_route())

        assert source.mapping_registry() is original_registry
        # Source's mapping names still include the language_model. prefix
        names = {m.megatron_param for m in source.mapping_registry().mappings}
        assert "language_model.embedding.word_embeddings.weight" in names

    def test_other_methods_delegate_to_source_class(self):
        """Only mapping_registry is overridden; everything else stays original."""
        registry = _vlm_source_registry()
        source = _FakeSourceBridge(registry)
        wrapped = make_route_local_bridge(source, _language_route())

        # Calls inherited class method via copied state
        assert wrapped.some_method() == "hello from original"
        # State copied
        assert wrapped.some_state == "original"

    def test_explicit_registry_used(self):
        registry = _vlm_source_registry()
        source = _FakeSourceBridge(registry)
        precomputed = build_route_local_registry(registry, _language_route())

        wrapped = make_route_local_bridge(source, _language_route(), route_local_registry=precomputed)

        assert wrapped.mapping_registry() is precomputed

    @pytest.mark.parametrize("route", [_language_route(), _vision_route()])
    def test_route_specific_registry_independent_per_call(self, route):
        registry = _vlm_source_registry()
        source = _FakeSourceBridge(registry)

        wrapped = make_route_local_bridge(source, route)
        names = {m.megatron_param for m in wrapped.mapping_registry().mappings}
        # Names contain no source prefix from this route
        for name in names:
            assert not name.startswith(route.source_prefix)
