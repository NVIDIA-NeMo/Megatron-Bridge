from unittest.mock import Mock, patch

from megatron.core.transformer.attention_layer_config import AttentionLayerConfig

from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronVLModelProvider


def test_provider_keeps_runtime_process_groups_out_of_nested_configs():
    class UncopyableProcessGroupCollection:
        def __deepcopy__(self, memo):
            raise TypeError("runtime process groups cannot be copied")

    provider = NemotronVLModelProvider(num_layers=2, vocab_size=1000)
    pg_collection = UncopyableProcessGroupCollection()
    provider._pg_collection = pg_collection

    def create_model(**kwargs):
        copied_config = AttentionLayerConfig.from_config(kwargs["language_transformer_config"])
        assert copied_config._pg_collection is None
        return Mock()

    with (
        patch(
            "megatron.bridge.models.nemotron_vl.nemotron_vl_provider.get_vit_layer_with_transformer_engine_spec",
            return_value=Mock(),
        ),
        patch(
            "megatron.bridge.models.nemotron_vl.nemotron_vl_provider.get_language_mlp_submodules",
            return_value=Mock(),
        ),
        patch(
            "megatron.bridge.models.nemotron_vl.nemotron_vl_provider.LLaVAModel",
            side_effect=create_model,
        ),
        patch("megatron.bridge.models.nemotron_vl.modeling_nemotron_vl.NemotronVLModel", return_value=Mock()),
    ):
        provider.provide(pre_process=True, post_process=True)

    assert provider._pg_collection is pg_collection
