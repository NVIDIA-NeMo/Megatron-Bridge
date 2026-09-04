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

import copy
from unittest.mock import Mock, patch

from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronVLModelProvider


def _copy_attention_config(config):
    try:
        from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
    except ModuleNotFoundError as error:
        if error.name != "megatron.core.transformer.attention_layer_config":
            raise
        return copy.deepcopy(config)
    return AttentionLayerConfig.from_config(config)


def test_provider_keeps_runtime_process_groups_out_of_nested_configs():
    class UncopyableProcessGroupCollection:
        def __deepcopy__(self, memo):
            raise TypeError("runtime process groups cannot be copied")

    provider = NemotronVLModelProvider(num_layers=2, vocab_size=1000)
    pg_collection = UncopyableProcessGroupCollection()
    provider._pg_collection = pg_collection

    def create_model(**kwargs):
        copied_config = _copy_attention_config(kwargs["language_transformer_config"])
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
