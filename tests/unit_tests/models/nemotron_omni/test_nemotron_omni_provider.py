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

import pytest

from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider


pytestmark = pytest.mark.unit


def test_vision_recompute_is_disabled_by_default():
    provider = NemotronOmniModelProvider(
        recompute_granularity="selective",
        recompute_modules=["core_attn", "mlp"],
    )

    vision_config = provider._build_vision_config(provider)

    assert vision_config.recompute_granularity is None
    assert vision_config.recompute_method is None
    assert vision_config.recompute_num_layers is None


def test_vision_recompute_inherits_selective_model_config_when_enabled():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        recompute_granularity="selective",
        recompute_modules=["core_attn", "mlp"],
    )

    vision_config = provider._build_vision_config(provider)

    assert vision_config.recompute_granularity == "selective"
    assert vision_config.recompute_modules == ["core_attn", "mlp"]
    assert vision_config.recompute_num_layers is None


def test_vision_recompute_inherits_full_model_config_when_enabled():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=4,
    )

    vision_config = provider._build_vision_config(provider)

    assert vision_config.recompute_granularity == "full"
    assert vision_config.recompute_method == "uniform"
    assert vision_config.recompute_num_layers == 4


def test_vision_recompute_can_override_full_model_config_with_selective():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=4,
        vision_recompute_granularity="selective",
    )

    vision_config = provider._build_vision_config(provider)

    assert provider.recompute_granularity == "full"
    assert vision_config.recompute_granularity == "selective"
    assert vision_config.recompute_method is None
    assert vision_config.recompute_num_layers is None


def test_vision_recompute_can_use_per_layer_full_recompute_independently():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        recompute_granularity="selective",
        recompute_modules=["core_attn", "mlp"],
        vision_recompute_granularity="full",
        vision_recompute_method="uniform",
        vision_recompute_num_layers=1,
    )

    vision_config = provider._build_vision_config(provider)

    assert provider.recompute_granularity == "selective"
    assert vision_config.recompute_granularity == "full"
    assert vision_config.recompute_method == "uniform"
    assert vision_config.recompute_num_layers == 1


def test_vision_recompute_requires_effective_granularity():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
    )

    with pytest.raises(ValueError, match="effective recompute granularity"):
        provider._build_vision_config(provider)


@pytest.mark.parametrize(
    ("method", "num_layers", "message"),
    [
        (None, 1, "requires a recompute method"),
        ("uniform", None, "requires a layer count"),
    ],
)
def test_full_vision_recompute_requires_complete_policy(method, num_layers, message):
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        recompute_granularity="selective",
        vision_recompute_granularity="full",
        vision_recompute_method=method,
        vision_recompute_num_layers=num_layers,
    )

    with pytest.raises(ValueError, match=message):
        provider._build_vision_config(provider)


def test_vision_recompute_rejects_invalid_granularity():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        vision_recompute_granularity="typo",
    )

    with pytest.raises(ValueError, match="must be 'full' or 'selective'"):
        provider._build_vision_config(provider)


def test_full_vision_recompute_rejects_invalid_method():
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        vision_recompute_granularity="full",
        vision_recompute_method="typo",
        vision_recompute_num_layers=1,
    )

    with pytest.raises(ValueError, match="must be 'uniform' or 'block'"):
        provider._build_vision_config(provider)


@pytest.mark.parametrize("num_layers", [0, 33, 1.5, True])
def test_full_vision_recompute_rejects_invalid_layer_count(num_layers):
    provider = NemotronOmniModelProvider(
        recompute_vision=True,
        radio_force_eval_mode=False,
        vision_recompute_granularity="full",
        vision_recompute_method="uniform",
        vision_recompute_num_layers=num_layers,
    )

    with pytest.raises(ValueError, match="integer between 1 and 32"):
        provider._build_vision_config(provider)


def test_vision_recompute_rejects_forced_radio_eval_mode():
    provider = NemotronOmniModelProvider(recompute_vision=True)

    with pytest.raises(ValueError, match="radio_force_eval_mode=False"):
        provider._build_vision_config(provider)
