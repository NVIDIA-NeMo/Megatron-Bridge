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

"""Exercise Bridge attention configuration against the installed MCore backend selector."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer import attention as attention_module
from megatron.core.transformer import utils as transformer_utils
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.transformer_config import (
    HeterogeneousTransformerConfig,
    MLATransformerConfig,
    TransformerConfig,
)


CONFIG_TYPES = (TransformerConfig, MLATransformerConfig, HeterogeneousTransformerConfig)
BACKEND_ENV = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
FLASH_VERSION_ENV = ("NVTE_FLASH_ATTN_V2", "NVTE_FLASH_ATTN_V3", "NVTE_FLASH_ATTN_V4")
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_attention_environment():
    with patch.dict(os.environ):
        for name in (*BACKEND_ENV, *FLASH_VERSION_ENV):
            os.environ.pop(name, None)
        yield


def make_config(config_type, backend, **overrides):
    kwargs = dict(num_layers=2, hidden_size=64, num_attention_heads=4, attention_backend=backend)
    kwargs.update(overrides)
    if config_type is HeterogeneousTransformerConfig:
        block = {
            "attention": {"no_op": False, "replace_with_linear": False, "num_query_groups": 4},
            "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 256},
        }
        kwargs["heterogeneous_layers_config_encoded_json"] = json.dumps({"block_configs": [block, block]})
    return config_type(**kwargs)


def configure_mcore_attention(config):
    if hasattr(transformer_utils, "set_attention_backend"):
        transformer_utils.set_attention_backend(config)
    else:
        # The unchanged dev pin keeps the selector on LanguageModule.
        LanguageModule._set_attention_backend(SimpleNamespace(config=config))


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
@pytest.mark.parametrize("backend", list(AttnBackend))
@pytest.mark.skipif(
    not hasattr(attention_module.Attention, "_resolve_flash_version"),
    reason="The frozen MCore dev pin predates FlashAttention version controls",
)
def test_explicit_fa2_disables_fa4_without_changing_backend(config_type, backend, monkeypatch):
    monkeypatch.setenv("NVTE_FLASH_ATTN_V4", "0")
    config = make_config(config_type, backend, flash_attention_version=2)

    config.finalize()
    configure_mcore_attention(config)

    assert config.flash_attention_version == 2
    assert config.attention_backend is backend
    assert tuple(os.environ[name] for name in FLASH_VERSION_ENV) == ("1", "0", "0")


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
@pytest.mark.skipif(
    not hasattr(attention_module.Attention, "_resolve_flash_version"),
    reason="The frozen MCore dev pin predates FlashAttention version controls",
)
def test_direct_attention_uses_explicit_fa2_even_when_fa4_is_installed(config_type, monkeypatch):
    monkeypatch.setattr(attention_module, "HAVE_FA4", True)
    monkeypatch.setattr(attention_module, "HAVE_FA3", True)
    config = make_config(config_type, AttnBackend.auto, flash_attention_version=2)
    config.finalize()

    attention = SimpleNamespace(flash_attention_version=config.flash_attention_version)
    assert attention_module.Attention._resolve_flash_version(attention) == (False, False)


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
@pytest.mark.parametrize("version", [None, 2, 3, 4])
@pytest.mark.skipif(
    not hasattr(attention_module.Attention, "_resolve_flash_version"),
    reason="The frozen MCore dev pin predates FlashAttention version controls",
)
def test_explicit_flash_version_is_preserved(config_type, version):
    config = make_config(config_type, AttnBackend.auto, flash_attention_version=version)

    config.finalize()

    assert config.flash_attention_version == version


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_default_flash_version_is_inherited_from_mcore(config_type):
    config = make_config(config_type, AttnBackend.auto)

    config.finalize()

    assert getattr(config, "flash_attention_version", None) is None
    assert all(name not in os.environ for name in FLASH_VERSION_ENV)


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_unset_backend_finalizes_to_mcore_auto(config_type):
    config = make_config(config_type, None)

    config.finalize()
    configure_mcore_attention(config)

    assert config.attention_backend is AttnBackend.auto
    assert tuple(os.environ[name] for name in BACKEND_ENV) == ("1", "1", "1")
    config.finalize()
    configure_mcore_attention(config)
    assert config.attention_backend is AttnBackend.auto


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
@pytest.mark.parametrize("backend", [AttnBackend.fused, "fused"])
def test_explicit_backend_is_preserved(config_type, backend):
    config = make_config(config_type, backend)

    config.finalize()

    assert config.attention_backend == backend


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_explicit_fused_backend_accepts_matching_environment(config_type, monkeypatch):
    expected = ("0", "1", "0")
    for name, value in zip(BACKEND_ENV, expected):
        monkeypatch.setenv(name, value)
    config = make_config(config_type, AttnBackend.fused)

    config.finalize()
    configure_mcore_attention(config)

    assert tuple(os.environ[name] for name in BACKEND_ENV) == expected


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_unset_backend_does_not_overwrite_environment_restrictions(config_type, monkeypatch):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    config = make_config(config_type, None)

    config.finalize()

    with pytest.raises(AssertionError, match="NVTE_FLASH_ATTN"):
        configure_mcore_attention(config)
    assert os.environ["NVTE_FLASH_ATTN"] == "0"
