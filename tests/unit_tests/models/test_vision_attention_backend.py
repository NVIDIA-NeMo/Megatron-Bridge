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

"""Vision config propagation must preserve MCore's process-wide backend contract."""

import os
from types import SimpleNamespace

import pytest
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.utils import set_attention_backend

from megatron.bridge.models.exaone.exaone45.exaone45_config import Exaone45ModelConfig
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_config import (
    get_vision_model_config as exaone_vision_config,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    get_vision_model_config as qwen_vision_config,
)
from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit

_BACKEND_FLAGS = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
_VERSION_FLAGS = tuple(f"NVTE_FLASH_ATTN_V{version}" for version in (2, 3, 4))


@pytest.fixture(autouse=True)
def isolated_attention_environment(monkeypatch):
    for name in (*_BACKEND_FLAGS, *_VERSION_FLAGS):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(params=[qwen_vision_config, exaone_vision_config], ids=["qwen", "exaone"])
def vision_factory(request):
    return request.param


def _vision_hf_config():
    return SimpleNamespace(
        depth=2,
        hidden_size=64,
        num_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        num_position_embeddings=256,
        out_hidden_size=64,
        deepstack_visual_indexes=[],
    )


def _language_config(backend, version=None):
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        attention_backend=backend,
        flash_attention_version=version,
    )


@pytest.mark.parametrize(
    "backend,expected_flags",
    [
        (AttnBackend.auto, ("1", "1", "1")),
        (AttnBackend.flash, ("1", "0", "0")),
        (AttnBackend.fused, ("0", "1", "0")),
        (AttnBackend.unfused, ("0", "0", "1")),
        ("fused", ("0", "1", "0")),
    ],
)
@pytest.mark.parametrize("version", [None, 2, 3, 4])
@pytest.mark.parametrize("vision_first", [True, False])
def test_vision_language_backend_construction(vision_factory, backend, expected_flags, version, vision_first):
    language = _language_config(backend, version)
    vision = vision_factory(_vision_hf_config(), language)

    # LanguageModule calls this same helper before constructing its layers.
    # VisionModule itself can be constructed without distributed groups or GPUs.
    for _ in range(2):
        if vision_first:
            VisionModule(vision)
            set_attention_backend(language)
        else:
            set_attention_backend(language)
            VisionModule(vision)

    assert tuple(os.environ[name] for name in _BACKEND_FLAGS) == expected_flags
    expected_versions = tuple(str(int(value == version)) for value in (2, 3, 4))
    assert tuple(os.environ.get(name) for name in _VERSION_FLAGS) == (
        expected_versions if version is not None else (None, None, None)
    )


@pytest.mark.parametrize("compatible", [True, False])
def test_vision_backend_respects_external_environment(vision_factory, monkeypatch, compatible):
    language = _language_config(AttnBackend.flash, 3)
    vision = vision_factory(_vision_hf_config(), language)
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0" if compatible else "1")

    if compatible:
        VisionModule(vision)
        set_attention_backend(language)
    else:
        with pytest.raises(AssertionError, match="NVTE_FUSED_ATTN"):
            VisionModule(vision)
        assert os.environ["NVTE_FUSED_ATTN"] == "1"


def test_vision_backend_rejects_conflicting_flash_version(vision_factory, monkeypatch):
    vision = vision_factory(_vision_hf_config(), _language_config(AttnBackend.flash, 3))
    monkeypatch.setenv("NVTE_FLASH_ATTN_V2", "1")
    with pytest.raises(AssertionError, match="NVTE_FLASH_ATTN_V2"):
        VisionModule(vision)


def test_vision_backend_rejects_unknown_backend(vision_factory):
    language = _language_config(AttnBackend.auto)
    language.attention_backend = "unknown_backend"
    vision = vision_factory(_vision_hf_config(), language)
    with pytest.raises(ValueError, match="Unknown attention backend"):
        VisionModule(vision)


def test_language_config_rejects_invalid_flash_version():
    with pytest.raises(AssertionError, match="flash_attention_version must be one of"):
        _language_config(AttnBackend.flash, 5).finalize()


@pytest.mark.parametrize(
    "model_config_class,factory",
    [(Qwen35VLModelConfig, qwen_vision_config), (Exaone45ModelConfig, exaone_vision_config)],
)
def test_builder_config_forwards_nested_attention_settings(model_config_class, factory):
    model_config = model_config_class(transformer=_language_config(AttnBackend.flash, 3))
    vision = factory(_vision_hf_config(), model_config)

    assert vision.attention_backend == model_config.transformer.attention_backend
    assert vision.flash_attention_version == model_config.transformer.flash_attention_version
    VisionModule(vision)
    set_attention_backend(model_config.transformer)
