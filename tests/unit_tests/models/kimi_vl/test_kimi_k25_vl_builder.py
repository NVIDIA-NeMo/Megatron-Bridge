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

"""Unit tests for the builder-backed Kimi K2.5 VL model config and builder."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.transformer.module import mark_keep_in_fp32
from transformers import PretrainedConfig

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.models.kimi_vl.kimi_k25_vl_builder import KimiK25VLModelBuilder
from megatron.bridge.models.kimi_vl.kimi_k25_vl_config import KimiK25VLModelConfig
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.model_load_save import load_model_config


pytestmark = pytest.mark.unit

_BUILDER_MODULE = "megatron.bridge.models.kimi_vl.kimi_k25_vl_builder"

_VISION_LANGUAGE_FIELDS = (
    "scatter_embedding_sequence_parallel",
    "vision_config",
    "hf_model_path",
    "trust_remote_code",
    "bos_token_id",
    "eos_token_id",
    "image_token_id",
    "media_placeholder_token_id",
    "pad_token_id",
    "ignore_index",
    "freeze_language_model",
    "freeze_vision_model",
    "freeze_vision_projection",
)


def _vision_config() -> PretrainedConfig:
    """Stand-in for the remote ``KimiK25VisionConfig``, whose vision settings are undeclared attributes."""
    return PretrainedConfig(
        patch_size=7,
        vt_num_hidden_layers=2,
        vt_hidden_size=64,
        vt_intermediate_size=128,
        vt_num_attention_heads=2,
        merge_kernel_size=[2, 2],
        mm_hidden_size=64,
        text_hidden_size=128,
    )


def _model_config() -> KimiK25VLModelConfig:
    transformer = MLATransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        params_dtype=torch.bfloat16,
    )
    return KimiK25VLModelConfig(
        transformer=transformer,
        vocab_size=1024,
        vision_config=_vision_config(),
        hf_model_path="/ckpt/kimi_k25",
        trust_remote_code=True,
        rotary_base=50000.0,
    )


class TestKimiK25VLModelConfig:
    def test_config_types(self):
        model_config = _model_config()

        assert isinstance(model_config, BridgeGPTMLAModelConfig)
        assert type(model_config.transformer) is MLATransformerConfig
        assert model_config.get_builder_cls() is KimiK25VLModelBuilder
        # MLA attention reads the rotary base from the nested transformer config.
        assert model_config.transformer.rotary_base == 50000.0

    def test_vision_language_defaults_match_the_provider(self):
        config_fields = {field.name: field for field in dataclasses.fields(KimiK25VLModelConfig)}
        provider_fields = {field.name: field for field in dataclasses.fields(KimiK25VLModelProvider)}

        for name in _VISION_LANGUAGE_FIELDS:
            assert config_fields[name].default == provider_fields[name].default, name

    def test_round_trip_keeps_the_vision_config(self):
        model_config = _model_config()
        model_config.freeze_vision_projection = True
        vision_config = model_config.vision_config

        restored = KimiK25VLModelConfig.from_dict(model_config.as_dict())

        assert type(restored) is KimiK25VLModelConfig
        assert restored.vision_config.to_dict() == vision_config.to_dict()
        assert restored.vision_config.patch_size == 7
        for owner, restored_owner in ((model_config, restored), (model_config.transformer, restored.transformer)):
            for field in dataclasses.fields(owner):
                if field.name not in {"transformer", "vision_config"}:
                    assert getattr(restored_owner, field.name) == getattr(owner, field.name), field.name
        # Serialization leaves the live config untouched.
        assert model_config.vision_config is vision_config

    def test_checkpoint_run_config_reload(self, tmp_path):
        """A config saved to run_config.yaml reloads as the same config."""
        model_config = _model_config()
        model_config.freeze_vision_model = True
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = model_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is KimiK25VLModelConfig
        assert type(loaded.transformer) is MLATransformerConfig
        assert loaded.get_builder_cls() is KimiK25VLModelBuilder
        assert (loaded.hf_model_path, loaded.trust_remote_code) == ("/ckpt/kimi_k25", True)
        assert loaded.freeze_vision_model is True
        assert loaded.transformer.rotary_base == 50000.0
        assert loaded.vision_config.to_dict() == model_config.vision_config.to_dict()

    def test_flat_overrides_route_to_their_owner(self):
        model_config = _model_config()

        model_config.freeze_vision_model = True
        model_config.tensor_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(model_config.transformer)
        assert model_config.freeze_vision_model is True
        assert model_config.transformer.tensor_model_parallel_size == 2
        with pytest.raises(AttributeError):
            model_config.vision_tower_config = None


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _pg_collection(pp_rank: int = 0, pp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pp=_FakeGroup(pp_rank, pp_size))


def _built_model(pre_process: bool) -> Mock:
    """A model stage with FP32 Hugging Face modules, as KimiK25VLModel builds them."""
    model = Mock(spec=["freeze", "vision_tower", "mm_projector"] if pre_process else ["freeze"])
    if pre_process:
        model.vision_tower = torch.nn.Linear(4, 4)
        model.mm_projector = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        mark_keep_in_fp32(model.mm_projector[0].weight)
    return model


@pytest.fixture
def model_class(monkeypatch):
    """Replace KimiK25VLModel in the builder, recording how the builder constructs it."""
    model_class = Mock(side_effect=lambda *args, **kwargs: _built_model(kwargs["pre_process"]))
    monkeypatch.setattr(f"{_BUILDER_MODULE}.KimiK25VLModel", model_class)
    return model_class


class TestKimiK25VLModelBuilder:
    def test_build_model_constructs_the_vision_language_model(self, model_class, monkeypatch):
        model_config = _model_config()
        pg_collection = _pg_collection()
        language_model = Mock()
        gpt_build_model = Mock(return_value=language_model)
        monkeypatch.setattr(GPTModelBuilder, "build_model", gpt_build_model)

        model = KimiK25VLModelBuilder(model_config).build_model(pg_collection, pre_process=True, post_process=True)

        (config,), kwargs = model_class.call_args
        assert config is model_config.transformer
        assert kwargs["model_config"] is model_config
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (True, True, None)
        assert kwargs["pg_collection"] is pg_collection
        model.freeze.assert_not_called()

        # The language model comes from the GPT builder, with the same process groups.
        assert kwargs["build_language_model"](pre_process=True, post_process=False, vp_stage=None) is language_model
        gpt_build_model.assert_called_once_with(pg_collection, pre_process=True, post_process=False, vp_stage=None)

    def test_build_model_casts_the_hugging_face_modules(self, model_class):
        """Conversion builds without Float16Module, so the builder casts what the provider path's wrapper does."""
        model = KimiK25VLModelBuilder(_model_config()).build_model(_pg_collection(), pre_process=True)

        assert model.vision_tower.weight.dtype is torch.bfloat16
        assert model.mm_projector[1].weight.dtype is torch.bfloat16
        assert model.mm_projector[0].bias.dtype is torch.bfloat16
        assert model.mm_projector[0].weight.dtype is torch.float32

    @pytest.mark.parametrize(
        ("pp_rank", "pp_size", "pre_process", "post_process"),
        [(0, 1, True, True), (0, 2, True, False), (1, 2, False, True)],
    )
    def test_build_model_derives_stage_flags(
        self, model_class, monkeypatch, pp_rank, pp_size, pre_process, post_process
    ):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        KimiK25VLModelBuilder(_model_config()).build_model(_pg_collection(pp_rank, pp_size))

        kwargs = model_class.call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == (pre_process, post_process)

    def test_build_model_freezes_requested_modules(self, model_class):
        model_config = _model_config()
        model_config.freeze_language_model = True
        model_config.freeze_vision_projection = True

        model = KimiK25VLModelBuilder(model_config).build_model(_pg_collection(), pre_process=False)

        model.freeze.assert_called_once_with(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=True
        )

    def test_builder_rejects_other_configs(self, model_class):
        model_config = _model_config()
        text_config = BridgeGPTMLAModelConfig(transformer=model_config.transformer, vocab_size=1024)

        with pytest.raises(TypeError, match="KimiK25VLModelConfig"):
            KimiK25VLModelBuilder(text_config).build_model(_pg_collection())
        model_class.assert_not_called()
