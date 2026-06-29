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

import importlib
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.data.energon.nemotron_omni_task_encoder import NemotronOmniTaskEncoder
from megatron.bridge.data.hf_datasets.provider import HFConversationDatasetProvider
from megatron.bridge.data.vlm_datasets.collate import nemotron_omni_collate_fn
from megatron.bridge.training.config import ConfigContainer


_recipe_module = importlib.import_module("megatron.bridge.recipes.nemotron_omni.nemotron_omni")

_PUBLIC_HF_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
_TEST_HF_ID = "unit-test/nemotron-omni"

_RECIPE_FUNCS = [
    _recipe_module.nemotron_omni_cord_v2_sft_config,
    _recipe_module.nemotron_omni_cord_v2_peft_config,
    _recipe_module.nemotron_omni_valor32k_sft_config,
    _recipe_module.nemotron_omni_valor32k_peft_config,
]


class _FakeModelCfg:
    def finalize(self):
        return None


class _FakeAutoBridge:
    hf_path = None
    kwargs = None
    load_weights = None

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        _FakeAutoBridge.hf_path = hf_path
        _FakeAutoBridge.kwargs = kwargs
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        _FakeAutoBridge.load_weights = load_weights
        return _FakeModelCfg()


@pytest.fixture
def fake_processor(monkeypatch: pytest.MonkeyPatch):
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=11),
        image_processor=SimpleNamespace(max_num_tiles=12),
    )

    import transformers

    monkeypatch.setattr(_recipe_module, "AutoBridge", _FakeAutoBridge)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *_, **__: processor)
    return processor


def _build_config(recipe_func: Callable, fake_processor) -> ConfigContainer:
    return recipe_func(hf_path=_TEST_HF_ID)


def _assert_common_config(cfg: ConfigContainer):
    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.dataset is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.checkpoint is not None
    assert cfg.mixed_precision == "bf16_mixed"

    assert _FakeAutoBridge.hf_path == _TEST_HF_ID
    assert _FakeAutoBridge.kwargs == {"trust_remote_code": True}
    assert _FakeAutoBridge.load_weights is False

    assert cfg.model.seq_length == 4096
    assert cfg.model.dynamic_resolution is True
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_sound_encoder is True
    assert cfg.model.transformer_impl == "transformer_engine"
    assert cfg.model.attention_backend == "flash"

    assert cfg.train.train_iters == 2000
    assert cfg.train.global_batch_size == 64
    assert cfg.train.micro_batch_size == 1
    assert cfg.validation.eval_interval == 200
    assert cfg.validation.eval_iters == 0
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.overlap_grad_reduce is False


def test_default_hf_path_is_public_model_id():
    assert _recipe_module._DEFAULT_HF_PATH == _PUBLIC_HF_ID


@pytest.mark.parametrize("recipe_func", _RECIPE_FUNCS)
def test_each_nemotron_omni_recipe_builds_valid_config(recipe_func: Callable, fake_processor):
    cfg = _build_config(recipe_func, fake_processor)

    _assert_common_config(cfg)
    assert cfg.dataset.seq_length == 4096


def test_cord_v2_sft_recipe_uses_hf_dataset_provider(fake_processor):
    cfg = _build_config(_recipe_module.nemotron_omni_cord_v2_sft_config, fake_processor)

    _assert_common_config(cfg)
    assert isinstance(cfg.dataset, HFConversationDatasetProvider)
    assert cfg.dataset.hf_processor_path == _TEST_HF_ID
    assert cfg.dataset.maker_name == "cord_v2"
    assert cfg.dataset.collate_impl is nemotron_omni_collate_fn
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.model.temporal_patch_dim == 1
    assert cfg.model.freeze_sound_projection is False
    assert cfg.peft is None


def test_cord_v2_peft_recipe_configures_lora_and_freezing(fake_processor):
    cfg = _build_config(_recipe_module.nemotron_omni_cord_v2_peft_config, fake_processor)

    _assert_common_config(cfg)
    assert isinstance(cfg.dataset, HFConversationDatasetProvider)
    assert cfg.peft is not None
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "in_proj", "out_proj"]
    assert cfg.peft.dim == 16
    assert cfg.peft.alpha == 32
    assert cfg.checkpoint.load is None
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.freeze_sound_projection is True


def test_valor32k_sft_recipe_uses_temporal_omni_task_encoder(fake_processor):
    cfg = _build_config(_recipe_module.nemotron_omni_valor32k_sft_config, fake_processor)

    _assert_common_config(cfg)
    assert isinstance(cfg.dataset, EnergonProvider)
    assert cfg.dataset.path == ""
    assert cfg.dataset.enable_in_batch_packing is False
    assert isinstance(cfg.dataset.task_encoder, NemotronOmniTaskEncoder)
    assert cfg.dataset.task_encoder.processor is fake_processor
    assert cfg.dataset.task_encoder.max_audio_duration == 10.0
    assert cfg.dataset.task_encoder.num_mel_bins == 128
    assert cfg.dataset.task_encoder.use_temporal_video_embedder is True
    assert cfg.dataset.task_encoder.patch_dim == 16
    assert cfg.model.temporal_patch_dim == 2
    assert cfg.model.separate_video_embedder is True
    assert cfg.model.temporal_ckpt_compat is True
    assert cfg.model.freeze_sound_projection is False
    assert cfg.peft is None


def test_valor32k_peft_recipe_configures_lora_and_freezing(fake_processor):
    cfg = _build_config(_recipe_module.nemotron_omni_valor32k_peft_config, fake_processor)

    _assert_common_config(cfg)
    assert isinstance(cfg.dataset, EnergonProvider)
    assert isinstance(cfg.dataset.task_encoder, NemotronOmniTaskEncoder)
    assert cfg.dataset.task_encoder.use_temporal_video_embedder is True
    assert cfg.peft is not None
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "in_proj", "out_proj"]
    assert cfg.peft.dim == 16
    assert cfg.peft.alpha == 32
    assert cfg.checkpoint.load is None
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.freeze_sound_projection is True
