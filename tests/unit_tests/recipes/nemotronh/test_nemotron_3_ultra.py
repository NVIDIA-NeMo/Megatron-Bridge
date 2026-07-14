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

import pytest

from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import (
    NEMOTRON_3_ULTRA_TOKENIZER_NAME,
    nemotron_3_ultra_peft_openmathinstruct2_packed_config,
    nemotron_3_ultra_pretrain_config,
    nemotron_3_ultra_sft_openmathinstruct2_packed_config,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


class _FakeUltraProvider:
    """Fake model provider for testing recipe field overrides without HF Hub I/O."""

    def __init__(self) -> None:
        self.vocab_size = 256

    def finalize(self) -> None:
        return None


class _FakeAutoBridge:
    """Fake AutoBridge that returns an Ultra provider without loading a model."""

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _FakeUltraProvider()


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch AutoBridge in the recipe module to avoid Hugging Face access."""
    mod = importlib.import_module("megatron.bridge.recipes.nemotronh.nemotron_3_ultra")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeAutoBridge)


@pytest.mark.unit
def test_pretrain_uses_initial_parallelism_values() -> None:
    cfg = nemotron_3_ultra_pretrain_config()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 3
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.sequence_parallel is True
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.mtp_loss_scaling_factor == 0.3
    assert cfg.model.mtp_use_repeated_layer is True

    assert cfg.optimizer.lr == 2.5e-4
    assert cfg.optimizer.min_lr == 2.5e-4
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.scheduler.lr_decay_style == "constant"
    assert cfg.scheduler.lr_warmup_iters == 0
    assert cfg.checkpoint.async_save is True
    assert cfg.checkpoint.async_strategy == "mcore"
    assert cfg.checkpoint.save_interval == 200

    assert cfg.train.global_batch_size == 3072
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.seq_length == 8192
    assert cfg.dataset.blend is None


@pytest.mark.unit
def test_openmath_sft_uses_initial_parallelism_values() -> None:
    cfg = nemotron_3_ultra_sft_openmathinstruct2_packed_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.sequence_parallel is True
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.recompute_modules == ["moe", "layernorm", "core_attn", "moe_act"]

    assert cfg.train.train_iters == 1000
    assert cfg.train.global_batch_size == 128
    assert cfg.checkpoint.async_save is True
    assert cfg.checkpoint.async_strategy == "mcore"
    assert cfg.dataset.hf_dataset.dataset_name == "openmathinstruct2"
    assert cfg.dataset.offline_packing_specs is not None
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 4096
    assert cfg.dataset.offline_packing_specs.tokenizer_model_name == NEMOTRON_3_ULTRA_TOKENIZER_NAME


@pytest.mark.unit
def test_openmath_peft_uses_validated_parallelism_values() -> None:
    cfg = nemotron_3_ultra_peft_openmathinstruct2_packed_config()

    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.sequence_parallel is True
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_method is None
    assert cfg.model.recompute_num_layers is None
    assert cfg.model.recompute_modules == ["moe", "layernorm", "core_attn", "moe_act", "mlp", "shared_experts"]

    assert cfg.optimizer.lr == 1e-4
    assert cfg.optimizer.min_lr == 1e-5
    assert cfg.train.train_iters == 1000
    assert cfg.train.global_batch_size == 128
    assert cfg.checkpoint.async_save is True
    assert cfg.checkpoint.async_strategy == "nvrx"


@pytest.mark.unit
def test_openmath_peft_none_disables_adapter() -> None:
    cfg = nemotron_3_ultra_peft_openmathinstruct2_packed_config(peft_scheme="none")
    assert cfg.peft is None


@pytest.mark.unit
def test_openmath_peft_recompute_modules_are_not_shared() -> None:
    cfg = nemotron_3_ultra_peft_openmathinstruct2_packed_config()
    cfg.model.recompute_modules.append("sentinel")

    fresh_cfg = nemotron_3_ultra_peft_openmathinstruct2_packed_config()
    assert fresh_cfg.model.recompute_modules == ["moe", "layernorm", "core_attn", "moe_act", "mlp", "shared_experts"]


@pytest.mark.unit
def test_openmath_sft_recompute_modules_are_not_shared() -> None:
    cfg = nemotron_3_ultra_sft_openmathinstruct2_packed_config()
    cfg.model.recompute_modules.append("sentinel")

    fresh_cfg = nemotron_3_ultra_sft_openmathinstruct2_packed_config()
    assert fresh_cfg.model.recompute_modules == ["moe", "layernorm", "core_attn", "moe_act"]
