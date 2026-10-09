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

"""Cover the public recipe contract without downloading model weights."""

import pytest
from transformers.models.diffusion_gemma import DiffusionGemmaConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.diffusion_gemma.data import DiffusionGemmaDatasetConfig, MockDiffusionGemmaDatasetConfig
from megatron.bridge.recipes.diffusion_gemma import diffusion_gemma_26b_sft_config


pytestmark = pytest.mark.unit


def _patch_provider(monkeypatch):
    provider = AutoBridge.from_hf_config(
        DiffusionGemmaConfig(
            architectures=["DiffusionGemmaForBlockDiffusion"],
            text_config={
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "num_global_key_value_heads": 2,
            },
        )
    ).to_megatron_provider(load_weights=False)
    calls = []

    class _Bridge:
        def to_megatron_provider(self, load_weights):
            calls.append(load_weights)
            return provider

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", lambda path: _Bridge())
    return calls


def test_recipe_has_trainable_ep8_distributed_optimizer_and_explicit_splits(monkeypatch):
    calls = _patch_provider(monkeypatch)
    cfg = diffusion_gemma_26b_sft_config(
        train_path="train.jsonl", validation_path="validation.jsonl", test_path="test.jsonl"
    )
    assert calls == [True]
    assert cfg.model.tensor_model_parallel_size == cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert not cfg.model.freeze_language_model
    assert not cfg.model.freeze_vision_model
    assert not cfg.model.freeze_vision_projection
    assert cfg.optimizer.use_distributed_optimizer and cfg.ddp.use_distributed_optimizer
    assert cfg.model.diffusion_encoder_loss_weight == cfg.model.diffusion_decoder_loss_weight == 1.0
    assert cfg.model.diffusion_self_conditioning_prob == 0.5
    assert cfg.model.moe_aux_loss_coeff == 0.0
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.recompute_num_layers == 1
    assert isinstance(cfg.dataset, DiffusionGemmaDatasetConfig)
    assert cfg.dataset.train_path == "train.jsonl"
    assert cfg.dataset.validation_path == "validation.jsonl"
    assert cfg.dataset.test_path == "test.jsonl"
    assert cfg.checkpoint.pretrained_checkpoint is None  # HF hook, not an invalid HF path masquerading as MCore
    assert cfg.checkpoint.ckpt_format == "torch_dist"
    assert cfg.mixed_precision == "bf16_mixed"


def test_recipe_defaults_to_synthetic_data(monkeypatch):
    calls = _patch_provider(monkeypatch)
    cfg = diffusion_gemma_26b_sft_config()
    assert calls == [True]
    assert isinstance(cfg.dataset, MockDiffusionGemmaDatasetConfig)


@pytest.mark.parametrize("split", ["validation_path", "test_path"])
def test_recipe_rejects_eval_split_without_train_split(split):
    with pytest.raises(ValueError, match="require train_path"):
        diffusion_gemma_26b_sft_config(**{split: f"{split}.jsonl"})
