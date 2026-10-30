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

"""
Unit tests for the Step-3.7-Flash recipe configuration builders.

Monkeypatch ``AutoBridge`` to a builder-only fake that returns a real, strict
``Step37ModelConfig`` built from a small Step-3.7 config, so the recipe wiring
runs without HF Hub I/O and every field the recipe sets must be declared by the
builder config. Also monkeypatches ``apply_flex_dispatcher_backend`` so the
recipe does not poke CUDA device properties during tests.
"""

import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.stepfun.configuration_step37 import Step37Config
from megatron.bridge.models.stepfun.step37_config import Step37ModelConfig
from megatron.bridge.recipes.stepfun import step37_sft_flickr8k_config, step37_sft_flickr8k_smoke_config
from megatron.bridge.recipes.stepfun.h100 import step37 as _step37_h100
from megatron.bridge.recipes.stepfun.h100.step37 import (
    step37_sft_4gpu_h100_bf16_flickr8k_smoke_config,
    step37_sft_64gpu_h100_bf16_flickr8k_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


def _step37_hf_config() -> Step37Config:
    """A small Step-3.7 architecture: dense-first MoE, hybrid attention, one MTP layer, 2-layer vision tower."""
    return Step37Config(
        text_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "moe_intermediate_size": 64,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "share_expert_dims": 64,
            "head_dim": 32,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "num_nextn_predict_layers": 1,
            "moe_layers_enum": "2,3",
            "rope_theta": [5e6, 1e4, 1e4, 1e4, 5e6],
            "sliding_window": 64,
            "use_moe_router_bias": True,
            "need_fp32_gate": True,
            "torch_dtype": "bfloat16",
        },
        vision_config={"width": 64, "layers": 2, "heads": 2, "image_size": 56, "patch_size": 14},
        architectures=["Step3p7ForConditionalGeneration"],
    )


class _FakeBridge:
    """Return a real strict model config while rejecting the provider API."""

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()

    def get_model_config(self) -> Step37ModelConfig:
        return AutoBridge.from_hf_config(_step37_hf_config()).get_model_config()

    def to_megatron_provider(self, load_weights: bool = False):
        raise AssertionError("Step-3.7 recipes must use get_model_config(), not the legacy provider API")


def _patch_recipe_env(monkeypatch):
    patch_recipe_module_global(monkeypatch, _step37_h100, "AutoBridge", _FakeBridge)
    patch_recipe_module_global(monkeypatch, _step37_h100, "apply_flex_dispatcher_backend", lambda *a, **kw: None)


def test_step37_sft_recipe_builds_model_config(monkeypatch):
    _patch_recipe_env(monkeypatch)

    cfg = step37_sft_64gpu_h100_bf16_flickr8k_config()

    assert isinstance(cfg, ConfigContainer)
    m = cfg.model
    assert isinstance(m, Step37ModelConfig)
    assert m.vision_config is not None
    assert (m.tensor_model_parallel_size, m.pipeline_model_parallel_size) == (1, 8)
    assert (m.expert_model_parallel_size, m.context_parallel_size) == (8, 1)
    assert m.sequence_parallel is True
    assert m.seq_length == 2048
    assert m.pipeline_dtype is torch.bfloat16
    assert (m.freeze_language_model, m.freeze_vision_model, m.freeze_vision_projection) == (False, False, False)
    assert m.moe_token_dispatcher_type == "alltoall"
    assert m.moe_hybridep_num_sms == 16
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.max_packing_seqlen == 2048


def test_step37_smoke_recipe_builds_model_config(monkeypatch):
    _patch_recipe_env(monkeypatch)

    cfg = step37_sft_4gpu_h100_bf16_flickr8k_smoke_config()

    m = cfg.model
    assert isinstance(m, Step37ModelConfig)
    assert m.mtp_num_layers == 0
    assert (m.freeze_language_model, m.freeze_vision_model, m.freeze_vision_projection) == (True, True, False)
    assert m.pipeline_model_parallel_size == 4
    assert (m.num_layers_in_first_pipeline_stage, m.num_layers_in_last_pipeline_stage) == (12, 9)
    assert m.expert_model_parallel_size == 1
    assert m.variable_seq_lengths is True
    assert m.rotary_percent == 0.5
    assert m.window_size == [512, 0]
    # The per-layer lists cover the 45 Step-3.7 decoder layers and live on the nested transformer config.
    assert len(m.transformer.rotary_base_per_layer) == 45
    assert len(m.transformer.window_attn_skip_freq) == 45
    assert len(m.transformer.moe_layer_freq) == 45
    assert m.recompute_granularity == "full"
    assert cfg.dataset.fixed_pack_idx == 0


def test_legacy_recipe_names_alias_h100_recipes():
    assert step37_sft_flickr8k_config is step37_sft_64gpu_h100_bf16_flickr8k_config
    assert step37_sft_flickr8k_smoke_config is step37_sft_4gpu_h100_bf16_flickr8k_smoke_config
