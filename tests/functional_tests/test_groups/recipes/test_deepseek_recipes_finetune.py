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

"""Functional smoke tests for DeepSeek-V4-Flash SFT (finetune) recipe configurations.

Mirrors ``test_deepseek_recipes_pretrain.py`` (toy HF model + DSv4-in-mcore guard) but
exercises the *finetune* path: build the SFT recipe on a 2-layer toy model, swap in a
mock dataset, and run 5 finetune iterations. Like the pretrain functional test, this
skips unless the synthetic DSv4 toy model is available and mcore ships the DSv4
prerequisites.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from megatron.bridge.recipes.deepseek import (
    deepseek_v4_flash_no_mtp_sft_config,
    deepseek_v4_flash_sft_config,
)
from megatron.bridge.recipes.deepseek.h100 import deepseek_v4 as deepseek_v4_h100_module


DEEPSEEK_V4_TEST_MODEL_ENV = "DEEPSEEK_V4_TOY_HF_PATH"
DEEPSEEK_V4_TEST_MODEL_PATH = Path("/home/TestData/megatron_bridge/models/deepseek_v4_toy")


def _has_dsv4_in_mcore() -> bool:
    try:
        return all(
            importlib.util.find_spec(mod) is not None
            for mod in (
                "megatron.core.transformer.hyper_connection",
                "megatron.core.transformer.experimental_attention_variant.csa",
                "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
            )
        )
    except ModuleNotFoundError:
        return False


def _deepseek_v4_toy_model_path() -> str:
    model_path = Path(os.environ.get(DEEPSEEK_V4_TEST_MODEL_ENV, DEEPSEEK_V4_TEST_MODEL_PATH))
    if not model_path.exists():
        pytest.skip(
            f"DeepSeek-V4 toy HF model not found at {model_path}. "
            f"Set {DEEPSEEK_V4_TEST_MODEL_ENV} or upload the synthetic model to CI test data."
        )
    return str(model_path)


# Shrink the Flash architecture to a 2-layer toy. Keep the validated SFT path
# (unfused mHC/rope); MTP off and recompute off for a fast smoke.
DEEPSEEK_V4_SFT_MODEL_OVERRIDES = {
    "num_layers": 2,
    "mtp_num_layers": None,
    "pipeline_model_parallel_layout": None,
    "num_moe_experts": 8,
    "moe_router_topk": 1,
    "moe_layer_freq": [0, 1],
    "csa_compress_ratios": [0, 0],
    "csa_backend": "unfused",
    "use_fused_mhc": False,
    "apply_rope_fusion": False,
    "dsa_indexer_loss_coeff": 0.0,
    "dsa_indexer_use_sparse_loss": False,
    "recompute_granularity": None,
    "recompute_modules": None,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
}

# (config_func, recipe_name, requires_blackwell)
DEEPSEEK_V4_SFT_RECIPES = [
    (deepseek_v4_flash_sft_config, "deepseek_v4_flash_sft", False),
    (deepseek_v4_flash_no_mtp_sft_config, "deepseek_v4_flash_no_mtp_sft", False),
]


class TestDeepSeekV4FinetuneRecipes:
    """Functional smoke tests for DeepSeek-V4-Flash SFT recipes."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.skipif(not _has_dsv4_in_mcore(), reason="megatron-core does not yet ship DSv4 prerequisites.")
    @pytest.mark.parametrize("config_func,recipe_name,requires_blackwell", DEEPSEEK_V4_SFT_RECIPES)
    def test_deepseek_v4_sft_recipes(self, config_func, recipe_name, requires_blackwell, tmp_path):
        """Build the SFT recipe on a toy model and run 5 finetune iters on mock data."""
        if requires_blackwell and torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("DeepSeek-V4 MXFP8 recipe requires Blackwell GPUs.")

        hf_path = _deepseek_v4_toy_model_path()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(deepseek_v4_h100_module, "DEEPSEEK_V4_FLASH_HF_PATH", hf_path)
            config = config_func()

        # Swap the shipped SQuAD config for a mock dataset (forward path is identical;
        # this keeps CI fast and offline).
        from megatron.bridge.training.config import MockGPTDatasetConfig

        seq_length = 512
        config.dataset = MockGPTDatasetConfig(
            random_seed=5678,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=0,
        )

        for attribute_name, attribute_value in DEEPSEEK_V4_SFT_MODEL_OVERRIDES.items():
            setattr(config.model, attribute_name, attribute_value)

        config.model.seq_length = seq_length
        config.train.train_iters = 5
        config.validation.eval_interval = 100  # skip mid-training eval
        config.validation.eval_iters = 1
        config.train.micro_batch_size = 1
        config.train.global_batch_size = 2
        config.scheduler.lr_warmup_iters = 1
        config.logger.dir = str(tmp_path)
        config.logger.name = recipe_name
        # Smoke test: train from the toy model's init (no pretrained checkpoint, no save).
        config.checkpoint.pretrained_checkpoint = None
        config.checkpoint.load = None
        config.checkpoint.save = None

        # Minimal dataset splits sized to the iteration counts above.
        train_samples = config.train.train_iters * config.train.global_batch_size
        eval_samples = config.validation.eval_iters * config.train.global_batch_size
        test_samples = 8
        total = train_samples + eval_samples + test_samples
        config.dataset.split = [train_samples / total, eval_samples / total, test_samples / total]

        from megatron.bridge.training.finetune import finetune
        from megatron.bridge.training.gpt_step import forward_step
        from tests.functional_tests.utils import clear_directories, initialize_distributed

        initialize_distributed()
        try:
            finetune(config, forward_step)
        finally:
            clear_directories(tmp_path)
