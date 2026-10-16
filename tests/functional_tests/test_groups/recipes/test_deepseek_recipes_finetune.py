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
exercises the *finetune* path: import the synthetic toy HF model into a Megatron
checkpoint, build the SFT recipe on that toy model, swap in a mock dataset, and run 5
finetune iterations from the imported checkpoint. Like the pretrain functional test,
this skips unless the synthetic DSv4 toy model is available and mcore ships the DSv4
prerequisites.
"""

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.deepseek import (
    deepseek_v4_flash_no_mtp_sft_config,
    deepseek_v4_flash_sft_config,
    deepseek_v4_flash_sft_openmath_thinking_packed_config,
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


def _replace_degenerate_hash_routing_tables(megatron_model) -> None:
    """Give every token distinct experts in hash-routing tables that repeat an expert.

    The synthetic toy model ships all-zero ``tid2eid`` tables, so each token selects
    expert 0 for every top-k slot. Megatron's dispatcher requires distinct experts per
    token, so use the round-robin table Megatron builds for an untrained hash router.
    """
    for model_chunk in megatron_model:
        for module in model_chunk.modules():
            tid2eid = getattr(module, "tid2eid", None)
            if not getattr(module, "is_hash_layer", False) or tid2eid is None:
                continue
            sorted_experts = tid2eid.sort(dim=1).values
            if not (sorted_experts[:, 1:] == sorted_experts[:, :-1]).any():
                continue
            token_ids = torch.arange(tid2eid.shape[0], device=tid2eid.device)
            expert_offsets = torch.arange(tid2eid.shape[1], device=tid2eid.device)
            tid2eid.copy_((token_ids[:, None] + expert_offsets) % module.num_experts)


# Finetuning loads the weights imported from the toy HF model, and on HybridModel the
# layer count, per-layer compression ratios, and MoE placement are derived from that toy
# config via the hybrid layer pattern. So, as in the pretrain test, do NOT override
# num_layers / num_moe_experts / moe_router_topk / moe_layer_freq / csa_compress_ratios:
# they would no longer match the imported checkpoint and would conflict with the
# pattern-derived values. Keep the validated SFT path (unfused mHC/rope/DSA); MTP off and
# recompute off for a fast smoke.
DEEPSEEK_V4_SFT_MODEL_OVERRIDES = {
    "mtp_num_layers": None,
    "pipeline_model_parallel_layout": None,
    "dsa_kernel_backend": "none",
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
    (deepseek_v4_flash_sft_openmath_thinking_packed_config, "deepseek_v4_flash_sft_openmath_thinking_packed", False),
]


class TestDeepSeekV4FinetuneRecipes:
    """Functional smoke tests for DeepSeek-V4-Flash SFT recipes."""

    @pytest.fixture(scope="class")
    def deepseek_v4_megatron_checkpoint(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        """Import the synthetic DSv4 toy HF model into a Megatron checkpoint to finetune from."""
        hf_path = _deepseek_v4_toy_model_path()

        from megatron.core import parallel_state
        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        from tests.functional_tests.utils import broadcast_path, initialize_distributed

        # import_ckpt uses a collective checkpoint save: initialize distributed first and
        # broadcast rank 0's path so every torchrun rank writes the same directory.
        initialize_distributed()
        temp_dir = tmp_path_factory.mktemp("deepseek_v4_megatron_ckpt")
        megatron_checkpoint_dir = broadcast_path(str(temp_dir / "megatron_checkpoint"))

        # Conversion and finetuning construct separate models with different attention backends.
        with patch.dict(os.environ):
            bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
            megatron_model = bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)
            _replace_degenerate_hash_routing_tables(megatron_model)
            bridge.save_megatron_model(
                megatron_model,
                megatron_checkpoint_dir,
                hf_tokenizer_path=hf_path,
                hf_tokenizer_kwargs={"trust_remote_code": True},
            )

        # Clean up global state so the finetune run initializes Megatron from scratch.
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        destroy_rerun_state_machine()

        return megatron_checkpoint_dir

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.skipif(not _has_dsv4_in_mcore(), reason="megatron-core does not yet ship DSv4 prerequisites.")
    @pytest.mark.parametrize("config_func,recipe_name,requires_blackwell", DEEPSEEK_V4_SFT_RECIPES)
    def test_deepseek_v4_sft_recipes(
        self, deepseek_v4_megatron_checkpoint, config_func, recipe_name, requires_blackwell, tmp_path
    ):
        """Build the SFT recipe on the toy model and finetune it for 5 iters on mock data."""
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
        # finetune() requires a pretrained checkpoint: start from the imported toy weights
        # rather than resuming from the recipe's default path, and skip saving.
        config.checkpoint.pretrained_checkpoint = deepseek_v4_megatron_checkpoint
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
