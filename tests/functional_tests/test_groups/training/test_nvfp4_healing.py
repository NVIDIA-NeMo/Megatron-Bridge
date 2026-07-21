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

"""Functional tests for NVFP4 training with FP8 healing (requires Blackwell GPUs)."""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import megatron.core.fp4_utils as fp4_utils
import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.nvfp4_healing import (
    NVFP4HealingCallback,
    NVFP4HealingConfig,
    _unwrap_model_chunk,
)
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import broadcast_path, clear_directories, initialize_distributed


SEQ_LENGTH = 512
GLOBAL_BATCH_SIZE = 8
PRETRAIN_ITERS = 4
FINETUNE_ITERS = 4
HEALING_ITER = 2


@dataclass
class TinyGPTProvider(GPTModelProvider):
    """Tiny GPT with all GEMM dimensions NVFP4-compatible (divisible by 32)."""

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    num_query_groups: int = 8
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    make_vocab_size_divisible_by: int = 128
    vocab_size: int | None = None
    seq_length: int = SEQ_LENGTH
    bf16: bool = True
    pipeline_dtype: torch.dtype = torch.bfloat16


class RecipeTrackingCallback(Callback):
    """Records the active FP4-path recipe class name at each step and the step losses."""

    def __init__(self):
        self.recipe_names: list[str] = []
        self.losses: list[float] = []

    def on_train_step_start(self, context: CallbackContext) -> None:
        model_config = _unwrap_model_chunk(context.model[0]).config
        recipe = fp4_utils.get_fp4_recipe(model_config)
        self.recipe_names.append(type(recipe).__name__)

    def on_train_step_end(self, context: CallbackContext) -> None:
        if context.loss_dict:
            self.losses.extend(float(value) for value in context.loss_dict.values())


def _base_config(model_cfg, train_iters, checkpoint_cfg, mixed_precision=None, peft=None):
    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=GLOBAL_BATCH_SIZE,
            micro_batch_size=1,
        ),
        validation=ValidationConfig(eval_interval=train_iters + 1, eval_iters=0),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            lr=3e-4,
            weight_decay=0.01,
            min_lr=1e-6,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=1,
            lr_warmup_init=0.0,
            lr_decay_iters=train_iters,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=MockGPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=SEQ_LENGTH,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(log_interval=1),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=10000),
        checkpoint=checkpoint_cfg,
        rng=RNGConfig(seed=1234),
        mixed_precision=mixed_precision,
        peft=peft,
    )


class TestNVFP4HealingEndToEnd:
    """Pretrain a tiny GPT, then LoRA-finetune in NVFP4 with FP8 healing."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("healing_recipe", ["delayed", "mxfp8"])
    def test_nvfp4_finetune_with_fp8_healing(self, healing_recipe, tmp_path):
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        pretrain_dir = os.path.join(shared_base_dir, f"pretrain_{healing_recipe}")
        lora_dir = os.path.join(shared_base_dir, f"lora_{healing_recipe}")
        torch.distributed.barrier()

        try:
            # Phase 1: BF16 pretrain to produce a base checkpoint.
            pretrain_cfg = _base_config(
                TinyGPTProvider(tensor_model_parallel_size=1, pipeline_model_parallel_size=1),
                PRETRAIN_ITERS,
                CheckpointConfig(save=pretrain_dir, save_interval=PRETRAIN_ITERS),
            )
            pretrain(pretrain_cfg, forward_step)
            torch.distributed.barrier()

            # Phase 2: LoRA finetune in NVFP4 with FP8 healing at HEALING_ITER.
            healing_callback = NVFP4HealingCallback(
                NVFP4HealingConfig(
                    healing_iter=HEALING_ITER,
                    healing_recipe=healing_recipe,
                    pre_quantize_base_weights=True,
                )
            )
            tracker = RecipeTrackingCallback()
            finetune_cfg = _base_config(
                TinyGPTProvider(tensor_model_parallel_size=1, pipeline_model_parallel_size=1),
                FINETUNE_ITERS,
                CheckpointConfig(
                    save=lora_dir,
                    save_interval=FINETUNE_ITERS,
                    pretrained_checkpoint=pretrain_dir,
                ),
                mixed_precision="bf16_with_nvfp4_mixed",
                peft=LoRA(dim=8, alpha=16, dropout=0.0, target_modules=["linear_qkv", "linear_proj"]),
            )
            finetune(finetune_cfg, forward_step, callbacks=[healing_callback, tracker])

            # Healing applied exactly at the configured boundary.
            assert healing_callback.healed, "healing did not fire"
            expected_healing_recipe = "DelayedScaling" if healing_recipe == "delayed" else "MXFP8BlockScaling"
            assert tracker.recipe_names[:HEALING_ITER] == ["NVFP4BlockScaling"] * HEALING_ITER, (
                f"pre-healing steps must run NVFP4, got {tracker.recipe_names}"
            )
            assert tracker.recipe_names[HEALING_ITER:] == [expected_healing_recipe] * (
                FINETUNE_ITERS - HEALING_ITER
            ), f"post-healing steps must run {expected_healing_recipe}, got {tracker.recipe_names}"

            # Recipe function restored after training.
            restored = fp4_utils.get_fp4_recipe(
                TinyGPTProvider(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            )
            assert type(restored).__name__ == "NVFP4BlockScaling"

            # Training remained numerically sane through the switch.
            assert len(tracker.losses) == FINETUNE_ITERS
            assert all(math.isfinite(loss) for loss in tracker.losses), f"non-finite losses: {tracker.losses}"

        finally:
            clear_directories(shared_base_dir)
