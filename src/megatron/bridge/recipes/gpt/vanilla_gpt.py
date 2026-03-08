# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Vanilla GPT recipe — a minimal baseline that mirrors Megatron-LM pretrain_gpt.py defaults.

Use this recipe for MLM <-> Bridge correlation testing.  All architectural
and training knobs are left at their Megatron-Core / pretrain_gpt.py defaults
so that the *only* source of difference between the two frameworks is what you
explicitly override on the CLI.

Example::

    torchrun --nproc_per_node=1 scripts/training/run_recipe.py \\
        --recipe vanilla_gpt_pretrain_config \\
        model.num_layers=2 model.hidden_size=256 model.num_attention_heads=4 \\
        model.activation_func=silu model.gated_linear_unit=true \\
        train.train_iters=10 train.global_batch_size=8 train.micro_batch_size=2
"""

import os

from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)


def vanilla_gpt_pretrain_config() -> ConfigContainer:
    """Minimal GPT pretrain config aligned with Megatron-LM pretrain_gpt.py defaults.

    The model provider uses bare GPTModelProvider defaults (LayerNorm, GeLU,
    learned_absolute position embeddings, etc.) so there are **no** hidden
    model-specific assumptions.  Override anything you need via CLI, including
    ``model.activation_func=silu`` and ``model.gated_linear_unit=true`` for
    SwiGLU activation.

    Returns:
        ConfigContainer with Megatron-LM-compatible defaults.
    """
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=None,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg = ConfigContainer(
        # Bare GPTModelProvider — all fields at their dataclass defaults.
        model=GPTModelProvider(),
        train=TrainingConfig(
            train_iters=300000,
            global_batch_size=32,
            micro_batch_size=2,
        ),
        validation=ValidationConfig(
            eval_interval=500,
            eval_iters=32,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=False,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            sequence_length=1024,
            blend=None,
            blend_per_split=None,
            split="9999,8,2",
            dataloader_type="single",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
        ),
        checkpoint=CheckpointConfig(
            save_interval=500,
            save=checkpoint_dir,
            ckpt_format="torch_dist",
        ),
        rng=RNGConfig(seed=1234),
        dist=DistributedInitConfig(),
        mixed_precision="bf16_mixed",
    )

    return cfg
