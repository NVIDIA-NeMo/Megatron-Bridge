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

"""ERNIE 5.5 recipe on H100: supervised fine-tuning of a Hugging Face checkpoint.

:func:`ernie55_sft_128gpu_h100_bf16_config` trains with the model, loss, optimizer and precision settings of
:mod:`megatron.bridge.recipes.ernie.ernie55` and changes what an 80 GB GPU needs: eight pipeline stages (exchanging
fp32) and EP16 on 16 nodes of 8 GPUs, packed rows of at most 4096 tokens, a global batch of 32 rows, and four times
the steps. Like the GB300 recipe it runs with the ERNIE 5.5 step
(``scripts/training/run_recipe.py --recipe <name> --step_func ernie55_step``) and requires Megatron-Core's ``dev``
branch at commit ``f78809f6d`` or later.

The recipe keeps the cuDNN DSA backend, whose sparse forward is FlashMLA's. The environment therefore needs a
FlashMLA build with its SM90 kernels; ``docker/Dockerfile.ci`` builds FlashMLA with ``FLASH_MLA_DISABLE_SM90=1``,
so an image for H100 must build it without that variable (see ``examples/models/ernie55/README.md``).
"""

from __future__ import annotations

import torch

from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDatasetConfig
from megatron.bridge.recipes.ernie.ernie55 import (
    _SYNTHETIC_TOKEN_RANGE,
    _ernie55_common_config,
    _ernie55_hf_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


__all__ = [
    "ernie55_sft_128gpu_h100_bf16_config",
]

# The row length the 80 GB layout is sized for: what one rank of the GB300 recipe holds of a 262144-token row.
_H100_SEQ_LENGTH = 4096
# Two micro-batches per rank (32 / 16 data-parallel ranks): with eight per rank (a global batch of 128), the 1F1B
# schedule keeps more forward passes in flight on the first stages than an 80 GB GPU holds.
_H100_GLOBAL_BATCH_SIZE = 32
# The GB300 recipe's schedule in row capacity: 2065 steps of 2 rows of 262144 tokens hold as many tokens as 8260
# steps of 32 rows of 4096 tokens, and 103 warmup steps as many as 412.
_H100_TRAIN_ITERS = 8260
_H100_LR_WARMUP_ITERS = 412


def ernie55_sft_128gpu_h100_bf16_config(*, hf_path: str | None = None) -> ConfigContainer:
    """Fine-tune a Hugging Face ERNIE 5.5 checkpoint on 16 nodes of 8 H100 GPUs.

    TP1 PP8 CP1 EP16 (DP16), micro-batch 1, global batch 32, rows of at most 4096 tokens, 8260 steps with 412
    warmup steps (the GB300 recipe's schedule in tokens of full rows), the reference's model, loss and optimizer
    settings (see :mod:`megatron.bridge.recipes.ernie.ernie55`), with the model built from the checkpoint's
    ``config.json`` and starting from its weights (``checkpoint.pretrained_checkpoint``). The data are SFT JSONL
    (``dataset.path``; text rows need ``dataset.hf_tokenizer``), several samples packed into a row, samples longer
    than 4096 tokens split into rows; without a path, every row is one synthetic 4096-token document. The final
    checkpoint holds the weights only, as the GB300 recipe's does.

    Args:
        hf_path: Directory of the Hugging Face ERNIE 5.5 checkpoint (``config.json`` and safetensors); default
            ``$ERNIE55_HF_PATH``. Required to train.

    Returns:
        The training config; run it with ``--step_func ernie55_step`` (``run_recipe.py`` infers ``finetune``).
        Without ``hf_path`` and ``$ERNIE55_HF_PATH`` it builds the placeholder
        :data:`~megatron.bridge.recipes.ernie.ernie55.ERNIE55_FULL_SIZE_HF_CONFIG` with no checkpoint, as Bridge's
        recipe matrix requires; finetuning then stops in setup.
    """
    hf_config, path = _ernie55_hf_config(hf_path)
    cfg = _ernie55_common_config(
        hf_config,
        seq_length=_H100_SEQ_LENGTH,
        # A 4096-token row stays on one rank.
        context_parallel_size=1,
        # EP16 spans the 16 data-parallel ranks of a pipeline stage: a rank holds a sixteenth of the routed experts.
        expert_model_parallel_size=16,
        train_iters=_H100_TRAIN_ITERS,
        lr_warmup_iters=_H100_LR_WARMUP_ITERS,
        global_batch_size=_H100_GLOBAL_BATCH_SIZE,
    )
    # Eight pipeline stages, so that a rank's share of the layers fits an 80 GB GPU. Ernie55ModelConfig splits the
    # layer pattern at ERNIE-layer boundaries and enables variable_seq_lengths for the packed rows.
    cfg.model.pipeline_model_parallel_size = 8
    # The stages exchange the hyper-connections' fp32 residual streams. bf16_mixed() writes its own pipeline_dtype
    # to the model config at startup, so it is set there too.
    cfg.model.pipeline_dtype = torch.float32
    cfg.mixed_precision.pipeline_dtype = torch.float32
    cfg.dataset = Ernie55SFTDatasetConfig(
        seq_length=_H100_SEQ_LENGTH,
        pad_token_id=hf_config.pad_token_id,
        vocab_size=hf_config.vocab_size,
        path=None,
        truncation="split",
        # Short samples share a row, so rows of real data are close to the 4096 tokens the layout is sized for.
        pack=True,
        # 2 x CP, as in the GB300 recipe (2 x 64 there).
        pad_to_multiple_of=2,
        mock_document_lengths=(_H100_SEQ_LENGTH,),
        mock_prompt_fraction=0.25,
        mock_token_range=_SYNTHETIC_TOKEN_RANGE,
        seed=42,
    )
    cfg.checkpoint.save_interval = cfg.train.train_iters
    cfg.checkpoint.pretrained_checkpoint = path
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    return cfg
