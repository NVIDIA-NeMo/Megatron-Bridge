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

"""ERNIE 5.5 recipe on GB200: supervised fine-tuning of a Hugging Face checkpoint.

:func:`ernie55_sft_64gpu_gb200_bf16_config` fine-tunes a Hugging Face ERNIE 5.5 checkpoint on 16 nodes of 4 GB200
GPUs. Its model, loss, optimizer, schedule, precision and tokenizer settings are those of the GB300 recipe
(:mod:`megatron.bridge.recipes.ernie.ernie55`, whose module docstring describes them); only the shape differs,
because the GB300 recipe's shape needs more memory per GPU than a GB200 has: CP8 instead of CP64, 16384-token
packed rows instead of one 262144-token sample per row, global batch 32 instead of 2, and a process group whose
object collectives run on CPU. The function's docstring gives the reason for each.

Like the GB300 recipe it requires Megatron-Core's ``dev`` branch at commit ``f78809f6d`` or later and trains with
the ERNIE 5.5 step: ``scripts/training/run_recipe.py --recipe <name> --step_func ernie55_step``.
"""

from __future__ import annotations

import dataclasses

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDatasetConfig
from megatron.bridge.recipes.ernie.ernie55 import (
    _SYNTHETIC_TOKEN_RANGE,
    _ernie55_common_config,
    _ernie55_hf_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


__all__ = [
    "ernie55_sft_64gpu_gb200_bf16_config",
]

_GB200_CONTEXT_PARALLEL_SIZE = 8
_GB200_SEQ_LENGTH = 16384


def ernie55_sft_64gpu_gb200_bf16_config(*, hf_path: str | None = None) -> ConfigContainer:
    """Fine-tune a Hugging Face ERNIE 5.5 checkpoint on 16 nodes of 4 GB200 GPUs.

    TP1 PP1 CP8 EP64 (DP8), micro-batch 1, global batch 32, 16384-token rows split contiguously over the 8
    context-parallel ranks (``cp_partition_mode="contiguous"``), 2065 steps with 103 warmup steps, with the model,
    loss and optimizer settings of
    :func:`~megatron.bridge.recipes.ernie.ernie55.ernie55_sft_64gpu_gb300_bf16_config`, the model built from the
    checkpoint's ``config.json`` and starting from its weights (``checkpoint.pretrained_checkpoint``). What differs
    from that recipe, and why:

    * **CP8, so DP8.** With CP64 on 64 GPUs the data-parallel size is 1 and every GPU holds the whole optimizer
      state of the parameters that expert parallelism does not split. With CP8 the layer-wise distributed
      optimizer spreads that state over 8 data-parallel ranks.
    * **16384-token rows.** Every context-parallel rank holds 2048 of a row's tokens. Longer rows have not been
      validated on GB200.
    * **Packed rows.** Samples shorter than a row share it (``dataset.pack``, the reference's ``--packing``);
      samples longer than 16384 tokens are split into rows, as in the GB300 recipe. Each document is padded to a
      multiple of 16 tokens (2 x CP). Without ``dataset.path`` every row is one synthetic 16384-token document.
    * **Global batch 32.** Four micro-batches on each of the 8 data-parallel ranks: 524288 tokens per step, as many
      as the GB300 recipe's two 262144-token rows, so its schedule carries over unchanged.
    * **Process group ``cpu:gloo,cuda:nccl``.** CUDA tensors still go through NCCL; the object collectives of a
      distributed checkpoint save go through gloo. On an NCCL-only group the save leaves about 24 GiB of
      point-to-point buffers on rank 0 at 64 ranks, and the next training step runs out of memory there.

    The final checkpoint holds the weights only, as the GB300 recipe's does.

    Needs Megatron-Core dev with the ``_post_o_group_proj`` hook: Megatron-Core main has no CP partition modes.

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
        seq_length=_GB200_SEQ_LENGTH,
        # DP = 64 GPUs / CP8 = 8: the layer-wise distributed optimizer shards its state over the DP ranks.
        context_parallel_size=_GB200_CONTEXT_PARALLEL_SIZE,
        expert_model_parallel_size=64,
        train_iters=2065,
        lr_warmup_iters=103,
        # Micro-batch 1 x DP8 x 4 micro-batches per step: 524288 tokens, as in the GB300 recipe.
        global_batch_size=32,
    )
    # cp_partition_mode is a Megatron-Core dev field. Without it (Megatron-Core main) the model config's finalize
    # rejects CP > 1 with a clear message instead of an AttributeError here.
    if "cp_partition_mode" in {field.name for field in dataclasses.fields(TransformerConfig)}:
        cfg.model.cp_partition_mode = "contiguous"
    cfg.dataset = Ernie55SFTDatasetConfig(
        seq_length=_GB200_SEQ_LENGTH,
        pad_token_id=hf_config.pad_token_id,
        vocab_size=hf_config.vocab_size,
        path=None,
        truncation="split",
        # Short samples share a row; at one sample per row most of a 16384-token row would be empty.
        pack=True,
        pad_to_multiple_of=2 * _GB200_CONTEXT_PARALLEL_SIZE,
        mock_document_lengths=(_GB200_SEQ_LENGTH,),
        mock_prompt_fraction=0.25,
        mock_token_range=_SYNTHETIC_TOKEN_RANGE,
        seed=42,
    )
    cfg.checkpoint.save_interval = cfg.train.train_iters
    cfg.checkpoint.pretrained_checkpoint = path
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    # The object collectives of a distributed checkpoint save run on CPU (gloo); CUDA tensors keep NCCL. On an
    # NCCL-only group they leave about 24 GiB of point-to-point buffers on rank 0 at 64 ranks, more than this
    # shape has to spare on a GB200.
    cfg.dist.distributed_backend = "cpu:gloo,cuda:nccl"
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    return cfg
