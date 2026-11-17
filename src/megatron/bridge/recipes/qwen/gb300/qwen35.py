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

"""Qwen3.5 text recipes for GB300 NVL72 systems."""

from megatron.bridge.recipes.qwen.gb200.qwen35 import (
    _CODERFORGE_REVISION,
    _QWEN35_35B_A3B,
    _QWEN35_35B_A3B_INSTRUCT_REVISION,
    qwen35_text_35b_a3b_fp8mx_dynamic_cp_common,
)
from megatron.bridge.recipes.utils.dataset_utils import default_coderforge_config
from megatron.bridge.training.config import ConfigContainer


_LONG_CONTEXT_SEQ_LENGTH = 131072
_LONG_CONTEXT_CP_SIZE = 8
_LONG_CONTEXT_EP_SIZE = 8


def _qwen35_text_35b_a3b_long_context_dynamic_cp_common(*, enable_offline_packing: bool) -> ConfigContainer:
    """Build the shared eight-GB300 128K CoderForge dynamic-CP configuration."""
    cfg = qwen35_text_35b_a3b_fp8mx_dynamic_cp_common(
        seq_length=_LONG_CONTEXT_SEQ_LENGTH,
        context_parallel_size=_LONG_CONTEXT_CP_SIZE,
        expert_model_parallel_size=_LONG_CONTEXT_EP_SIZE,
    )
    cfg.tokenizer.tokenizer_model = _QWEN35_35B_A3B
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _QWEN35_35B_A3B_INSTRUCT_REVISION}

    cfg.dataset = default_coderforge_config(
        seq_length=_LONG_CONTEXT_SEQ_LENGTH,
        enable_offline_packing=enable_offline_packing,
        # Offline rows are packed on lengths aligned to 2 x CP, which also truncates a longer trajectory to the
        # row instead of dropping it.
        pad_seq_to_mult=2 * _LONG_CONTEXT_CP_SIZE,
    )
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _CODERFORGE_REVISION}
    cfg.dataset.enable_global_batch_packing = True
    # Samples are read in file order, so static and dynamic CP runs see the same steps.
    cfg.dataset.dataloader_type = "single"
    # 256-wide attention heads run on FlashAttention, which needs bins without padding between sequences.
    cfg.dataset.fold_alignment_padding = True
    return cfg


def qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_dynamic_cp_config() -> ConfigContainer:
    """Return a full Qwen3.5-35B-A3B 128K SFT config for eight GB300 GPUs with dynamic CP on CoderForge.

    Fine-tunes all 40 decoder layers plus one MTP layer on the CoderForge Preview SWE-Rebench trajectories with TP1,
    PP1, base CP8, EP8 and ETP1 at a 131,072-token context, micro batch 1, global batch 16 samples, per-token loss,
    MXFP8 compute, FP32 Adam state offloaded to the host, HybridEP and uniform full recompute. A trajectory of at
    most 16,384 tokens runs on one GPU, and global-batch packing runs every packed bin on a CP1, CP2, CP4 or CP8
    group sized for its longest sequence (``default_dynamic_cp``). Override
    ``model.dynamic_context_parallel=false`` for the static-CP8 baseline on the same packing path. Longer
    trajectories are truncated to the context.

    Set ``checkpoint.pretrained_checkpoint`` to a converted Megatron text checkpoint before training. On more
    GPUs the extra ranks become data-parallel replicas that join the dynamic-CP pool; scale
    ``train.global_batch_size`` with them.

    Requires Megatron-Core with dynamic context parallelism through hybrid (Gated DeltaNet) layers,
    https://github.com/NVIDIA/Megatron-LM/pull/7048, and a large host-memory allocation for the CPU optimizer.
    """
    cfg = _qwen35_text_35b_a3b_long_context_dynamic_cp_common(enable_offline_packing=False)
    # GPT-SFT tokenizes each trajectory lazily (seconds per sample), and the scheduler pulls a whole step's
    # samples at once; two workers leave the GPUs idle.
    cfg.dataset.num_workers = 16
    cfg.train.global_batch_size = 16
    return cfg


def qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_offline_dynamic_cp_config() -> ConfigContainer:
    """Return :func:`qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_dynamic_cp_config` on offline rows.

    The CoderForge trajectories are tokenized and packed ahead of training into 131,072-token rows. Each step takes
    a global batch of eight rows, and the global-batch scheduler unpacks them and runs every packed bin on a CP1,
    CP2, CP4 or CP8 group sized for its longest sequence. Offline packing decides which trajectories share a step,
    so a step carries a nearly constant number of tokens, and no trajectory is tokenized during training.

    Prepare the rows before training with ``scripts/training/prepare_gpt_sft_packed_data.py``; otherwise rank 0
    tokenizes the whole split while the other ranks wait.
    """
    cfg = _qwen35_text_35b_a3b_long_context_dynamic_cp_common(enable_offline_packing=True)
    cfg.train.global_batch_size = 8
    return cfg
