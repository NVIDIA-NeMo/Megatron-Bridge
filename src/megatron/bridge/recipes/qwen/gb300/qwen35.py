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
    _QWEN35_35B_A3B_INSTRUCT_REVISION,
    qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config,
)
from megatron.bridge.training.config import ConfigContainer


def qwen35_text_35b_a3b_sft_long_context_16gpu_gb300_fp8mx_dynamic_cp_config(
    seq_length: int = 131072,
    *,
    hf_path: str = "Qwen/Qwen3.5-35B-A3B",
    hf_revision: str | None = _QWEN35_35B_A3B_INSTRUCT_REVISION,
) -> ConfigContainer:
    """MXFP8 128K text SFT on CoderForge with dynamic context parallelism on 16 GB300 GPUs.

    This is :func:`qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config` (TP1, base CP8, EP16, one MTP
    layer, offline packed CoderForge rows, selective recompute, NCCL EP) with global-batch packing and dynamic
    context parallelism switched on: the same rows, global batch, optimizer and schedule. Evaluation is shorter
    (two global batches instead of eight), because global-batch packing does not cycle the validation rows, and
    the Gated DeltaNet layers run without the fused pre-delta-rule kernels, which cannot run a 16-rank group.
    Each step, the scheduler unpacks the step's rows and runs every packed bin on a CP1, CP2, CP4 or CP8 group
    sized for its longest sequence (``default_dynamic_cp``); a sequence of at most ``seq_length // 8`` tokens runs
    on one GPU. Every bin is padded to its group's capacity, so each rank always runs ``seq_length // 8`` tokens.
    Override ``model.dynamic_context_parallel=false`` for static CP8 on the same packing path.

    Set ``checkpoint.pretrained_checkpoint`` to a converted Megatron text checkpoint before training. The 16 ranks
    form the dynamic-CP pool; Megatron-Core splits Gated DeltaNet's 16 key heads across a bin's group, so the pool
    cannot grow beyond 16 ranks for this model.

    Args:
        seq_length: Packed token budget, a positive multiple of 16 for CP8.
        hf_path: HF model ID or extracted local text checkpoint.
        hf_revision: Matching model/tokenizer revision, or None for local inputs.

    Returns:
        Training configuration with global-batch packing over offline rows and dynamic context parallelism.
    """
    cfg = qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config(
        seq_length, hf_path=hf_path, hf_revision=hf_revision
    )

    # The scheduler re-bins the sequences of each step's offline rows across the DP x CP pool.
    cfg.dataset.enable_global_batch_packing = True
    cfg.dataset.dataloader_type = "single"
    # 256-wide attention heads run on FlashAttention, which needs bins without padding between sequences.
    cfg.dataset.fold_alignment_padding = True
    cfg.train.micro_batch_size = 1
    cfg.ddp.average_in_collective = False
    # Global-batch packing draws the rows of every evaluation of the run from the validation split without
    # cycling: 11 evaluations of 2 x 32 rows (704) fit the 5% split of the CoderForge trajectories (about 1,650
    # rows), which 11 evaluations of 8 x 32 rows (2,816) do not.
    cfg.validation.eval_iters = 2

    # The dataset switch above selects the default_dynamic_cp scheduler at validation.
    cfg.model.dynamic_context_parallel = True
    cfg.model.min_dynamic_context_parallel_size = 1
    cfg.model.max_seqlen_per_dp_cp_rank = seq_length // cfg.model.context_parallel_size
    # Pad every packed bin to its group's full capacity with a masked dummy sequence, so a rank always runs
    # max_seqlen_per_dp_cp_rank tokens, as it does on the static recipe's fixed-width rows. Triton and CuTe DSL
    # kernels on this path are autotuned or compiled per sequence length: with the bins' own lengths, nearly every
    # microbatch pays that again while the other ranks wait.
    cfg.model.pad_packed_seq_alignment = "max"
    # A bin can run on the whole 16-rank pool, where each rank holds one Gated DeltaNet key head and two value heads.
    # Megatron-Core's fused pre-delta-rule kernels need the channels a rank holds to be a multiple of 8 and fail
    # there (772 channels); the unfused path runs every group size and is a few percent slower.
    cfg.model.gdn_pre_gated_delta_rule_fusion = False
    return cfg
