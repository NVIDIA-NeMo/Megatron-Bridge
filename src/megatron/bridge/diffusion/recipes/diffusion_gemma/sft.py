# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Offline full-parameter SFT recipe using native Megatron checkpoints."""

from pathlib import Path

from megatron.bridge.data.builders import GPTSFTDatasetConfig, PromptCompletionSFTPreprocessingConfig
from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig


def diffusion_gemma_sft_config(
    *,
    dataset_root: str,
    tokenizer_model: str,
    pretrained_checkpoint: str | None,
    experiment_dir: str,
    seq_length: int = 2048,
    micro_batch_size: int = 1,
    global_batch_size: int = 8,
    train_iters: int = 1000,
    model: DiffusionGemmaModelProvider | None = None,
    allow_random_init: bool = False,
) -> ConfigContainer:
    """Build BF16, unpacked SFT with frozen routers and no MoE auxiliary loss.

    Args:
        dataset_root: Existing local GPTSFT train/validation JSONL directory.
        tokenizer_model: Existing tokenizer directory; no automatic download.
        pretrained_checkpoint: Native Megatron checkpoint matching this provider.
        experiment_dir: Explicit directory for checkpoints and TensorBoard logs.
        seq_length: Dataset sequence length before final canvas EOS fill.
        micro_batch_size: Number of rows per microbatch.
        global_batch_size: Accumulated global batch size.
        train_iters: Number of optimizer steps.
        model: Optional native provider, primarily for small acceptance fixtures.
        allow_random_init: Allow missing pretrained weights for a fixture only.

    Returns:
        ConfigContainer for the existing Bridge finetune/pretrain entrypoint.

    The objective averages independently DP-normalized diffusion and encoder
    token means per microbatch, then averages those microbatches through the
    standard accumulation schedule. Packing, PEFT, and model parallelism are
    outside this bounded implementation. HF weight conversion is not provided.
    """
    if not all(str(value).strip() for value in (dataset_root, tokenizer_model, experiment_dir)):
        raise ValueError("dataset_root, tokenizer_model and experiment_dir must be explicit nonempty paths")
    if not pretrained_checkpoint and not allow_random_init:
        raise ValueError("a native pretrained checkpoint is required; random initialization is fixture-only")
    if min(seq_length, micro_batch_size, global_batch_size, train_iters) < 1:
        raise ValueError("sequence length, batch sizes and train_iters must be positive")
    cfg = _sft_common()
    cfg.model = model if model is not None else DiffusionGemmaModelProvider()
    if cfg.model.overlap_moe_expert_parallel_comm or cfg.model.recompute_granularity is not None:
        raise ValueError("DiffusionGemma does not support overlap schedule plans or activation recompute")
    if cfg.model.calculate_per_token_loss:
        raise ValueError("DiffusionGemma requires calculate_per_token_loss=False")
    cfg.model.seq_length = seq_length
    cfg.model.calculate_per_token_loss = False
    cfg.model.apply_rope_fusion = False
    cfg.model.freeze_router = True
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_load_balancing_type = "none"
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cuda_graph_impl = "none"
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.global_batch_size = global_batch_size
    cfg.train.train_iters = train_iters
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=dataset_root,
        seq_length=seq_length,
        do_test=False,
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
            loss_mode="completion",
            add_eos=True,
        ),
        dataset_kwargs={"pad_to_max_length": False},
    )
    cfg.tokenizer = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)
    output = Path(experiment_dir)
    cfg.checkpoint.save = str(output / "checkpoints")
    cfg.checkpoint.load = cfg.checkpoint.save
    cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint
    cfg.checkpoint.save_rng_state_per_dp_rank = True
    cfg.rng.data_parallel_random_init = True
    cfg.logger.tensorboard_dir = str(output / "tb_logs")
    cfg.mixed_precision = "bf16_mixed"
    cfg.peft = None
    return cfg
