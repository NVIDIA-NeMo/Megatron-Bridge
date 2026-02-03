#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
E2E test for MIMO training using MIMO configs, model, data, training loop, and optimizer.

Run with torchrun on an even number of GPUs (>=2):
    torchrun --nproc_per_node=2 tests/e2e/mimo/test_mimo_training_e2e.py
"""

import logging
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from megatron.core import parallel_state
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
try:
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_transformer_engine_spec as get_vit_layer_spec,
    )
except Exception:
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_local_spec as get_vit_layer_spec,
    )
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules

from megatron.bridge.data.mimo import MockMimoProvider
from megatron.bridge.data.loaders import build_train_valid_test_data_loaders
from megatron.bridge.models.mimo import MimoModelProvider
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mimo_step import forward_step as mimo_forward_step
from megatron.bridge.training.pretrain_mimo import pretrain_mimo
from megatron.bridge.training.state import GlobalState, TrainState

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def build_data_iterators_fn(cfg: ConfigContainer, mimo_infra) -> Tuple:
    _ = mimo_infra
    train_state = TrainState()
    
    train_loader, valid_loader, _ = build_train_valid_test_data_loaders(
        cfg=cfg,
        train_state=train_state,
        build_train_valid_test_datasets_provider=lambda *_: None,
        dp_group=None,
    )
    
    def _wrap_iter(data_iter):
        if data_iter is None:
            return None
        for batch in data_iter:
            if batch is None:
                yield None
                continue
            modality_inputs = batch.get("modality_inputs", {})
            if "vision" in modality_inputs and isinstance(modality_inputs["vision"], dict):
                vision_inputs = modality_inputs["vision"]
                # CLIPViTModel expects "x" (images tensor), not "pixel_values".
                if "pixel_values" in vision_inputs:
                    vision_inputs = {"x": vision_inputs["pixel_values"]}
                batch["modality_inputs"]["vision"] = {"clip": vision_inputs}
            if "loss_mask" not in batch:
                batch["loss_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)
            # Let GPTModel generate its own causal mask; avoid 2D mask shape mismatch.
            batch["attention_mask"] = None
            yield batch

    train_iter = iter(train_loader) if train_loader is not None else None
    valid_iter = iter(valid_loader) if valid_loader is not None else None
    
    return _wrap_iter(train_iter), _wrap_iter(valid_iter)


def run_e2e_test() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    # Initialize Megatron Core global process groups for modules that
    # fall back to MPU process groups on non-participating ranks.
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )

    if world_size < 2 or world_size % 2 != 0:
        if rank == 0:
            logger.error("This test requires an even world size >= 2.")
        dist.destroy_process_group()
        sys.exit(1)

    logger.info(f"[Rank {rank}/{world_size}] Starting MIMO training e2e test")

    # Model/data sizes
    hidden_size = 64
    special_token_id = 32000
    image_size = (3, 224, 224)
    patch_dim = 16
    add_class_token = True
    encoder_seq_length = (
        (image_size[1] // patch_dim) * (image_size[2] // patch_dim)
        + (1 if add_class_token else 0)
    )
    seq_length = 256

    tokenizer_path = os.getenv("MIMO_TEST_TOKENIZER", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pad vocab size to be divisible by max TP we'll test (4)
    # This is required because VocabParallelEmbedding shards the embedding table
    max_tp = 4
    vocab_size = tokenizer.vocab_size
    if vocab_size % max_tp != 0:
        vocab_size = ((vocab_size // max_tp) + 1) * max_tp

    # Parallelism config from environment variables (defaults to DP-only split)
    default_dp = world_size // 2
    llm_tp = int(os.getenv("MIMO_LLM_TP", "1"))
    llm_pp = int(os.getenv("MIMO_LLM_PP", "1"))
    llm_dp = int(os.getenv("MIMO_LLM_DP", str(default_dp)))
    llm_offset = int(os.getenv("MIMO_LLM_OFFSET", "0"))

    vision_tp = int(os.getenv("MIMO_VISION_TP", "1"))
    vision_pp = int(os.getenv("MIMO_VISION_PP", "1"))
    vision_dp = int(os.getenv("MIMO_VISION_DP", str(default_dp)))
    vision_offset = int(os.getenv("MIMO_VISION_OFFSET", str(llm_tp * llm_pp * llm_dp)))

    llm_ranks = llm_tp * llm_pp * llm_dp
    vision_ranks = vision_tp * vision_pp * vision_dp
    total_ranks_needed = max(llm_offset + llm_ranks, vision_offset + vision_ranks)

    if total_ranks_needed > world_size:
        if rank == 0:
            logger.error(
                f"Parallelism config requires {total_ranks_needed} ranks, but only {world_size} available. "
                f"LLM: TP={llm_tp}, PP={llm_pp}, DP={llm_dp}, offset={llm_offset} ({llm_ranks} ranks). "
                f"Vision: TP={vision_tp}, PP={vision_pp}, DP={vision_dp}, offset={vision_offset} ({vision_ranks} ranks)."
            )
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        logger.info(
            f"Parallelism config: LLM(TP={llm_tp}, PP={llm_pp}, DP={llm_dp}, offset={llm_offset}), "
            f"Vision(TP={vision_tp}, PP={vision_pp}, DP={vision_dp}, offset={vision_offset})"
        )

    mimo_parallelism_config = MimoParallelismConfig(
        module_parallelisms={
            "llm": ModuleParallelismConfig(
                tensor_model_parallel_size=llm_tp,
                pipeline_model_parallel_size=llm_pp,
                data_parallel_size=llm_dp,
                rank_offset=llm_offset,
            ),
            "vision": ModuleParallelismConfig(
                tensor_model_parallel_size=vision_tp,
                pipeline_model_parallel_size=vision_pp,
                data_parallel_size=vision_dp,
                rank_offset=vision_offset,
            ),
        },
        special_token_ids={"vision": special_token_id},
    )

    # Use LLM's DP for global_batch_size (defines training semantics)
    dp_per_module = llm_dp

    if rank == 0:
        logger.info("Precision: fp32")

    encoder_config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    vision_encoder_spec = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": encoder_config,
            "transformer_layer_spec": get_vit_layer_spec(),
            "add_class_token": add_class_token,
            "class_token_len": 1,
            "patch_dim": patch_dim,
            "img_h": image_size[1],
            "img_w": image_size[2],
            "model_subtype": "clip",
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip": vision_encoder_spec},
        },
    )

    lm_config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": vocab_size,
            "max_sequence_length": seq_length,
            "pre_process": True,
            "post_process": True,
        },
    )

    mimo_provider = MimoModelProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"vision": vision_submodule_spec},
        special_token_ids={"vision": special_token_id},
        mimo_parallelism_config=mimo_parallelism_config,
        bf16=False,
        fp16=False,
    )

    processor_path = os.getenv("MIMO_TEST_PROCESSOR", "openai/clip-vit-base-patch16")
    dataset_provider = MockMimoProvider(
        seq_length=seq_length,
        processor_paths={"vision": processor_path},
        tokenizer_path=tokenizer_path,
        special_token_ids={"vision": special_token_id},
        encoder_seq_lengths={"vision": encoder_seq_length},
        modality_configs={"vision": {"type": "image", "width": image_size[1], "height": image_size[2]}},
        num_workers=0,
        pin_memory=False,
    )
    if not hasattr(dataset_provider, "drop_last"):
        dataset_provider.drop_last = True

    micro_batch_size = 2
    num_microbatches = 1
    train_iters = 2
    global_batch_size = micro_batch_size * dp_per_module * num_microbatches

    # Calculate train_samples based on max DP to ensure all modules have enough data.
    # Vision with higher DP (4) needs more total samples than LLM with DP=2.
    # train_samples = train_iters * micro_batch_size * max_dp * num_microbatches
    max_dp = max(llm_dp, vision_dp)
    train_samples = train_iters * micro_batch_size * max_dp * num_microbatches

    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
        train_samples=train_samples,  # Override to ensure enough data for all modules
        eval_interval=1000,
        eval_iters=1,
    )
    # MIMO training loop expects these runtime fields.
    train_cfg.num_microbatches = num_microbatches
    train_cfg.grad_reduce_in_fp32 = False
    train_cfg.overlap_grad_reduce = False
    train_cfg.use_distributed_optimizer = True
    train_cfg.check_for_nan_in_grad = False

    optimizer_cfg = OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=False,
        use_distributed_optimizer=True,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=optimizer_cfg,
        scheduler=SchedulerConfig(lr_decay_style="constant"),
        dataset=dataset_provider,
        logger=LoggerConfig(timing_log_level=0, timing_log_option="minmax", log_interval=1),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer"),
        checkpoint=CheckpointConfig(save=None, save_interval=None),
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=True),
    )
    cfg.data_parallel_size = dp_per_module
    cfg.dist.use_gloo_process_groups = False

    global_state = GlobalState()
    global_state.cfg = cfg
    global_state.train_state = TrainState()

    logger.info(f"[Rank {rank}] Starting pretrain_mimo")
    try:
        pretrain_mimo(
            cfg=cfg,
            mimo_provider=mimo_provider,
            forward_step_func=mimo_forward_step,
            build_data_iterators_fn=build_data_iterators_fn,
            opt_config=optimizer_cfg,
            schedulers={},
            global_state=global_state,
        )
        logger.info(f"[Rank {rank}] pretrain_mimo completed successfully")
    except Exception as exc:
        logger.error(f"[Rank {rank}] Error during MIMO e2e test: {exc}")
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    logger.info(f"[Rank {rank}] MIMO e2e test PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    run_e2e_test()
