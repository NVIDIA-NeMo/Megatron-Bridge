from dataclasses import dataclass

import torch

from megatron.bridge.models.llama import Llama3ModelProvider, Llama31ModelProvider
from megatron.bridge.training.config import (
    MockGPTDatasetConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


@dataclass
class Llama3ModelProviderFSDP145M(Llama3ModelProvider):
    """Small Llama3 model configuration for FSDP testing - matches the test."""

    rotary_base: int = 500_000
    seq_length: int = 1024
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int = 4096
    # Disable gradient accumulation fusion for FSDP
    gradient_accumulation_fusion: bool = False
    # Model parallel settings
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    attention_softmax_in_fp32: bool = True
    make_vocab_size_divisible_by: int = 128
    bf16: bool = True
    pipeline_dtype: torch.dtype = torch.bfloat16


@dataclass
class Llama31ModelProviderFSDP145M(Llama31ModelProvider):
    """Small Llama31 model - test if Llama31 (like HF bridge uses) is the issue."""

    rotary_base: int = 500_000
    seq_length: int = 1024
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int = 4096
    gradient_accumulation_fusion: bool = False
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    attention_softmax_in_fp32: bool = True
    make_vocab_size_divisible_by: int = 128
    bf16: bool = True
    pipeline_dtype: torch.dtype = torch.bfloat16


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing full Llama 3.2 1B model with FSDP + checkpointing")
    print("=" * 80)

    from megatron.bridge.recipes.llama import llama32_1b_pretrain_config as pretrain_config
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    # Create config with FSDP enabled
    cfg = pretrain_config(seq_length=1024, use_megatron_fsdp=True)

    # Minimal required overrides for testing
    cfg.train.train_iters = 2
    cfg.scheduler.lr_warmup_iters = 1
    cfg.scheduler.lr_decay_iters = 2
    cfg.train.eval_iters = 0
    cfg.logger.log_interval = 1

    # FSDP-specific configurations
    # cfg.model.gradient_accumulation_fusion = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.average_in_collective = False

    # CPU offloading configuration
    # cpu_offloading_num_layers must be set and must be < num_layers (16 for Llama 3.2 1B)
    # Offload all but the last layer for maximum memory savings
    # cfg.model.cpu_offloading = True
    # cfg.model.cpu_offloading_num_layers = 15  # Must be < 16 (num_layers)
    cfg.model.gradient_accumulation_fusion = False

    # Checkpoint configuration for FSDP
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.save = "/tmp/fsdp_checkpoint"
    cfg.checkpoint.save_interval = 2  # Save at the end
    cfg.checkpoint.load = None

    print(f"\nModel: {cfg.model.__class__.__name__}")
    print(f"Num layers: {cfg.model.num_layers}")
    print(f"Hidden size: {cfg.model.hidden_size}")
    print(f"Vocab size: {cfg.model.vocab_size}")
    print(f"\nDataset type: {cfg.dataset.__class__.__name__}")
    print(f"Mixed precision: {cfg.mixed_precision}")
    print(f"Comm overlap: {cfg.comm_overlap}")

    # FIX: Set mixed_precision to None (recipe sets "bf16_mixed" string, test has None)
    print("\nClearing mixed_precision config (recipe sets 'bf16_mixed', test uses None)...")
    cfg.mixed_precision = None

    # FIX: Replace GPTDatasetConfig with MockGPTDatasetConfig (like the test uses)
    print("Replacing dataset with MockGPTDatasetConfig...")
    cfg.dataset = MockGPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=1024,
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
    )

    pretrain(cfg, forward_step)
