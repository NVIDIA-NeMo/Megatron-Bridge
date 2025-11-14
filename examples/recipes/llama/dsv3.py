"""Minimal example for running a DeepSeek recipe with debug-friendly settings."""

# uv run python -m torch.distributed.run --nproc_per_node=8 examples/recipes/llama/dsv3.py

from __future__ import annotations

import torch

from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import bf16_with_fp8_subchannel_scaling_mixed
from megatron.bridge.training.pretrain import pretrain

def deepseek_v3_pretrain_config_debug() -> ConfigContainer:
    """Run a tiny DeepSeek training job with subchannel FP8 precision."""
    from megatron.bridge.recipes.deepseek import deepseek_v3_pretrain_config_32nodes

    cfg: ConfigContainer = deepseek_v3_pretrain_config_32nodes(
        dir="/chcui/mbridge_home/exp/interactive/1112-deepseek_v3_pretrain_debug",
        enable_deepep=True,
    )

    # small values for debugging
    cfg.model.mtp_num_layers = 0
    cfg.model.num_layers = 20
    cfg.model.moe_layer_freq = [0] + [1] * 19
    cfg.model.seq_length = 512
    cfg.model.num_moe_experts = 16
    cfg.model.ffn_hidden_size = 2048
    cfg.model.hidden_size = 2048
    cfg.dataset.sequence_length = 512
    cfg.train.train_iters = 3
    cfg.logger.log_interval = 1
    cfg.train.global_batch_size = 8
    cfg.scheduler.lr_warmup_iters = 100
    cfg.scheduler.lr_decay_iters = 1000

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    # Do not save checkpoint during debug run
    cfg.checkpoint.save_interval = 0  # disables checkpointing
    cfg.checkpoint.save = None        # no checkpoint output dir
    cfg.checkpoint.load = None        # do not load checkpoint
    # cfg.mixed_precision = bf16_with_fp8_subchannel_scaling_mixed()
    
    # Disable overlap_grad_reduce for MoE with FP8 to avoid gradient sync issues
    # when some experts don't receive tokens. Must set via CommOverlapConfig
    # to prevent validate() from re-enabling it.
    # cfg.comm_overlap = CommOverlapConfig(
    #     tp_comm_overlap=False,
    #     overlap_grad_reduce=False,  # Critical for MoE with FP8
    #     overlap_param_gather=True,  # Keep other optimizations
    # )
    return cfg

def nemotron_nano_9b_v2_finetune() -> ConfigContainer:
    from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config
    cfg: ConfigContainer = nemotron_nano_9b_v2_finetune_config(
        dir="/chcui/mbridge_home/exp/interactive/1110-nemotron_nano_9b_v2_finetune",
        pretrained_checkpoint="/chcui/mbridge_home/models/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base",
        peft=None,
    )
    return cfg

def qwen3_600m_finetune() -> ConfigContainer:
    from megatron.bridge.recipes.qwen import qwen3_600m_finetune_config
    cfg: ConfigContainer = qwen3_600m_finetune_config(
        pretrained_checkpoint="/chcui/mbridge_home/models/Qwen/Qwen3-0.6B",
        peft="lora",
    )
    return cfg

def main() -> None:
    
    cfg = deepseek_v3_pretrain_config_debug()
    cfg.validate()
    pretrain(config=cfg, forward_step_func=forward_step)

    # Cleanup process group
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

    # RC5
    # Without DEEPEP: Allocated: 13.39 GB, Reserved: 18.54 GB
    # With DEEPEP: 13.39 GB, Reserved: 18.33 GB
