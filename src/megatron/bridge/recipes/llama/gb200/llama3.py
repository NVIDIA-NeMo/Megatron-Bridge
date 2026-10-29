# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 training configurations for GB200."""

from megatron.bridge.recipes.llama.h100.llama3 import llama31_405b_pretrain_256gpu_h100_bf16_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


def llama31_405b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Configure 405B MXFP8 pretraining at 8K on 64 GB200 GPUs.

    Uses TP8/PP8/VPP8/CP1 with sequence parallelism, no activation recomputation,
    and no CUDA graphs. Tensor-parallel communication overlap is enabled.
    Place the sixteen four-GPU nodes within one NVLink domain. At 64 GPUs,
    DP is one: global batch 1536 and microbatch one accumulate 1536 microbatches
    per optimizer update. Keep the global batch fixed when sizing a launch.

    MXFP8 parameter gather reuses the gradient buffer. Gradient accumulation
    is BF16; optimizer master parameters and Adam moments remain FP32. No
    optimizer or activation offload is enabled. The training schedule,
    checkpointing, and numerical checks retain their training defaults.
    Configure a tokenizer and indexed dataset for real-text training; the
    default dataset is synthetic.

    Validated for 10 real-text updates in NeMo 26.10.rc10: approximately 1718
    model TFLOP/s/GPU, 289 seconds/update, and 154.6 GiB peak allocated memory.
    This is a capacity/stability check, not a long-term convergence result.
    MXFP8 checkpoint save/resume was not exercised in this validation.
    """
    cfg = llama31_405b_pretrain_256gpu_h100_bf16_config()
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192
    cfg.model.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.tokenizer.vocab_size = 128256
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = []
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = []
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True
    cfg.comm_overlap.tp_comm_overlap = True
    cfg.comm_overlap.tp_comm_overlap_cfg = None

    cfg.env_vars = {
        **cfg.env_vars,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg
