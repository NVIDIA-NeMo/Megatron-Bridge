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

import os

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class Qwen3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str
    # Dataset configuration
    data_paths: list[str] | None
    data_args_path: str | None
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    per_split_data_args_path: str | None
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    use_null_tokenizer: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


class Qwen3FinetuneKwargs(Qwen3CommonKwargs, total=False):
    """Typed options accepted by Qwen3 finetuning recipe helper functions."""

    # Core finetuning options
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Training params
    finetune_lr: float

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None


def qwen3_600m_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 0.6B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-0.6B").to_megatron_provider(load_weights=False)

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-0.6B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_1p7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 1.7B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-1.7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-1.7B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_4b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 4B.

    Recommended parallelism: TP=2, PP=1.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-4B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-4B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 8B.

    Recommended parallelism: TP=4, PP=1.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-8B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_14b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 14B.

    Recommended parallelism: TP=8, PP=1.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-14B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-14B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config (paths set in _pretrain_common)
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_32b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 32B.

    Recommended parallelism: TP=8, PP=2 with recompute enabled for memory optimization.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-32B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-32B"

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading) - ENABLED for 32B
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_600m_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 600M.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-0.6B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_1p7b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 1.7B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-1.7B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_4b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 4B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-4B",
        "tensor_model_parallel_size": 2 if is_full_sft else 1,  # Match NeMo2: higher TP for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_8b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 8B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=4, PP=1, LR=5e-6
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-8B",
        "tensor_model_parallel_size": 4 if is_full_sft else 1,  # Match NeMo2: TP=4 for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_14b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 14B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=1, LR=5e-6
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-14B",
        "tensor_model_parallel_size": 8 if is_full_sft else 1,  # Match NeMo2: TP=8 for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_32b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 32B.

    Default configuration: 2 nodes, 16 GPUs total
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4 (with recompute)
    - Full SFT: TP=8, PP=2, LR=5e-6 (with recompute)
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-32B",
        "tensor_model_parallel_size": 8 if is_full_sft else 1,  # Match NeMo2: TP=8 for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 2 if is_full_sft else 1,  # PP=2 for SFT, PP=1 for LoRA
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    config = _qwen3_finetune_common(**combined_kwargs)

    # Enable recompute for 32B model
    config.model.recompute_granularity = "full"
    config.model.recompute_method = "uniform"
    config.model.recompute_num_layers = 1

    return config


def _qwen3_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Core model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int | None = None,  # Auto-select based on packed_sequence if None
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,  # Let config handle this
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for all Qwen3 models."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select global_batch_size based on packed_sequence
    if global_batch_size is None:
        global_batch_size = 8 if packed_sequence else 128

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )

    # PEFT config
    peft_config = default_peft_config(peft)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Always use HF tokenizer for finetuning
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    pad_seq_to_mult = context_parallel_size * 2 if packed_sequence and context_parallel_size > 1 else 1

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=32,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(check_for_nan_in_grad=True),
        dataset=default_squad_config(seq_length, packed_sequence, pad_seq_to_mult),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
