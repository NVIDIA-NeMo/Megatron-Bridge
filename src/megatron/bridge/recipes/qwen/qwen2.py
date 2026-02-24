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
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
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
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class Qwen2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen2/2.5 recipe helper functions."""

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
    check_for_nan_in_grad: bool
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
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


# =============================================================================
# Qwen2 Pretrain Configs
# =============================================================================


def qwen2_500m_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2 0.5B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-0.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    # Qwen2 uses NullTokenizer by default for pretraining
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen2_1p5b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2 1.5B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-1.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen2_7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2 7B.

    Recommended parallelism: TP=2, PP=1.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen2_72b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2 72B.

    Recommended parallelism: TP=8, PP=4.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-72B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
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
    cfg.model.cross_entropy_fusion_impl = "native"

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
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# Qwen2.5 Pretrain Configs
# =============================================================================


def qwen25_500m_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2.5 0.5B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-0.5B").to_megatron_provider(load_weights=False)

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_1p5b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2.5 1.5B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-1.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2.5 7B.

    Recommended parallelism: TP=2, PP=1.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_14b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2.5 14B.

    Recommended parallelism: TP=4, PP=1.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-14B").to_megatron_provider(load_weights=False)

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

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
    cfg.model.cross_entropy_fusion_impl = "native"

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
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_32b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2.5 32B.

    Recommended parallelism: TP=8, PP=2.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-32B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
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
    cfg.model.cross_entropy_fusion_impl = "native"

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

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_72b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen2.5 72B.

    Recommended parallelism: TP=8, PP=4.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-72B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
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
    cfg.model.cross_entropy_fusion_impl = "native"

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

    # DDP config (Qwen2.5 uses simpler DDP settings than _pretrain_common)
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


class Qwen2FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen2/2.5 finetuning recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str

    # Finetuning-specific
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Training hyperparameters
    train_iters: int
    global_batch_size: int | None
    micro_batch_size: int
    seq_length: int | None
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: float | None
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None

    # Precision
    precision_config: MixedPrecisionConfig | str | None


# Qwen2 Finetuning Configs
def qwen2_500m_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2 500M.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    return _qwen2_finetune_common(hf_path="Qwen/Qwen2-0.5B", **user_kwargs)


def qwen2_1p5b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2 1.5B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    return _qwen2_finetune_common(hf_path="Qwen/Qwen2-1.5B", **user_kwargs)


def qwen2_7b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2 7B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 2 if is_full_sft else 1

    return _qwen2_finetune_common(hf_path="Qwen/Qwen2-7B", **user_kwargs)


def qwen2_72b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2 72B.

    Default configuration: 4 nodes (SFT) or 1 node (LoRA), 8 GPUs per node
    - LoRA/DoRA: TP=8, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=4, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 8
    if "pipeline_model_parallel_size" not in user_kwargs:
        user_kwargs["pipeline_model_parallel_size"] = 4 if is_full_sft else 1

    return _qwen2_finetune_common(hf_path="Qwen/Qwen2-72B", **user_kwargs)


# Qwen2.5 Finetuning Configs
def qwen25_500m_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2.5 500M.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    return _qwen2_finetune_common(hf_path="Qwen/Qwen2.5-0.5B", **user_kwargs)


def qwen25_1p5b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2.5 1.5B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    return _qwen2_finetune_common(hf_path="Qwen/Qwen2.5-1.5B", **user_kwargs)


def qwen25_7b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2.5 7B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 2 if is_full_sft else 1

    return _qwen2_finetune_common(hf_path="Qwen/Qwen2.5-7B", **user_kwargs)


def qwen25_14b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2.5 14B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=4, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 4 if is_full_sft else 1

    return _qwen2_finetune_common(hf_path="Qwen/Qwen2.5-14B", **user_kwargs)


def qwen25_32b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2.5 32B.

    Default configuration: 2 nodes (SFT) or 1 node (LoRA), 8 GPUs per node
    - LoRA/DoRA: TP=8, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=2, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 8
    if "pipeline_model_parallel_size" not in user_kwargs:
        user_kwargs["pipeline_model_parallel_size"] = 2 if is_full_sft else 1

    return _qwen2_finetune_common(hf_path="Qwen/Qwen2.5-32B", **user_kwargs)


def qwen25_72b_finetune_config(**user_kwargs: Unpack[Qwen2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen2.5 72B.

    Default configuration: 4 nodes (SFT) or 1 node (LoRA), 8 GPUs per node
    - LoRA/DoRA: TP=8, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=4, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 8
    if "pipeline_model_parallel_size" not in user_kwargs:
        user_kwargs["pipeline_model_parallel_size"] = 4 if is_full_sft else 1

    return _qwen2_finetune_common(hf_path="Qwen/Qwen2.5-72B", **user_kwargs)


def _qwen2_finetune_common(
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
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 100,
    global_batch_size: int | None = None,
    micro_batch_size: int = 1,
    seq_length: int | None = None,
    eval_interval: int = 50,
    save_interval: int = 100,
    # Optimizer
    finetune_lr: float | None = None,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 10,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for all Qwen2/2.5 models."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select sequence length
    if seq_length is None:
        seq_length = 2048 if packed_sequence else 4096

    # Auto-select global_batch_size
    if global_batch_size is None:
        global_batch_size = 128

    # Auto-select learning rate
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4

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

    # Precision configuration
    if precision_config is None:
        precision_config = bf16_mixed()
    elif isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    # Optimizer and scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
    )

    # PEFT config
    peft_config = default_peft_config(peft) if not is_full_sft else None

    pad_seq_to_mult = context_parallel_size * 2 if packed_sequence and context_parallel_size > 1 else 1

    # Dataset config
    dataset_config = default_squad_config(seq_length, packed_sequence, pad_seq_to_mult)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Tokenizer
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    # DDP config
    ddp_cfg = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=False if is_full_sft else True,
        overlap_grad_reduce=True if is_full_sft else False,
        overlap_param_gather=True if is_full_sft else False,
        average_in_collective=True if is_full_sft else False,
        use_distributed_optimizer=True if is_full_sft else False,
    )

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=10,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
        dataset=dataset_config,
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
        mixed_precision=precision_config,
    )
