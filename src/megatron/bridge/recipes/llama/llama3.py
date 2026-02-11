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
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class Llama3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Llama3 family recipe helpers."""

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
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    adam_eps: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


class Llama3FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Llama3 finetuning recipe helper functions.

    This is separate from Llama3CommonKwargs to avoid confusion - finetuning
    uses SQuAD dataset by default, not the data path fields.
    """

    # Core identifiers
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


# Sequence length constants
SEQUENCE_LENGTH_16K: int = 16384
SEQUENCE_LENGTH_64K: int = 65536
SEQUENCE_LENGTH_128K: int = 131072


# =============================================================================
# Llama 3.2 Pretrain Configs
# =============================================================================


def llama32_1b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3.2 1B.

    Recommended parallelism: TP=1, PP=1, CP=1.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B").to_megatron_provider(load_weights=False)

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 8192

    # Training config
    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000

    # Logger config
    cfg.logger.log_timers_to_tensorboard = True

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
    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def llama32_3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3.2 3B.

    Recommended parallelism: TP=1, PP=1, CP=1.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-3B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 8192

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


# =============================================================================
# Llama 3 8B Pretrain Configs
# =============================================================================


def llama3_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B.

    Recommended parallelism: TP=1, PP=1, CP=2.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 8192

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def llama3_8b_16k_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B 16K.

    Recommended parallelism: TP=4, PP=2, CP=2, SP=True.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_16K

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_16K

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def llama3_8b_64k_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B 64K.

    Recommended parallelism: TP=4, PP=2, CP=4, SP=True.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_64K

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_64K

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def llama3_8b_128k_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 8B 128K.

    Recommended parallelism: TP=4, PP=2, CP=8, SP=True.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_128K

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_128K

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def llama3_8b_low_precision_pretrain_config(mixed_precision_recipe: str) -> ConfigContainer:
    """Return a low precision (FP8 Current Scaling/MXFP8/NVFP4) pre-training config for Llama 3 8B.

    Args:
        mixed_precision_recipe (str): The mixed precision recipe to use. Valid options are:
            - "bf16_with_mxfp8_mixed"
            - "bf16_with_fp8_current_scaling_mixed"
            - "bf16_with_nvfp4_mixed"

    Returns:
        ConfigContainer: The pre-training configuration for Llama 3 8B.
    """
    assert mixed_precision_recipe in [
        "bf16_with_mxfp8_mixed",
        "bf16_with_fp8_current_scaling_mixed",
        "bf16_with_nvfp4_mixed",
    ], f"Invalid low precision recipe: {mixed_precision_recipe}. This recipe has not been tested yet."

    precision_config = get_mixed_precision_config(mixed_precision_recipe)
    if mixed_precision_recipe == "bf16_with_nvfp4_mixed":
        # For llama3-8B nvfp4 recipe, we use BF16 for the last 4 layers
        precision_config.first_last_layers_bf16 = True
        precision_config.num_layers_at_start_in_bf16 = 0
        precision_config.num_layers_at_end_in_bf16 = 4

    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 8192

    # Low precision specific training params
    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Low precision specific optimizer params
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Low precision specific optimizer params
    cfg.optimizer.lr = 6e-4
    cfg.optimizer.min_lr = 6e-6
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Set the precision config
    cfg.mixed_precision = precision_config

    return cfg


# =============================================================================
# Llama 3 70B Pretrain Configs
# =============================================================================


def llama3_70b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B.

    Recommended parallelism: TP=4, PP=4, VPP=5, CP=2, SP=True with CommOverlap.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-70B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 8192

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Communication overlap for 70B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    )

    # Mixed precision - explicitly use bf16_mixed
    cfg.mixed_precision = bf16_mixed()

    return cfg


def llama3_70b_16k_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B 16K.

    Recommended parallelism: TP=8, PP=2, CP=2, SP=True with CommOverlap.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-70B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_16K

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_16K

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Communication overlap for 70B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    )

    cfg.mixed_precision = bf16_mixed()

    return cfg


def llama3_70b_64k_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3 70B 64K.

    Recommended parallelism: TP=8, PP=4, CP=8, SP=True with CommOverlap.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-70B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_64K

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_64K

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Communication overlap for 70B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    )

    cfg.mixed_precision = bf16_mixed()

    return cfg


# =============================================================================
# Llama 3.1 Pretrain Configs
# =============================================================================


def llama31_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3.1 8B.

    Recommended parallelism: TP=1, PP=1, CP=2.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-8B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 8192

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def llama31_70b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3.1 70B.

    Recommended parallelism: TP=4, PP=4, VPP=5, CP=2, SP=True with CommOverlap, seq=128K.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-70B").to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_128K

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_128K

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Communication overlap for 70B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    )

    cfg.mixed_precision = bf16_mixed()

    return cfg


def llama31_405b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Llama 3.1 405B.

    Recommended parallelism: TP=8, PP=8, VPP=2, CP=4, SP=True with CommOverlap, seq=128K.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3.1-405B").to_megatron_provider(
        load_weights=False
    )

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = SEQUENCE_LENGTH_128K

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = SEQUENCE_LENGTH_128K

    # Large model specific pipeline split configurations
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1  # 405B uses micro_batch_size=1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = True

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.checkpoint.save_interval = 500
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Communication overlap for 405B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    )

    cfg.mixed_precision = bf16_mixed()

    return cfg


# ============================================================================
# Finetuning Configurations
# ============================================================================


def llama32_1b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.2 1B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=1, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Llama-3.2-1B", **user_kwargs)

    # Model-specific parallelism settings
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.context_parallel_size = 1
    config.model.sequence_parallel = False

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False

    return config


def llama32_3b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.2 3B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=1, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Llama-3.2-3B", **user_kwargs)

    # Model-specific parallelism settings (match NeMo pattern)
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.context_parallel_size = 1
    config.model.sequence_parallel = False

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False

    return config


def llama3_8b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3 8B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=2, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    if "finetune_lr" not in user_kwargs:
        user_kwargs["finetune_lr"] = 5e-6 if is_full_sft else 1e-4

    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3-8B", **user_kwargs)

    # Parallelism settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 2
        config.model.pipeline_model_parallel_size = 1
        config.peft = None
    else:
        config.model.tensor_model_parallel_size = 1
        config.model.pipeline_model_parallel_size = 1
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
        else:
            config.peft = peft
        config.optimizer.use_distributed_optimizer = False
        config.model.cross_entropy_loss_fusion = False

    return config


def llama31_8b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.1 8B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=2, PP=1, LR=5e-6
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3.1-8B", **user_kwargs)

    # Parallelism settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 2
        config.model.pipeline_model_parallel_size = 1
        config.peft = None
    else:
        config.model.tensor_model_parallel_size = 1
        config.model.pipeline_model_parallel_size = 1
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
        else:
            config.peft = peft
        config.optimizer.use_distributed_optimizer = False
        config.model.cross_entropy_loss_fusion = False

    return config


def llama3_70b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3 70B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=8, PP=1, LR=1e-4, dim=16, alpha=32
    - Full SFT (peft=None): TP=8, PP=4, VPP=5, LR=5e-6 (requires 4 nodes)
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3-70B", **user_kwargs)

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 8
        config.model.pipeline_model_parallel_size = 4
        config.peft = None
    else:
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 16
            config.peft.alpha = 32
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False
        config.model.tensor_model_parallel_size = 8

    return config


def llama31_70b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.1 70B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=8, PP=1, LR=1e-4, dim=16, alpha=32
    - DoRA: TP=8, PP=1, LR=1e-4, dim=16, alpha=32
    - Full SFT (peft=None): TP=8, PP=4, VPP=5, LR=5e-6 (requires 4 nodes)
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3.1-70B", **user_kwargs)

    if is_full_sft:
        config.model.tensor_model_parallel_size = 8
        config.model.pipeline_model_parallel_size = 4
        config.peft = None
    else:
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 16
            config.peft.alpha = 32
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False
        config.model.tensor_model_parallel_size = 8

    return config


def llama31_405b_finetune_config(**user_kwargs: Unpack[Llama3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Llama 3.1 405B.

    Default configuration: 4 nodes (LoRA) or 16 nodes (Full SFT), 8 GPUs per node
    - LoRA (default): TP=4, PP=8, VPP=8, CP=1, LR=1e-4, dim=16, alpha=32, GBS=32, SP=True
      Total: 32 GPUs (4 nodes)
      Note: 128 effective layers ÷ 8 = 16 layers/rank, VPP=8 splits into 2 layers/virtual stage
    - DoRA: TP=4, PP=8, VPP=8, CP=1, LR=1e-4, dim=16, alpha=32, GBS=32, SP=True
      Total: 32 GPUs (4 nodes)
    - Full SFT (peft=None): TP=8, PP=16, VPP=None, CP=1, LR=5e-6, GBS=6, SP=True
      Total: 128 GPUs (16 nodes)
      Note: 128 effective layers ÷ 16 = 8 layers/rank
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    if "finetune_lr" not in user_kwargs:
        user_kwargs["finetune_lr"] = 5e-6 if is_full_sft else 1e-4

    if "global_batch_size" not in user_kwargs:
        user_kwargs["global_batch_size"] = 16 if is_full_sft else 32

    if "seq_length" not in user_kwargs:
        user_kwargs["seq_length"] = 2048

    config = _llama3_finetune_common(hf_path="meta-llama/Meta-Llama-3.1-405B", **user_kwargs)

    # Parallelism settings
    if is_full_sft:
        config.model.tensor_model_parallel_size = 8
        config.model.pipeline_model_parallel_size = 16
        config.model.virtual_pipeline_model_parallel_size = None
        config.model.sequence_parallel = True
        config.peft = None
        config.ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        )
        config.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=True, defer_embedding_wgrad_compute=True, wgrad_deferral_limit=22
        )
    else:
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 16
            config.peft.alpha = 32
            config.peft.target_modules = ["linear_qkv"]
        else:
            config.peft = peft

        config.optimizer.use_distributed_optimizer = False
        config.model.cross_entropy_loss_fusion = False
        config.model.tensor_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 8
        config.model.virtual_pipeline_model_parallel_size = 8
        config.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    config.mixed_precision = get_mixed_precision_config(config.mixed_precision)
    config.mixed_precision.grad_reduce_in_fp32 = False
    config.ddp.grad_reduce_in_fp32 = False
    config.model.sequence_parallel = True

    config.train.manual_gc = True
    config.train.manual_gc_interval = 100
    config.train.manual_gc_eval = 100

    config.optimizer.use_precision_aware_optimizer = False

    config.model.account_for_embedding_in_pipeline_split = True
    config.model.account_for_loss_in_pipeline_split = True

    return config


def _llama3_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int | None = None,
    micro_batch_size: int = 1,
    seq_length: int | None = None,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
) -> ConfigContainer:
    """Minimal common finetuning configuration.

    This function provides only the basic setup. Individual model configs handle parallelism settings
    depending on PEFT or full SFT.
    """

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select seq_length based on packed_sequence
    # For unpacked sequence, most samples in SQuAD dataset are shorter than 2K
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    # Auto-select global_batch_size based on packed_sequence
    if global_batch_size is None:
        global_batch_size = 8 if packed_sequence else 128

    # Create basic model config from HF
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.seq_length = seq_length

    # Basic optimizer configuration
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )
    ddp_cfg = DistributedDataParallelConfig(check_for_nan_in_grad=True)
    pad_seq_to_mult = (
        model_cfg.context_parallel_size * 2 if packed_sequence and model_cfg.context_parallel_size > 1 else 1
    )

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=32,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
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
        peft=None,
        comm_overlap=None,
        mixed_precision=precision_config,
    )
